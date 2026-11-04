"""Test degli adattatori dei dataset (righe -> record nel formato di prepare_data). Nessun download, nessun pyarrow."""
import json
import tempfile
import unittest
from pathlib import Path

import common
import sources

KB = common.load_kb()

VULN = "int f(char *d, char *s)\n{\n    strcpy(d, s);\n    return 0;\n}"
FIXED = "int f(char *d, char *s)\n{\n    strncpy(d, s, 15);\n    return 0;\n}"


def sv_row(idx, vulnerable, body, fixed_idx=float("nan"), **kw):
    row = {"idx": idx, "project": "linux", "project_url": "https://github.com/torvalds/linux", "filepath": "drivers/x.c",
           "commit_id": "c1", "commit_message": "[media] fix uaf\n\nlong body", "is_vulnerable": vulnerable, "func_name": "f",
           "func_body": body, "cve_list": ["CVE-2017-1"], "cwe_list": ["CWE-120", "NVD-CWE-noinfo"], "fixed_func_idx": fixed_idx}
    row.update(kw)
    return row


class SecVulEvalTests(unittest.TestCase):
    def test_vulnerable_and_fixed_rows_become_linked_records(self):
        recs = list(sources.secvuleval_records([sv_row(0, True, VULN, 1.0), sv_row(1, False, FIXED)]))
        vuln, fixed = recs
        self.assertEqual((vuln["target"], vuln["func"], vuln["counterpart"]), (1, VULN, FIXED))
        self.assertEqual((fixed["target"], fixed["func"], fixed["counterpart"]), (0, FIXED, VULN), "il link inverso si ricava dalla riga vulnerabile")
        self.assertEqual(vuln["cwe"], ["CWE-120"], "le etichette NVD-CWE-* non sono CWE")
        self.assertEqual((vuln["cve"], vuln["file_name"], vuln["commit_id"], vuln["lang"], vuln["source"]), ("CVE-2017-1", "drivers/x.c", "c1", "c", "secvuleval"))
        self.assertEqual(vuln["note"], "[media] fix uaf")

    def test_unlinked_functions_have_no_counterpart_and_extension_picks_the_language(self):
        recs = list(sources.secvuleval_records([sv_row(5, False, FIXED, filepath="src/a.cpp"), sv_row(6, True, VULN, filepath="src/b.cc")]))
        self.assertEqual([(r["counterpart"], r["lang"]) for r in recs], [(None, "cpp"), (None, "cpp")])

    def test_rows_without_a_function_body_are_skipped(self):
        self.assertEqual(list(sources.secvuleval_records([sv_row(1, True, "")])), [])


JULIET_BAD = (
    "#include <stdio.h>\n\n#ifndef OMITBAD\n\n"
    "void CWE690_NULL_Deref_From_Return__char_realloc_01_bad()\n{\n    char * data;\n    /* FLAW: no check */\n    strcpy(data, \"x\");\n}\n\n#endif\n"
    "int main(int argc, char * argv[])\n{\n    return 0;\n}\n"
)
JULIET_GOOD = (
    "static void goodG2B()\n{\n    char * data = malloc(10);\n    /* FIX: check first */\n    if (data) { free(data); }\n}\n\n"
    "void CWE690_NULL_Deref_From_Return__char_realloc_01_good()\n{\n    goodG2B();\n}\n"
)


class JulietTests(unittest.TestCase):
    ROW = {"filename": "testcases/CWE690_NULL_Deref_From_Return/s01/CWE690_NULL_Deref_From_Return__char_realloc_01.c", "bad": JULIET_BAD, "good": JULIET_GOOD}

    def test_flawed_function_and_its_good_variant_are_extracted(self):
        recs = list(sources.juliet_records([self.ROW]))
        bad = [r for r in recs if r["target"] == 1]
        good = [r for r in recs if r["target"] == 0]
        self.assertEqual(len(bad), 1)
        self.assertEqual(len(good), 1, "il wrapper good() non contiene né FIX né sink: non è un esempio")
        b = bad[0]
        self.assertEqual((b["cwe"], b["kind"], b["source"], b["synthetic"]), (["CWE-690"], "vulnerable", "juliet", True))
        self.assertEqual(b["sink_hint"], {"callee": "strcpy", "line": 5})
        self.assertTrue(b["func"].startswith("void CWE690_NULL_Deref_From_Return__char_realloc_01_bad()"))
        g = good[0]
        self.assertEqual((g["kind"], g["sink_hint"]["callee"]), ("patched", "free"))
        self.assertTrue(g["func"].startswith("static void goodG2B()"))
        self.assertEqual(b["commit_id"], g["commit_id"], "stesso gruppo: le due versioni finiscono nello stesso split")

    def test_good_function_without_fix_comment_still_gives_a_negative_when_it_uses_the_cwe_sink(self):
        good = "static void goodB2G()\n{\n    char buf[10];\n    strncpy(buf, \"ok\", 9);\n}\n\nvoid CWE121_x_01_good()\n{\n    goodB2G();\n}\n"
        row = {"filename": "testcases/CWE121_Stack/s01/CWE121_x_01.c", "bad": "", "good": good}
        recs = list(sources.juliet_records([row], kb=KB))
        self.assertEqual([(r["target"], r["kind"], r["sink_hint"]) for r in recs], [(0, "patched", {"callee": "strncpy", "line": 4})])
        self.assertEqual(list(sources.juliet_records([row])), [], "senza KB si usano solo i commenti FIX")

    def test_functions_without_a_flaw_comment_and_main_are_not_examples(self):
        row = dict(self.ROW, bad="void helper()\n{\n    run();\n}\nint main()\n{\n    return 0;\n}\n", good="")
        self.assertEqual(list(sources.juliet_records([row])), [])

    def test_per_cwe_cap_keeps_the_first_cases_and_their_good_twins(self):
        rows = [dict(self.ROW, filename=f"testcases/CWE690_x/s01/CWE690_x_{i:02d}.c") for i in range(5)]
        recs = list(sources.juliet_records(rows, per_cwe=2))
        self.assertEqual(len({r["commit_id"] for r in recs}), 2)


def py_row(**kw):
    row = {"source_dataset": "secureIT-project/CVEfixes", "cve_id": "CVE-2021-1", "cve_description": "SQL injection in q()", "cwe_id": "CWE-89",
           "function_name": "q", "code_before": "def q(cur, uid):\n    cur.execute(f\"SELECT * FROM t WHERE id={uid}\")\n",
           "code_after": "def q(cur, uid):\n    cur.execute(\"SELECT * FROM t WHERE id=%s\", (uid,))\n", "commit_hash": "abc", "repo_url": "https://github.com/o/r",
           "programming_language": "python"}
    row.update(kw)
    return row


class PythonPairTests(unittest.TestCase):
    def test_pair_becomes_vulnerable_and_fixed_records(self):
        vuln, fixed = list(sources.python_pair_records([py_row()], KB))
        self.assertEqual((vuln["target"], vuln["cwe"], vuln["lang"], vuln["source"]), (1, ["CWE-89"], "python", "cvefixes-python"))
        self.assertEqual((vuln["func"], vuln["counterpart"]), (py_row()["code_before"], py_row()["code_after"]))
        self.assertEqual((fixed["target"], fixed["func"], fixed["counterpart"]), (0, py_row()["code_after"], py_row()["code_before"]))
        self.assertEqual((vuln["commit_id"], vuln["project"], vuln["cve"]), ("abc", "r", "CVE-2021-1"))
        self.assertEqual(vuln["commit_id"], fixed["commit_id"])

    def test_missing_cwe_is_inferred_from_the_sink_the_fix_touches(self):
        recs = list(sources.python_pair_records([py_row(cwe_id=None, source_dataset="billquan/PyVul")], KB))
        self.assertEqual([r["cwe"] for r in recs], [["CWE-89"], ["CWE-89"]])

    def test_nvd_placeholder_cwe_counts_as_missing(self):
        recs = list(sources.python_pair_records([py_row(cwe_id="NVD-CWE-noinfo")], KB))
        self.assertEqual(recs[0]["cwe"], ["CWE-89"])

    def test_pair_without_label_or_localizable_sink_is_dropped(self):
        row = py_row(cwe_id=None, code_before="def f(x):\n    return x + 1\n", code_after="def f(x):\n    return x + 2\n")
        self.assertEqual(list(sources.python_pair_records([row], KB)), [])

    def test_row_without_fixed_code_yields_only_the_vulnerable_side(self):
        recs = list(sources.python_pair_records([py_row(code_after=None)], KB))
        self.assertEqual([(r["target"], r["counterpart"]) for r in recs], [(1, None)])


CODEQL_PY = """from flask import Flask, request
app = Flask(__name__)

@app.route("/users/<username>")
def show_user(username): # $ Source
    with connection.cursor() as cursor:
        # GOOD -- Using parameters
        cursor.execute("SELECT * FROM users WHERE username = %s", username)

        # BAD -- Using string formatting
        cursor.execute("SELECT * FROM users WHERE username = '%s'" % username) # $ Alert

def helper():
    cursor.execute("SELECT 1") # $ MISSING: Alert
"""

CODEQL_C = """void bad_copy(char *s) {
    char buf[10];
    strcpy(buf, s); // $ Alert
}

void good_copy(char *s) {
    char buf[10];
    strncpy(buf, s, 9); // GOOD
}
"""


class CodeqlTests(unittest.TestCase):
    PY = "python/ql/test/query-tests/Security/CWE-089-SqlInjection/sql_injection.py"
    C = "cpp/ql/test/query-tests/Security/CWE/CWE-120/test.c"

    def records(self, path, text):
        return list(sources.codeql_records([(path, text)], KB))

    def test_each_sink_in_a_function_gets_its_own_verdict(self):
        recs = self.records(self.PY, CODEQL_PY)
        self.assertEqual(len(recs), 2, "helper() ha un'annotazione MISSING: ambigua, esclusa")
        good, bad = sorted(recs, key=lambda r: r["target"])
        self.assertEqual((bad["target"], bad["kind"], bad["cwe"], bad["lang"], bad["source"], bad["synthetic"]),
                         (1, "vulnerable", ["CWE-89"], "python", "codeql-tests", True))
        self.assertEqual((good["target"], good["kind"]), (0, "patched"))
        self.assertEqual(bad["sink_hint"]["callee"], "cursor.execute")
        self.assertEqual(good["sink_hint"]["callee"], "cursor.execute")
        self.assertNotEqual(bad["sink_hint"]["line"], good["sink_hint"]["line"])
        self.assertEqual(bad["func"], good["func"], "stessa funzione, sink diverso")

    def test_label_comments_never_reach_the_prompt_and_hint_lines_still_point_at_the_calls(self):
        bad = max(self.records(self.PY, CODEQL_PY), key=lambda r: r["target"])
        for leak in ("$ Alert", "GOOD", "BAD", "$ Source"):
            self.assertNotIn(leak, bad["func"])
        line = bad["func"].split("\n")[bad["sink_hint"]["line"] - 1]
        self.assertIn("% username", line)

    def test_c_functions_get_their_own_verdicts_from_the_cwe_directory(self):
        recs = self.records(self.C, CODEQL_C)
        by_name = {localize_name(r["func"]): r for r in recs}
        self.assertEqual(set(by_name), {"bad_copy", "good_copy"})
        self.assertEqual((by_name["bad_copy"]["target"], by_name["bad_copy"]["sink_hint"]["callee"], by_name["bad_copy"]["cwe"]), (1, "strcpy", ["CWE-120"]))
        self.assertEqual((by_name["good_copy"]["target"], by_name["good_copy"]["sink_hint"]["callee"]), (0, "strncpy"))
        self.assertEqual(by_name["bad_copy"]["lang"], "c")

    def test_python2_directories_and_unknown_cwe_directories_are_skipped(self):
        self.assertEqual(self.records("python/ql/test/query-tests/Security/CWE-078-CommandInjection-py2/a.py", CODEQL_PY), [])
        self.assertEqual(self.records("python/ql/test/query-tests/Security/NoCwe/a.py", CODEQL_PY), [])

    def test_group_is_the_file_so_all_its_functions_land_in_one_split(self):
        recs = self.records(self.PY, CODEQL_PY)
        self.assertEqual(len({r["commit_id"] for r in recs}), 1)


def localize_name(code):
    import localize

    return localize.extract_function_name(code)


class ReaderTests(unittest.TestCase):
    def test_jsonl_and_csv_are_read_without_pyarrow(self):
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "rows.jsonl"
            j.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
            self.assertEqual([r["a"] for r in sources.read_rows(j)], [1, 2])
            self.assertEqual(len(list(sources.read_rows(j, limit=1))), 1)
            c = Path(d) / "rows.csv"
            c.write_text('filename,bad,good\nx.c,"int a;\nint b;",ok\n', encoding="utf-8")
            self.assertEqual(list(sources.read_rows(c))[0]["bad"], "int a;\nint b;")

    def test_unknown_extension_is_an_error(self):
        with self.assertRaises(ValueError):
            list(sources.read_rows(Path("/tmp/rows.xyz")))


if __name__ == "__main__":
    unittest.main()
