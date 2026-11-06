"""Test di prepare_data.py con record PrimeVul sintetici (nessun download)."""
import contextlib
import io
import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import common
import prepare_data as pd
from c_anonymize import LEAK_RE

KB = common.load_kb()

VULN_FUNC = """static int copy_name(char *dst, const char *src)
{
    strcpy(dst, src);
    return helper(dst);
}"""

FIXED_FUNC = """static int copy_name(char *dst, const char *src)
{
    snprintf(dst, 16, "%s", src);
    return helper(dst);
}"""


def record(**kw):
    base = {
        "idx": 1,
        "project": "proj",
        "commit_id": "c1",
        "target": 1,
        "func": VULN_FUNC,
        "file_name": "a.c",
        "cwe": ["CWE-787"],
        "cve": "CVE-2020-1",
        "cve_desc": "Buffer overflow in copy_name",
    }
    base.update(kw)
    return base


class ParseTests(unittest.TestCase):
    def test_parse_cwes_accepts_lists_and_string_reprs(self):
        self.assertEqual(pd.parse_cwes(["CWE-787", "CWE-20"]), ["CWE-787", "CWE-20"])
        self.assertEqual(pd.parse_cwes("['CWE-704']"), ["CWE-704"])
        self.assertEqual(pd.parse_cwes("None"), [])
        self.assertEqual(pd.parse_cwes(None), [])
        self.assertEqual(pd.parse_cwes([]), [])

    def test_function_name_extraction(self):
        cases = {
            VULN_FUNC: "copy_name",
            "void Foo::bar(int n)\n{\n}": "bar",
            "char *dup_str(const char *s)\n{\n return 0;\n}": "dup_str",
            "MODULE_EXPORT int\nmulti_line(int a,\n           int b)\n{\n}": "multi_line",
            "garbage without parens": "unknown",
        }
        for code, expected in cases.items():
            with self.subTest(expected):
                self.assertEqual(pd.extract_function_name(code), expected)

    def test_language_guess(self):
        self.assertEqual(pd.guess_lang("a.c", "int f(){}"), "c")
        self.assertEqual(pd.guess_lang("a.cc", "int f(){}"), "cpp")
        self.assertEqual(pd.guess_lang("x.hpp", "int f(){}"), "cpp")
        self.assertEqual(pd.guess_lang("None", "void A::f() {}"), "cpp")
        self.assertEqual(pd.guess_lang("None", "int f() {}"), "c")
        self.assertEqual(pd.guess_lang("app/db.py", "def f(): pass"), "python")
        self.assertEqual(pd.guess_lang("None", "def f(a):\n    return a"), "python")


class CallDetectionTests(unittest.TestCase):
    def test_comments_strings_and_keywords_are_ignored(self):
        code = (
            "int f(char *s)\n"
            "{\n"
            "    /* strcpy(a, b); */\n"
            '    puts("memcpy(x)");\n'
            "    // system(cmd);\n"
            "    if (s) {\n"
            "        strcpy(buf, s);\n"
            "    }\n"
            "    return sizeof(buf);\n"
            "}\n"
        )
        calls = pd.find_calls(code)
        self.assertEqual([c for c, _ in calls], ["puts", "strcpy"])
        self.assertEqual(dict(calls)["strcpy"], 7)

    def test_member_and_namespace_callees_keep_their_path(self):
        calls = pd.find_calls("void f() {\n  ctx->run(1);\n  std::memcpy(a, b, n);\n  obj.close();\n}")
        self.assertEqual([c for c, _ in calls], ["ctx->run", "std::memcpy", "obj.close"])

    def test_the_function_own_header_is_not_a_call(self):
        self.assertEqual(pd.find_calls("int helper(char *p) { return p[0]; }"), [])

    def test_detect_sink_honours_the_preferred_cwe(self):
        code = "void f() {\n  char *p = malloc(n);\n  strcpy(p, s);\n}"
        first = pd.detect_sink(KB, code, "c")
        self.assertEqual((first["entry"]["id"], first["callee"], first["line"]), ("CWE-190", "malloc", 2))
        wanted = pd.detect_sink(KB, code, "c", preferred_cwe="CWE-120")
        self.assertEqual((wanted["entry"]["id"], wanted["callee"], wanted["line"]), ("CWE-120", "strcpy", 3))
        self.assertIsNone(pd.detect_sink(KB, code, "c", preferred_cwe="CWE-78"))


class BuildExampleTests(unittest.TestCase):
    def build(self, rec, **kw):
        return pd.build_example(rec, KB, patched={}, max_chars=12000, **kw)

    def test_vulnerable_record_becomes_a_validated_training_example(self):
        ex, reason = self.build(record())
        self.assertIsNone(reason)
        roles = [m["role"] for m in ex["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(ex["messages"][0]["content"], common.system_prompt())

        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual(common.validate_finding(finding), [])
        self.assertEqual(
            (finding["vulnerable"], finding["cwe_id"], finding["severity"], finding["sink_function"], finding["sink_file"], finding["sink_line"], finding["vex_status"]),
            (True, "CWE-120", "HIGH", "strcpy", "a.c", 3, "affected"),
        )
        entry = next(e for e in KB["entries"] if e["id"] == "CWE-120")
        self.assertEqual(finding["reasoning"], f"The sink `strcpy` at line 3 of a.c receives unvalidated data and is vulnerable to CWE-120 ({entry['name']}).")
        self.assertEqual(finding["suggested_fix"], entry["fix_hint"])

        user = ex["messages"][1]["content"]
        self.assertIn("CWE-120 — Buffer Copy without Checking Size of Input", user)
        self.assertIn("- Function: strcpy", user)
        self.assertIn("- Line: 3", user)
        self.assertIn("### a.c · copy_name (lines 1-5, depth 0)", user)
        self.assertIn("    3 |     strcpy(dst, src);", user)
        self.assertEqual(ex["meta"]["label"], 1)
        self.assertEqual(ex["meta"]["kind"], "vulnerable")

    def test_assistant_json_keeps_the_schema_field_order(self):
        ex, _ = self.build(record())
        self.assertEqual(list(json.loads(ex["messages"][2]["content"]).keys()), list(common.load_schema()["properties"].keys()))

    def test_patched_function_is_a_hard_negative(self):
        rec = record(target=0, func=FIXED_FUNC.replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)"))
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["vulnerable"], finding["severity"], finding["vex_status"], finding["cwe_id"]), (False, "NONE", "not_affected", "CWE-120"))
        self.assertEqual(finding["suggested_fix"], "No change needed.")
        self.assertEqual(ex["meta"]["kind"], "patched")
        self.assertEqual(common.validate_finding(finding), [])

    def test_plain_benign_function_uses_the_first_matching_sink(self):
        rec = record(target=0, cwe=[], cve="None", cve_desc="None", commit_id="None", func="void g(char *p) {\n  free(p);\n}")
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-416", "free", 2))
        self.assertEqual(ex["meta"]["kind"], "benign")

    def test_vulnerable_suggested_fix_is_catalog_hint_not_upstream_diff(self):
        """Anche con la versione corretta nota (mappa `patched` o `counterpart`) il fix è il suggerimento del catalogo."""
        hint = next(e for e in KB["entries"] if e["id"] == "CWE-120")["fix_hint"]
        patched = {("c1", "copy_name"): FIXED_FUNC}
        for name, ex in (("patched map", pd.build_example(record(), KB, patched=patched, max_chars=12000)[0]),
                         ("counterpart", pd.build_example(record(counterpart=FIXED_FUNC), KB, patched={}, max_chars=12000)[0])):
            with self.subTest(name):
                fix = json.loads(ex["messages"][2]["content"])["suggested_fix"]
                self.assertEqual(fix, hint)
                for diff_line in ("@@", "upstream", "-    strcpy(dst, src);", '+    snprintf(dst, 16, "%s", src);'):
                    self.assertNotIn(diff_line, fix)

    def test_skip_reasons(self):
        cases = [
            (record(cwe=["CWE-703"]), "cwe_not_in_kb"),
            (record(func="void f() {\n  helper();\n}"), "no_sink"),
            (record(func="void f() {\n  strcpy(a, b);\n" + "  x = 1;\n" * 5000 + "}"), "too_long"),
            (record(func=""), "empty_function"),
            (record(target=0, cwe=[], cve="None", func="void f() { helper(); }"), "no_sink"),
        ]
        for rec, expected in cases:
            with self.subTest(expected):
                ex, reason = self.build(rec)
                self.assertIsNone(ex)
                self.assertEqual(reason, expected)

    def test_sinkless_function_mode_uses_the_function_itself_as_sink(self):
        rec = record(func="void f(struct a *p) {\n  p->x = 1;\n}", cwe=["CWE-476"])
        ex, reason = pd.build_example(rec, KB, patched={}, max_chars=12000, sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"], finding["vulnerable"]), ("CWE-476", "f", 1, True))
        self.assertEqual(ex["meta"]["sink_kind"], "function")
        self.assertIn("- Function: f", ex["messages"][1]["content"])
        self.assertEqual(common.validate_finding(finding), [])

    def test_call_sinks_are_tagged_and_sinkless_mode_does_not_change_them(self):
        ex, _ = pd.build_example(record(), KB, patched={}, max_chars=12000, sinkless="function")
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["sink_function"]), ("call", "strcpy"))

    def test_sinkless_mode_still_needs_a_known_cwe_and_never_applies_to_unlabeled_code(self):
        unknown = record(func="void f() {\n  helper();\n}", cwe=["CWE-9999"])
        self.assertEqual(pd.build_example(unknown, KB, patched={}, max_chars=12000, sinkless="function")[1], "cwe_not_in_kb")
        unlabeled = record(target=0, cwe=[], cve="None", commit_id="None", func="void f() { helper(); }")
        self.assertEqual(pd.build_example(unlabeled, KB, patched={}, max_chars=12000, sinkless="function")[1], "no_sink")

    # ---- sink localizzati --------------------------------------------------------------------------------------
    def test_diff_localized_sink_beats_an_earlier_call_on_an_unchanged_line(self):
        vuln = "static int f(char *d, char *s)\n{\n    memcpy(tmp, s, 4);\n    strcpy(d, s);\n    return 0;\n}"
        fixed = vuln.replace("strcpy(d, s)", "strncpy(d, s, 15)")
        ex, reason = self.build(record(func=vuln, counterpart=fixed), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["sink_function"], finding["sink_line"]), ("strcpy", 4))
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["localized"]), ("call", "diff"))

    def test_function_level_sink_points_at_the_first_changed_line(self):
        vuln = "int f(struct a *p)\n{\n    int x = p->v;\n    use(p->w);\n    return x;\n}"
        fixed = vuln.replace("    use(p->w);", "    if (!p)\n        return -1;\n    use(p->w);")
        ex, reason = self.build(record(func=vuln, counterpart=fixed, cwe=["CWE-476"]), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["sink_function"], finding["sink_line"]), ("f", 4))
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["localized"]), ("function", "diff"))

    def test_with_a_counterpart_unrelated_call_sinks_are_never_promoted_to_the_sink(self):
        vuln = "int f(char *d, char *s)\n{\n    memcpy(d, s, 4);\n    int y = 1;\n    return y;\n}"
        fixed = vuln.replace("int y = 1;", "int y = 2;")
        self.assertEqual(self.build(record(func=vuln, counterpart=fixed), sinkless="skip")[1], "no_sink")
        ex, _ = self.build(record(func=vuln, counterpart=fixed), sinkless="function")
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["sink_line"]), ("function", 4))

    def test_negative_twin_is_localized_on_its_own_changed_line(self):
        fixed = FIXED_FUNC.replace('snprintf(dst, 16, "%s", src)', "strncpy(dst, src, 15)")
        ex, reason = self.build(record(target=0, func=fixed, counterpart=VULN_FUNC), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["vulnerable"], finding["sink_function"], finding["sink_line"]), (False, "strncpy", 3))
        self.assertEqual(ex["meta"]["kind"], "patched")

    def test_the_vulnerable_lookup_provides_the_counterpart_of_a_patched_function(self):
        fixed = FIXED_FUNC.replace('snprintf(dst, 16, "%s", src)', "strncpy(dst, src, 15)")
        ex, _ = pd.build_example(record(target=0, func=fixed), KB, patched={}, max_chars=12000, sinkless="function",
                                 vulnerable_map={("c1", "copy_name"): VULN_FUNC})
        self.assertEqual((ex["meta"]["sink_function"], ex["meta"]["localized"]), ("strncpy", "diff"))

    JULIET_BAD = ("void CWE690_x_bad()\n{\n    char * data;\n    /* FLAW: no check */\n    strcpy(data, \"x\");\n    free(data);\n}")

    def test_juliet_flaw_hint_is_the_sink_and_the_example_is_marked_synthetic(self):
        rec = {"idx": "j1", "project": "juliet", "commit_id": "None", "target": 1, "func": self.JULIET_BAD, "file_name": "CWE690_x_01.c",
               "cwe": ["CWE-690"], "cve": "None", "sink_hint": {"callee": "strcpy", "line": 5}, "source": "juliet", "synthetic": True, "kind": "vulnerable"}
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-476", "strcpy", 5))
        self.assertEqual((ex["meta"]["localized"], ex["meta"]["source"], ex["meta"]["synthetic"]), ("comment", "juliet", True))

    def test_explicit_kind_marks_juliet_good_functions_as_hard_negatives(self):
        good = "static void goodG2B()\n{\n    char *p = malloc(10);\n    if (p) free(p);\n}"
        rec = {"idx": "j2", "project": "juliet", "commit_id": "None", "target": 0, "func": good, "cwe": ["CWE-416"], "cve": "None",
               "kind": "patched", "sink_hint": {"callee": "free", "line": 4}, "source": "juliet", "synthetic": True}
        ex, _ = self.build(rec)
        self.assertEqual(ex["meta"]["kind"], "patched")

    def test_python_records_use_python_call_detection(self):
        vuln = "def q(cur, uid):\n    cur.execute(f\"SELECT * FROM t WHERE id={uid}\")\n    return cur.fetchall()"
        fixed = vuln.replace('f"SELECT * FROM t WHERE id={uid}"', '"SELECT * FROM t WHERE id=%s", (uid,)')
        ex, reason = self.build(record(func=vuln, counterpart=fixed, cwe=["CWE-89"], file_name="db.py"), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-89", "cur.execute", 2))
        self.assertEqual(ex["meta"]["lang"], "python")
        self.assertIn("```python", ex["messages"][1]["content"])

    def test_record_source_defaults_to_primevul_and_is_not_synthetic(self):
        ex, _ = self.build(record())
        self.assertEqual((ex["meta"]["source"], ex["meta"]["synthetic"]), ("primevul", False))

    def test_vulnerable_reasoning_has_no_cve_id_description_or_commit_note(self):
        rec = record(cve="CVE-2020-1", cve_desc="boom", note="fix x")
        for name, kw in (("with counterpart", {"counterpart": FIXED_FUNC}), ("without", {})):
            with self.subTest(name):
                ex, _ = self.build({**rec, **kw})
                assistant = ex["messages"][2]["content"]
                self.assertEqual(re.findall(r"CVE-\d{4}-\d+", assistant), [])
                for leaked in ("CVE-", "boom", "fix x", "Description", "Upstream"):
                    self.assertNotIn(leaked, assistant)
                self.assertEqual(ex["meta"]["cve"], "CVE-2020-1", "il CVE resta nei metadati, non nel testo del target")

    def test_safe_reasoning_has_no_cve_note_or_label_words(self):
        """Patch e benigni: stessa frase per entrambi, nessun materiale CVE, nessuna parola-spia (`fix`, `good`, ...)."""
        fixed = FIXED_FUNC.replace('snprintf(dst, 16, "%s", src)', "strncpy(dst, src, 15)")
        patched = record(target=0, func=fixed, cve="CVE-2020-1", cve_desc="boom", note="fix x", counterpart=VULN_FUNC)
        benign = record(target=0, cwe=[], cve="None", cve_desc="None", commit_id="None", func="void g(char *p) {\n  free(p);\n}")
        entry120 = next(e for e in KB["entries"] if e["id"] == "CWE-120")
        entry416 = next(e for e in KB["entries"] if e["id"] == "CWE-416")
        for rec, kind, (callee, line), entry in ((patched, "patched", ("strncpy", 3), entry120), (benign, "benign", ("free", 2), entry416)):
            with self.subTest(kind):
                ex, reason = self.build(rec, sinkless="function")
                self.assertIsNone(reason)
                finding = json.loads(ex["messages"][2]["content"])
                self.assertEqual(ex["meta"]["kind"], kind)
                self.assertEqual(finding["reasoning"], f"The sink `{callee}` at line {line} of a.c does not receive untrusted data: the value reaching it is "
                                 f"constant or checked before use, so it is not vulnerable to {entry['id']} ({entry['name']}).")
                self.assertEqual(finding["suggested_fix"], pd.NO_FIX)
                for leaked in ("CVE-", "boom", "fix x", "fixed version"):
                    self.assertNotIn(leaked, finding["reasoning"])

    def test_no_reasoning_for_any_catalog_entry_contains_a_label_word(self):
        """`_reasoning` non deve dire «fixed», «good», ...: sarebbe la risposta, non una prova."""
        sink = {"callee": "f", "line": 1}
        for entry in KB["entries"]:
            for kind in ("vulnerable", "patched", "benign"):
                with self.subTest(entry["id"], kind=kind):
                    text = pd._reasoning(kind, sink, "a.c", entry)
                    if kind != "vulnerable":
                        self.assertIsNone(LEAK_RE.search(text), text)
                    self.assertIsNone(re.search(r"CVE-\d", text))

    def test_missing_file_name_gets_a_stable_label(self):
        ex, _ = self.build(record(file_name="None"))
        self.assertEqual(json.loads(ex["messages"][2]["content"])["sink_file"], "proj/copy_name.c")


class SplitAndBalanceTests(unittest.TestCase):
    def test_hash_split_is_deterministic_and_groups_by_commit(self):
        self.assertEqual(pd.is_validation("c1", 10, seed=1), pd.is_validation("c1", 10, seed=1))
        share = sum(pd.is_validation(f"commit{i}", 10, seed=1) for i in range(2000)) / 2000
        self.assertTrue(0.06 < share < 0.14, share)

    def test_negative_plan_prefers_hard_negatives_but_respects_the_ratio(self):
        # (vulnerabili, patch, benigni disponibili, rapporto) -> (patch tenute, benigni tenuti)
        self.assertEqual(pd.negative_plan(10, 4, 100, 1.0), (4, 6))
        self.assertEqual(pd.negative_plan(10, 14, 100, 1.0), (10, 0))
        self.assertEqual(pd.negative_plan(10, 0, 100, 2.5), (0, 25))
        self.assertEqual(pd.negative_plan(10, 4, 3, 1.0), (4, 3), "non si inventano esempi che non ci sono")
        self.assertEqual(pd.negative_plan(0, 5, 5, 1.0), (0, 0))

    def test_reservoir_selection_is_seeded_and_exact(self):
        rng = random.Random(7)
        picked = pd.choose_indices(1000, 50, seed=7)
        self.assertEqual(len(picked), 50)
        self.assertEqual(picked, pd.choose_indices(1000, 50, seed=7))
        self.assertNotEqual(picked, pd.choose_indices(1000, 50, seed=8))
        self.assertEqual(len(pd.choose_indices(10, 50, seed=1)), 10)
        del rng


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        def rows(prefix, n_vuln, n_benign):
            out = []
            for i in range(n_vuln):
                name = f"copy_{prefix}{i}"
                out.append(record(idx=f"{prefix}v{i}", commit_id=f"{prefix}c{i}", func=VULN_FUNC.replace("copy_name", name)))
                out.append(record(idx=f"{prefix}p{i}", commit_id=f"{prefix}c{i}", target=0,
                                  func=FIXED_FUNC.replace("copy_name", name).replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")))
            for i in range(n_benign):
                out.append(record(idx=f"{prefix}b{i}", commit_id="None", target=0, cwe=[], cve="None", cve_desc="None", func=f"void g{i}(char *p) {{\n  free(p);\n}}"))
            out.append(record(idx=f"{prefix}x", target=0, cwe=[], cve="None", commit_id="None", func="void nosink() { helper(); }"))
            return out

        write_jsonl(root / "train.jsonl", rows("t", 6, 30))
        write_jsonl(root / "valid.jsonl", rows("v", 2, 10))
        write_jsonl(root / "test.jsonl", rows("e", 3, 20))

    def run_main(self, raw: Path, out: Path):
        return pd.main([
            "--train", str(raw / "train.jsonl"),
            "--valid", str(raw / "valid.jsonl"),
            "--test", str(raw / "test.jsonl"),
            "--out-dir", str(out),
            "--seed", "3",
        ])

    def test_pipeline_writes_balanced_splits_and_stats(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            self.make_inputs(raw)
            self.assertEqual(self.run_main(raw, out), 0)

            for name, n_vuln in (("train", 6), ("val", 2), ("test", 3)):
                lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                examples = [json.loads(l) for l in lines]
                labels = [e["meta"]["label"] for e in examples]
                self.assertEqual(sum(labels), n_vuln, name)
                self.assertEqual(len(labels) - sum(labels), n_vuln, f"{name}: classi bilanciate (ratio 1.0)")
                kinds = {e["meta"]["kind"] for e in examples}
                self.assertIn("patched", kinds)
                for e in examples:
                    self.assertEqual(common.validate_finding(json.loads(e["messages"][2]["content"])), [])
            stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train"]["kept"]["vulnerable"], 6)
            self.assertEqual(stats["train"]["skipped"]["no_sink"], 1)
            self.assertFalse((out / ".benign_tmp.jsonl").exists())

    def test_benign_ratio_adds_plain_negatives_on_top_of_patched_ones(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            self.make_inputs(raw)
            pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "valid.jsonl"), "--out-dir", str(out), "--seed", "3", "--benign-ratio", "2", "--label-ratio", "0"])
            examples = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            kinds = [e["meta"]["kind"] for e in examples]
            self.assertEqual((kinds.count("vulnerable"), kinds.count("patched"), kinds.count("benign")), (6, 6, 6))

    def test_patched_negatives_are_downsampled_to_the_requested_ratio(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            fixed = FIXED_FUNC.replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")
            rows = [record(idx="v0", commit_id="c0", func=VULN_FUNC), record(idx="v1", commit_id="c1", func=VULN_FUNC.replace("copy_name", "copy_two"))]
            rows += [record(idx=f"p{i}", commit_id="c0", target=0, func=fixed.replace("copy_name", f"other_{i}")) for i in range(6)]
            write_jsonl(raw / "train.jsonl", rows)
            pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "train.jsonl"), "--out-dir", str(out), "--benign-ratio", "1"])
            kinds = [json.loads(l)["meta"]["kind"] for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual((kinds.count("vulnerable"), kinds.count("patched")), (2, 2))

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as o1, tempfile.TemporaryDirectory() as o2:
            raw = Path(raw_dir)
            self.make_inputs(raw)
            self.run_main(raw, Path(o1))
            self.run_main(raw, Path(o2))
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((Path(o1) / name).read_bytes(), (Path(o2) / name).read_bytes(), name)

    def test_without_a_valid_file_the_validation_set_is_carved_from_train_by_commit(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            rows = []
            for i in range(40):  # abbastanza commit perché entrambi gli split risultino popolati
                rows.append(record(idx=f"v{i}", commit_id=f"c{i}", func=VULN_FUNC.replace("copy_name", f"copy_{i}")))
                rows.append(record(idx=f"p{i}", commit_id=f"c{i}", target=0,
                                   func=FIXED_FUNC.replace("copy_name", f"copy_{i}").replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")))
            write_jsonl(raw / "train.jsonl", rows)
            code = pd.main(["--train", str(raw / "train.jsonl"), "--out-dir", str(out), "--seed", "3", "--val-pct", "50"])
            self.assertEqual(code, 0)
            train_commits = {json.loads(l)["meta"]["commit_id"] for l in (out / "train.jsonl").read_text().splitlines()}
            val_commits = {json.loads(l)["meta"]["commit_id"] for l in (out / "val.jsonl").read_text().splitlines()}
            self.assertTrue(val_commits)
            self.assertFalse((train_commits - {"None"}) & (val_commits - {"None"}), "una commit non può stare in entrambi gli split")
            self.assertFalse((out / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()


# ======================================================================================================== multi-sorgente
import collections  # noqa: E402

import sources  # noqa: E402


class SplitAndDedupTests(unittest.TestCase):
    def test_split_of_is_deterministic_and_respects_the_percentages(self):
        counts = collections.Counter(pd.split_of(f"g{i}", 1, 10, 20) for i in range(4000))
        self.assertTrue(0.16 < counts["test"] / 4000 < 0.24, counts)
        self.assertTrue(0.07 < counts["val"] / 4000 < 0.13, counts)
        self.assertEqual(pd.split_of("g1", 1, 10, 20), pd.split_of("g1", 1, 10, 20))

    def test_without_a_test_share_nothing_goes_to_test(self):
        self.assertNotIn("test", {pd.split_of(f"g{i}", 1, 10, 0) for i in range(2000)})

    def test_func_hash_ignores_whitespace(self):
        self.assertEqual(pd.func_hash("a  b\n  c"), pd.func_hash("a b c"))
        self.assertNotEqual(pd.func_hash("a b"), pd.func_hash("a c"))

    def test_examples_carry_the_hashes_of_function_and_counterpart(self):
        ex, _ = pd.build_example(record(counterpart=FIXED_FUNC), KB, patched={}, max_chars=12000, sinkless="function")
        self.assertEqual(ex["meta"]["func_hash"], pd.func_hash(VULN_FUNC))
        self.assertEqual(ex["meta"]["counterpart_hash"], pd.func_hash(FIXED_FUNC))

    def test_dedupe_drops_what_the_test_set_already_contains_including_the_other_twin(self):
        def ex(code, other=None):
            return {"meta": {"func_hash": pd.func_hash(code), "counterpart_hash": pd.func_hash(other) if other else None, "label": 1}}

        test = [ex("T1", "T1fixed")]
        train = [ex("T1"), ex("T1fixed", "T1"), ex("other"), ex("other")]
        kept, removed = pd.dedupe(train, pd.forbidden_hashes(test))
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, {"in_test": 2, "duplicate": 1})


class FunctionShareTests(unittest.TestCase):
    @staticmethod
    def make(n_call, n_function):
        def ex(kind, label, i):
            return {"meta": {"sink_kind": kind, "label": label, "func_hash": f"{kind}{label}{i}"}}

        return ([ex("call", i % 2, i) for i in range(n_call)] + [ex("function", i % 2, i) for i in range(n_function)])

    def test_function_level_examples_are_capped_to_the_requested_share_keeping_label_balance(self):
        kept, dropped = pd.limit_function_level(self.make(10, 100), 0.5, seed=1)
        fn = [e for e in kept if e["meta"]["sink_kind"] == "function"]
        self.assertEqual((len(kept) - len(fn), len(fn), dropped), (10, 10, 90))
        self.assertEqual(sorted(collections.Counter(e["meta"]["label"] for e in fn).values()), [5, 5])

    def test_share_one_keeps_everything_and_zero_keeps_only_call_sinks(self):
        data = self.make(10, 100)
        self.assertEqual(pd.limit_function_level(data, 1.0, seed=1), (data, 0))
        kept, dropped = pd.limit_function_level(data, 0.0, seed=1)
        self.assertEqual((len(kept), dropped), (10, 100))

    def test_selection_is_deterministic_and_never_adds_examples(self):
        data = self.make(4, 6)
        a, _ = pd.limit_function_level(data, 0.6, seed=3)
        self.assertEqual(a, pd.limit_function_level(data, 0.6, seed=3)[0])
        self.assertLessEqual(len(a), len(data))

    def test_without_call_sinks_nothing_is_dropped_by_the_cap(self):
        data = self.make(0, 20)
        self.assertEqual(pd.limit_function_level(data, 0.5, seed=1), (data, 0))


class LabelSkewTests(unittest.TestCase):
    @staticmethod
    def ex(label, synthetic, i):
        return {"meta": {"label": label, "synthetic": synthetic, "func_hash": f"{label}{synthetic}{i}"}}

    def test_the_majority_label_is_trimmed_dropping_synthetic_examples_first(self):
        data = ([self.ex(1, True, i) for i in range(10)] + [self.ex(1, False, i) for i in range(4)] + [self.ex(0, False, i) for i in range(8)])
        kept, dropped = pd.limit_label_skew(data, 1.25, seed=1)
        self.assertEqual(dropped, 4)
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 1), 10)
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 1 and not e["meta"]["synthetic"]), 4, "gli esempi reali restano")
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 0), 8)

    def test_real_examples_are_trimmed_only_when_synthetic_ones_are_not_enough(self):
        data = [self.ex(1, False, i) for i in range(20)] + [self.ex(0, False, i) for i in range(8)]
        kept, dropped = pd.limit_label_skew(data, 1.0, seed=1)
        self.assertEqual((dropped, len(kept)), (12, 16))

    def test_balanced_or_disabled_means_untouched_and_order_is_preserved(self):
        data = [self.ex(i % 2, False, i) for i in range(10)]
        self.assertEqual(pd.limit_label_skew(data, 1.25, seed=1), (data, 0))
        skewed = [self.ex(1, True, i) for i in range(10)] + [self.ex(0, False, 0)]
        self.assertEqual(pd.limit_label_skew(skewed, 0, seed=1), (skewed, 0))
        kept, _ = pd.limit_label_skew(skewed, 1.25, seed=1)
        self.assertEqual([e["meta"]["func_hash"] for e in kept], sorted((e["meta"]["func_hash"] for e in kept), key=lambda h: [x["meta"]["func_hash"] for x in skewed].index(h)))


class MultiSourceEndToEndTests(unittest.TestCase):
    PY_BEFORE = 'def q(cur, uid):\n    cur.execute(f"SELECT * FROM t WHERE id={uid}")\n'
    PY_AFTER = 'def q(cur, uid):\n    cur.execute("SELECT * FROM t WHERE id=%s", (uid,))\n'
    JULIET_BAD = ("void CWE690_x_%02d_bad()\n{\n    char * data;\n    /* FLAW: no check */\n    strcpy(data, \"x\");\n}\n")
    JULIET_GOOD = ("static void goodG2B()\n{\n    char * data = malloc(10);\n    /* FIX: check first */\n    if (data) { free(data); }\n}\n")

    def write_inputs(self, root: Path):
        sv = []
        for i in range(60):
            vuln = f"int f{i}(char *d, char *s)\n{{\n    strcpy(d, s);\n    return {i};\n}}"
            fixed = vuln.replace("strcpy(d, s)", "strncpy(d, s, 15)")
            base = {"project": "linux", "project_url": "u", "filepath": "d/x.c", "commit_id": f"sv{i}", "commit_message": "fix", "func_name": f"f{i}",
                    "cve_list": ["CVE-1"], "cwe_list": ["CWE-120"]}
            sv.append({**base, "idx": 2 * i, "is_vulnerable": True, "func_body": vuln, "fixed_func_idx": 2 * i + 1.0})
            sv.append({**base, "idx": 2 * i + 1, "is_vulnerable": False, "func_body": fixed, "fixed_func_idx": None})
        write_jsonl(root / "sv.jsonl", sv)
        py = [{"source_dataset": "secureIT-project/CVEfixes", "cve_id": f"CVE-{i}", "cve_description": "sqli", "cwe_id": "CWE-89", "function_name": "q",
               "code_before": self.PY_BEFORE.replace("uid", f"uid{i}"), "code_after": self.PY_AFTER.replace("uid", f"uid{i}"), "commit_hash": f"py{i}",
               "repo_url": "https://github.com/o/r", "programming_language": "python"} for i in range(60)]
        write_jsonl(root / "py.jsonl", py)
        import csv

        with (root / "juliet.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["filename", "bad", "good"])
            w.writeheader()
            for i in range(30):
                w.writerow({"filename": f"testcases/CWE690_x/s01/CWE690_x_{i:02d}.c", "bad": self.JULIET_BAD % i, "good": self.JULIET_GOOD})

    def run_main(self, raw: Path, out: Path, *extra):
        return pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "valid.jsonl"), "--test", str(raw / "test.jsonl"),
                        "--secvuleval", str(raw / "sv.jsonl"), "--python-pairs", str(raw / "py.jsonl"), "--juliet", str(raw / "juliet.csv"),
                        "--out-dir", str(out), "--seed", "3", "--holdout-pct", "25", "--val-pct", "10", *extra])

    def setUp(self):
        self._raw, self._out = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.raw, self.out = Path(self._raw.name), Path(self._out.name)
        EndToEndTests().make_inputs(self.raw)
        self.write_inputs(self.raw)

    def tearDown(self):
        self._raw.cleanup()
        self._out.cleanup()

    def load(self, name):
        return [json.loads(l) for l in (self.out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_every_source_contributes_and_juliet_never_reaches_the_test_set(self):
        self.assertEqual(self.run_main(self.raw, self.out), 0)
        sources_by_split = {name: collections.Counter(e["meta"]["source"] for e in self.load(name)) for name in ("train", "val", "test")}
        self.assertEqual(set(sources_by_split["train"]), {"primevul", "secvuleval", "cvefixes-python", "juliet"})
        self.assertNotIn("juliet", sources_by_split["test"])
        self.assertTrue({"primevul", "secvuleval", "cvefixes-python"} <= set(sources_by_split["test"]), sources_by_split["test"])

    def test_no_function_is_shared_between_training_and_test(self):
        self.run_main(self.raw, self.out)
        test_hashes = {e["meta"]["func_hash"] for e in self.load("test")}
        for name in ("train", "val"):
            self.assertFalse(test_hashes & {e["meta"]["func_hash"] for e in self.load(name)}, name)

    def test_stats_break_down_by_source(self):
        self.run_main(self.raw, self.out)
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertIn("secvuleval", stats["train"]["by_source"])
        self.assertGreater(stats["train"]["by_source"]["secvuleval"]["kept"]["vulnerable"], 0)
        self.assertIn("duplicates_removed", stats["train"])

    def test_localized_call_sinks_dominate_the_synthetic_and_diff_based_sources(self):
        self.run_main(self.raw, self.out)
        sv = [e for e in self.load("train") if e["meta"]["source"] == "secvuleval" and e["meta"]["kind"] == "vulnerable"]
        self.assertTrue(sv)
        self.assertTrue(all(e["meta"]["sink_function"] == "strcpy" and e["meta"]["localized"] == "diff" for e in sv))
        for e in self.load("train"):
            self.assertEqual(common.validate_finding(json.loads(e["messages"][2]["content"])), [])

    def test_output_is_deterministic_with_all_sources(self):
        self.run_main(self.raw, self.out)
        with tempfile.TemporaryDirectory() as other:
            self.run_main(self.raw, Path(other))
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((self.out / name).read_bytes(), (Path(other) / name).read_bytes(), name)

    def write_codeql(self):
        base = self.raw / "codeql" / "python" / "ql" / "test" / "query-tests" / "Security" / "CWE-089-SqlInjection"
        base.mkdir(parents=True)
        for i in range(40):
            (base / f"case{i}.py").write_text(
                f"def view{i}(cursor, name):\n"
                f"    # GOOD\n"
                f"    cursor.execute('SELECT {i} WHERE n = %s', name)\n"
                f"    cursor.execute('SELECT {i} WHERE n = %s' % name)  # $ Alert\n",
                encoding="utf-8",
            )
        return self.raw / "codeql"

    def test_codeql_tests_are_a_source_with_per_sink_verdicts_and_no_leaked_labels(self):
        root = self.write_codeql()
        self.run_main(self.raw, self.out, "--codeql", str(root))
        found = [e for name in ("train", "val", "test") for e in self.load(name) if e["meta"]["source"] == "codeql-tests"]
        self.assertTrue(found)
        self.assertEqual({e["meta"]["localized"] for e in found}, {"comment"})
        self.assertEqual({e["meta"]["label"] for e in found}, {0, 1})
        for e in found:
            user = e["messages"][1]["content"]
            self.assertNotIn("$ Alert", user)
            self.assertNotIn("# GOOD", user)
        test_sources = {e["meta"]["source"] for e in self.load("test")}
        self.assertIn("codeql-tests", test_sources)

    def test_function_share_only_trims_training_and_validation(self):
        def function_level(prefix, n):
            return [record(idx=f"{prefix}{i}", commit_id=f"{prefix}fc{i}", func=f"void {prefix}fn{i}(struct a *p)\n{{\n  p->x = {i};\n}}", cwe=["CWE-476"])
                    for i in range(n)]

        for name, prefix in (("train", "t"), ("test", "e")):
            path = self.raw / f"{name}.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n".join(json.dumps(r) for r in function_level(prefix + "x", 6)) + "\n", encoding="utf-8")
        self.run_main(self.raw, self.out, "--function-share", "0")
        self.assertEqual({e["meta"]["sink_kind"] for e in self.load("train")}, {"call"})
        self.assertIn("function", {e["meta"]["sink_kind"] for e in self.load("test")}, "il test non viene ridotto")
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertGreater(stats["train"]["function_level_dropped"], 0)
        self.assertEqual(stats["test"]["function_level_dropped"], 0)

    def test_juliet_per_cwe_cap(self):
        self.run_main(self.raw, self.out, "--juliet-per-cwe", "3")
        groups = {e["meta"]["commit_id"] for name in ("train", "val") for e in self.load(name) if e["meta"]["source"] == "juliet"}
        self.assertLessEqual(len(groups), 3)

    # ---- --graph-dir: esempi con slice -----------------------------------------------------------------------------
    def write_graph_dir(self) -> Path:
        graph = self.raw / "graph"
        write_jsonl(graph / "a_train.jsonl", [graph_example(i, i % 2, n_blocks=[1, 2, 3, 2, 1, 2][i], view="sink_only" if i == 3 else "callers") for i in range(6)])
        write_jsonl(graph / "b_train.jsonl", [graph_example(10 + i, i % 2, source="py-synth", track="py_synth") for i in range(2)])
        write_jsonl(graph / "a_val.jsonl", [graph_example(20 + i, i % 2) for i in range(3)])
        write_jsonl(graph / "c_test.jsonl", [graph_example(30, 1)])  # un file di valutazione: mai letto
        (graph / "juliet_report.json").write_text("{}", encoding="utf-8")
        return graph

    def read_lines(self, out: Path, name: str) -> list[dict]:
        return [json.loads(l) for l in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_juliet_is_no_longer_a_default_source(self):
        self.assertNotIn("juliet", pd.ALL_SOURCES)

    def test_graph_dir_examples_go_to_train_and_val_only_and_test_is_unchanged(self):
        self.run_main(self.raw, self.out)
        graph = self.write_graph_dir()
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            self.run_main(self.raw, other, "--graph-dir", str(graph))
            self.assertEqual((self.out / "test.jsonl").read_bytes(), (other / "test.jsonl").read_bytes(), "il test non cambia nemmeno di un byte")
            test_sources = lambda out: collections.Counter(e["meta"]["source"] for e in self.read_lines(out, "test"))  # noqa: E731
            self.assertEqual(test_sources(self.out), test_sources(other))
            self.assertNotIn("py-synth", test_sources(other))

            for name, expected in (("train", {"juliet-nist": 6, "py-synth": 2}), ("val", {"juliet-nist": 3})):
                got = collections.Counter(e["meta"]["source"] for e in self.read_lines(other, name) if e["meta"].get("track"))
                self.assertEqual(dict(got), expected, name)
                old = lambda out: sorted(json.dumps(e, sort_keys=True) for e in self.read_lines(out, name) if not e["meta"].get("track"))  # noqa: E731
                self.assertEqual(old(self.out), old(other), f"{name}: gli esempi delle fonti vecchie non cambiano")
            self.assertNotIn("c_test", json.dumps(self.read_lines(other, "train")))
            stats = json.loads((other / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train"]["graph_added"], 8)
            self.assertEqual(stats["val"]["graph_added"], 3)
            self.assertEqual(stats["test"]["graph_added"], 0)
            self.assertEqual(stats["train"]["by_source"]["juliet-nist"]["kept"], {"vulnerable": 3, "patched": 3})

    def test_output_is_deterministic_with_a_graph_dir(self):
        graph = self.write_graph_dir()
        self.run_main(self.raw, self.out, "--graph-dir", str(graph))
        with tempfile.TemporaryDirectory() as other:
            self.run_main(self.raw, Path(other), "--graph-dir", str(graph))
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((self.out / name).read_bytes(), (Path(other) / name).read_bytes(), name)

    def test_graph_examples_colliding_with_the_test_set_or_already_collected_are_dropped(self):
        self.run_main(self.raw, self.out)
        test_user = next(m["content"] for m in self.read_lines(self.out, "test")[0]["messages"] if m["role"] == "user")
        train_user = next(m["content"] for m in self.read_lines(self.out, "train")[0]["messages"] if m["role"] == "user")
        clash_test, clash_train, fresh = graph_example(1, 1), graph_example(2, 1), graph_example(3, 1)
        for ex, text in ((clash_test, test_user), (clash_train, train_user)):
            ex["messages"][1]["content"] = text
            ex["meta"]["func_hash"] = pd.func_hash(text)
        clash_train["meta"]["label"] = self.read_lines(self.out, "train")[0]["meta"]["label"]
        graph = self.raw / "graph"
        write_jsonl(graph / "a_train.jsonl", [clash_test, clash_train, fresh, fresh])
        with tempfile.TemporaryDirectory() as other:
            self.run_main(self.raw, Path(other), "--graph-dir", str(graph))
            added = [e for e in self.read_lines(Path(other), "train") if e["meta"].get("track")]
            self.assertEqual([e["meta"]["func_hash"] for e in added], [fresh["meta"]["func_hash"]])
            stats = json.loads((Path(other) / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train"]["graph_removed"], {"in_test": 1, "duplicate": 2})
            self.assertEqual(stats["train"]["graph_added"], 1)

    def test_graph_dir_that_does_not_exist_is_a_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self.run_main(self.raw, self.out, "--graph-dir", str(self.raw / "nope"))
        self.assertIn("non è una cartella", err.getvalue())

    def write_codeql_with_c(self):
        root = self.write_codeql()
        c_dir = root / "cpp" / "ql" / "test" / "query-tests" / "Security" / "CWE" / "CWE-120"
        c_dir.mkdir(parents=True)
        for i in range(40):
            (c_dir / f"case{i}.c").write_text(
                f"void bad_copy{i}(char *s) {{\n    char buf[10];\n    strcpy(buf, s); // $ Alert\n}}\n\n"
                f"void good_copy{i}(char *s) {{\n    char buf[10];\n    strncpy(buf, s, 9); // GOOD\n}}\n",
                encoding="utf-8",
            )
        return root

    def test_graph_dir_skips_python_codeql_but_keeps_c(self):
        root = self.write_codeql_with_c()
        graph = self.write_graph_dir()
        self.run_main(self.raw, self.out, "--codeql", str(root))
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            self.run_main(self.raw, other, "--codeql", str(root), "--graph-dir", str(graph))
            codeql = lambda out, name: collections.Counter(  # noqa: E731
                (e["meta"]["lang"], e["meta"]["kind"]) for e in self.read_lines(out, name) if e["meta"]["source"] == "codeql-tests")
            self.assertTrue(any(lang == "python" for lang, _ in codeql(self.out, "train")), "senza graph-dir i Python di CodeQL stanno nel train")
            for name in ("train", "val"):
                self.assertEqual({lang for lang, _ in codeql(other, name)} - {"c"}, set(), f"{name}: niente Python di CodeQL")
            self.assertTrue(any(lang == "c" for lang, _ in codeql(other, "train")), "i C di CodeQL restano")
            self.assertEqual(codeql(self.out, "test"), codeql(other, "test"), "la composizione del test non cambia")
            self.assertTrue(any(lang == "python" for lang, _ in codeql(other, "test")))
            self.assertEqual((self.out / "test.jsonl").read_bytes(), (other / "test.jsonl").read_bytes())

    def test_graph_dir_length_filter_drops_and_counts_per_source(self):
        limit = int(8192 * 3.5)
        graph = self.raw / "graph"
        short = graph_example(1, 1)
        edge = graph_example(2, 0)  # il prompt è lungo quanto serve per arrivare al limite esatto
        edge["messages"][1]["content"] ="u" * (limit - len(edge["messages"][2]["content"]))
        edge["meta"]["func_hash"] = pd.func_hash(edge["messages"][1]["content"])
        over = graph_example(3, 1, source="py-synth", track="py_synth")
        over["messages"][1]["content"] = "v" * (limit - len(over["messages"][2]["content"]) + 1)
        over["meta"]["func_hash"] = pd.func_hash(over["messages"][1]["content"])
        write_jsonl(graph / "a_train.jsonl", [short, edge, over])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.run_main(self.raw, self.out, "--graph-dir", str(graph))
        kept = {e["meta"]["func_hash"] for e in self.read_lines(self.out, "train") if e["meta"].get("track")}
        self.assertEqual(kept, {short["meta"]["func_hash"], edge["meta"]["func_hash"]}, "esattamente al limite resta, un carattere oltre no")
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["train"]["too_long_dropped"], {"py-synth": 1})
        self.assertEqual(stats["val"]["too_long_dropped"], {})
        self.assertIn("py-synth: 1", err.getvalue())

    def test_length_filter_applies_to_train_and_val_of_every_source_but_not_to_the_test(self):
        self.run_main(self.raw, self.out)
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            self.run_main(self.raw, other, "--max-seq-length", "450")  # 1.575 caratteri: taglia quasi tutto
            stats = json.loads((other / "stats.json").read_text(encoding="utf-8"))
            for name in ("train", "val"):
                limit = 450 * 3.5
                for e in self.read_lines(other, name):
                    self.assertLessEqual(sum(len(m["content"]) for m in e["messages"] if m["role"] != "system"), limit)
                dropped = sum(stats[name]["too_long_dropped"].values())
                self.assertGreater(dropped, 0, name)
                self.assertEqual(len(self.read_lines(other, name)) + dropped, len(self.read_lines(self.out, name)), name)
                self.assertIn("primevul", stats[name]["too_long_dropped"])
            self.assertEqual((self.out / "test.jsonl").read_bytes(), (other / "test.jsonl").read_bytes())
            self.assertEqual(stats["test"]["too_long_dropped"], {})

    def test_report_lists_sizes_source_counts_and_trace_shares(self):
        graph = self.write_graph_dir()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.run_main(self.raw, self.out, "--graph-dir", str(graph))
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test"):
            self.assertEqual(stats[name]["file_bytes"], (self.out / f"{name}.jsonl").stat().st_size, name)
            self.assertFalse(stats[name]["over_size_limit"])
            self.assertEqual(stats[name]["source_counts"], dict(collections.Counter(e["meta"]["source"] for e in self.read_lines(self.out, name))), name)
        shares = stats["train"]["trace_shares"]
        self.assertEqual(shares["juliet"], {"examples": 6, "multi_block": 4, "multi_block_share": round(4 / 6, 4), "sink_only": 1, "sink_only_share": round(1 / 6, 4)})
        self.assertEqual(shares["py_synth"]["multi_block"], 2)
        text = err.getvalue()
        self.assertIn("train.jsonl", text)
        self.assertIn("juliet-nist", text)
        self.assertIn("66.7%", text)
        self.assertNotIn("WARNING", text)

    def test_report_warns_for_files_over_50_mb(self):
        self.assertEqual(pd.MAX_FILE_BYTES, 50 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as d:
            small, big = Path(d) / "small.jsonl", Path(d) / "big.jsonl"
            small.write_bytes(b"x" * 5)
            big.write_bytes(b"x" * 20)
            sizes, warnings = pd.size_report([small, big], limit=10)
            self.assertEqual(sizes, {"small.jsonl": 5, "big.jsonl": 20})
            self.assertEqual(len(warnings), 1)
            self.assertIn("WARNING", warnings[0])
            self.assertIn("big.jsonl", warnings[0])
            self.assertEqual(pd.size_report([small, big], limit=20)[1], [], "il limite stesso non è superato")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(pd, "MAX_FILE_BYTES", 10):
            self.run_main(self.raw, self.out)
        self.assertIn("WARNING", err.getvalue())
        self.assertIn("train.jsonl", [l for l in err.getvalue().splitlines() if "WARNING" in l][0])
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertTrue(stats["train"]["over_size_limit"])


def graph_example(i: int, label: int, *, source: str = "juliet-nist", track: str = "juliet", view: str = "callers",
                  n_blocks: int = 2) -> dict:
    """Esempio con slice minimo (la forma che scrivono i file `data/graph/*_{train,val}.jsonl`): solo i campi che main legge."""
    user = f"### graph prompt {i} {source}"
    finding = {"vulnerable": bool(label), "cwe_id": "CWE-78", "severity": "HIGH" if label else "NONE", "sink_function": "system",
               "sink_file": "src/a.c", "sink_line": 4, "vex_status": "affected" if label else "not_affected", "reasoning": "r",
               "suggested_fix": "fix" if label else pd.NO_FIX}
    blocks = [{"depth": d, "name": f"fn_{d}"} for d in range(n_blocks - 1, -1, -1)]
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(finding)}],
            "meta": {"source": source, "synthetic": False, "track": track, "group": f"g{i}", "variant": "01", "label": label,
                     "kind": "vulnerable" if label else "patched", "cwe": "CWE-78", "lang": "c", "sink_function": "system", "sink_line": 4,
                     "sink_kind": "call", "func_hash": pd.func_hash(user), "view": view, "pair_id": None,
                     "trace": {"cwe_id": "CWE-78", "sink": {}, "blocks": blocks}}}


class GraphHelpersTests(unittest.TestCase):
    def test_load_reads_train_and_val_files_in_name_order_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            graph = Path(d)
            (graph / "b_train.jsonl").write_text(json.dumps(graph_example(2, 1)) + "\n\n   \n" + json.dumps(graph_example(3, 0)) + "\n", encoding="utf-8")
            write_jsonl(graph / "a_train.jsonl", [graph_example(1, 1)])
            write_jsonl(graph / "a_val.jsonl", [graph_example(4, 0)])
            write_jsonl(graph / "z_test.jsonl", [graph_example(5, 0)])
            (graph / "juliet_report.json").write_text("{not json", encoding="utf-8")
            loaded = pd.load_graph_examples(graph)
            self.assertEqual([e["meta"]["group"] for e in loaded["train"]], ["g1", "g2", "g3"])
            self.assertEqual([e["meta"]["group"] for e in loaded["val"]], ["g4"])
            self.assertEqual(set(loaded), {"train", "val"})

    def test_load_rejects_lines_without_messages_or_meta(self):
        good = graph_example(1, 1)
        for name, bad in (("no meta", {"messages": good["messages"]}), ("no messages", {"meta": good["meta"]}),
                          ("meta not a dict", {**good, "meta": []}), ("not an object", [1]),
                          ("no assistant", {**good, "messages": good["messages"][:2]})):
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                (Path(d) / "x_train.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError) as ctx:
                    pd.load_graph_examples(Path(d))
                self.assertIn("x_train.jsonl:2", str(ctx.exception))

    def test_an_empty_graph_dir_gives_empty_lists(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(pd.load_graph_examples(Path(d)), {"train": [], "val": []})

    def test_merge_removes_test_collisions_and_duplicates_but_keeps_order(self):
        existing = [graph_example(1, 1), graph_example(2, 0)]
        in_test = graph_example(3, 1)
        test_prompts = {pd.prompt_hash(in_test)}
        new = [graph_example(4, 1), in_test, graph_example(1, 1), graph_example(4, 1), graph_example(5, 0)]
        merged, removed = pd.merge_graph_examples(existing, new, test_prompts, set())
        self.assertEqual([e["meta"]["group"] for e in merged], ["g1", "g2", "g4", "g5"])
        self.assertEqual(removed, {"in_test": 1, "duplicate": 2})
        self.assertEqual(pd.merge_graph_examples([], [], set(), set()), ([], {}))
        # lo stesso prompt con l'etichetta opposta non è un duplicato esatto (e `drop_conflicts` non è affare di questo passo)
        flipped = graph_example(1, 0)
        self.assertEqual(len(pd.merge_graph_examples(existing, [flipped], set(), set())[0]), 3)
        # il func_hash di un esempio di test (impronta della funzione) blocca anche un esempio con slice che porta lo stesso hash
        clash = graph_example(6, 1)
        self.assertEqual(pd.merge_graph_examples([], [clash], set(), {clash["meta"]["func_hash"]}), ([], {"in_test": 1}))

    def test_drop_too_long_counts_per_source_and_keeps_the_exact_limit(self):
        def ex(source, chars):
            return {"messages": [{"role": "system", "content": "s" * 10_000}, {"role": "user", "content": "u" * (chars - 5)},
                                 {"role": "assistant", "content": "a" * 5}], "meta": {"source": source}}

        kept, dropped = pd.drop_too_long([ex("a", 100), ex("a", 101), ex("b", 150), ex("b", 99), ex("a", 200)], 100)
        self.assertEqual([len(e["messages"][1]["content"]) + 5 for e in kept], [100, 99])
        self.assertEqual(dropped, {"a": 2, "b": 1})
        self.assertEqual(pd.drop_too_long([], 100), ([], {}))
        self.assertEqual(pd.max_example_chars(8192), 8192 * 3.5)

    def test_trace_shares_per_track(self):
        examples = [graph_example(1, 1, n_blocks=3), graph_example(2, 0, n_blocks=1), graph_example(3, 0, view="sink_only"),
                    graph_example(4, 1, track="py_codeql", source="codeql-tests"),
                    {"messages": [], "meta": {"source": "primevul"}}]  # senza track: non conta
        examples[1]["meta"].pop("trace")
        shares = pd.trace_shares(examples)
        self.assertEqual(set(shares), {"juliet", "py_codeql"})
        self.assertEqual(shares["juliet"], {"examples": 3, "multi_block": 2, "multi_block_share": round(2 / 3, 4), "sink_only": 1, "sink_only_share": round(1 / 3, 4)})
        self.assertEqual(shares["py_codeql"]["multi_block_share"], 1.0)
        self.assertEqual(pd.trace_shares([]), {})
