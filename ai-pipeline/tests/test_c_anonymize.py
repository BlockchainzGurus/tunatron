"""Test dell'anonimizzazione del C delle slice Juliet: nessuna etichetta deve passare per nomi, commenti o stringhe."""
import copy
import json
import re
import unittest

import c_anonymize
from c_anonymize import LEAK_RE, anonymize_slice, leaky_identifier, strip_comments

MACROS = {"SYSTEM": "system"}
P = "CWE78_OS_Command_Injection__char_environment_system_54"

# Blocchi 54d e 54e come li restituisce `sast-engine slice` (testo di Juliet, righe assolute dei file originali).
BAD_54D = f"""void {P}d_badSink(char * data)
{{
    {P}e_badSink(data);
}}"""
BAD_54E = f"""void {P}e_badSink(char * data)
{{
    /* POTENTIAL FLAW: Execute command in data possibly leading to command injection */
    if (SYSTEM(data) != 0)
    {{
        printLine("command execution failed!");
        exit(1);
    }}
}}"""
GOOD_54D = BAD_54D.replace("badSink", "goodG2BSink")
GOOD_54E = BAD_54E.replace("badSink", "goodG2BSink")


def record(kind: str, function: str = "system") -> dict:
    d, e = (BAD_54D, BAD_54E) if kind == "bad" else (GOOD_54D, GOOD_54E)
    d_line, e_line = (49, 46) if kind == "bad" else (62, 61)
    return {
        "sink": {"function": function, "file": f"s02/{P}e.c", "line": e_line + 3, "cwe_id": "CWE-78"},
        "blocks": [
            {"file": f"s02/{P}d.c", "name": f"{P}d_{kind}Sink", "start_line": d_line, "end_line": d_line + 3,
             "depth": 1, "lang": "c", "code": d},
            {"file": f"s02/{P}e.c", "name": f"{P}e_{kind}Sink", "start_line": e_line, "end_line": e_line + 8,
             "depth": 0, "lang": "c", "code": e},
        ],
        "truncated": False,
    }


BAD = record("bad")
GOOD = record("goodG2B")

# Sorgente «buona» di Juliet (54a): stringa costante, commento FIX, stringa letterale rivelatrice, variabile `dataGoodBuffer`.
GOOD_SOURCE = """static void goodG2B()
{
    char * data;
    char dataGoodBuffer[100] = FULL_COMMAND;
    data = dataGoodBuffer;
    /* FIX: Append a fixed string to data (not user / external input) */
    strcat(data, "*.*");
    printLine("Benign, fixed string");
    if (globalReturnsTrue()) { size_t prefix_len = strlen(data); }
    CWE78_OS_Command_Injection__char_environment_system_54b_goodG2BSink(data);
}"""


def with_source(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    rec["blocks"].insert(0, {"file": f"s02/{P}a.c", "name": "goodG2B", "start_line": 74, "end_line": 83,
                             "depth": 2, "lang": "c", "code": GOOD_SOURCE})
    return rec


def block(code: str, start_line: int = 10, name: str = "f") -> dict:
    """Record a un solo blocco (profondità 0) con il sink alla riga che contiene `system(`."""
    offset = next(i for i, line in enumerate(code.replace("\r\n", "\n").split("\n")) if "system(" in line)
    return {"sink": {"function": "system", "file": "a.c", "line": start_line + offset, "cwe_id": "CWE-78"},
            "blocks": [{"file": "a.c", "name": name, "start_line": start_line, "end_line": start_line + code.count("\n"),
                        "depth": 0, "lang": "c", "code": code}],
            "truncated": False}


def sink_code_line(out: dict) -> str:
    b = next(b for b in out["blocks"] if b["depth"] == 0)
    return b["code"].split("\n")[out["sink"]["line"] - b["start_line"]]


class StripCommentsTests(unittest.TestCase):
    def test_strip_comments_removes_both_kinds_and_keeps_string_content(self):
        self.assertEqual(strip_comments('x = "/* no */"; /* c */ // d')[0], 'x = "/* no */";')

    def test_strip_comments_line_map_points_to_original_lines(self):
        code = "int f()\n{\n    /* c */\n    a();\n\n\n    b(); // x\n}"
        new, line_map = strip_comments(code)
        self.assertEqual(new, "int f()\n{\n    a();\n\n    b();\n}")
        self.assertEqual(line_map, [0, 1, 3, 4, 6, 7])
        old = code.split("\n")
        for i, line in enumerate(new.split("\n")):
            self.assertTrue(old[line_map[i]].startswith(line), (i, line))

    def test_whole_line_comment_leaves_no_blank_line(self):
        self.assertEqual(strip_comments("a;\n// c\n  /* d */  \nb;"), ("a;\nb;", [0, 3]))

    def test_multiline_block_comment_keeps_the_line_map(self):
        code = "a; /* uno\n * due\n\n tre */ b;\nc;"
        self.assertEqual(strip_comments(code), ("a;\n b;\nc;", [0, 3, 4]))

    def test_comment_markers_inside_literals_are_not_comments(self):
        code = "s = \"a // b\"; c = '\"'; q = '\\''; e = \"x\\\"/* y\"; // via\nt = \"/*\"; z = 1; /* via */"
        self.assertEqual(strip_comments(code)[0], "s = \"a // b\"; c = '\"'; q = '\\''; e = \"x\\\"/* y\";\nt = \"/*\"; z = 1;")

    def test_comment_between_tokens_becomes_a_space(self):
        self.assertEqual(strip_comments("int/**/x;")[0], "int x;")

    def test_crlf_input_is_normalized(self):
        self.assertEqual(strip_comments("a;\r\n// c\r\nb;\r\n"), ("a;\nb;", [0, 2]))

    def test_all_comment_code_becomes_empty(self):
        self.assertEqual(strip_comments("/* a */\n// b"), ("", []))


class LeakyIdentifierTests(unittest.TestCase):
    def test_leaky_identifier_matches_parts_not_substrings(self):
        for name in ("dataGoodBuffer", "goodG2BSource", "badSink", "CWE78_x_54e_badSink", "dataG2B", "B2G_sink"):
            self.assertTrue(leaky_identifier(name), name)
        for name in ("badge_count", "fixture", "prefix", "data", "globalReturnsTrue", "big2bar"):
            self.assertFalse(leaky_identifier(name), name)

    def test_extra_parts_extend_the_check(self):
        self.assertFalse(leaky_identifier("unsafe_query"))
        self.assertTrue(leaky_identifier("unsafe_query", frozenset({"safe", "unsafe"})))
        self.assertFalse(leaky_identifier("safety", frozenset({"safe"})))


class AnonymizeSliceTests(unittest.TestCase):
    def test_defined_functions_get_one_consistent_name_across_blocks(self):
        out = anonymize_slice(BAD, "s", MACROS)
        caller, sink = out["blocks"]
        self.assertRegex(sink["name"], r"^fn_[0-9a-f]{6}$")
        self.assertRegex(caller["name"], r"^fn_[0-9a-f]{6}$")
        self.assertNotEqual(caller["name"], sink["name"])
        self.assertIn(f"{sink['name']}(data);", caller["code"], "la chiamata nel blocco a profondità 1 usa il nome nuovo")
        self.assertTrue(sink["code"].startswith(f"void {sink['name']}(char * data)"))

    def test_macro_is_rewritten_and_sink_line_follows(self):
        out = anonymize_slice(BAD, "s", MACROS)
        self.assertNotIn("SYSTEM", json.dumps(out))
        self.assertIn("system(data)", sink_code_line(out))
        self.assertEqual(out["sink"]["line"] - out["blocks"][1]["start_line"], 2, "il commento tolto sposta il sink in su")
        for b in out["blocks"]:
            self.assertEqual(b["end_line"], b["start_line"] + len(b["code"].split("\n")) - 1)

    def test_macro_applies_to_the_sink_callee(self):
        out = anonymize_slice(record("bad", function="SYSTEM"), "s", MACROS)
        self.assertEqual(out["sink"]["function"], "system")

    def test_file_names_are_hidden_and_consistent(self):
        out = anonymize_slice(with_source(BAD), "s", MACROS)
        files = [b["file"] for b in out["blocks"]]
        for f in files:
            self.assertRegex(f, r"^src/[0-9a-f]{8}\.c$")
        self.assertEqual(len(set(files)), 3, "un nome per file originale")
        self.assertEqual(out["sink"]["file"], out["blocks"][-1]["file"])
        twice = copy.deepcopy(BAD)
        twice["blocks"][0]["file"] = twice["blocks"][1]["file"]
        self.assertEqual(len({b["file"] for b in anonymize_slice(twice, "s", MACROS)["blocks"]}), 1)

    def test_leaky_literal_replaced_but_neutral_literal_kept(self):
        out = anonymize_slice(with_source(GOOD), "s", MACROS)
        code = "\n".join(b["code"] for b in out["blocks"])
        self.assertIn('printLine("text");', code)
        self.assertNotIn("Benign", code)
        self.assertIn('printLine("command execution failed!");', code)
        self.assertIn('strcat(data, "*.*");', code)

    def test_deterministic_per_seed_and_different_across_seeds(self):
        self.assertEqual(anonymize_slice(BAD, "s", MACROS), anonymize_slice(BAD, "s", MACROS))
        a, b = anonymize_slice(BAD, "s", MACROS), anonymize_slice(BAD, "t", MACROS)
        self.assertNotEqual([x["name"] for x in a["blocks"]], [x["name"] for x in b["blocks"]])
        self.assertNotEqual([x["file"] for x in a["blocks"]], [x["file"] for x in b["blocks"]])

    def test_input_record_is_not_modified(self):
        rec = with_source(BAD)
        before = copy.deepcopy(rec)
        anonymize_slice(rec, "s", MACROS)
        self.assertEqual(rec, before)

    def test_no_leak_pattern_survives(self):
        for rec in (BAD, GOOD, with_source(BAD), with_source(GOOD)):
            out = anonymize_slice(rec, "s", MACROS)
            self.assertIsNone(LEAK_RE.search(json.dumps(out)), json.dumps(out))

    def test_c_identifiers_with_a_leak_substring_are_renamed(self):
        """Regola R5: in C basta la sottostringa (il cancello finale è a sottostringa), anche se `leaky_identifier` dice no."""
        code = "void f(char *d)\n{\n    size_t suffix_len = 1;\n    char prefixBuf[4];\n    system(d);\n}"
        out = anonymize_slice(block(code), "s", MACROS)["blocks"][0]["code"]
        self.assertFalse(leaky_identifier("prefix"))
        self.assertNotIn("suffix_len", out)
        self.assertNotIn("prefixBuf", out)
        self.assertRegex(out, r"size_t v_[0-9a-f]{6} = 1;")
        self.assertRegex(out, r"char v_[0-9a-f]{6}\[4\];")

    def test_variables_are_renamed_consistently_and_neutral_names_kept(self):
        out = anonymize_slice(with_source(GOOD), "s", MACROS)["blocks"][0]["code"]
        renamed = re.search(r"char (v_[0-9a-f]{6})\[100\] = FULL_COMMAND;", out)
        self.assertIsNotNone(renamed, out)
        self.assertIn(f"data = {renamed.group(1)};", out)
        for kept in ("globalReturnsTrue()", "strlen(data)", "char * data;", "if ("):
            self.assertIn(kept, out)
        self.assertRegex(out, r"^void fn_[0-9a-f]{6}\(\)", "la `static` della definizione si toglie")

    def test_nothing_is_renamed_inside_literals(self):
        code = "void data(char *d)\n{\n    printLine(\"data\");\n    char c = 'x';\n    system(d);\n}"
        out = anonymize_slice(block(code, name="data"), "s", MACROS)["blocks"][0]["code"]
        self.assertRegex(out, r"^void fn_[0-9a-f]{6}\(char \*d\)")
        self.assertIn('printLine("data");', out, "il nome di funzione `data` cambia ovunque tranne che nella stringa")
        self.assertIn("char c = 'x';", out)

    def test_leaky_char_literal_and_include_are_neutralized(self):
        code = "void f(char *d)\n{\n#include \"CWE78_helper.h\"\n    int tag = 'BAD';\n    system(d);\n}"
        out = anonymize_slice(block(code), "s", MACROS)["blocks"][0]["code"]
        self.assertIn('#include "text"', out)
        self.assertIn("int tag = 't';", out)

    def test_crlf_record_keeps_the_sink_line(self):
        code = "void f(char *d)\r\n{\r\n    /* FLAW */\r\n    system(d);\r\n}"
        out = anonymize_slice(block(code, start_line=20), "s", MACROS)
        self.assertNotIn("\r", out["blocks"][0]["code"])
        self.assertEqual(sink_code_line(out), "    system(d);")
        self.assertEqual(out["sink"]["line"] - out["blocks"][0]["start_line"], 2)

    def test_block_empty_after_stripping_is_kept_as_one_empty_line(self):
        rec = copy.deepcopy(BAD)
        rec["blocks"][0]["code"] = "/* solo un commento */"
        out = anonymize_slice(rec, "s", MACROS)
        self.assertEqual(out["blocks"][0]["code"], "")
        self.assertEqual(out["blocks"][0]["end_line"], out["blocks"][0]["start_line"])

    def test_sink_line_outside_the_cleaned_block_raises(self):
        rec = copy.deepcopy(BAD)
        rec["sink"]["line"] = 46 + 2  # la riga del commento FLAW: non esiste più
        with self.assertRaises(ValueError):
            anonymize_slice(rec, "s", MACROS)

    def test_hash_collisions_are_disambiguated(self):
        """Un identificatore già presente con lo stesso nome canonico costringe a rifare l'hash."""
        clash = "fn_" + c_anonymize._digest("s", "fn:0:0")[:6]
        out = anonymize_slice(block(f"void f(char *d)\n{{\n    {clash}(d);\n    system(d);\n}}"), "s", MACROS)
        name = out["blocks"][0]["name"]
        self.assertRegex(name, r"^fn_[0-9a-f]{6}$")
        self.assertNotEqual(name, clash)
        self.assertIn(f"{clash}(d);", out["blocks"][0]["code"], "l'identificatore esterno resta com'è")

    def test_new_names_never_spell_a_leak_word_in_hex(self):
        """`bad` è fatto di cifre esadecimali: con il seme `s1736`, `fn:0:0` dà `38badf…` e `d462.c` dà `bad90f45…`."""
        self.assertIn("bad", c_anonymize._digest("s1736", "fn:0:0")[:6])
        self.assertIn("bad", c_anonymize._digest("s1736", "d462.c")[:8])
        rec = block("void f(char *d)\n{\n    system(d);\n}")
        rec["blocks"][0]["file"] = rec["sink"]["file"] = "d462.c"
        out = anonymize_slice(rec, "s1736", MACROS)
        self.assertRegex(out["blocks"][0]["name"], r"^fn_[0-9a-f]{6}$")
        self.assertRegex(out["blocks"][0]["file"], r"^src/[0-9a-f]{8}\.c$")
        self.assertIsNone(LEAK_RE.search(json.dumps(out)), json.dumps(out))


def twin(name: str, start_line: int, file: str = "s02/x_54e.c") -> dict:
    """Record a un blocco con il corpo dei sink gemelli di Juliet (12 righe, `SYSTEM(data)` alla quarta)."""
    code = f"""void {name}(char * data)
{{
    /* POTENTIAL FLAW: Execute command in data possibly leading to command injection */
    if (SYSTEM(data) != 0)
    {{
        printLine("command execution failed!");
        exit(1);
    }}
    // fine
    data = NULL;
    return;
}}"""
    return {"sink": {"function": "system", "file": file, "line": start_line + 3, "cwe_id": "CWE-78"},
            "blocks": [{"file": file, "name": name, "start_line": start_line, "end_line": start_line + 11, "depth": 0,
                        "lang": "c", "code": code}],
            "truncated": False}


class CanonicalRenamingTests(unittest.TestCase):
    """Regole R8/R9: l'uscita non dipende da nomi originali, posizioni nel file o nomi di variabile rivelatori."""

    def test_twin_sinks_with_different_names_and_positions_become_identical(self):
        bad = anonymize_slice(twin(f"{P}e_badSink", 20), "s", MACROS)
        good = anonymize_slice(twin(f"{P}e_goodG2BSink", 40), "s", MACROS)
        self.assertEqual(bad["blocks"], good["blocks"])
        self.assertEqual(bad["sink"], good["sink"])
        self.assertIn("system(data)", sink_code_line(bad))

    def test_original_line_position_is_not_encoded(self):
        self.assertEqual(anonymize_slice(twin("f", 5), "s", MACROS), anonymize_slice(twin("f", 500), "s", MACROS))

    def test_synthetic_start_line_follows_the_formula(self):
        out = anonymize_slice(BAD, "s", MACROS)
        for b, rank in zip(out["blocks"], ("1:0", "0:0")):
            self.assertEqual(b["start_line"], 1 + int(c_anonymize._digest("s", f"line:{rank}")[:4], 16) % 900)

    def test_leaky_variable_names_do_not_change_output(self):
        body = "void f(char * data)\n{{\n    char {0}[50];\n    data = {0};\n    system(data);\n}}"
        bad = anonymize_slice(block(body.format("dataBadBuffer")), "s", MACROS)
        good = anonymize_slice(block(body.format("dataGoodBuffer")), "s", MACROS)
        self.assertEqual(bad, good)
        self.assertIn("data = v_" + c_anonymize._digest("s", "v:0")[:6] + ";", bad["blocks"][0]["code"])

    def test_block_function_names_depend_on_position_not_original_name(self):
        def rec(outer: str, inner: str) -> dict:
            r = block(f"void {inner}(char *d)\n{{\n    system(d);\n}}", name=inner)
            r["blocks"].insert(0, {"file": "a.c", "name": outer, "start_line": 1, "end_line": 1, "depth": 1, "lang": "c",
                                   "code": f"void {outer}(char *d) {{ {inner}(d); }}"})
            return r

        self.assertEqual(anonymize_slice(rec("alpha", "beta"), "s", MACROS), anonymize_slice(rec("beta", "alpha"), "s", MACROS))
        out = anonymize_slice(rec("alpha", "beta"), "s", MACROS)
        self.assertEqual(out["blocks"][1]["name"], "fn_" + c_anonymize._digest("s", "fn:0:0")[:6])
        self.assertEqual(out["blocks"][0]["name"], "fn_" + c_anonymize._digest("s", "fn:1:0")[:6])

    def test_static_storage_class_of_the_definition_is_dropped(self):
        """In Juliet `bad()` è globale e `goodG2B()` è `static`: la parola chiave direbbe l'etichetta (Task 6)."""
        body = "void f(char * data)\n{\n    static int calls = 0;\n    system(data);\n}"
        plain = anonymize_slice(block(body), "s", MACROS)
        for variant in ("static " + body, "static\t" + body, "/* goodG2B uses the GoodSource */\nstatic " + body):
            self.assertEqual(anonymize_slice(block(variant), "s", MACROS), plain, variant)
        code = plain["blocks"][0]["code"]
        self.assertTrue(code.startswith("void fn_"), code)
        self.assertIn("static int calls = 0;", code, "una `static` dentro il corpo resta")
        self.assertIn("system(data)", sink_code_line(plain))
        self.assertEqual(anonymize_slice(block("staticky_t " + body), "s", MACROS)["blocks"][0]["code"][:11], "staticky_t ")


if __name__ == "__main__":
    unittest.main()
