"""Test di `codeql_graph`: albero dei test CodeQL Python ripulito, etichette dai `$ Alert`, nomi e commenti senza parole-spia.

Albero finto in cartella temporanea e record di slice scritti a mano; l'ultimo test usa il motore vero se esiste.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import build_graph_data
import codeql_graph as cg
import common
import graph_examples
import leak_audit
import localize
import prepare_data
import slice_io

KB = common.load_kb()
SECURITY = "python/ql/test/query-tests/Security"
REL = f"{SECURITY}/CWE-078-CommandInjection/test.py"

# Le righe sono numerate nei commenti: le annotazioni stanno alle righe 2 e 6, il commento `GOOD` alla 7.
PY_SRC = "\n".join([
    "import os",                                # 1
    "from flask import request  # $ Source",    # 2
    "",                                         # 3
    "",                                         # 4
    "def view(cmd):",                           # 5
    "    os.system(cmd)  # $ Alert",            # 6
    "    # GOOD: a constant",                   # 7
    "    os.system('ls')",                      # 8
    "",                                         # 9
    "",                                         # 10
    "def middle(x):",                           # 11
    "    view(x)",                              # 12
]) + "\n"


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return root


class PrepareTreeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def prepare(self, files: dict[str, str]):
        root = write_tree(self.tmp / "codeql", files)
        notes = cg.prepare_tree(root, self.tmp / "tree", KB)
        return notes, self.tmp / "tree"

    def test_prepare_tree_strips_annotations_and_keeps_line_numbers(self):
        notes, tree = self.prepare({REL: PY_SRC})
        copied = (tree / REL).read_text(encoding="utf-8")
        self.assertNotIn("$", copied)
        self.assertNotIn("GOOD", copied)
        self.assertEqual(copied.count("\n"), PY_SRC.count("\n"), "stesso numero di righe")
        lines = copied.split("\n")
        self.assertEqual(lines[5], "    os.system(cmd)")
        self.assertEqual(lines[7], "    os.system('ls')")
        self.assertEqual(notes[REL]["alert"], {6})
        self.assertEqual(notes[REL]["ambiguous"], set())
        self.assertTrue(notes[REL]["annotated"])
        self.assertEqual(notes[REL]["root"], f"{SECURITY}/CWE-078-CommandInjection")

    def test_cwe_comes_from_directory_name(self):
        notes, _ = self.prepare({f"{SECURITY}/CWE-078-CommandInjection/a.py": "x = 1  # $ Alert\n",
                                 f"{SECURITY}/CWE-089-SqlInjection/b.py": "x = 1  # $ Alert\n",
                                 f"{SECURITY}/CWE-601-UrlRedirect/c.py": "x = 1  # $ Alert\n"})
        self.assertEqual({Path(k).parent.name.split("-")[1]: v["cwe"] for k, v in notes.items()},
                         {"078": "CWE-78", "089": "CWE-89", "601": "CWE-601"})

    def test_cwes_outside_the_data_flow_set_or_the_catalog_are_left_out(self):
        """R23: 327/377 (algoritmo debole, file temporaneo), 611 e le CWE fuori catalogo non entrano; `CVE-…` neanche."""
        notes, tree = self.prepare({f"{SECURITY}/CWE-327-BrokenCryptoAlgorithm/a.py": "x = 1  # $ Alert\n",
                                    f"{SECURITY}/CWE-377-InsecureTemporaryFile/a.py": "x = 1  # $ Alert\n",
                                    f"{SECURITY}/CWE-611-Xxe/a.py": "x = 1  # $ Alert\n",
                                    f"{SECURITY}/CWE-312-CleartextLogging/a.py": "x = 1  # $ Alert\n",
                                    f"{SECURITY}/CVE-2018-1281/a.py": "x = 1  # $ Alert\n",
                                    REL: PY_SRC})
        self.assertEqual(list(notes), [REL])
        self.assertFalse((tree / SECURITY / "CWE-327-BrokenCryptoAlgorithm").exists())

    def test_py2_paths_are_ignored(self):
        notes, _ = self.prepare({f"{SECURITY}/CWE-078-CommandInjection-py2/a.py": "x = 1  # $ Alert\n",
                                 f"{SECURITY}/CWE-078-CommandInjection/py2/b.py": "x = 1  # $ Alert\n",
                                 f"{SECURITY}/CWE-078-CommandInjection/c.txt": "not python\n",
                                 REL: PY_SRC})
        self.assertEqual(list(notes), [REL])

    def test_file_without_inline_annotations_is_marked_not_annotated(self):
        """R22: in un file senza `$` il sink non annotato non è per forza sicuro."""
        notes, _ = self.prepare({REL: "import os\n\ndef view(cmd):\n    os.system(cmd)\n"})
        self.assertFalse(notes[REL]["annotated"])
        self.assertEqual(notes[REL]["alert"], set())

    def test_file_with_only_source_annotations_counts_as_annotated(self):
        notes, _ = self.prepare({REL: "from flask import request  # $ Source\n\ndef view():\n    return request.args\n"})
        self.assertTrue(notes[REL]["annotated"])

    def test_crlf_is_normalized_so_engine_lines_match(self):
        notes, tree = self.prepare({REL: "import os\r\ndef view(cmd):\r\n    os.system(cmd)  # $ Alert\r\n"})
        self.assertNotIn("\r", (tree / REL).read_text(encoding="utf-8"))
        self.assertEqual(notes[REL]["alert"], {3})

    def test_ambiguous_lines_are_collected(self):
        notes, _ = self.prepare({REL: "import os\ndef view(cmd):\n    os.system(cmd)  # $ MISSING: Alert\n"})
        self.assertEqual((notes[REL]["alert"], notes[REL]["ambiguous"]), (set(), {3}))


def slice_rec(line: int, *, file: str = "test.py", function: str = "os.system", cwe: str = "CWE-78", start: int = 5,
              code: str = "def view(cmd):\n    os.system(cmd)\n    os.system('ls')") -> dict:
    block = {"file": file, "name": "view", "start_line": start, "end_line": start + code.count("\n"), "depth": 0,
             "lang": "python", "code": code}
    return {"sink": {"function": function, "file": file, "line": line, "cwe_id": cwe}, "blocks": [block], "truncated": False}


NOTES = {"test.py": {"alert": {6}, "ambiguous": {12}, "cwe": "CWE-78", "annotated": True, "root": "d"}}


class LabelSlicesTests(unittest.TestCase):
    def test_label_slices_alert_is_1_plain_sink_is_0_ambiguous_skipped(self):
        stats: Counter = Counter()
        out = cg.label_slices([slice_rec(6), slice_rec(7), slice_rec(12)], NOTES, stats=stats)
        self.assertEqual([(r["sink"]["line"], label) for r, label in out], [(6, 1), (7, 0)])
        self.assertEqual(stats["skipped_ambiguous"], 1)

    def test_a_sink_of_another_cwe_is_skipped_even_on_an_alert_line(self):
        stats: Counter = Counter()
        out = cg.label_slices([slice_rec(6, cwe="CWE-89"), slice_rec(7, cwe="CWE-89")], NOTES, stats=stats)
        self.assertEqual(out, [])
        self.assertEqual(stats["skipped_other_cwe"], 2)

    def test_not_annotated_and_unknown_files_give_no_labels(self):
        notes = {"test.py": {**NOTES["test.py"], "annotated": False}}
        stats: Counter = Counter()
        self.assertEqual(cg.label_slices([slice_rec(7), slice_rec(7, file="other.py")], notes, stats=stats), [])
        self.assertEqual((stats["skipped_not_annotated"], stats["skipped_unknown_file"]), (1, 1))

    def test_alert_on_a_later_line_of_a_multi_line_call_still_labels_the_call_as_vulnerable(self):
        code = "def view(cmd):\n    os.system(\n        cmd)\n    os.system('ls')"
        notes = {"test.py": {**NOTES["test.py"], "alert": {22}, "ambiguous": set()}}
        out = cg.label_slices([slice_rec(21, start=20, code=code), slice_rec(23, start=20, code=code)], notes)
        self.assertEqual([(r["sink"]["line"], label) for r, label in out], [(21, 1), (23, 0)])

    def test_ambiguous_line_inside_a_multi_line_call_skips_it(self):
        code = "def view(cmd):\n    os.system(\n        cmd)"
        notes = {"test.py": {**NOTES["test.py"], "alert": set(), "ambiguous": {22}}}
        self.assertEqual(cg.label_slices([slice_rec(21, start=20, code=code)], notes), [])

    def test_prefix_joins_slice_files_to_notes_of_the_whole_tree(self):
        notes = {"dir/test.py": NOTES["test.py"]}
        self.assertEqual(cg.label_slices([slice_rec(6)], notes), [])
        out = cg.label_slices([slice_rec(6)], notes, prefix="dir")
        self.assertEqual([label for _, label in out], [1])

    def test_slice_without_blocks_falls_back_to_the_sink_line(self):
        rec = slice_rec(7)
        rec["blocks"] = []
        out = cg.label_slices([rec], NOTES)
        self.assertEqual([label for _, label in out], [0], "senza blocco si ricade sulla sola riga del sink")


def pyblock(name: str, depth: int, code: str, start: int, file: str = "test.py") -> dict:
    return {"file": file, "name": name, "start_line": start, "end_line": start + code.count("\n"), "depth": depth,
            "lang": "python", "code": code}


def pyrec(blocks: list[dict], sink_line: int, function: str = "os.system", cwe: str = "CWE-78") -> dict:
    sink_block = next(b for b in blocks if b["depth"] == 0)
    return {"sink": {"function": function, "file": sink_block["file"], "line": sink_line, "cwe_id": cwe},
            "blocks": blocks, "truncated": False}


class ScrubPythonTests(unittest.TestCase):
    def test_comments_are_removed_but_a_hash_inside_a_string_is_not_a_comment(self):
        code, _ = cg.scrub_python('x = "# not a comment"  # real comment\ny = 1\n')
        self.assertEqual(code, 'x = "# not a comment"\ny = 1')

    def test_whole_line_comment_leaves_no_blank_line_and_the_map_points_to_original_rows(self):
        code, line_map = cg.scrub_python("def f():\n    # note\n    a = 1\n\n\n    return a\n")
        self.assertEqual(code, "def f():\n    a = 1\n\n    return a")
        self.assertEqual(line_map, [0, 2, 3, 5])

    def test_leaky_string_is_replaced_and_neutral_string_is_kept(self):
        code, _ = cg.scrub_python('a = "unsafe value"\nb = "ls -l"\nc = b"Good"\n')
        self.assertEqual(code, 'a = "text"\nb = "ls -l"\nc = "text"')

    def test_leaky_docstring_spanning_lines_collapses_to_one_line_and_rows_are_mapped(self):
        src = 'def f():\n    """Is this a good\n    sanitizer?"""\n    return 1\n'
        code, line_map = cg.scrub_python(src)
        self.assertEqual(code, 'def f():\n    "text"\n    return 1')
        self.assertEqual(line_map, [0, 1, 3])

    def test_blank_lines_inside_a_kept_multiline_string_are_untouched(self):
        code, _ = cg.scrub_python('q = """\nSELECT 1\n\n\nFROM t\n"""  # c\n')
        self.assertEqual(code, 'q = """\nSELECT 1\n\n\nFROM t\n"""')

    def test_fstring_text_is_left_alone_for_the_audit_to_judge(self):
        code, _ = cg.scrub_python('print(f"{x} unsafe")\n')
        self.assertEqual(code, 'print(f"{x} unsafe")')

    def test_non_ascii_text_does_not_shift_the_replacements(self):
        code, _ = cg.scrub_python('s = "è"; t = "bad"  # è\n')
        self.assertEqual(code, 's = "è"; t = "text"')

    def test_indented_method_code_is_accepted(self):
        code, line_map = cg.scrub_python("    def m(self):\n        # c\n        return 1\n")
        self.assertEqual(code, "    def m(self):\n        return 1")
        self.assertEqual(line_map, [0, 2])

    def test_untokenizable_code_raises_value_error(self):
        with self.assertRaises(ValueError):
            cg.scrub_python("x = (1,\n")

    def test_crlf_becomes_lf(self):
        self.assertEqual(cg.scrub_python("a = 1\r\nb = 2\r\n")[0], "a = 1\nb = 2")


class ScrubRecordTests(unittest.TestCase):
    def record(self) -> dict:
        caller = pyblock("middle", 1, "def middle(x):\n    # pass it on\n    view(x)", 11)
        sink = pyblock("view", 0, 'def view(cmd):\n    """A good sanitizer"""\n    # GOOD\n    os.system(cmd)', 5)
        return pyrec([caller, sink], sink_line=8)

    def test_scrub_record_cleans_blocks_and_recomputes_sink_line_end_line_and_start_lines(self):
        rec = self.record()
        before = json.dumps(rec, sort_keys=True)
        out = cg.scrub_record(rec, "seed")
        self.assertEqual(json.dumps(rec, sort_keys=True), before, "il record in ingresso non cambia")
        caller, sink = out["blocks"]
        self.assertEqual(caller["code"], "def middle(x):\n    view(x)")
        self.assertEqual(sink["code"], 'def view(cmd):\n    "text"\n    os.system(cmd)')
        for b in out["blocks"]:
            self.assertTrue(1 <= b["start_line"] <= 900, b["start_line"])
            self.assertEqual(b["end_line"], b["start_line"] + b["code"].count("\n"))
        self.assertEqual(out["sink"]["line"], sink["start_line"] + 2, "la riga del sink segue il codice ripulito")
        self.assertEqual(out["sink"]["file"], sink["file"])

    def test_start_lines_are_synthetic_deterministic_and_seed_dependent(self):
        rec = self.record()
        again = cg.scrub_record(rec, "seed")["blocks"]
        self.assertEqual([b["start_line"] for b in cg.scrub_record(rec, "seed")["blocks"]], [b["start_line"] for b in again])
        starts = {tuple(b["start_line"] for b in cg.scrub_record(rec, f"s{i}")["blocks"]) for i in range(6)}
        self.assertGreater(len(starts), 1, "il seme cambia le righe: nessuna posizione originale nel prompt")

    def test_sink_line_that_is_no_longer_in_the_cleaned_code_raises(self):
        caller = pyblock("middle", 1, "def middle(x):\n    view(x)", 11)
        sink = pyblock("view", 0, "def view(cmd):\n    # only a comment here\n    os.system(cmd)", 5)
        with self.assertRaises(ValueError):
            cg.scrub_record(pyrec([caller, sink], sink_line=6), "seed")

    def test_no_label_word_survives_in_comments_or_docstrings(self):
        out = cg.scrub_record(self.record(), "seed")
        self.assertEqual(leak_audit.forbidden_hits(
            [{"meta": {"trace": {"sink": out["sink"], "blocks": out["blocks"]}}, "messages": []}],
            mode="parts", extra_parts=cg.PY_EXTRA_PARTS), [])


class RenameTests(unittest.TestCase):
    CALLER = 'def good_caller(req):\n    fixture = req.args\n    prefix = "unsafe"\n    bad_cmd(fixture)\n'
    SINK = "def bad_cmd(x):\n    os.system(x)\n"

    def record(self) -> dict:
        caller = pyblock("good_caller", 1, self.CALLER, 11, file="bad_example.py")
        sink = pyblock("bad_cmd", 0, self.SINK, 20, file="good_example.py")
        return pyrec([caller, sink], sink_line=21)

    def rename_one(self, code: str, seed: str = "s") -> str:
        """Codice del blocco dopo la rinomina, col nome (sempre hash, R25) rimesso a `handler` per confrontare il resto."""
        block = cg.rename_leaky_names(pyrec([pyblock("handler", 0, code, 5)], sink_line=6), seed)["blocks"][0]
        return block["code"].replace(block["name"], "handler")

    def test_rename_leaky_names_hides_bad_good_safe_unsafe_and_keeps_calls_consistent(self):
        out = cg.rename_leaky_names(self.record(), "seed")
        caller, sink = out["blocks"]
        self.assertRegex(caller["name"], r"^fn_[0-9a-f]{6}$")
        self.assertRegex(sink["name"], r"^fn_[0-9a-f]{6}$")
        self.assertIn(f"def {caller['name']}(req):", caller["code"])
        self.assertIn(f"{sink['name']}(fixture)", caller["code"], "la chiamata segue la definizione")
        self.assertIn(f"def {sink['name']}(x):", sink["code"])
        self.assertIn("fixture = req.args", caller["code"], "`fixture` non è una parola-spia")
        self.assertIn('prefix = "unsafe"', caller["code"], "`prefix` resta e le stringhe non si toccano")
        self.assertNotIn("bad_cmd", json.dumps(out))
        self.assertNotIn("good_caller", json.dumps(out))

    def test_every_block_function_is_hashed_even_when_its_name_looks_innocent(self):
        """R25: `ok8`, `not_ok7`, `full_ssrf`, `command_injection1` sono metadati d'autore: nessuna lista di parole li copre."""
        caller = pyblock("not_ok7", 1, "def not_ok7():\n    full_ssrf(1)\n", 20)
        sink = pyblock("full_ssrf", 0, "def full_ssrf(u):\n    requests.get(u)\n", 5)
        out = cg.rename_leaky_names(pyrec([caller, sink], sink_line=6, function="requests.get", cwe="CWE-918"), "s")
        names = [b["name"] for b in out["blocks"]]
        self.assertTrue(all(re.fullmatch(r"fn_[0-9a-f]{6}", n) for n in names), names)
        self.assertIn(f"{names[1]}(1)", out["blocks"][0]["code"], "la chiamata segue la definizione")
        self.assertIn("requests.get(u)", out["blocks"][1]["code"], "i nomi d'API non si toccano")
        for word in ("not_ok7", "full_ssrf"):
            self.assertNotIn(word, json.dumps(out))

    def test_a_method_call_through_an_attribute_keeps_its_name(self):
        code = "def run(self):\n    self.get(1)\n    requests.get(2)\n"
        self.assertIn("self.get(1)", self.rename_one(code), "un attributo non si riscrive: potrebbe essere `requests.get`")

    def test_variables_and_parameters_defined_in_the_slice_are_renamed_everywhere_in_the_block(self):
        text = self.rename_one("def handler(unsafe_arg):\n    unsafe_cmd = 'ls ' + unsafe_arg\n    os.system(unsafe_cmd)\n"
                               "    return unsafe_cmd.strip()\n")
        self.assertNotIn("unsafe", text)
        self.assertEqual(len(set(re.findall(r"\bv_[0-9a-f]{6}\b", text))), 2)
        self.assertEqual(re.sub(r"\bv_[0-9a-f]{6}\b", "V", text),
                         "def handler(V):\n    V = 'ls ' + V\n    os.system(V)\n    return V.strip()\n")

    def test_keyword_argument_of_a_renamed_parameter_follows_it(self):
        text = cg.rename_leaky_names(pyrec([
            pyblock("outer", 1, "def outer():\n    inner(safe_value=1)\n", 20),
            pyblock("inner", 0, "def inner(safe_value):\n    os.system(safe_value)\n", 5)], sink_line=6), "s")
        param = re.search(r"def fn_[0-9a-f]{6}\((v_[0-9a-f]{6})\)", text["blocks"][1]["code"]).group(1)
        self.assertIn(f"{text['blocks'][1]['name']}({param}=1)", text["blocks"][0]["code"])

    def test_attributes_and_names_not_defined_in_the_slice_are_left_alone(self):
        code = "def handler(p):\n    yaml.safe_load(p)\n    unsafe_cache.append(p)\n    obj.unsafe = p\n"
        self.assertEqual(self.rename_one(code), code, "nulla di ciò che è definito qui è una parola-spia")

    def test_a_leaky_attribute_named_like_a_renamed_variable_is_not_touched(self):
        text = self.rename_one("def handler(p):\n    unsafe = p\n    return obj.unsafe + unsafe\n")
        self.assertIn("obj.unsafe + v_", text)
        self.assertEqual(text.count("unsafe"), 1)

    def test_imports_are_not_renamed(self):
        code = "def handler(p):\n    from yaml import safe_load\n    return safe_load(p)\n"
        self.assertEqual(self.rename_one(code), code, "un nome d'API non si riscrive: se è una spia lo scarta il controllo finale")

    def test_identifiers_with_leak_substrings_stay_whole(self):
        code = "def handler(prefix, fixture, badge_count):\n    return prefix + fixture + badge_count\n"
        self.assertEqual(self.rename_one(code), code)

    @unittest.skipUnless(sys.version_info >= (3, 12), "le espressioni delle f-string sono token solo da Python 3.12")
    def test_names_inside_fstring_expressions_are_renamed(self):
        text = self.rename_one("def handler(unsafe_x):\n    os.system(f'echo {unsafe_x}')\n")
        self.assertNotIn("unsafe", text)
        self.assertRegex(text, r"def handler\((v_[0-9a-f]{6})\):\n    os\.system\(f'echo \{\1\}'\)")

    def test_rename_is_deterministic_and_depends_on_the_seed(self):
        names = lambda seed: [b["name"] for b in cg.rename_leaky_names(self.record(), seed)["blocks"]]  # noqa: E731
        self.assertEqual(names("a"), names("a"))
        self.assertGreater(len({tuple(names(f"s{i}")) for i in range(5)}), 1)

    def test_files_become_src_hash_py_and_the_sink_file_follows_the_depth_zero_block(self):
        out = cg.rename_leaky_names(self.record(), "seed")
        files = [b["file"] for b in out["blocks"]]
        self.assertTrue(all(re.fullmatch(r"src/[0-9a-f]{8}\.py", f) for f in files), files)
        self.assertEqual(len(set(files)), 2, "un nome per file originale")
        self.assertEqual(out["sink"]["file"], out["blocks"][1]["file"])
        same = cg.rename_leaky_names(pyrec([pyblock("a", 1, "def a():\n    b()\n", 30, file="t.py"),
                                            pyblock("b", 0, "def b():\n    os.system(1)\n", 5, file="t.py")], 6), "seed")
        self.assertEqual(same["blocks"][0]["file"], same["blocks"][1]["file"])

    def test_lines_and_the_input_record_are_not_changed(self):
        rec = self.record()
        before = json.dumps(rec, sort_keys=True)
        out = cg.rename_leaky_names(rec, "seed")
        self.assertEqual(json.dumps(rec, sort_keys=True), before)
        self.assertEqual([(b["start_line"], b["end_line"]) for b in out["blocks"]],
                         [(b["start_line"], b["end_line"]) for b in rec["blocks"]])
        self.assertEqual(out["sink"]["line"], 21)

    def test_unparsable_block_raises_value_error(self):
        with self.assertRaises(ValueError):
            cg.rename_leaky_names(pyrec([pyblock("handler", 0, "def handler(:\n    pass\n", 5)], sink_line=6), "s")


class AnonymizePythonTests(unittest.TestCase):
    def test_anonymize_python_leaves_no_leaky_identifier_comment_or_string(self):
        caller = pyblock("good_caller", 1, 'def good_caller(req):\n    # BAD: unsafe path\n    unsafe_cmd = req.args\n'
                         '    """A good sanitizer"""\n    run_bad(unsafe_cmd)', 11)
        sink = pyblock("run_bad", 0, "def run_bad(c):\n    os.system(c)", 30)
        out = cg.anonymize_python(pyrec([caller, sink], sink_line=31), "seed")
        example = {"meta": {"trace": {"sink": out["sink"], "blocks": out["blocks"]}}, "messages": []}
        self.assertEqual(leak_audit.forbidden_hits([example], mode="parts", extra_parts=cg.PY_EXTRA_PARTS), [])
        self.assertEqual(out["sink"]["line"], out["blocks"][1]["start_line"] + 1)

    def test_api_names_whose_safe_word_is_the_semantics_are_exempt(self):
        self.assertTrue({"SafeLoader", "safe_load"} <= cg.API_LEAKY_NAMES)
        for name in cg.API_LEAKY_NAMES:
            self.assertTrue(cg.has_leaky_word(name), f"{name} è nell'elenco solo se il controllo lo scarterebbe")

    def test_catalog_sink_names_with_a_leak_part_are_exempt_vocabulary(self):
        self.assertTrue({"mark_safe", "unsafe_load"} <= cg.catalog_leaky_names(KB))
        self.assertNotIn("system", cg.catalog_leaky_names(KB))


CMD_SRC = """import os
from flask import request  # $ Source


def good_run(cmd):
    # BAD: runs whatever it gets
    os.system(cmd)  # $ Alert


def handler():
    data = request.args.get("c")
    good_run(data)


def constant():
    os.system("ls")
"""
POPEN_SRC = "import os\n\n\ndef view(p):\n    os.popen(p)  # $ Alert\n"
SQL_SRC = """def q(cur, uid):
    cur.execute("SELECT * FROM t WHERE id = " + uid)  # $ Alert


def q2(cur):
    cur.execute("SELECT 1")
"""
YAML_SRC = """import yaml
from flask import request  # $ Source


def load_it():
    return yaml.load(request.data, Loader=yaml.SafeLoader)
"""
HIDDEN_SRC = """import os


def helper(p):
    os.system(get_safe_path(p))  # $ Alert
"""
CODEQL_FILES = {
    f"{SECURITY}/CWE-078-CommandInjection/cmd.py": CMD_SRC,
    f"{SECURITY}/CWE-078-CommandInjection/popen.py": POPEN_SRC,
    f"{SECURITY}/CWE-078-CommandInjection/hidden.py": HIDDEN_SRC,
    f"{SECURITY}/CWE-089-SqlInjection/sql.py": SQL_SRC,
    f"{SECURITY}/CWE-502-UnsafeDeserialization/yaml_test.py": YAML_SRC,
    f"{SECURITY}/CWE-089-SqlInjection/plain.py": "def r(cur, x):\n    cur.execute(x)\n",  # nessuna annotazione
    f"{SECURITY}/CWE-078-CommandInjection-py2/old.py": "import os\nos.system('x')  # $ Alert\n",
    f"{SECURITY}/CWE-327-BrokenCryptoAlgorithm/c.py": "import hashlib\nhashlib.md5(b'x')  # $ Alert\n",
}
FAKE_ENGINE = """#!/bin/sh
d="$(dirname "$0")"
r="$(basename "$2")"
echo "$r" >> "$d/roots.log"
if [ -f "$d/$r.jsonl" ]; then cat "$d/$r.jsonl"; fi
exit 0
"""


def fake_slices(root: Path) -> list[dict]:
    """Come il motore: una slice per chiamata a un sink del catalogo, con i chiamanti diretti dello stesso file."""
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        funcs = localize.split_python_functions(path.read_text(encoding="utf-8"))
        blocks = {f["name"]: {"file": rel, "name": f["name"], "start_line": f["start_line"], "lang": "python", "code": f["code"],
                              "end_line": f["start_line"] + f["code"].count("\n")} for f in funcs}
        for f in funcs:
            for callee, line in localize.find_calls(f["code"], "python"):
                entry = common.match_sink(KB, "python", callee)
                if entry is None:
                    continue
                chain = [dict(blocks[f["name"]], depth=0)]
                chain += [dict(blocks[g["name"]], depth=1) for g in funcs
                          if g["name"] != f["name"] and re.search(rf"\b{f['name']}\(", g["code"])]
                chain.sort(key=lambda b: -b["depth"])
                out.append({"sink": {"function": callee, "file": rel, "line": f["start_line"] + line - 1, "cwe_id": entry["id"]},
                            "blocks": chain, "truncated": False})
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


class CodeqlTrackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.codeql = write_tree(self.dir / "raw" / "codeql-tests", CODEQL_FILES)
        prep = self.dir / "prep"
        notes = cg.prepare_tree(self.codeql, prep, KB)
        engine_dir = self.dir / "engine"
        engine_dir.mkdir()
        self.slices: dict[str, list[dict]] = {}
        for root_rel in sorted({n["root"] for n in notes.values()}):
            root = prep / root_rel
            self.slices[root.name] = fake_slices(root)
            (engine_dir / f"{root.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.slices[root.name]), encoding="utf-8")
        self.engine = engine_dir / "fake-engine"
        self.engine.write_text(FAKE_ENGINE, encoding="utf-8")
        self.engine.chmod(0o755)
        self.raw, self.out, self.eval = self.dir / "raw", self.dir / "graph", self.dir / "eval"

    def build(self, splits: dict[str, str] | None = None, **kw) -> dict:
        args = dict(raw_dir=self.raw, out_dir=self.out, eval_dir=self.eval, seed=42, engine_bin=self.engine)
        args.update(kw)
        patch = mock.patch.object(prepare_data, "split_of", side_effect=lambda g, *_: splits.get(g, "train")) if splits else contextlib.nullcontext()
        with patch, contextlib.redirect_stderr(io.StringIO()):
            return build_graph_data.build_codeql(**args)

    def outputs(self) -> dict[str, list[dict]]:
        return {"train": read_jsonl(self.out / "py_codeql_train.jsonl"), "val": read_jsonl(self.out / "py_codeql_val.jsonl"),
                "test": read_jsonl(self.eval / "py_codeql.jsonl")}

    def test_writes_train_val_eval_and_report(self):
        report = self.build()
        for path in (self.out / "py_codeql_train.jsonl", self.out / "py_codeql_val.jsonl", self.eval / "py_codeql.jsonl",
                     self.out / "py_codeql_report.json"):
            self.assertTrue(path.exists(), path)
        self.assertEqual(json.loads((self.out / "py_codeql_report.json").read_text(encoding="utf-8")), report)

    def test_split_group_is_the_file_with_the_prepare_data_key(self):
        """R21: stesso gruppo e stesso `split_of` di `prepare_data`, così un file del test.jsonl non entra mai in train/val."""
        cmd = f"codeql:{SECURITY}/CWE-078-CommandInjection/cmd.py"
        popen = f"codeql:{SECURITY}/CWE-078-CommandInjection/popen.py"
        self.build(splits={cmd: "test", popen: "val"})
        out = self.outputs()
        sinks = {k: {ex["meta"]["group"] for ex in v} for k, v in out.items()}
        self.assertEqual(sinks["test"], {cmd})
        self.assertEqual(sinks["val"], {popen})
        self.assertNotIn(cmd, sinks["train"] | sinks["val"])
        self.assertNotIn(popen, sinks["train"] | sinks["test"])
        calls = []
        with mock.patch.object(prepare_data, "split_of", side_effect=lambda *a: calls.append(a) or "train"), \
                contextlib.redirect_stderr(io.StringIO()):
            build_graph_data.build_codeql(self.raw, self.out, self.eval, seed=7, engine_bin=self.engine, val_pct=3, test_pct=11)
        self.assertTrue(calls and all(c[1:] == (7, 3, 11) for c in calls), calls)
        self.assertEqual({c[0] for c in calls}, {cmd, popen, f"codeql:{SECURITY}/CWE-089-SqlInjection/sql.py",
                                                  f"codeql:{SECURITY}/CWE-502-UnsafeDeserialization/yaml_test.py"})

    def test_labels_follow_alert_annotations_and_files_without_annotations_give_nothing(self):
        self.build()
        examples = [ex for exs in self.outputs().values() for ex in exs]
        labels = Counter((ex["meta"]["group"].rsplit("/", 1)[-1], ex["meta"]["label"]) for ex in examples)
        self.assertEqual(labels[("cmd.py", 1)], 1)
        self.assertEqual(labels[("cmd.py", 0)], 1)
        self.assertEqual(labels[("sql.py", 1)], 1)
        self.assertEqual(labels[("sql.py", 0)], 1)
        self.assertEqual(labels[("popen.py", 1)], 1)
        self.assertEqual(labels[("yaml_test.py", 0)], 1, "yaml.SafeLoader è la semantica dell'API: non si scarta (R26)")
        self.assertNotIn("plain.py", {g for g, _ in labels}, "senza `$` non si può dire cosa è sicuro")
        self.assertFalse({g for g, _ in labels} & {"old.py", "c.py"}, "py2 e CWE-327 restano fuori")

    def test_examples_carry_track_source_and_no_labels_in_the_prompt(self):
        self.build()
        allow = cg.catalog_leaky_names(KB) | cg.API_LEAKY_NAMES
        examples = [ex for exs in self.outputs().values() for ex in exs]
        self.assertTrue(examples)
        for ex in examples:
            self.assertEqual((ex["meta"]["track"], ex["meta"]["source"], ex["meta"]["synthetic"], ex["meta"]["lang"]),
                             (graph_examples.PY_CODEQL_TRACK, build_graph_data.PY_SOURCE, False, "python"))
            user = ex["messages"][1]["content"]
            for marker in ("$ Alert", "# BAD", "# GOOD", "$ Source", "good_run", "get_safe_path"):
                self.assertNotIn(marker, user)
            self.assertEqual(common.validate_finding(json.loads(ex["messages"][2]["content"])), [])
        self.assertEqual(leak_audit.forbidden_hits(examples, mode="parts", extra_parts=cg.PY_EXTRA_PARTS, allow=allow), [])

    def test_record_calling_a_hidden_function_with_a_leaky_name_is_dropped_and_counted(self):
        report = self.build()
        self.assertEqual(report["leaky_dropped"], 1)
        self.assertEqual(report["leaky_words"], {"get_safe_path": 1})
        groups = {ex["meta"]["group"].rsplit("/", 1)[-1] for exs in self.outputs().values() for ex in exs}
        self.assertNotIn("hidden.py", groups)

    def test_engine_runs_once_per_query_directory_with_annotated_files(self):
        self.build()
        roots = (self.dir / "engine" / "roots.log").read_text(encoding="utf-8").split()
        self.assertEqual(sorted(roots), ["CWE-078-CommandInjection", "CWE-089-SqlInjection", "CWE-502-UnsafeDeserialization"])

    def test_truncated_slices_are_dropped_and_counted(self):
        for rec in self.slices["CWE-078-CommandInjection"]:
            if rec["sink"]["function"] == "os.popen":
                rec["truncated"] = True
        (self.dir / "engine" / "CWE-078-CommandInjection.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.slices["CWE-078-CommandInjection"]), encoding="utf-8")
        report = self.build()
        self.assertEqual(report["truncated_dropped"], 1)
        self.assertNotIn("popen.py", {ex["meta"]["group"].rsplit("/", 1)[-1] for exs in self.outputs().values() for ex in exs})

    def test_sink_only_views_appear_only_for_slices_with_callers_and_keep_the_full_trace(self):
        self.build(p_callers=0.0, val_pct=0, test_pct=0)
        examples = [ex for exs in self.outputs().values() for ex in exs]
        only = [ex for ex in examples if ex["meta"]["view"] == "sink_only"]
        self.assertTrue(only, "con p_callers=0 almeno la slice con chiamanti passa alla view solo-sink")
        for ex in only:
            self.assertGreater(len(ex["meta"]["trace"]["blocks"]), 1, "solo-sink solo se la slice ha chiamanti")
            self.assertEqual(ex["messages"][1]["content"].count("(lines "), 1)

    def test_output_is_deterministic(self):
        self.build()
        first = {p.name: p.read_bytes() for p in [*self.out.glob("py_codeql*"), *self.eval.glob("py_codeql*")]}
        report = self.build()
        report.pop("seconds", None)
        second = {p.name: p.read_bytes() for p in [*self.out.glob("py_codeql*"), *self.eval.glob("py_codeql*")]}
        self.assertEqual({k: v for k, v in first.items() if not k.endswith("report.json")},
                         {k: v for k, v in second.items() if not k.endswith("report.json")})

    def test_report_counts_files_records_and_skips(self):
        report = self.build()
        self.assertEqual(report["files"], 6, "py2 e CWE-327 non sono nell'albero")
        self.assertEqual(report["files_not_annotated"], 1)
        self.assertEqual(report["skipped_not_annotated"], 1, "il sink di plain.py sta in una radice annotata ma il suo file no")
        self.assertEqual(report["roots"], 3)
        self.assertEqual(report["records"], sum(len(v) for v in self.slices.values()))
        self.assertEqual(report["labels"]["1"] + report["labels"]["0"], sum(report["written"].values()))
        self.assertEqual(sum(v["train"] for v in report["per_cwe"].values()), report["written"]["train"])

    def test_cli_runs_the_codeql_subcommand(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = build_graph_data.main(["codeql", "--raw-dir", str(self.raw), "--out-dir", str(self.out), "--eval-dir", str(self.eval),
                                          "--engine-bin", str(self.engine)])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "py_codeql_report.json").exists())

    @unittest.skipUnless(slice_io.engine_bin(), "serve il binario di sast-engine")
    def test_end_to_end_with_real_engine(self):
        tree = write_tree(self.dir / "real", {REL: PY_SRC})
        notes = cg.prepare_tree(tree, self.dir / "real_prep", KB)
        records, _ = slice_io.run_slice(self.dir / "real_prep" / notes[REL]["root"])
        labelled = cg.label_slices(records, notes, prefix=notes[REL]["root"])
        self.assertEqual(sorted(label for _, label in labelled), [0, 1], "un sink vulnerabile e uno sicuro")


if __name__ == "__main__":
    unittest.main()
