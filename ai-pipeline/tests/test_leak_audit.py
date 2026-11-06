"""Test di `leak_audit`: parole-spia nei blocchi, token sbilanciati, sacco di token e classificatore di sola struttura.

Gli esempi si costruiscono con `graph_examples.build_example` da record di slice scritti a mano: niente motore né Juliet.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import common
import leak_audit
from graph_examples import JULIET_TRACK, build_example

KB = common.load_kb()
HERE = Path(__file__).resolve().parents[1]
META = {"source": "juliet-nist", "synthetic": False, "track": JULIET_TRACK, "group": "g", "variant": "54"}


def block(name: str, depth: int, code: str, start: int = 10, file: str = "src/0a1b2c3d.c") -> dict:
    return {"file": file, "name": name, "start_line": start, "end_line": start + code.count("\n"),
            "depth": depth, "lang": "c", "code": code}


def example(label: int, sink_body: str, callers: list[str] = (), *, view: str = "callers", group: str = "g") -> dict:
    """Esempio CWE-78 con il sink `system` nel blocco a profondità 0 e i chiamanti dati (dal più esterno)."""
    sink_code = "void fn_a0a0a0(char * data)\n{\n    " + sink_body + "\n}"
    n = len(callers)
    blocks = [block(f"fn_c{d}c{d}c{d}", d, code, 100 * (n - d + 1)) for d, code in zip(range(n, 0, -1), callers)]
    blocks.append(block("fn_a0a0a0", 0, sink_code, 500))
    rec = {"sink": {"function": "system", "file": "src/0a1b2c3d.c", "line": 502, "cwe_id": "CWE-78"},
           "blocks": blocks, "truncated": False}
    return build_example(rec, KB, label=label, variant_kind="bad" if label else "g2b", origin=None,
                         meta=dict(META, group=group), view=view)


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.write_text("".join(json.dumps(ex) + "\n" for ex in examples), encoding="utf-8")


class ForbiddenHitsTests(unittest.TestCase):
    def test_forbidden_hits_finds_leak_word_in_a_block(self):
        leaky = example(1, "system(data);", ["void fn_c1c1c1(void)\n{\n    goodG2B(data);\n}"])
        clean = example(0, "system(data);", ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"])
        hits = leak_audit.forbidden_hits([clean, leaky])
        self.assertTrue(hits)
        self.assertTrue(all("goodG2B" in h for h in hits), hits)
        self.assertEqual(leak_audit.forbidden_hits([clean]), [])

    def test_forbidden_hits_looks_at_block_names_and_files(self):
        ex = example(1, "system(data);")
        ex["meta"]["trace"]["blocks"][0]["name"] = "CWE78_x_badSink"
        ex["meta"]["trace"]["blocks"][0]["file"] = "s01/CWE78_x_54e.c"
        hits = leak_audit.forbidden_hits([ex])
        self.assertTrue(any("name" in h for h in hits), hits)
        self.assertTrue(any("file" in h for h in hits), hits)

    def test_parts_mode_ignores_words_that_only_contain_a_leak_substring(self):
        ex = example(1, "system(prefix);", ["def fn_c1c1c1(fixture):\n    return fixture"])
        self.assertTrue(leak_audit.forbidden_hits([ex], mode="substring"))
        self.assertEqual(leak_audit.forbidden_hits([ex], mode="parts"), [])
        bad = example(1, "system(data);", ["def fn_c1c1c1(bad_value):\n    return bad_value"])
        self.assertTrue(leak_audit.forbidden_hits([bad], mode="parts"))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            leak_audit.forbidden_hits([], mode="regex")


class TokenTests(unittest.TestCase):
    def test_top_tokens_puts_a_token_only_in_safe_examples_first(self):
        exs = [example(1, "system(data);", group=f"v{i}") for i in range(10)]
        exs += [example(0, "sanitize(data); system(data);", group=f"s{i}") for i in range(10)]
        top = leak_audit.top_tokens(exs, k=5)
        token, log_odds, n_vuln, n_safe = top[0]
        self.assertEqual(token, "sanitize")
        self.assertLess(log_odds, 0, "log-odds negativo = sbilanciato verso le sicure")
        self.assertEqual((n_vuln, n_safe), (0, 10))
        self.assertLessEqual(len(top), 5)
        magnitudes = [abs(t[1]) for t in top]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_top_tokens_reads_only_the_blocks_shown_by_the_view(self):
        callers = ["void fn_c1c1c1(void)\n{\n    hidden_token(data);\n}"]
        exs = [example(0, "system(data);", callers, view="sink_only", group=f"s{i}") for i in range(4)]
        exs += [example(1, "run(data); system(data);", group=f"v{i}") for i in range(4)]
        self.assertNotIn("hidden_token", [t[0] for t in leak_audit.top_tokens(exs, k=50)])

    def test_bow_accuracy_is_one_on_a_separable_case(self):
        train = [example(1, "system(data);", group=f"v{i}") for i in range(8)]
        train += [example(0, "sanitize(data); system(data);", group=f"s{i}") for i in range(8)]
        heldout = [example(1, "system(data);", group="hv"), example(0, "sanitize(data); system(data);", group="hs")]
        self.assertEqual(leak_audit.bow_accuracy(train, heldout), 1.0)

    def test_bow_accuracy_of_empty_heldout_is_zero(self):
        self.assertEqual(leak_audit.bow_accuracy([example(1, "system(data);")], []), 0.0)


class StructureTests(unittest.TestCase):
    def test_structure_accuracy_learns_block_count(self):
        callers = ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"]
        train = [example(1, "system(data);", group=f"v{i}") for i in range(5)]
        train += [example(0, "system(data);", callers, group=f"s{i}") for i in range(5)]
        heldout = [example(1, "system(data);", group="hv"), example(0, "system(data);", callers, group="hs")]
        self.assertEqual(leak_audit.structure_accuracy(train, heldout), 1.0)

    def test_structure_accuracy_is_near_chance_when_shapes_do_not_depend_on_the_label(self):
        callers = ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"]
        train = [example(label, "system(data);", callers if i % 2 else [], group=f"t{label}{i}")
                 for label in (0, 1) for i in range(6)]
        heldout = [example(label, "system(data);", callers if i % 2 else [], group=f"h{label}{i}")
                   for label in (0, 1) for i in range(4)]
        self.assertLessEqual(leak_audit.structure_accuracy(train, heldout), 0.5)

    def test_structure_uses_the_blocks_shown_by_the_view(self):
        callers = ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"]
        ex = example(1, "system(data);", callers, view="sink_only")
        self.assertEqual(leak_audit.structure_table([ex]), {"1": {"0": 0, "1": 1}})

    def test_structure_table_counts_labels_by_block_count(self):
        callers = ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"]
        exs = [example(1, "system(data);"), example(0, "system(data);", callers), example(0, "system(data);", callers)]
        self.assertEqual(leak_audit.structure_table(exs), {"1": {"0": 0, "1": 1}, "2": {"0": 2, "1": 0}})


class CliTests(unittest.TestCase):
    def run_cli(self, train: list[dict], heldout: list[dict]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            tr, he = Path(d) / "train.jsonl", Path(d) / "heldout.jsonl"
            write_jsonl(tr, train)
            write_jsonl(he, heldout)
            return subprocess.run([sys.executable, str(HERE / "leak_audit.py"), "--train", str(tr), "--heldout", str(he)],
                                  capture_output=True, text=True, cwd=HERE)

    def balanced(self, prefix: str, n: int) -> list[dict]:
        return [example(label, "system(data);" if i % 2 else "run(data); system(data);", group=f"{prefix}{label}{i}")
                for label in (0, 1) for i in range(n)]

    def test_cli_exits_one_on_leaks(self):
        leaky = example(1, "system(data);", ["void fn_c1c1c1(void)\n{\n    badSink(data);\n}"])
        out = self.run_cli(self.balanced("t", 4) + [leaky], self.balanced("h", 2))
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("badSink", out.stdout + out.stderr)

    def test_cli_exits_zero_without_leaks_and_prints_the_measures(self):
        out = self.run_cli(self.balanced("t", 4), self.balanced("h", 2))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        for word in ("forbidden", "top_tokens", "bow_accuracy", "structure_accuracy", "structure_table"):
            self.assertIn(word, out.stdout)

    def test_cli_exits_one_when_structure_alone_predicts_the_label(self):
        callers = ["void fn_c1c1c1(void)\n{\n    fn_a0a0a0(data);\n}"]
        train = [example(1, "system(data);", group=f"v{i}") for i in range(5)]
        train += [example(0, "system(data);", callers, group=f"s{i}") for i in range(5)]
        heldout = [example(1, "system(data);", group="hv"), example(0, "system(data);", callers, group="hs")]
        out = self.run_cli(train, heldout)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("structure", (out.stdout + out.stderr).lower())


if __name__ == "__main__":
    unittest.main()
