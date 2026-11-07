"""Test di `synth_python` (traccia P3) e del sottocomando `synth` di `build_graph_data` con un motore finto.

Il generatore scrive piccoli pacchetti Python a coppie di gemelli: stessa catena di funzioni fino al sink, cambia solo il
valore che il chiamante in cima passa (input non fidato oppure costante o controllato), così decidono i chiamanti.
"""
import ast
import contextlib
import io
import json
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import build_graph_data
import c_anonymize
import codeql_graph
import common
import graph_examples
import leak_audit
import localize
import synth_python as sp

KB = common.load_kb()
LABEL_PARTS = frozenset({"safe", "unsafe", "tainted", "vuln", "vulnerable", "clean", "const", "bad", "good"})
ALL_PARTS = LABEL_PARTS | codeql_graph.PY_EXTRA_PARTS
WORD_RE = re.compile(r"[^\W\d]\w*")


class Generated:
    """Cartella temporanea con `n` pacchetti generati e il manifesto."""

    def __init__(self, n: int = 64, seed: int = 1):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.manifest = sp.generate(self.dir, n, seed)

    def text(self, entry: dict) -> str:
        return (self.dir / entry["file"]).read_text(encoding="utf-8")

    def close(self) -> None:
        self._tmp.cleanup()


def functions_of(text: str) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def source_of(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def sink_function(text: str, line: int) -> str:
    return next(name for name, node in functions_of(text).items() if node.lineno <= line <= node.end_lineno)


def callers_of(text: str, name: str) -> list[str]:
    return [other for other, node in functions_of(text).items()
            if other != name and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
                                     for c in ast.walk(node))]


def chain_top(text: str, line: int) -> str:
    """La funzione più in alto della catena di chiamanti che porta al sink (una sola: la catena è lineare)."""
    top = sink_function(text, line)
    while callers_of(text, top):
        (top,) = callers_of(text, top)
    return top


def chain_depth(text: str, line: int) -> int:
    """Quante funzioni stanno sopra quella del sink, seguendo i chiamanti per nome."""
    depth, level = 0, [sink_function(text, line)]
    while True:
        level = [c for name in level for c in callers_of(text, name)]
        if not level:
            return depth
        depth += 1


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = Generated(64, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.gen.close()

    def test_sources_parse_and_manifest_line_holds_the_callee(self):
        m = self.gen.manifest
        self.assertEqual(len(m), 64)
        for entry in m:
            self.assertLessEqual({"file", "line", "callee", "cwe_id", "label", "chain_len"}, set(entry))
            text = self.gen.text(entry)
            ast.parse(text)
            self.assertIn(entry["callee"].split(".")[-1], text.split("\n")[entry["line"] - 1], entry)
        self.assertEqual(len({e["file"] for e in m}), 64, "un pacchetto, un sink")

    def test_each_package_is_a_directory_with_one_module(self):
        for entry in self.gen.manifest:
            path = Path(entry["file"])
            self.assertEqual(len(path.parts), 2, entry["file"])
            self.assertEqual([p.name for p in (self.gen.dir / path.parent).iterdir()], [path.name])

    def test_callees_are_catalog_sinks(self):
        for entry in self.gen.manifest:
            found = common.match_sink(KB, "python", entry["callee"])
            self.assertIsNotNone(found, entry["callee"])
            self.assertEqual(found["id"], entry["cwe_id"])
        self.assertEqual({e["cwe_id"] for e in self.gen.manifest}, {"CWE-601", "CWE-78", "CWE-89", "CWE-22"})

    def test_labels_are_balanced_and_chain_lengths_vary(self):
        labels = Counter(e["label"] for e in self.gen.manifest)
        self.assertLessEqual(abs(labels[1] - labels[0]), 1)
        self.assertEqual({e["chain_len"] for e in self.gen.manifest}, {0, 1, 2, 3})
        for cwe in ("CWE-601", "CWE-78", "CWE-89", "CWE-22"):
            self.assertEqual({e["label"] for e in self.gen.manifest if e["cwe_id"] == cwe}, {0, 1}, cwe)

    def test_chain_len_is_the_number_of_functions_above_the_sink(self):
        for entry in self.gen.manifest:
            self.assertEqual(chain_depth(self.gen.text(entry), entry["line"]), entry["chain_len"], entry["file"])

    def test_deterministic_for_same_seed_and_different_for_another(self):
        other = Generated(64, seed=1)
        another = Generated(64, seed=2)
        try:
            self.assertEqual(self.gen.manifest, other.manifest)
            self.assertEqual([self.gen.text(e) for e in self.gen.manifest], [other.text(e) for e in other.manifest])
            self.assertNotEqual([another.text(e) for e in another.manifest], [self.gen.text(e) for e in self.gen.manifest])
        finally:
            other.close()
            another.close()

    def test_no_identifier_or_string_carries_label_words(self):
        for entry in self.gen.manifest:
            tree = ast.parse(self.gen.text(entry))
            words = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    words.add(node.name)
                elif isinstance(node, ast.Name):
                    words.add(node.id)
                elif isinstance(node, ast.arg):
                    words.add(node.arg)
                elif isinstance(node, ast.keyword) and node.arg:
                    words.add(node.arg)
                elif isinstance(node, ast.Attribute):
                    words.add(node.attr)
                elif isinstance(node, ast.alias):
                    words.update(node.name.split("."))
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    words.update(WORD_RE.findall(node.value))
            leaky = sorted(w for w in words if c_anonymize.leaky_identifier(w, ALL_PARTS))
            self.assertEqual(leaky, [], entry["file"])

    def test_twins_share_the_chain_down_to_the_sink_and_differ_only_in_the_value_at_the_top(self):
        groups: dict[str, list[dict]] = defaultdict(list)
        for entry in self.gen.manifest:
            groups[entry["group"]].append(entry)
        self.assertGreater(len(groups), 20)
        for group, pair in groups.items():
            self.assertEqual(len(pair), 2, group)
            self.assertEqual(sorted(e["label"] for e in pair), [0, 1])
            a, b = pair
            self.assertEqual((a["callee"], a["cwe_id"], a["chain_len"]), (b["callee"], b["cwe_id"], b["chain_len"]))
            ta, tb = self.gen.text(a), self.gen.text(b)
            self.assertNotEqual(ta, tb)
            if a["chain_len"] == 0:
                continue
            fa, fb = functions_of(ta), functions_of(tb)
            self.assertEqual(sorted(fa), sorted(fb), "stesse funzioni, stessi nomi")
            differing = {n for n in fa if source_of(ta, fa[n]) != source_of(tb, fb[n])}
            self.assertEqual(differing, {chain_top(ta, a["line"])}, f"{group}: cambia solo la funzione in cima alla catena")

    def test_negatives_include_validated_and_long_constant_forms_and_positives_are_tainted(self):
        kinds = Counter((e["label"], e["kind"]) for e in self.gen.manifest)
        self.assertEqual({k for lab, k in kinds if lab == 1}, {"tainted"})
        self.assertEqual({k for lab, k in kinds if lab == 0}, {"constant", "validated", "long_constant", "decoy_source"})
        for entry in self.gen.manifest:
            text = self.gen.text(entry)
            has_source = any(marker in text for marker in sp.SOURCE_MARKERS)
            if entry["kind"] in ("tainted", "validated", "decoy_source"):
                self.assertTrue(has_source, f"{entry['kind']} senza sorgente: {entry['file']}")
            else:
                self.assertFalse(has_source, f"{entry['kind']} con una sorgente: {entry['file']}")

    def test_a_source_in_the_file_does_not_decide_the_label(self):
        """Un classificatore che guarda solo se c'è una sorgente non fidata (e un controllo) sbaglierebbe molti negativi."""
        negatives = [e for e in self.gen.manifest if e["label"] == 0]
        with_source = [e for e in negatives if any(m in self.gen.text(e) for m in sp.SOURCE_MARKERS)]
        self.assertGreaterEqual(len(with_source) / len(negatives), 0.4)

    def test_decoy_source_is_read_but_never_reaches_the_sink(self):
        decoys = [e for e in self.gen.manifest if e["kind"] == "decoy_source"]
        self.assertTrue(decoys)
        for entry in decoys:
            text = self.gen.text(entry)
            tree = ast.parse(text)
            sink_fn = functions_of(text)[sink_function(text, entry["line"])]
            if entry["chain_len"] > 0:  # a catena zero il sink sta nella stessa funzione che legge la sorgente
                self.assertFalse(any(m in source_of(text, sink_fn) for m in sp.SOURCE_MARKERS), entry["file"])
            top = functions_of(text)[chain_top(text, entry["line"])]
            assigned = {t.id: n.value for n in ast.walk(top) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            self.assertTrue(any(any(m in ast.get_source_segment(text, v) for m in sp.SOURCE_MARKERS) for v in assigned.values()), entry["file"])
            self.assertIsNotNone(tree)

    def test_length_is_not_a_label_tell(self):
        """R19: la sorgente visibile e la lunghezza non devono dire il verdetto."""
        lines = {label: [self.gen.text(e).count("\n") for e in self.gen.manifest if e["label"] == label] for label in (0, 1)}
        mean = {label: sum(v) / len(v) for label, v in lines.items()}
        ratio = mean[1] / mean[0]
        self.assertTrue(0.8 <= ratio <= 1.25, (mean, ratio))

    def test_odd_n_still_balances_labels(self):
        gen = Generated(9, seed=3)
        try:
            labels = Counter(e["label"] for e in gen.manifest)
            self.assertEqual(len(gen.manifest), 9)
            self.assertLessEqual(abs(labels[1] - labels[0]), 1)
        finally:
            gen.close()


class CapShareTests(unittest.TestCase):
    @staticmethod
    def examples(n: int, *, grouped: bool = False) -> list[dict]:
        return [{"id": i, "meta": {"group": f"g{i // 2}" if grouped else f"s{i}", "label": i % 2}} for i in range(n)]

    def test_cap_share_never_exceeds_fifteen_percent(self):
        real = self.examples(100)
        for n in (5, 17, 18, 100, 400):
            kept = sp.cap_share(self.examples(n), real)
            self.assertLessEqual(len(kept) / (len(real) + len(kept)), 0.15, n)
        self.assertEqual(len(sp.cap_share(self.examples(400), real)), 17)

    def test_cap_share_without_real_examples_keeps_none(self):
        self.assertEqual(sp.cap_share(self.examples(10), []), [])

    def test_cap_share_keeps_whole_groups_in_the_original_order_and_is_deterministic(self):
        synthetic = self.examples(60, grouped=True)
        kept = sp.cap_share(synthetic, self.examples(100))
        self.assertEqual(kept, sp.cap_share(synthetic, self.examples(100)))
        ids = [e["id"] for e in kept]
        self.assertEqual(ids, sorted(ids))
        groups = Counter(e["meta"]["group"] for e in kept)
        self.assertTrue(groups and set(groups.values()) == {2}, "gemelli sempre insieme")
        self.assertLessEqual(len(kept), 17)

    def test_cap_share_one_keeps_everything(self):
        synthetic = self.examples(30)
        self.assertEqual(sp.cap_share(synthetic, self.examples(3), share=1.0), synthetic)


# --------------------------------------------------------------------------- sottocomando `synth`
FAKE_ENGINE = """#!/bin/sh
d="$(dirname "$0")"
r="$(basename "$2")"
echo "$r" >> "$d/roots.log"
if [ -f "$d/$r.jsonl" ]; then cat "$d/$r.jsonl"; fi
exit 0
"""


def fake_record(root: Path, entry: dict) -> dict:
    """Slice come la farebbe il motore: la funzione del sink a profondità 0 e i suoi chiamanti per nome fino a 5."""
    rel = Path(entry["file"]).name
    text = (root / rel).read_text(encoding="utf-8")
    funcs = {f["name"]: f for f in localize.split_python_functions(text)}

    def block(name: str, depth: int) -> dict:
        f = funcs[name]
        return {"file": rel, "name": name, "start_line": f["start_line"], "end_line": f["start_line"] + f["code"].count("\n"),
                "depth": depth, "lang": "python", "code": f["code"]}

    start = sink_function(text, entry["line"])
    blocks, level, seen = [block(start, 0)], [start], {start}
    for depth in range(1, 6):
        level = [c for n in level for c in callers_of(text, n) if c not in seen]
        seen.update(level)
        blocks += [block(n, depth) for n in level]
    blocks.sort(key=lambda b: -b["depth"])
    return {"sink": {"function": entry["callee"], "file": rel, "line": entry["line"], "cwe_id": entry["cwe_id"]},
            "blocks": blocks, "truncated": False}


class SynthTrackTests(unittest.TestCase):
    N = 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        gen_dir = self.dir / "gen"
        self.manifest = sp.generate(gen_dir, self.N, 42)
        engine_dir = self.dir / "engine"
        engine_dir.mkdir()
        for entry in self.manifest:
            root = gen_dir / Path(entry["file"]).parent
            (engine_dir / f"{root.name}.jsonl").write_text(json.dumps(fake_record(root, entry)) + "\n", encoding="utf-8")
        self.engine = engine_dir / "fake-engine"
        self.engine.write_text(FAKE_ENGINE, encoding="utf-8")
        self.engine.chmod(0o755)
        self.out, self.eval = self.dir / "graph", self.dir / "eval"

    def write_real(self, n: int) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"meta": {"track": "py_codeql", "label": i % 2}}) for i in range(n)]
        (self.out / "py_codeql_train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def build(self, **kw) -> dict:
        args = dict(out_dir=self.out, n=self.N, seed=42, engine_bin=self.engine)
        args.update(kw)
        with contextlib.redirect_stderr(io.StringIO()):
            return build_graph_data.build_synth(**args)

    def written(self) -> list[dict]:
        return [json.loads(line) for name in ("py_synth_train.jsonl", "py_synth_val.jsonl")
                for line in (self.out / name).read_text(encoding="utf-8").split("\n") if line.strip()]

    def test_writes_train_val_and_report_and_never_an_eval_file(self):
        self.write_real(200)
        report = self.build(val_pct=20)
        for name in ("py_synth_train.jsonl", "py_synth_val.jsonl", "py_synth_report.json"):
            self.assertTrue((self.out / name).exists(), name)
        self.assertFalse(self.eval.exists(), "i sintetici non entrano mai nella valutazione")
        self.assertEqual(json.loads((self.out / "py_synth_report.json").read_text(encoding="utf-8")), report)

    def test_share_of_synthetic_among_python_examples_is_capped_by_the_real_ones(self):
        self.write_real(100)
        report = self.build()
        self.assertEqual(report["real_python_examples"], 100)
        self.assertEqual(report["cap"], 17)
        self.assertLessEqual(sum(report["written"].values()), 17)
        self.assertLessEqual(sum(report["written"].values()) / (100 + sum(report["written"].values())), 0.15)
        self.assertGreater(sum(report["written"].values()), 0)

    def test_without_real_python_examples_nothing_is_written(self):
        report = self.build()
        self.assertEqual((report["real_python_examples"], report["cap"], sum(report["written"].values())), (0, 0, 0))

    def test_real_count_can_be_given_explicitly(self):
        report = self.build(real_examples=1000)
        self.assertEqual((report["real_python_examples"], report["cap"]), (1000, 176))

    def test_examples_are_synthetic_python_slices_with_the_pair_as_group_and_twins_in_the_same_split(self):
        self.write_real(400)
        report = self.build(val_pct=30)
        examples = self.written()
        self.assertTrue(examples)
        for ex in examples:
            meta = ex["meta"]
            self.assertEqual((meta["track"], meta["source"], meta["synthetic"], meta["lang"]),
                             (graph_examples.PY_SYNTH_TRACK, "py-synth", True, "python"))
            self.assertEqual(len(meta["trace"]["blocks"]), meta["chain_len"] + 1)
            self.assertIn(meta["synth_kind"], {"tainted", *sp.NEGATIVE_KINDS})
            self.assertEqual(meta["synth_kind"] == "tainted", meta["label"] == 1)
            self.assertEqual(common.validate_finding(json.loads(ex["messages"][2]["content"])), [])
        split_of = {}
        for name in ("train", "val"):
            for line in (self.out / f"py_synth_{name}.jsonl").read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    split_of.setdefault(json.loads(line)["meta"]["group"], set()).add(name)
        self.assertTrue(all(len(s) == 1 for s in split_of.values()), "un gruppo sta in un solo insieme")
        self.assertEqual({"train", "val"}, {next(iter(s)) for s in split_of.values()})
        self.assertEqual(report["labels"]["1"] + report["labels"]["0"], len(examples))

    def test_no_label_word_in_any_example(self):
        self.write_real(400)
        self.build()
        self.assertEqual(leak_audit.forbidden_hits(self.written(), mode="parts", extra_parts=ALL_PARTS,
                                                    allow=codeql_graph.catalog_leaky_names(KB)), [])

    def test_sink_only_view_is_never_chosen_when_twins_share_the_sink_function(self):
        self.write_real(400)
        self.build(p_callers=0.0)
        for ex in self.written():
            if ex["meta"]["view"] == "sink_only":
                self.fail("i gemelli hanno lo stesso solo-sink con etichette opposte: nessuna view solo-sink")

    def test_engine_runs_once_per_package_with_one_target(self):
        self.write_real(400)
        self.build()
        roots = (self.dir / "engine" / "roots.log").read_text(encoding="utf-8").split()
        self.assertEqual(len(roots), self.N)
        self.assertEqual(len(set(roots)), self.N)

    def test_cli_runs_the_synth_subcommand(self):
        self.write_real(200)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = build_graph_data.main(["synth", "--out-dir", str(self.out), "--n", str(self.N), "--engine-bin", str(self.engine)])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "py_synth_report.json").exists())

    def test_output_is_deterministic(self):
        self.write_real(300)
        self.build()
        first = {p.name: p.read_bytes() for p in self.out.glob("py_synth_t*.jsonl")}
        self.build()
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.out.glob("py_synth_t*.jsonl")})


if __name__ == "__main__":
    unittest.main()
