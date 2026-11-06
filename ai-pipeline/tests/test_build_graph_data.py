"""Test di `build_graph_data juliet` con un motore finto (script `sh`) e un archivio Juliet in miniatura.

Lo zip contiene pochi test case CWE-78 scritti come quelli di Juliet; il motore finto restituisce, per ogni radice, record
di slice costruiti qui dai file estratti (catene di chiamanti come quelle del motore vero, `main` e dispatcher `…_good`
compresi). Niente rete, niente Rust.

Test case del finto archivio:
- `_52` (due file): sink gemelli `52b_badSink`/`52b_goodG2BSink` con corpo identico; decidono i chiamanti in `52a`.
- `_44`: sink statici gemelli chiamati per puntatore a funzione, senza chiamanti: tutti i prompt in conflitto.
- `_01`: `bad` e `goodG2B` con sorgente e sink nella stessa funzione (corpi diversi).
- `_03`: un record troncato, uno la cui funzione a profondità 0 non ha etichetta, un target saltato dal motore.
- una radice CWE-190 senza sink del catalogo: nessun target, il motore non va lanciato.
"""
import contextlib
import io
import json
import random
import re
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

import build_graph_data
import c_anonymize
import common
import graph_examples
import juliet_graph
import leak_audit
import localize
from c_anonymize import LEAK_RE

KB = common.load_kb()
HEADER_RE = re.compile(r"^### .*depth \d+\)$", re.M)
P = "CWE78_OS_Command_Injection__char_environment_system"
Q = "CWE78_OS_Command_Injection__char_console_system"

SINK_BODY = """{
    /* POTENTIAL FLAW: Execute command in data possibly leading to command injection */
    EXECL(COMMAND_INT_PATH, COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG3, NULL);
}
"""

FILES = {
    f"CWE78_OS_Command_Injection/s01/{P}_52a.c": f"""/* TEMPLATE GENERATED TESTCASE FILE
Filename: {P}_52a.c
*/
#include "std_testcase.h"

#ifndef OMITBAD

void {P}_52b_badSink(char * data);

void {P}_52_bad()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    {{
        /* POTENTIAL FLAW: Read data from an environment variable */
        char * environment = GETENV(ENV_VARIABLE);
        if (environment != NULL)
        {{
            strncat(data, environment, 100-1);
        }}
    }}
    {P}_52b_badSink(data);
}}

#endif /* OMITBAD */

#ifndef OMITGOOD

void {P}_52b_goodG2BSink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    /* FIX: Append a fixed string to data (not user / external input) */
    strcat(data, "*.*");
    {P}_52b_goodG2BSink(data);
}}

void {P}_52_good()
{{
    goodG2B();
}}

#endif /* OMITGOOD */

int main(int argc, char * argv[])
{{
    printLine("Calling good()...");
    {P}_52_good();
    printLine("Calling bad()...");
    {P}_52_bad();
    return 0;
}}
""",
    f"CWE78_OS_Command_Injection/s01/{P}_52b.c": f"""#include "std_testcase.h"

#ifndef OMITBAD

void {P}_52b_badSink(char * data)
{SINK_BODY}
#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void {P}_52b_goodG2BSink(char * data)
{SINK_BODY}
#endif /* OMITGOOD */
""",
    f"CWE78_OS_Command_Injection/s01/{P}_44.c": f"""#include "std_testcase.h"

#ifndef OMITBAD

static void badSink(char * data)
{SINK_BODY}
void {P}_44_bad()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    void (*funcPtr) (char *) = badSink;
    data = data_buf;
    strncat(data, GETENV(ENV_VARIABLE), 100-1);
    funcPtr(data);
}}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void goodG2BSink(char * data)
{SINK_BODY}
static void goodG2B()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    void (*funcPtr) (char *) = goodG2BSink;
    data = data_buf;
    strcat(data, "*.*");
    funcPtr(data);
}}

void {P}_44_good()
{{
    goodG2B();
}}

#endif /* OMITGOOD */
""",
    f"CWE78_OS_Command_Injection/s01/{P}_01.c": f"""#include "std_testcase.h"

#ifndef OMITBAD

void {P}_01_bad()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    strncat(data, GETENV(ENV_VARIABLE), 100-1);
    if (SYSTEM(data) != 0)
    {{
        exit(1);
    }}
}}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void goodG2B()
{{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    /* FIX: Benign, fixed string */
    strcat(data, "*.*");
    if (SYSTEM(data) != 0)
    {{
        exit(1);
    }}
}}

void {P}_01_good()
{{
    goodG2B();
}}

#endif /* OMITGOOD */

int main(int argc, char * argv[])
{{
    {P}_01_good();
    {P}_01_bad();
    return 0;
}}
""",
    f"CWE78_OS_Command_Injection/s02/{Q}_03.c": f"""#include "std_testcase.h"

void {Q}_03_bad()
{{
    char data[100] = "";
    fgets(data, 100, stdin);
    SYSTEM(data);
}}

static void goodG2B1()
{{
    char data[100] = "ls";
    SYSTEM(data);
}}

static void goodG2B2()
{{
    char data[100] = "dir";
    SYSTEM(data);
}}
""",
    "CWE190_Integer_Overflow/s01/CWE190_Integer_Overflow__int_rand_add_01.c": """#include "std_testcase.h"

void CWE190_Integer_Overflow__int_rand_add_01_bad()
{
    int data = RAND32();
    int result = data + 1;
    printIntLine(result);
}
""",
}

# Catene dei chiamanti (dal chiamante diretto verso l'esterno) per la funzione che contiene il sink, come le dà il motore.
CHAINS = {
    f"{P}_52b_badSink": [(f"{P}_52a.c", f"{P}_52_bad"), (f"{P}_52a.c", "main")],
    f"{P}_52b_goodG2BSink": [(f"{P}_52a.c", "goodG2B"), (f"{P}_52a.c", f"{P}_52_good"), (f"{P}_52a.c", "main")],
    "badSink": [],
    "goodG2BSink": [],
    f"{P}_01_bad": [(f"{P}_01.c", "main")],
    "goodG2B": [(f"{P}_01.c", f"{P}_01_good"), (f"{P}_01.c", "main")],
    f"{Q}_03_bad": [],
    "goodG2B1": [],
}

FAKE_ENGINE = """#!/bin/sh
d="$(dirname "$0")"
r="$(basename "$2")"
echo "$r" >> "$d/roots.log"
if [ -f "$d/$r.err" ]; then cat "$d/$r.err" >&2; fi
if [ -f "$d/$r.jsonl" ]; then cat "$d/$r.jsonl"; fi
exit 0
"""


def make_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, text in FILES.items():
            z.writestr(f"C/testcases/{name}", text.replace("\n", "\r\n"))


def functions_by_file(root: Path) -> dict[str, dict[str, dict]]:
    """{nome file: {nome funzione: blocco come lo restituirebbe il motore (senza profondità)}}."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(root.rglob("*.c")):
        rel = path.relative_to(root).as_posix()
        for f in localize.split_c_functions(path.read_text(encoding="utf-8")):
            out.setdefault(path.name, {})[f["name"]] = {
                "file": rel, "name": f["name"], "start_line": f["start_line"],
                "end_line": f["start_line"] + f["code"].count("\n"), "lang": "c", "code": f["code"]}
    return out


def fake_records(root: Path) -> tuple[list[dict], str]:
    """Record del motore finto per i target di `root` e il suo stderr (con un target «fuori da ogni funzione»)."""
    funcs = functions_by_file(root)
    records, err = [], ""
    for n, t in enumerate(juliet_graph.sink_targets(root, KB), start=1):
        fname = t["file"].rsplit("/", 1)[-1]
        holder = next(b for b in funcs[fname].values() if b["start_line"] <= t["line"] <= b["end_line"])
        if holder["name"] == "goodG2B2":
            err += f"riga {n}: {t['file']}:{t['line']} fuori da ogni funzione\n"
            continue
        blocks = [dict(holder, depth=0)]
        for depth, (file_name, name) in enumerate(CHAINS[holder["name"]], start=1):
            blocks.append(dict(funcs[file_name][name], depth=depth))
        blocks.sort(key=lambda b: (-b["depth"], b["file"], b["start_line"]))
        rec = {"sink": {"function": t["callee"], "file": t["file"], "line": t["line"], "cwe_id": t["cwe_id"]},
               "blocks": blocks, "truncated": False}
        if holder["name"] == f"{Q}_03_bad":
            rec["truncated"] = True
        if holder["name"] == "goodG2B1":
            rec["blocks"][0]["name"] = "helperBad"
        records.append(rec)
    return records, err


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


class JulietTrackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.zip = self.dir / "juliet.zip"
        make_zip(self.zip)
        prep = self.dir / "prep"
        engine_dir = self.dir / "engine"
        engine_dir.mkdir()
        for root in juliet_graph.extract_c_sources(self.zip, prep, KB):
            records, err = fake_records(root)
            if records or err:
                (engine_dir / f"{root.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                (engine_dir / f"{root.name}.err").write_text(err, encoding="utf-8")
        self.engine = engine_dir / "fake-engine"
        self.engine.write_text(FAKE_ENGINE, encoding="utf-8")
        self.engine.chmod(0o755)
        self.raw, self.out, self.eval = self.dir / "raw", self.dir / "graph", self.dir / "eval"

    def build(self, **kw) -> dict:
        args = dict(raw_dir=self.raw, out_dir=self.out, eval_dir=self.eval, seed=42, max_train=100, max_eval=100,
                    zip_path=self.zip, engine_bin=self.engine, val_pct=0, test_pct=100, holdout_pct=0)
        args.update(kw)
        with contextlib.redirect_stderr(io.StringIO()):
            return build_graph_data.build_juliet(**args)

    def outputs(self) -> dict[str, list[dict]]:
        return {
            "train": read_jsonl(self.out / "juliet_train.jsonl"),
            "val": read_jsonl(self.out / "juliet_val.jsonl"),
            "test": read_jsonl(self.eval / "juliet_test.jsonl"),
            "unseen": read_jsonl(self.eval / "juliet_unseen_variants.jsonl"),
            "pairs": read_jsonl(self.eval / "juliet_pairs.jsonl"),
        }

    def test_cli_writes_all_output_files(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = build_graph_data.main(["juliet", "--raw-dir", str(self.raw), "--out-dir", str(self.out),
                                          "--eval-dir", str(self.eval), "--seed", "42", "--max-train", "50",
                                          "--max-eval", "20", "--zip", str(self.zip), "--engine-bin", str(self.engine)])
        self.assertEqual(code, 0)
        for path in (self.out / "juliet_train.jsonl", self.out / "juliet_val.jsonl", self.out / "juliet_report.json",
                     self.eval / "juliet_test.jsonl", self.eval / "juliet_unseen_variants.jsonl",
                     self.eval / "juliet_pairs.jsonl"):
            self.assertTrue(path.is_file(), path)

    def test_no_leak_word_or_original_path_in_any_example(self):
        self.build()
        everything = [ex for exs in self.outputs().values() for ex in exs]
        self.assertTrue(everything)
        self.assertEqual(leak_audit.forbidden_hits(everything), [])
        for ex in everything:
            for b in ex["meta"]["trace"]["blocks"]:
                self.assertIsNone(LEAK_RE.search(b["code"] + b["name"] + b["file"]), b)
                self.assertRegex(b["file"], r"^src/[0-9a-f]{8}\.c$")
            prompt = ex["messages"][1]["content"]
            self.assertNotIn("s01/", prompt)
            self.assertNotIn("OS_Command_Injection__", prompt)
            self.assertRegex(prompt, HEADER_RE)

    def test_twins_form_a_pair_with_equal_pair_id_and_opposite_labels(self):
        self.build()
        pairs = self.outputs()["pairs"]
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["meta"]["pair_id"], pairs[1]["meta"]["pair_id"])
        self.assertIsNotNone(pairs[0]["meta"]["pair_id"])
        self.assertEqual(sorted(ex["meta"]["label"] for ex in pairs), [0, 1])
        self.assertEqual({ex["meta"]["group"] for ex in pairs}, {f"{P}_52"})
        self.assertTrue(all(ex["meta"]["view"] == "callers" for ex in pairs))
        # Il blocco a profondità 0 dei gemelli è identico byte per byte dopo l'anonimizzazione.
        sinks = [[b for b in ex["meta"]["trace"]["blocks"] if b["depth"] == 0][0] for ex in pairs]
        self.assertEqual(sinks[0], sinks[1])
        self.assertEqual(pairs[0]["meta"]["trace"]["sink"], pairs[1]["meta"]["trace"]["sink"])

    def test_command_arg3_macro_is_shown_as_data(self):
        """In Juliet CWE-78 `#define COMMAND_ARG3 data`: senza riscriverla il sink `execl` non mostra il dato."""
        self.build()
        sink_block = [b for b in self.outputs()["pairs"][0]["meta"]["trace"]["blocks"] if b["depth"] == 0][0]
        self.assertIn("execl(COMMAND_INT_PATH, COMMAND_INT_PATH, COMMAND_ARG1, data, NULL);", sink_block["code"])
        self.assertNotIn("COMMAND_ARG3", sink_block["code"])

    def test_report_counts_drops_conflicts_and_testcases_without_examples(self):
        report = self.build()
        on_disk = json.loads((self.out / "juliet_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report, on_disk)
        self.assertEqual(report["truncated_dropped"], 1)
        self.assertEqual(report["unlabelled_dropped"], 1)
        self.assertEqual(report["targets_skipped_by_engine"], 1)
        self.assertEqual(report["targets_without_record"], 1)
        self.assertEqual(report["examples_before_conflicts"], 6)
        self.assertEqual(report["conflicts_dropped"], 2)
        self.assertAlmostEqual(report["undecidable_share"], 2 / 6, places=4)
        # `_44` (tutto in conflitto) e `_03` (nulla di utilizzabile) non danno esempi.
        self.assertEqual(report["testcases_without_examples"], 2)
        self.assertIn(f"{P}_44", report["testcases_without_examples_sample"])
        self.assertEqual(report["pairs"], 1)
        for key in ("bow_accuracy", "structure_accuracy", "structure_table", "top_tokens", "sink_only_share",
                    "per_cwe", "per_cwe_kind_split", "held_out_variants", "val_pct", "test_pct",
                    "dispatcher_blocks_dropped", "main_blocks_dropped", "candidates", "multi_block_share",
                    "train_twin_sets", "holdout_strata", "unseen_variants_blocks", "source_in_callee_dropped",
                    "source_in_callee_by_variant", "undecidable_or_callee_share", "stump_accuracies",
                    "hidden_callee_dropped", "hidden_callee_by_variant", "pairs_different_length_within_15pct",
                    "pairs_different_length", "pairs_vulnerable_longer", "pairs_length_within_15pct"):
            self.assertIn(key, report)

    def test_all_conflict_testcase_produces_no_examples(self):
        self.build()
        groups = {ex["meta"]["group"] for exs in self.outputs().values() for ex in exs}
        self.assertNotIn(f"{P}_44", groups)
        self.assertNotIn(f"{Q}_03", groups)
        self.assertEqual(groups, {f"{P}_52", f"{P}_01"})

    def test_dispatcher_and_main_blocks_are_dropped(self):
        report = self.build()
        self.assertEqual(report["dispatcher_blocks_dropped"], 2)  # `_52_good` e `_01_good`
        self.assertEqual(report["main_blocks_dropped"], 4)  # `main` nelle catene di `_52` e `_01`
        test = self.outputs()["test"]
        shape = {(ex["meta"]["group"], ex["meta"]["label"]): sorted(b["depth"] for b in ex["meta"]["trace"]["blocks"])
                 for ex in test}
        self.assertEqual(shape[(f"{P}_52", 1)], [0, 1])
        self.assertEqual(shape[(f"{P}_52", 0)], [0, 1])
        self.assertEqual(shape[(f"{P}_01", 1)], [0])
        self.assertEqual(shape[(f"{P}_01", 0)], [0])

    def test_each_record_is_anonymized_with_its_testcase_seed(self):
        calls = []
        real = c_anonymize.anonymize_slice

        def spy(rec, seed, macros):
            calls.append((juliet_graph.testcase_group(next(b for b in rec["blocks"] if b["depth"] == 0)["file"]), seed))
            return real(rec, seed, macros)

        with mock.patch.object(build_graph_data.c_anonymize, "anonymize_slice", side_effect=spy):
            self.build(seed=7)
        self.assertEqual(len(calls), 6)
        for group, seed in calls:
            self.assertEqual(seed, f"7:{group}")

    def test_labels_and_metadata_come_from_original_names(self):
        self.build()
        test = self.outputs()["test"]
        by_key = {(ex["meta"]["group"], ex["meta"]["label"]): ex["meta"] for ex in test}
        bad = by_key[(f"{P}_52", 1)]
        self.assertEqual((bad["variant_kind"], bad["variant"], bad["origin"]), ("bad", "52", "environment"))
        self.assertEqual(by_key[(f"{P}_52", 0)]["variant_kind"], "g2b")
        for meta in by_key.values():
            self.assertEqual((meta["source"], meta["synthetic"], meta["track"]), ("juliet-nist", False, "juliet"))

    def test_held_out_variants_appear_only_in_unseen_variants(self):
        report = self.build(holdout_pct=12, test_pct=50, val_pct=0, holdout_min_examples=1)
        held = set(report["held_out_variants"])
        # Uno strato per forma: `_52` ha chiamanti, `_01` è a un blocco; ciascuno dà almeno una variante (R17).
        self.assertEqual(report["holdout_strata"], {"caller": ["52"], "single": ["01"], "ineligible": []})
        self.assertEqual(held, {"01", "52"})
        self.assertEqual(report["unseen_variants_blocks"], {"single": 2, "multi": 2})
        out = self.outputs()
        self.assertTrue(out["unseen"])
        self.assertTrue(all(ex["meta"]["variant"] in held for ex in out["unseen"]))
        for name in ("train", "val", "test"):
            self.assertFalse([ex for ex in out[name] if ex["meta"]["variant"] in held], name)
        self.assertTrue(all(ex["meta"]["view"] == "callers" for ex in out["unseen"] + out["test"]))

    def test_train_receives_examples_when_nothing_goes_to_test(self):
        report = self.build(test_pct=0, val_pct=0)
        out = self.outputs()
        self.assertEqual(len(out["train"]), 4)
        self.assertEqual(out["test"], [])
        self.assertEqual(out["pairs"], [])
        self.assertIn("overall", report["sink_only_share"])

    def test_root_without_targets_is_not_sliced(self):
        report = self.build()
        roots = (self.engine.parent / "roots.log").read_text(encoding="utf-8").split()
        self.assertEqual(roots, ["CWE78_OS_Command_Injection"])
        self.assertEqual(report["roots_without_targets"], ["CWE190_Integer_Overflow"])


def _pool_block(name: str, depth: int, code: str, file: str) -> dict:
    return {"file": file, "name": name, "start_line": 10 * (depth + 1), "end_line": 10 * (depth + 1) + code.count("\n"),
            "depth": depth, "lang": "c", "code": code}


def _pool_example(group: str, variant: str, label: int, kind: str, caller: str | None, sink: str,
                  cwe: str = "CWE-78") -> dict:
    """Esempio vero (`build_example`) con un chiamante opzionale; il file dipende dal gruppo come dopo l'anonimizzazione."""
    file = f"src/{graph_examples._sha(group)[:8]}.c"
    sink_code = f"void fn_a0a0a0(char * data)\n{{\n    {sink}\n    system(data);\n}}"
    blocks = [_pool_block("fn_a0a0a0", 0, sink_code, file)]
    if caller is not None:
        blocks.insert(0, _pool_block("fn_b1b1b1", 1, f"void fn_b1b1b1(void)\n{{\n    {caller}\n    fn_a0a0a0(data);\n}}", file))
    rec = {"sink": {"function": "system", "file": file, "line": 13, "cwe_id": cwe}, "blocks": blocks, "truncated": False}
    meta = {"source": "juliet-nist", "synthetic": False, "track": graph_examples.JULIET_TRACK, "group": group,
            "variant": variant}
    return graph_examples.build_example(rec, KB, label=label, variant_kind=kind, origin=None, meta=meta)


def synthetic_pool() -> list[dict]:
    """Gruppi a catena (gemelli bad/g2b con lo stesso sink, a volte anche un b2g) e molti esempi a un blocco sbilanciati."""
    pool = []
    for i in range(60):
        group, variant = f"CWE78_x_{i}_5{i % 4}", f"5{i % 4}"
        pool.append(_pool_example(group, variant, 1, "bad", f"data = getenv(\"A{i}\");", f"use({i});"))
        pool.append(_pool_example(group, variant, 0, "g2b", f"data = \"ls{i}\";", f"use({i});"))
        if i % 3 == 0:
            pool.append(_pool_example(group, variant, 0, "b2g", f"data = getenv(\"A{i}\");", f"check({i});"))
    for i in range(400):
        label = int(i % 8 < 3)  # 150 vulnerabili, 250 sicuri
        pool.append(_pool_example(f"CWE78_y_{i}_0{i % 5}", f"0{i % 5}", label, "bad" if label else "g2b", None,
                                  f"single({i});"))
    return pool


def blocks_of(ex: dict) -> int:
    return len(ex["meta"]["trace"]["blocks"])


class CapTrainTests(unittest.TestCase):
    """R16: campione di train ricco di chiamanti e contrastivo (gruppi interi di gemelli)."""

    @classmethod
    def setUpClass(cls):
        cls.pool = synthetic_pool()

    def test_twins_of_a_selected_group_stay_together(self):
        out = build_graph_data.cap_train(self.pool, 100, 0.5, seed=3)
        chosen = {graph_examples.prompt_key(ex) for ex in out}
        groups = {ex["meta"]["group"] for ex in out if blocks_of(ex) >= 2}
        self.assertTrue(groups)
        for ex in self.pool:
            if ex["meta"]["group"] in groups and ex["meta"]["variant_kind"] in ("bad", "g2b"):
                self.assertIn(graph_examples.prompt_key(ex), chosen, ex["meta"]["group"])

    def test_multi_block_share_follows_the_option_and_labels_stay_balanced(self):
        for share in (0.3, 0.5, 0.7):
            out = build_graph_data.cap_train(self.pool, 100, share, seed=3)
            self.assertEqual(len(out), 100)
            multi = sum(blocks_of(ex) >= 2 for ex in out) / len(out)
            self.assertLessEqual(abs(multi - share), 0.05, (share, multi))
            labels = Counter(ex["meta"]["label"] for ex in out)
            self.assertLessEqual(abs(labels[0] - labels[1]), 1, labels)

    def test_cap_train_is_deterministic_and_order_independent(self):
        out = build_graph_data.cap_train(self.pool, 100, 0.5, seed=3)
        self.assertEqual(out, build_graph_data.cap_train(self.pool, 100, 0.5, seed=3))
        shuffled = list(self.pool)
        random.Random(0).shuffle(shuffled)
        again = build_graph_data.cap_train(shuffled, 100, 0.5, seed=3)
        self.assertEqual({graph_examples.prompt_key(e) for e in again}, {graph_examples.prompt_key(e) for e in out})

    def test_multi_share_zero_is_the_old_uniform_cap(self):
        self.assertEqual(build_graph_data.cap_train(self.pool, 100, 0.0, seed=3),
                         graph_examples.cap_balanced(self.pool, 100, 3))

    def test_twin_sets_are_counted_per_group(self):
        out = build_graph_data.cap_train(self.pool, 100, 0.5, seed=3)
        groups = {ex["meta"]["group"] for ex in out if blocks_of(ex) >= 2}
        self.assertEqual(build_graph_data.twin_sets(out, KB), len(groups))
        self.assertEqual(build_graph_data.twin_sets([ex for ex in out if ex["meta"]["label"] == 1], KB), 0)


class StratifiedHoldoutTests(unittest.TestCase):
    """R17: varianti tenute fuori scelte in ciascuno strato (con chiamanti / a un blocco)."""

    def fake(self, variant: str, n_blocks: int, i: int) -> dict:
        blocks = [{"depth": d} for d in range(n_blocks)]
        return {"meta": {"variant": variant, "label": i % 2, "trace": {"blocks": blocks}}}

    def pool(self) -> list[dict]:
        pool = []
        for v in range(1, 19):  # 18 varianti a un blocco
            pool += [self.fake(f"{v:02d}", 1, i) for i in range(120)]
        for v in (51, 52, 53, 54, 63, 64):  # 6 varianti con chiamanti (60% a 2+ blocchi)
            pool += [self.fake(str(v), 2 if i % 5 < 3 else 1, i) for i in range(150)]
        pool += [self.fake("44", 1, i) for i in range(6)]  # troppo pochi esempi: non eleggibile
        pool += [self.fake("42", 3, i) for i in range(99)]
        return pool

    def test_each_stratum_contributes_held_out_variants(self):
        held, strata = build_graph_data.stratified_holdout(self.pool(), seed=42, pct=12)
        self.assertEqual(set(strata["caller"]), {"51", "52", "53", "54", "63", "64"})
        self.assertEqual(set(strata["single"]), {f"{v:02d}" for v in range(1, 19)})
        self.assertEqual(set(strata["ineligible"]), {"42", "44"})
        self.assertGreaterEqual(len(held & set(strata["caller"])), 1)
        self.assertEqual(len(held & set(strata["single"])), 2)  # round(0.12 · 18)
        self.assertFalse(held & {"42", "44"})
        self.assertEqual(held, build_graph_data.stratified_holdout(self.pool(), seed=42, pct=12)[0])

    def test_a_small_stratum_still_gives_one_variant(self):
        pool = [self.fake("51", 2, i) for i in range(100)] + [self.fake(f"{v:02d}", 1, i) for v in range(1, 9)
                                                               for i in range(100)]
        held, strata = build_graph_data.stratified_holdout(pool, seed=1, pct=12)
        self.assertEqual(strata["caller"], ["51"])
        self.assertIn("51", held)
        self.assertEqual(len(held & set(strata["single"])), 1)


class SourceInCalleeTests(unittest.TestCase):
    """F1/R18: una sorgente chiamata ma non mostrata (`data = badSource(data)`) rende il record non decidibile."""

    P42 = "CWE78_OS_Command_Injection__char_environment_system_42"
    P61 = "CWE415_Double_Free__malloc_free_char_61"

    def rec(self, blocks: list[tuple[str, str, int, str]], sink_line: int, callee: str, cwe: str) -> tuple[dict, dict]:
        """(record, target) da blocchi (file, nome, profondità, codice); ogni blocco parte dalla riga 10."""
        out = [{"file": f, "name": n, "start_line": 10, "end_line": 10 + c.count("\n"), "depth": d, "lang": "c", "code": c}
               for f, n, d, c in blocks]
        depth0 = next(b for b in out if b["depth"] == 0)
        target = {"file": depth0["file"], "line": sink_line, "callee": callee, "cwe_id": cwe}
        return {"sink": {"function": callee, "file": depth0["file"], "line": sink_line, "cwe_id": cwe},
                "blocks": out, "truncated": False}, target

    def calling_source(self, kind: str) -> tuple[dict, dict]:
        holder = f"{self.P42}_bad" if kind == "bad" else "goodG2B"
        source = "badSource" if kind == "bad" else "goodG2BSource"
        code = (f"void {holder}()\n{{\n    char * data;\n    char data_buf[100] = FULL_COMMAND;\n    data = data_buf;\n"
                f"    data = {source}(data);\n    SYSTEM(data);\n}}")
        return self.rec([(f"s01/{self.P42}.c", holder, 0, code)], 16, "system", "CWE-78")

    def test_records_calling_a_source_that_is_not_shown_are_dropped_for_both_labels(self):
        stats = Counter()
        pairs = [self.calling_source("bad"), self.calling_source("good")]
        out = build_graph_data.examples_from_records([r for r, _ in pairs], [t for _, t in pairs], KB, 42, stats)
        self.assertEqual(out, [])
        self.assertEqual(stats["source_in_callee_dropped"], 2)
        self.assertEqual(stats["source_in_callee:42"], 2)
        self.assertEqual(stats["hidden_callee_dropped"], 2)
        self.assertEqual(stats["hidden_callee:42"], 2)

    def test_the_block_defining_the_source_is_kept(self):
        source = (f"char * {self.P61}b_badSource(char * data)\n{{\n    data = (char *)malloc(100*sizeof(char));\n"
                  "    if (data == NULL) {exit(-1);}\n    free(data);\n    return data;\n}")
        caller = (f"void {self.P61}_bad()\n{{\n    char * data;\n    data = NULL;\n"
                  f"    data = {self.P61}b_badSource(data);\n    free(data);\n}}")
        rec, target = self.rec([(f"s01/{self.P61}a.c", f"{self.P61}_bad", 1, caller),
                                (f"s01/{self.P61}b.c", f"{self.P61}b_badSource", 0, source)], 14, "free", "CWE-416")
        stats = Counter()
        out = build_graph_data.examples_from_records([rec], [target], KB, 42, stats)
        self.assertEqual(len(out), 1)
        self.assertEqual(stats["source_in_callee_dropped"], 0)

    def test_library_input_calls_are_untouched(self):
        code = (f"void {self.P42.replace('_42', '_01')}_bad()\n{{\n    char data[100] = \"\";\n"
                "    fgets(data, 100, stdin);\n    SYSTEM(data);\n}")
        rec, target = self.rec([(f"s01/{self.P42.replace('_42', '_01')}.c", f"{self.P42.replace('_42', '_01')}_bad", 0,
                                 code)], 14, "system", "CWE-78")
        stats = Counter()
        self.assertEqual(len(build_graph_data.examples_from_records([rec], [target], KB, 42, stats)), 1)
        self.assertEqual(stats["source_in_callee_dropped"], 0)


class HiddenCalleeTests(unittest.TestCase):
    """N1/R18b: ogni chiamata NON risolta a una funzione dal nome rivelatore (`…Sink`, `…Source`, …) scarta il record."""

    P21 = "CWE78_OS_Command_Injection__char_environment_system_21"
    rec = SourceInCalleeTests.rec

    def calling_sink(self, kind: str) -> tuple[dict, dict]:
        holder = f"{self.P21}_bad" if kind == "bad" else "goodB2G1"
        sink = "badSink" if kind == "bad" else "goodB2G1Sink"
        code = (f"void {holder}()\n{{\n    char * data;\n    char data_buf[100] = FULL_COMMAND;\n    data = data_buf;\n"
                f"    SYSTEM(data);\n    {sink}(data);\n}}")
        return self.rec([(f"s01/{self.P21}.c", holder, 0, code)], 15, "system", "CWE-78")

    def test_unresolved_leaky_callee_drops_the_record_for_both_labels(self):
        stats = Counter()
        pairs = [self.calling_sink("bad"), self.calling_sink("good")]
        out = build_graph_data.examples_from_records([r for r, _ in pairs], [t for _, t in pairs], KB, 42, stats)
        self.assertEqual(out, [])
        self.assertEqual(stats["hidden_callee_dropped"], 2)
        self.assertEqual(stats["hidden_callee:21"], 2)
        self.assertEqual(stats["source_in_callee_dropped"], 0, "nessuna `…Source` fra i callee nascosti")

    def test_a_shown_leaky_callee_is_kept(self):
        caller = (f"static void goodB2G1()\n{{\n    char * data;\n    char data_buf[100] = FULL_COMMAND;\n"
                  f"    data = data_buf;\n    goodB2G1Sink(data);\n}}")
        sink = "static void goodB2G1Sink(char * data)\n{\n    SYSTEM(data);\n}"
        rec, target = self.rec([(f"s01/{self.P21}.c", "goodB2G1", 1, caller),
                                (f"s01/{self.P21}.c", "goodB2G1Sink", 0, sink)], 12, "system", "CWE-78")
        stats = Counter()
        self.assertEqual(len(build_graph_data.examples_from_records([rec], [target], KB, 42, stats)), 1)
        self.assertEqual(stats["hidden_callee_dropped"], 0)

    def test_library_helpers_and_literals_are_untouched(self):
        code = (f"void {self.P21}_bad()\n{{\n    char data[100] = \"\";\n    strcpy(data, \"ls\");\n"
                "    printLine(\"Benign, fixed string\");\n    if (globalReturnsTrueOrFalse()) {}\n    SYSTEM(data);\n}")
        rec, target = self.rec([(f"s01/{self.P21}.c", f"{self.P21}_bad", 0, code)], 16, "system", "CWE-78")
        stats = Counter()
        self.assertEqual(len(build_graph_data.examples_from_records([rec], [target], KB, 42, stats)), 1)
        self.assertEqual(stats["hidden_callee_dropped"], 0)

    def test_harness_blocks_are_not_scanned(self):
        """`main` chiama sia `…_good()` sia `…_bad()`: va tolto prima del controllo, non far scartare il record."""
        rec, target = self.calling_sink("bad")
        rec["blocks"][0]["code"] = rec["blocks"][0]["code"].replace(f"    badSink(data);\n", "")
        main = f"int main()\n{{\n    {self.P21}_good();\n    {self.P21}_bad();\n    return 0;\n}}"
        rec["blocks"].insert(0, {"file": f"s01/{self.P21}.c", "name": "main", "start_line": 100, "end_line": 105,
                                 "depth": 1, "lang": "c", "code": main})
        stats = Counter()
        self.assertEqual(len(build_graph_data.examples_from_records([rec], [target], KB, 42, stats)), 1)
        self.assertEqual(stats["hidden_callee_dropped"], 0)


class CapPairsTests(unittest.TestCase):
    def test_cap_pairs_keeps_whole_pairs_in_input_order(self):
        pairs = [{"meta": {"pair_id": f"p{i // 2}", "cwe": "CWE-78", "variant": str(50 + i // 4), "label": i % 2}}
                 for i in range(10)]
        out = build_graph_data.cap_pairs(pairs, 5, seed=1)
        ids = Counter(ex["meta"]["pair_id"] for ex in out)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(n == 2 for n in ids.values()))
        self.assertEqual(out, [ex for ex in pairs if ex["meta"]["pair_id"] in ids])
        self.assertEqual(build_graph_data.cap_pairs(pairs, 100, seed=1), pairs)

    def test_pair_lengths_compare_the_two_prompts(self):
        def ex(pid: str, label: int, n: int) -> dict:
            return {"messages": [{"role": "user", "content": "x" * n}], "meta": {"pair_id": pid, "label": label}}

        pairs = [ex("a", 1, 200), ex("a", 0, 100), ex("b", 1, 100), ex("b", 0, 110), ex("c", 1, 50), ex("c", 0, 50)]
        self.assertEqual(build_graph_data.pair_lengths(pairs), {"pairs_different_length": 2, "pairs_vulnerable_longer": 1,
                                                                "pairs_length_within_15pct": 2,
                                                                "pairs_different_length_within_15pct": 1})


if __name__ == "__main__":
    unittest.main()
