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
                    "dispatcher_blocks_dropped", "main_blocks_dropped", "candidates"):
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
        report = self.build(holdout_pct=50, test_pct=50, val_pct=0)
        held = set(report["held_out_variants"])
        self.assertEqual(len(held), 1)
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


if __name__ == "__main__":
    unittest.main()
