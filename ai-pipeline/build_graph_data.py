"""Dati di training e valutazione con slice vere del grafo delle chiamate.

Sottocomando `juliet` (traccia J): dall'archivio NIST di Juliet (SARD, test suite 112, dominio pubblico / CC0) agli
esempi di chat con i chiamanti reali calcolati da `sast-engine slice`:

1. scarica/verifica lo zip ed estrae i `.c` delle CWE usate (`juliet_graph`);
2. per ogni radice (una directory di CWE Juliet) elenca i sink come `--targets` e lancia `slice_io.run_slice` a
   profondità 5; le radici senza target (CWE190/191/197: il catalogo vede solo le allocazioni) si saltano;
3. ogni record si riporta al suo target con `(file, riga, callee)`; etichetta, `variant_kind`, test case, variante di
   flusso e `source_token` vengono dai nomi ORIGINALI (funzione a profondità 0 e suo file), prima di anonimizzare;
4. si tolgono i blocchi dell'impalcatura di Juliet che esistono solo per lanciare il test e rivelerebbero l'etichetta
   (il dispatcher `…_good()`, presente solo nelle catene sicure, e `main()`, che chiama sia `…_good()` sia `…_bad()`);
5. anonimizzazione con un seme per test case (`f"{seme}:{gruppo}"`: i gemelli di un test case condividono nomi e righe),
   esempi completi, `drop_conflicts`, `make_candidates` UNA volta su tutto, divisione per test case e varianti tenute
   fuori, `choose_views` su train e val, tetti bilanciati;
6. scrive `juliet_train.jsonl`, `juliet_val.jsonl`, `juliet_report.json` in `--out-dir` e `juliet_test.jsonl`,
   `juliet_unseen_variants.jsonl`, `juliet_pairs.jsonl` in `--eval-dir`.

Uso (sotto un tetto di memoria, il motore è pesante):
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \\
    python3 build_graph_data.py juliet --raw-dir data/raw --out-dir data/graph --eval-dir data/eval --max-train 1300

Solo libreria standard.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import c_anonymize
import common
import graph_examples
import juliet_graph
import leak_audit
import slice_io

SOURCE = "juliet-nist"
DEPTH = 5
# Le stesse percentuali che `prepare_data.split_of` usa per le altre fonti: default di `--val-pct` e `--holdout-pct`
# in `prepare_data.py`.
VAL_PCT = 5
TEST_PCT = 10
HOLDOUT_PCT = 12
SAMPLE = 20
# Riscritture del codice nelle slice: le macro-sink di Juliet più `COMMAND_ARG3`, che in tutti i file CWE-78 è
# `#define COMMAND_ARG3 data`; senza, `execl(…, COMMAND_ARG3, NULL)` non mostra che il sink riceve il dato.
CODE_MACROS = {**juliet_graph.MACROS, "COMMAND_ARG3": "data"}

_ENGINE_SKIP_RE = re.compile(r"fuori da ogni funzione")


# --------------------------------------------------------------------------- impalcatura di Juliet
def is_dispatcher(name: str) -> bool:
    """Il dispatcher `CWE…_good()` (chiama `goodG2B()`/`goodB2G()`…): esiste solo nelle catene sicure."""
    return name == "good" or name.endswith("_good")


def is_harness(name: str) -> bool:
    """Blocchi che servono solo a lanciare il test: il dispatcher `…_good()` e `main()`."""
    return name == "main" or is_dispatcher(name)


def strip_harness(rec: dict) -> tuple[dict, int, int]:
    """Copia del record senza i blocchi d'impalcatura: (record, dispatcher tolti, `main` tolti).

    Le profondità dei blocchi rimasti non cambiano: il dispatcher e `main` sono sempre i chiamanti più esterni, quindi
    toglierli non lascia buchi nella catena.
    """
    out = copy.copy(rec)
    out["blocks"] = [b for b in rec["blocks"] if not is_harness(b["name"])]
    dispatchers = sum(is_dispatcher(b["name"]) for b in rec["blocks"])
    mains = sum(b["name"] == "main" for b in rec["blocks"])
    return out, dispatchers, mains


# --------------------------------------------------------------------------- record → esempi
def _dedupe_targets(targets: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for t in targets:
        seen.setdefault((t["file"], t["line"], t["callee"]), t)
    return list(seen.values())


def examples_from_records(records: list[dict], targets: list[dict], kb: dict, seed: int, stats: Counter) -> list[dict]:
    """Esempi completi (view `callers`) dai record di una radice; i conteggi degli scarti vanno in `stats`."""
    by_key = {(t["file"], t["line"], t["callee"]): t for t in targets}
    matched: set[tuple] = set()
    examples = []
    for rec in records:
        sink = rec["sink"]
        key = (sink["file"], sink["line"], sink["function"])
        if key not in by_key:
            stats["records_without_target"] += 1
            continue
        matched.add(key)
        at_sink = [b for b in rec["blocks"] if b["depth"] == 0]
        if len(at_sink) != 1:
            stats["malformed_dropped"] += 1
            continue
        name, file_name = at_sink[0]["name"], at_sink[0]["file"]
        label = juliet_graph.sink_label(name)
        if label is None:
            stats["unlabelled_dropped"] += 1
            continue
        if rec.get("truncated"):
            stats["truncated_dropped"] += 1
            continue
        if is_harness(name):
            stats["harness_sink_dropped"] += 1
            continue
        clean, dispatchers, mains = strip_harness(rec)
        stats["dispatcher_blocks_dropped"] += dispatchers
        stats["main_blocks_dropped"] += mains
        group = juliet_graph.testcase_group(file_name)
        try:
            anon = c_anonymize.anonymize_slice(clean, f"{seed}:{group}", CODE_MACROS)
        except ValueError:
            stats["anonymize_errors"] += 1
            continue
        meta = {"source": SOURCE, "synthetic": False, "track": graph_examples.JULIET_TRACK, "group": group,
                "variant": juliet_graph.flow_variant(file_name)}
        examples.append(graph_examples.build_example(
            anon, kb, label=label, variant_kind=juliet_graph.variant_kind(name),
            origin=juliet_graph.source_token(file_name), meta=meta))
    stats["targets_without_record"] += len(by_key) - len(matched)
    return examples


def dedupe_examples(examples: list[dict]) -> tuple[list[dict], int]:
    """Toglie i doppioni esatti (stesso prompt, stessa etichetta), tenendo il primo: (rimasti, tolti)."""
    seen: set[tuple[str, int]] = set()
    kept = []
    for ex in examples:
        key = (graph_examples.prompt_key(ex), ex["meta"]["label"])
        if key not in seen:
            seen.add(key)
            kept.append(ex)
    return kept, len(examples) - len(kept)


# --------------------------------------------------------------------------- tetti e conteggi
def cap_pairs(pairs: list[dict], max_examples: int, seed: int) -> list[dict]:
    """Al più `max_examples // 2` coppie INTERE, a turno fra i gruppi (cwe, variante), in ordine sha256(seme, pair_id).

    Una coppia non si spezza mai; l'uscita conserva l'ordine d'ingresso.
    """
    members: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(pairs):
        members[ex["meta"]["pair_id"]].append(i)
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for pid, idx in members.items():
        meta = pairs[idx[0]]["meta"]
        buckets[(str(meta.get("cwe")), str(meta.get("variant")))].append(pid)
    queues = [sorted(buckets[b], key=lambda p: (graph_examples._sha(f"{seed}:pairs:{p}"), p))
              for b in sorted(buckets, key=lambda b: (graph_examples._sha(f"{seed}:pairbucket:{b[0]}:{b[1]}"), b))]
    chosen: list[str] = []
    limit = max_examples // 2
    for r in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if len(chosen) < limit and r < len(q):
                chosen.append(q[r])
    keep = {i for pid in chosen for i in members[pid]}
    return [ex for i, ex in enumerate(pairs) if i in keep]


def sink_only_share(examples: list[dict]) -> dict:
    """Quota di esempi con la view `sink_only`: in tutto, per etichetta e per traccia."""
    def share(exs: list[dict]) -> float:
        return round(sum(ex["meta"]["view"] == "sink_only" for ex in exs) / len(exs), 4) if exs else 0.0

    by_label = {str(label): share([ex for ex in examples if ex["meta"]["label"] == label]) for label in (0, 1)}
    tracks = sorted({ex["meta"]["track"] for ex in examples})
    by_track = {t: share([ex for ex in examples if ex["meta"]["track"] == t]) for t in tracks}
    return {"overall": share(examples), "by_label": by_label, "by_track": by_track}


def cwe_kind_split(splits: dict[str, list[dict]]) -> dict:
    """`{cwe: {variant_kind: {split: n}}}`."""
    out: dict = defaultdict(lambda: defaultdict(Counter))
    for split, exs in splits.items():
        for ex in exs:
            out[ex["meta"]["cwe"]][ex["meta"]["variant_kind"]][split] += 1
    return {cwe: {kind: dict(c) for kind, c in sorted(kinds.items())} for cwe, kinds in sorted(out.items())}


def by_variant(before: list[dict], after: list[dict]) -> dict:
    """`{variante: {"examples": prima di drop_conflicts, "conflicts": tolti}}`, per vedere dove il grafo non basta."""
    total = Counter(ex["meta"]["variant"] for ex in before)
    left = Counter(ex["meta"]["variant"] for ex in after)
    return {v: {"examples": total[v], "conflicts": total[v] - left[v]} for v in sorted(total)}


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- traccia J
def build_juliet(raw_dir: Path, out_dir: Path, eval_dir: Path, *, seed: int = 42, max_train: int = 1300,
                 max_eval: int = 600, zip_path: Path | None = None, engine_bin: Path | None = None,
                 val_pct: int = VAL_PCT, test_pct: int = TEST_PCT, holdout_pct: int = HOLDOUT_PCT) -> dict:
    """Costruisce la traccia J, scrive i file e restituisce il rapporto (lo stesso di `juliet_report.json`).

    `zip_path` salta `fetch_juliet` (nessun controllo dello sha256: serve ai test); altrimenti lo zip si cerca o
    scarica in `raw_dir/juliet-nist`. Il val si limita in proporzione al train
    (`max_train · val_pct / (100 − val_pct − test_pct)`), test e varianti tenute fuori a `max_eval` ciascuno, le coppie
    a `max_eval` esempi (coppie intere, scelte fra TUTTI gli esempi completi di test e varianti tenute fuori).
    """
    started = time.monotonic()
    raw_dir, out_dir, eval_dir = Path(raw_dir), Path(out_dir), Path(eval_dir)
    kb = common.load_kb()
    zip_path = Path(zip_path) if zip_path else juliet_graph.fetch_juliet(raw_dir / "juliet-nist")
    roots = juliet_graph.extract_c_sources(zip_path, raw_dir / "juliet-nist" / "src", kb)

    stats: Counter = Counter()
    per_root: list[dict] = []
    roots_without_targets: list[str] = []
    targets_by_group: Counter = Counter()
    fulls: list[dict] = []
    for root in roots:
        targets = juliet_graph.sink_targets(root, kb)
        if not targets:
            roots_without_targets.append(root.name)
            print(f"{root.name}: nessun target, radice saltata", file=sys.stderr)
            continue
        unique = _dedupe_targets(targets)
        stats["targets"] += len(unique)
        stats["targets_duplicate"] += len(targets) - len(unique)
        for t in unique:
            targets_by_group[juliet_graph.testcase_group(t["file"])] += 1
        t0 = time.monotonic()
        records, stderr = slice_io.run_slice(root, unique, depth=DEPTH, binary=engine_bin)
        skipped = len(_ENGINE_SKIP_RE.findall(stderr))
        stats["records"] += len(records)
        stats["targets_skipped_by_engine"] += skipped
        examples = examples_from_records(records, unique, kb, seed, stats)
        del records
        fulls.extend(examples)
        seconds = round(time.monotonic() - t0, 1)
        per_root.append({"root": root.name, "targets": len(unique), "skipped_by_engine": skipped,
                         "examples": len(examples), "seconds": seconds})
        print(f"{root.name}: {len(unique)} target, {len(examples)} esempi, {skipped} saltati dal motore, {seconds} s",
              file=sys.stderr)

    fulls, duplicates = dedupe_examples(fulls)
    kept, conflicts = graph_examples.drop_conflicts(fulls)
    candidates = graph_examples.make_candidates(kept, kb)
    variants_present = sorted({c["full"]["meta"]["variant"] for c in candidates})
    held = graph_examples.holdout_variants(variants_present, seed, holdout_pct)
    split_cands: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "unseen_variants": []}
    for cand in candidates:
        meta = cand["full"]["meta"]
        split_cands[graph_examples.split_of_juliet(meta["group"], meta["variant"], seed, val_pct, test_pct, held)].append(cand)

    train_all = graph_examples.choose_views(split_cands["train"], seed)
    val_all = graph_examples.choose_views(split_cands["val"], seed)
    test_all = [c["full"] for c in split_cands["test"]]
    unseen_all = [c["full"] for c in split_cands["unseen_variants"]]
    max_val = round(max_train * val_pct / max(1, 100 - val_pct - test_pct))
    written = {
        "train": graph_examples.cap_balanced(train_all, max_train, seed),
        "val": graph_examples.cap_balanced(val_all, max_val, seed),
        "test": graph_examples.cap_balanced(test_all, max_eval, seed),
        "unseen_variants": graph_examples.cap_balanced(unseen_all, max_eval, seed),
    }
    all_pairs = graph_examples.link_pairs(test_all + unseen_all, kb)
    pairs = cap_pairs(all_pairs, max_eval, seed)

    write_jsonl(out_dir / "juliet_train.jsonl", written["train"])
    write_jsonl(out_dir / "juliet_val.jsonl", written["val"])
    write_jsonl(eval_dir / "juliet_test.jsonl", written["test"])
    write_jsonl(eval_dir / "juliet_unseen_variants.jsonl", written["unseen_variants"])
    write_jsonl(eval_dir / "juliet_pairs.jsonl", pairs)

    kept_groups = {ex["meta"]["group"] for ex in kept}
    without = sorted(g for g in targets_by_group if g not in kept_groups)
    everything = [ex for exs in written.values() for ex in exs] + pairs
    report = {
        "source": SOURCE,
        "seed": seed,
        "depth": DEPTH,
        "roots": per_root,
        "roots_without_targets": roots_without_targets,
        "candidates": stats["targets"],
        "targets_duplicate": stats["targets_duplicate"],
        "records": stats["records"],
        "targets_skipped_by_engine": stats["targets_skipped_by_engine"],
        "targets_without_record": stats["targets_without_record"],
        "records_without_target": stats["records_without_target"],
        "malformed_dropped": stats["malformed_dropped"],
        "unlabelled_dropped": stats["unlabelled_dropped"],
        "truncated_dropped": stats["truncated_dropped"],
        "harness_sink_dropped": stats["harness_sink_dropped"],
        "anonymize_errors": stats["anonymize_errors"],
        "dispatcher_blocks_dropped": stats["dispatcher_blocks_dropped"],
        "main_blocks_dropped": stats["main_blocks_dropped"],
        "duplicates_dropped": duplicates,
        "examples_before_conflicts": len(fulls),
        "conflicts_dropped": conflicts,
        "examples_after_conflicts": len(kept),
        "undecidable_share": round(conflicts / len(fulls), 4) if fulls else 0.0,
        "conflicts_by_variant": by_variant(fulls, kept),
        "testcases_with_targets": len(targets_by_group),
        "testcases_without_examples": len(without),
        "testcases_without_examples_sample": without[:SAMPLE],
        "per_cwe": {cwe: {"before_conflicts": n, "after_conflicts": sum(ex["meta"]["cwe"] == cwe for ex in kept),
                          **{split: sum(ex["meta"]["cwe"] == cwe for ex in exs) for split, exs in written.items()}}
                    for cwe, n in sorted(Counter(ex["meta"]["cwe"] for ex in fulls).items())},
        "per_cwe_kind_split_available": cwe_kind_split({"train": train_all, "val": val_all, "test": test_all,
                                                        "unseen_variants": unseen_all}),
        "per_cwe_kind_split": cwe_kind_split(written),
        "val_pct": val_pct,
        "test_pct": test_pct,
        "holdout_pct": holdout_pct,
        "variants_present": variants_present,
        "held_out_variants": sorted(held),
        "available": {"train": len(train_all), "val": len(val_all), "test": len(test_all),
                      "unseen_variants": len(unseen_all)},
        "written": {split: len(exs) for split, exs in written.items()} | {"pairs": len(pairs)},
        "caps": {"max_train": max_train, "max_val": max_val, "max_eval": max_eval},
        "pairs": len(pairs) // 2,
        "pairs_available": len(all_pairs) // 2,
        "pairs_cap": "coppie intere, scelte fra tutti gli esempi completi di test e varianti tenute fuori",
        "sink_only_share": sink_only_share(written["train"]),
        "sink_only_share_val": sink_only_share(written["val"]),
        "sink_only_share_train_before_cap": sink_only_share(train_all),
        "forbidden_hits": len(leak_audit.forbidden_hits(everything)),
        "bow_accuracy": round(leak_audit.bow_accuracy(written["train"], written["test"]), 4),
        "bow_accuracy_unseen_variants": round(leak_audit.bow_accuracy(written["train"], written["unseen_variants"]), 4),
        "structure_accuracy": round(leak_audit.structure_accuracy(written["train"], written["test"]), 4),
        "structure_accuracy_unseen_variants": round(
            leak_audit.structure_accuracy(written["train"], written["unseen_variants"]), 4),
        "structure_table": leak_audit.structure_table(written["train"]),
        "structure_table_test": leak_audit.structure_table(written["test"]),
        "top_tokens": [list(t) for t in leak_audit.top_tokens(written["train"], 30)],
        "seconds": round(time.monotonic() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "juliet_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="track", required=True)
    j = sub.add_parser("juliet", help="traccia J: Juliet C dall'archivio NIST")
    j.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    j.add_argument("--out-dir", type=Path, default=Path("data/graph"))
    j.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    j.add_argument("--seed", type=int, default=42)
    j.add_argument("--max-train", type=int, default=1300)
    j.add_argument("--max-eval", type=int, default=600)
    j.add_argument("--zip", type=Path, default=None, help="zip di Juliet già scaricato (salta il controllo dello sha256)")
    j.add_argument("--engine-bin", type=Path, default=None, help="binario di sast-engine (default: slice_io.engine_bin())")
    j.add_argument("--val-pct", type=int, default=VAL_PCT)
    j.add_argument("--test-pct", type=int, default=TEST_PCT)
    j.add_argument("--holdout-pct", type=int, default=HOLDOUT_PCT, help="percentuale di varianti di flusso tenute fuori")
    args = ap.parse_args(argv)

    report = build_juliet(args.raw_dir, args.out_dir, args.eval_dir, seed=args.seed, max_train=args.max_train,
                          max_eval=args.max_eval, zip_path=args.zip, engine_bin=args.engine_bin,
                          val_pct=args.val_pct, test_pct=args.test_pct, holdout_pct=args.holdout_pct)
    summary = {k: report[k] for k in ("candidates", "examples_before_conflicts", "conflicts_dropped", "undecidable_share",
                                      "written", "pairs", "forbidden_hits", "bow_accuracy", "structure_accuracy")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
