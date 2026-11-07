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
   fuori (stratificate fra varianti con chiamanti e a un blocco), `choose_views` su train e val, tetti bilanciati; il
   train dà metà del budget a gruppi interi con chiamanti, così i gemelli dello stesso sink arrivano insieme;
6. scrive `juliet_train.jsonl`, `juliet_val.jsonl`, `juliet_report.json` in `--out-dir` e `juliet_test.jsonl`,
   `juliet_unseen_variants.jsonl`, `juliet_pairs.jsonl` in `--eval-dir`.

Sottocomando `codeql` (traccia P1): i test CodeQL Python di `data/raw/codeql-tests` con le slice del motore (una radice
per directory di query) e l'etichetta dai `$ Alert` (`codeql_graph`). Il gruppo di divisione è il FILE, con la stessa chiave
e lo stesso `split_of` di `prepare_data` (`codeql:{percorso}`, R21): i file che `prepare_data` tiene nel test finiscono in
`data/eval/py_codeql.jsonl` e mai in train/val. Scrive `py_codeql_train.jsonl`, `py_codeql_val.jsonl`, `py_codeql_report.json`
in `--out-dir` e `py_codeql.jsonl` in `--eval-dir`.

Uso (sotto un tetto di memoria, il motore è pesante):
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \\
    python3 build_graph_data.py juliet --raw-dir data/raw --out-dir data/graph --eval-dir data/eval --max-train 1300
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \\
    python3 build_graph_data.py codeql --raw-dir data/raw --out-dir data/graph --eval-dir data/eval

Solo libreria standard.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import c_anonymize
import codeql_graph
import common
import graph_examples
import juliet_graph
import leak_audit
import prepare_data
import slice_io

SOURCE = "juliet-nist"
PY_SOURCE = "codeql-python"
DEPTH = 5
# Le stesse percentuali che `prepare_data.split_of` usa per le altre fonti: default di `--val-pct` e `--holdout-pct`
# in `prepare_data.py`.
VAL_PCT = 5
TEST_PCT = 10
HOLDOUT_PCT = 12
# Varianti eleggibili per essere tenute fuori: almeno tanti esempi non in conflitto (R17).
HOLDOUT_MIN_EXAMPLES = 100
# Quota del budget del train per gli esempi con chiamanti, presi a gruppi interi (R16).
MULTI_SHARE = 0.5
SAMPLE = 20
# Riscritture del codice nelle slice: le macro-sink di Juliet più `COMMAND_ARG3`, che in tutti i file CWE-78 è
# `#define COMMAND_ARG3 data`; senza, `execl(…, COMMAND_ARG3, NULL)` non mostra che il sink riceve il dato.
CODE_MACROS = {**juliet_graph.MACROS, "COMMAND_ARG3": "data"}

_ENGINE_SKIP_RE = re.compile(r"fuori da ogni funzione")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_LITERAL_RE = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'")


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
def hidden_callees(rec: dict) -> set[str]:
    """Funzioni chiamate dai blocchi, non definite da nessun blocco, il cui nome ORIGINALE soddisfa `LEAK_RE` (R18/R18b).

    Nelle varianti di Juliet con la sorgente o il secondo sink in un callee (`data = badSource(data)`,
    `goodB2G1Sink(data)`, …) il codice che decide l'etichetta sta in una funzione che la backward slice non mostra: il
    prompt non può sostenere l'etichetta, per nessuna delle due. Si guarda il codice originale senza commenti né
    letterali; le funzioni di libreria o di supporto (`printLine`, `strcpy`, `globalReturnsTrueOrFalse`) non contano.
    Il chiamante passa il record SENZA impalcatura: `main` e il dispatcher chiamano funzioni fuori dalla catena.
    """
    names = {b["name"] for b in rec["blocks"]}
    hidden: set[str] = set()
    for b in rec["blocks"]:
        code, _ = c_anonymize.strip_comments(b["code"])
        code = _LITERAL_RE.sub('""', code)
        hidden |= {m.group(1) for m in _CALL_RE.finditer(code)
                   if m.group(1) not in names and c_anonymize.LEAK_RE.search(m.group(1))}
    return hidden


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
        hidden = hidden_callees(clean)
        if hidden:
            variant = juliet_graph.flow_variant(file_name)
            stats["hidden_callee_dropped"] += 1
            stats[f"hidden_callee:{variant}"] += 1
            if any(name.endswith("Source") for name in hidden):
                stats["source_in_callee_dropped"] += 1
                stats[f"source_in_callee:{variant}"] += 1
            continue
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


# --------------------------------------------------------------------------- campione di train (R16)
def n_blocks(example: dict) -> int:
    """Blocchi della TRACCIA (non della view): ≥ 2 vuol dire che la slice ha chiamanti."""
    return len(example["meta"]["trace"]["blocks"])


def _sink_only_keys(examples: list[dict], kb: dict) -> list[str]:
    return [graph_examples._sha(graph_examples.messages_from_trace(ex["meta"]["trace"], kb, sink_only=True)[1]["content"])
            for ex in examples]


def _twin_members(examples: list[dict], kb: dict) -> set[int]:
    """Indici degli esempi che hanno, nello stesso gruppo, un gemello con la stessa chiave solo-sink ed etichetta opposta."""
    labels: dict[tuple, set[int]] = defaultdict(set)
    keys = _sink_only_keys(examples, kb)
    for ex, key in zip(examples, keys):
        labels[(ex["meta"]["group"], key)].add(ex["meta"]["label"])
    return {i for i, (ex, key) in enumerate(zip(examples, keys)) if len(labels[(ex["meta"]["group"], key)]) == 2}


def twin_sets(examples: list[dict], kb: dict) -> int:
    """Gruppi con un insieme gemello completo: un vulnerabile e un sicuro con la stessa chiave solo-sink."""
    return len({examples[i]["meta"]["group"] for i in _twin_members(examples, kb)})


def cap_train(examples: list[dict], max_train: int, multi_share: float, seed: int, kb: dict | None = None) -> list[dict]:
    """Tetto del train ricco di chiamanti e contrastivo (R16); con `multi_share <= 0` è `cap_balanced` come prima.

    - Fino a `round(max_train · multi_share)` posti vanno agli esempi con ≥ 2 blocchi nella traccia, presi a GRUPPI
      interi (test case): un gruppo per cella (cwe, variante) a turno, in ordine sha256(seme, gruppo); di un gruppo si
      prendono tutti gli esempi a più blocchi, così i gemelli bad/g2b/b2g dello stesso sink arrivano insieme (si può
      sforare di meno di un gruppo).
    - Il resto va agli esempi a un blocco, a turno fra le celle (cwe, variante) come `cap_balanced`, per etichetta
      quanto serve a pareggiare.
    - Etichette bilanciate (differenza ≤ 1) e al più `max_train`: si taglia l'etichetta in eccesso partendo dagli esempi
      a un blocco, poi dagli esempi a più blocchi non gemelli, per ultimi dai gemelli (gruppi scelti per ultimi prima).

    Deterministico e indipendente dall'ordine d'ingresso; l'uscita conserva l'ordine d'ingresso.
    """
    if multi_share <= 0:
        return graph_examples.cap_balanced(examples, max_train, seed)
    kb = common.load_kb() if kb is None else kb
    multi = [i for i, ex in enumerate(examples) if n_blocks(ex) >= 2]
    single = [i for i, ex in enumerate(examples) if n_blocks(ex) < 2]

    by_group: dict[str, list[int]] = defaultdict(list)
    for i in multi:
        by_group[examples[i]["meta"]["group"]].append(i)
    cells: dict[tuple, list[str]] = defaultdict(list)
    for group, idx in by_group.items():
        meta = examples[idx[0]]["meta"]
        cells[(str(meta.get("cwe")), str(meta.get("variant")))].append(group)
    queues = [sorted(cells[c], key=lambda g: (graph_examples._sha(f"{seed}:group:{g}"), g))
              for c in sorted(cells, key=lambda c: (graph_examples._sha(f"{seed}:cell:{c[0]}:{c[1]}"), c))]
    budget = round(max_train * multi_share)
    chosen_groups: list[str] = []
    taken = 0
    for r in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if taken < budget and r < len(q):
                chosen_groups.append(q[r])
                taken += len(by_group[q[r]])

    picked = [i for g in chosen_groups for i in sorted(by_group[g], key=lambda i: graph_examples.prompt_key(examples[i]))]
    twins = {picked[j] for j in _twin_members([examples[i] for i in picked], kb)}
    # Priorità per etichetta (si tiene un prefisso): gemelli, poi altri esempi a più blocchi, poi esempi a un blocco.
    priority: dict[int, list[int]] = {}
    for label in (0, 1):
        mine = [i for i in picked if examples[i]["meta"]["label"] == label]
        singles = [i for i in single if examples[i]["meta"]["label"] == label]
        priority[label] = ([i for i in mine if i in twins] + [i for i in mine if i not in twins]
                           + graph_examples._round_robin(singles, examples, len(singles), seed))
    k = min(max_train // 2, len(priority[0]), len(priority[1]))
    quota = {0: k, 1: k}
    if max_train - 2 * k >= 1:
        richer = max((1, 0), key=lambda label: len(priority[label]) - k)
        if len(priority[richer]) > k:
            quota[richer] += 1
    # Prefisso di ogni lista: gli esempi a un blocco riempiono ciò che i gruppi scelti lasciano, e un eccesso si taglia
    # dalla coda (prima gli esempi a un blocco, poi i non gemelli, per ultimi i gemelli dei gruppi scelti per ultimi).
    keep = {i for label in (0, 1) for i in priority[label][:quota[label]]}
    return [ex for i, ex in enumerate(examples) if i in keep]


# --------------------------------------------------------------------------- varianti tenute fuori (R17)
def stratified_holdout(examples: list[dict], seed: int, pct: int = HOLDOUT_PCT,
                       min_examples: int = 100) -> tuple[set[str], dict[str, list[str]]]:
    """Varianti di flusso tenute fuori, scelte in ciascuno strato: (varianti, strati).

    Strati sugli esempi non in conflitto: «caller» se almeno metà degli esempi della variante ha ≥ 2 blocchi nella
    traccia, «single» altrimenti; «ineligible» se la variante ha meno di `min_examples` esempi. In ogni strato si applica
    `graph_examples.holdout_variants(…, pct)`; se dà zero varianti ma lo strato non è vuoto, si prende la prima per lo
    stesso ordine sha256 (almeno una per strato).
    """
    total: Counter = Counter()
    multi: Counter = Counter()
    for ex in examples:
        variant = ex["meta"]["variant"]
        total[variant] += 1
        multi[variant] += n_blocks(ex) >= 2
    strata: dict[str, list[str]] = {"caller": [], "single": [], "ineligible": []}
    for v in sorted(total):
        if total[v] < min_examples:
            strata["ineligible"].append(v)
        else:
            strata["caller" if multi[v] * 2 >= total[v] else "single"].append(v)
    held: set[str] = set()
    for name in ("caller", "single"):
        stratum = strata[name]
        chosen = graph_examples.holdout_variants(stratum, seed, pct)
        if not chosen and stratum:
            chosen = {min(stratum, key=lambda v: (graph_examples._sha(f"{seed}:holdout:{v}"), v))}
        held |= chosen
    return held, strata


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


def pair_lengths(pairs: list[dict]) -> dict[str, int]:
    """Lunghezze (caratteri del prompt utente) dei due membri di ogni coppia S1 (F2/R19).

    `pairs_different_length`: coppie con lunghezze diverse; `pairs_vulnerable_longer`: coppie col vulnerabile più lungo;
    `pairs_length_within_15pct`: coppie la cui differenza è al più il 15% del prompt più lungo (comprese quelle di
    lunghezza identica); `pairs_different_length_within_15pct`: le stesse, ma solo fra quelle di lunghezza diversa.
    """
    members: dict[str, dict[int, int]] = defaultdict(dict)
    for ex in pairs:
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        members[ex["meta"]["pair_id"]][ex["meta"]["label"]] = len(user)
    whole = [m for m in members.values() if set(m) == {0, 1}]
    within = [m for m in whole if abs(m[1] - m[0]) <= 0.15 * max(m[0], m[1])]
    return {
        "pairs_different_length": sum(m[0] != m[1] for m in whole),
        "pairs_vulnerable_longer": sum(m[1] > m[0] for m in whole),
        "pairs_length_within_15pct": len(within),
        "pairs_different_length_within_15pct": sum(m[0] != m[1] for m in within),
    }


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
                 val_pct: int = VAL_PCT, test_pct: int = TEST_PCT, holdout_pct: int = HOLDOUT_PCT,
                 multi_share: float = MULTI_SHARE, holdout_min_examples: int = HOLDOUT_MIN_EXAMPLES) -> dict:
    """Costruisce la traccia J, scrive i file e restituisce il rapporto (lo stesso di `juliet_report.json`).

    `zip_path` salta `fetch_juliet` (nessun controllo dello sha256: serve ai test); altrimenti lo zip si cerca o
    scarica in `raw_dir/juliet-nist`. Il train si limita con `cap_train` (`multi_share` del budget a gruppi interi con
    chiamanti, R16), il val in proporzione al train (`max_train · val_pct / (100 − val_pct − test_pct)`), test e
    varianti tenute fuori a `max_eval` ciascuno, le coppie a `max_eval` esempi (coppie intere, scelte fra TUTTI gli
    esempi completi di test e varianti tenute fuori). Le varianti tenute fuori sono stratificate (`stratified_holdout`,
    R17).
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
    held, strata = stratified_holdout(kept, seed, holdout_pct, holdout_min_examples)
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
        "train": cap_train(train_all, max_train, multi_share, seed, kb),
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
        "source_in_callee_dropped": stats["source_in_callee_dropped"],
        "source_in_callee_by_variant": {k.split(":", 1)[1]: v for k, v in sorted(stats.items())
                                        if isinstance(k, str) and k.startswith("source_in_callee:")},
        "hidden_callee_dropped": stats["hidden_callee_dropped"],
        "hidden_callee_by_variant": {k.split(":", 1)[1]: v for k, v in sorted(stats.items())
                                     if isinstance(k, str) and k.startswith("hidden_callee:")},
        "undecidable_or_callee_share": round((stats["hidden_callee_dropped"] + conflicts) / stats["targets"], 4)
        if stats["targets"] else 0.0,
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
        "holdout_strata": strata,
        "holdout_min_examples": holdout_min_examples,
        "unseen_variants_blocks": {
            "single": sum(n_blocks(ex) < 2 for ex in written["unseen_variants"]),
            "multi": sum(n_blocks(ex) >= 2 for ex in written["unseen_variants"])},
        "multi_share": multi_share,
        "multi_block_share": round(sum(n_blocks(ex) >= 2 for ex in written["train"]) / len(written["train"]), 4)
        if written["train"] else 0.0,
        "train_twin_sets": twin_sets(written["train"], kb),
        "available": {"train": len(train_all), "val": len(val_all), "test": len(test_all),
                      "unseen_variants": len(unseen_all)},
        "written": {split: len(exs) for split, exs in written.items()} | {"pairs": len(pairs)},
        "caps": {"max_train": max_train, "max_val": max_val, "max_eval": max_eval},
        "pairs": len(pairs) // 2,
        "pairs_available": len(all_pairs) // 2,
        "pairs_cap": "coppie intere, scelte fra tutti gli esempi completi di test e varianti tenute fuori",
        **pair_lengths(pairs),
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
        "stump_accuracies": {name: leak_audit.stump_accuracies(written["train"], exs)
                             for name, exs in (("test", written["test"]), ("unseen_variants", written["unseen_variants"]),
                                               ("pairs", pairs))},
        "top_tokens": [list(t) for t in leak_audit.top_tokens(written["train"], 30)],
        "seconds": round(time.monotonic() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "juliet_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- traccia P1
def _audit(train: list[dict], heldout: list[dict]) -> dict | None:
    """Sacco di token, sola struttura e classificatori a un solo numero di `train` su `heldout` (None se manca uno dei due)."""
    if not train or not heldout:
        return None
    return {"bow_accuracy": round(leak_audit.bow_accuracy(train, heldout), 4),
            "structure_accuracy": round(leak_audit.structure_accuracy(train, heldout), 4),
            "stump_accuracies": leak_audit.stump_accuracies(train, heldout)}


def build_codeql(raw_dir: Path, out_dir: Path, eval_dir: Path, *, seed: int = 42, engine_bin: Path | None = None,
                 val_pct: int = VAL_PCT, test_pct: int = TEST_PCT, p_callers: float = 0.7,
                 codeql_root: Path | None = None) -> dict:
    """Costruisce la traccia P1, scrive i file e restituisce il rapporto (lo stesso di `py_codeql_report.json`).

    L'albero dei test (`raw_dir/codeql-tests`, o `codeql_root`) si copia ripulito in una cartella temporanea; il motore
    gira una volta per directory di query che ha almeno un file con aspettative inline; ogni slice si etichetta
    (`codeql_graph.label_slices`), si ripulisce (`anonymize_python`) e diventa un esempio completo. Un esempio che dopo la
    ripulitura ha ancora una parola-spia (un nome d'API o di funzione non definita nella slice, una f-string…) si scarta
    e si conta: il controllo finale `forbidden_hits(mode="parts")` sui file scritti è vuoto per costruzione. Gruppo di
    divisione = file (`codeql:{percorso}`) con `prepare_data.split_of(gruppo, seed, val_pct, test_pct)`: i file del test
    di `prepare_data` finiscono solo in `py_codeql.jsonl`. Poi doppioni, conflitti, `make_candidates` una volta su tutto e
    `choose_views` (`p_callers` slice completa, il resto solo sink dove è decidibile) su train e val.
    """
    started = time.monotonic()
    raw_dir, out_dir, eval_dir = Path(raw_dir), Path(out_dir), Path(eval_dir)
    kb = common.load_kb()
    codeql_root = Path(codeql_root) if codeql_root else raw_dir / "codeql-tests"
    allow = codeql_graph.catalog_leaky_names(kb) | codeql_graph.API_LEAKY_NAMES

    stats: Counter = Counter()
    leaky_words: Counter = Counter()
    survivors: list[tuple[str, dict]] = []
    per_root: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="codeql-tree-") as tmp:
        tree = Path(tmp)
        notes = codeql_graph.prepare_tree(codeql_root, tree, kb)
        stats["files"] = len(notes)
        stats["files_not_annotated"] = sum(not n["annotated"] for n in notes.values())
        files_of: dict[str, list[str]] = defaultdict(list)
        for rel, note in notes.items():
            files_of[note["root"]].append(rel)
        for root_rel in sorted(files_of):
            if not any(notes[rel]["annotated"] for rel in files_of[root_rel]):
                stats["roots_skipped_not_annotated"] += 1
                continue
            t0 = time.monotonic()
            records, _stderr = slice_io.run_slice(tree / root_rel, None, depth=DEPTH, binary=engine_bin)
            stats["roots"] += 1
            stats["records"] += len(records)
            labelled = codeql_graph.label_slices(records, notes, prefix=root_rel, stats=stats)
            built = 0
            for rec, label in labelled:
                if rec.get("truncated"):
                    stats["truncated_dropped"] += 1
                    continue
                group = f"codeql:{root_rel}/{rec['sink']['file']}"
                meta = {"source": PY_SOURCE, "synthetic": False, "track": graph_examples.PY_CODEQL_TRACK, "group": group}
                try:
                    anon = codeql_graph.anonymize_python(rec, f"{seed}:{group}")
                    example = graph_examples.build_example(anon, kb, label=label, variant_kind="bad" if label else "good",
                                                           origin=None, meta=meta)
                except ValueError:
                    stats["anonymize_errors"] += 1
                    continue
                hits = leak_audit.forbidden_hits([example], mode="parts", extra_parts=codeql_graph.PY_EXTRA_PARTS, allow=allow)
                if hits:
                    stats["leaky_dropped"] += 1
                    leaky_words.update({h.rsplit(": ", 1)[1] for h in hits})
                    continue
                survivors.append((prepare_data.split_of(group, seed, val_pct, test_pct), example))
                built += 1
            seconds = round(time.monotonic() - t0, 1)
            per_root.append({"root": Path(root_rel).name, "records": len(records), "labelled": len(labelled),
                             "examples": built, "seconds": seconds})
            print(f"{Path(root_rel).name}: {len(records)} slice, {len(labelled)} etichettate, {built} esempi, {seconds} s",
                  file=sys.stderr)

    split_of_example = {id(ex): split for split, ex in survivors}
    fulls, duplicates = dedupe_examples([ex for _, ex in survivors])
    kept, conflicts = graph_examples.drop_conflicts(fulls)
    candidates = graph_examples.make_candidates(kept, kb)
    by_split = {split: [c for c in candidates if split_of_example[id(c["full"])] == split] for split in ("train", "val", "test")}
    written = {
        "train": graph_examples.choose_views(by_split["train"], seed, p_callers),
        "val": graph_examples.choose_views(by_split["val"], seed, p_callers),
        "test": [c["full"] for c in by_split["test"]],
    }
    write_jsonl(out_dir / "py_codeql_train.jsonl", written["train"])
    write_jsonl(out_dir / "py_codeql_val.jsonl", written["val"])
    write_jsonl(eval_dir / "py_codeql.jsonl", written["test"])

    everything = [ex for exs in written.values() for ex in exs]
    report = {
        "source": PY_SOURCE,
        "track": graph_examples.PY_CODEQL_TRACK,
        "seed": seed,
        "depth": DEPTH,
        "split_group": "codeql:{percorso del file} (R21)",
        "val_pct": val_pct,
        "test_pct": test_pct,
        "p_callers": p_callers,
        "files": stats["files"],
        "files_not_annotated": stats["files_not_annotated"],
        "roots": stats["roots"],
        "roots_skipped_not_annotated": stats["roots_skipped_not_annotated"],
        "per_root": per_root,
        "records": stats["records"],
        "skipped_unknown_file": stats["skipped_unknown_file"],
        "skipped_not_annotated": stats["skipped_not_annotated"],
        "skipped_other_cwe": stats["skipped_other_cwe"],
        "skipped_ambiguous": stats["skipped_ambiguous"],
        "truncated_dropped": stats["truncated_dropped"],
        "anonymize_errors": stats["anonymize_errors"],
        "leaky_dropped": stats["leaky_dropped"],
        "leaky_words": dict(leaky_words.most_common(40)),
        "duplicates_dropped": duplicates,
        "conflicts_dropped": conflicts,
        "examples_before_conflicts": len(fulls),
        "examples_after_conflicts": len(kept),
        "written": {split: len(exs) for split, exs in written.items()},
        "labels": {str(label): sum(ex["meta"]["label"] == label for ex in everything) for label in (1, 0)},
        "labels_by_split": {split: {str(label): sum(ex["meta"]["label"] == label for ex in exs) for label in (1, 0)}
                            for split, exs in written.items()},
        "per_cwe": {cwe: {split: sum(ex["meta"]["cwe"] == cwe for ex in exs) for split, exs in written.items()}
                    for cwe in sorted({ex["meta"]["cwe"] for ex in everything})},
        "groups": {split: len({ex["meta"]["group"] for ex in exs}) for split, exs in written.items()},
        "multi_block_share": round(sum(n_blocks(ex) >= 2 for ex in written["train"]) / len(written["train"]), 4)
        if written["train"] else 0.0,
        "sink_only_share": sink_only_share(written["train"]),
        "forbidden_hits": len(leak_audit.forbidden_hits(everything, mode="parts", extra_parts=codeql_graph.PY_EXTRA_PARTS,
                                                         allow=allow)),
        "audit_vs_test": _audit(written["train"], written["test"]),
        "seconds": round(time.monotonic() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "py_codeql_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    j.add_argument("--holdout-pct", type=int, default=HOLDOUT_PCT,
                   help="percentuale di varianti di flusso tenute fuori, in ciascuno strato")
    j.add_argument("--multi-share", type=float, default=MULTI_SHARE,
                   help="quota del train con chiamanti, a gruppi interi (0 = tetto uniforme di prima)")
    c = sub.add_parser("codeql", help="traccia P1: test CodeQL Python con le slice del motore")
    c.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="contiene codeql-tests/")
    c.add_argument("--out-dir", type=Path, default=Path("data/graph"))
    c.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    c.add_argument("--seed", type=int, default=42, help="lo stesso di prepare_data.py: decide quali file sono del test")
    c.add_argument("--engine-bin", type=Path, default=None, help="binario di sast-engine (default: slice_io.engine_bin())")
    c.add_argument("--val-pct", type=int, default=VAL_PCT, help="come --val-pct di prepare_data.py")
    c.add_argument("--test-pct", type=int, default=TEST_PCT, help="come --holdout-pct di prepare_data.py")
    c.add_argument("--p-callers", type=float, default=0.7, help="quota di slice complete (il resto solo sink dove è decidibile)")
    args = ap.parse_args(argv)

    if args.track == "codeql":
        report = build_codeql(args.raw_dir, args.out_dir, args.eval_dir, seed=args.seed, engine_bin=args.engine_bin,
                              val_pct=args.val_pct, test_pct=args.test_pct, p_callers=args.p_callers)
        keys = ("files", "files_not_annotated", "roots", "records", "leaky_dropped", "truncated_dropped", "conflicts_dropped",
                "written", "labels", "groups", "multi_block_share", "forbidden_hits")
        print(json.dumps({k: report[k] for k in keys}, ensure_ascii=False, indent=2))
        return 0

    report = build_juliet(args.raw_dir, args.out_dir, args.eval_dir, seed=args.seed, max_train=args.max_train,
                          max_eval=args.max_eval, zip_path=args.zip, engine_bin=args.engine_bin,
                          val_pct=args.val_pct, test_pct=args.test_pct, holdout_pct=args.holdout_pct,
                          multi_share=args.multi_share)
    summary = {k: report[k] for k in ("candidates", "examples_before_conflicts", "conflicts_dropped", "undecidable_share",
                                      "written", "pairs", "multi_block_share", "train_twin_sets", "held_out_variants",
                                      "forbidden_hits", "bow_accuracy", "structure_accuracy")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
