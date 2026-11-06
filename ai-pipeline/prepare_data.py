#!/usr/bin/env python3
"""Converte PrimeVul nel formato `Slicing + CWE -> JSON VEX` usato per il fine-tuning.

Fonti (tutte opzionali, vedi --help): PrimeVul, SecVulEval (C/C++, coppie vulnerabile/corretta), CVEfixes+PyVul (Python,
coppie prima/dopo) e i test di sicurezza di CodeQL (MIT, sink annotati a mano). Il sink di ogni esempio è *localizzato*: commento
FLAW, oppure chiamata sulle righe che la fix tocca (diff), oppure la funzione stessa alla riga del bug (vedi localize.py).
`--juliet` (il CSV di Juliet con le parole-etichetta nel codice) resta come lettore legacy: Juliet vero entra da `--graph-dir`.

`--graph-dir DIR` aggiunge a train e val gli esempi con slice già costruiti (`DIR/*_train.jsonl`, `DIR/*_val.jsonl`, scritti da
`build_graph_data.py` e dalle tracce Python): non entrano mai nel test, e i test CodeQL Python di train/val sono quelli con slice.

Il target non contiene id CVE, descrizioni CVE, messaggi di commit né diff di fix di altri progetti: il reasoning è una frase
fissa sul sink e `suggested_fix` è il suggerimento del catalogo (vulnerabili) oppure `NO_FIX`.

Per ogni funzione con un sink noto alla KB (`sast-engine/cwe_kb.json`) produce un esempio di chat:
  system : stesso system prompt di sast-engine
  user   : regola CWE + sink candidato + traccia di codice (un solo blocco: la funzione)
  assistant : JSON conforme a `shared/vex_schema.json`

Il prompt è generato da `common.py` con gli stessi template (e golden test) del motore Rust.
Limite noto: PrimeVul contiene funzioni isolate, quindi in training la traccia ha profondità 0;
la catena di chiamanti (profondità 1-3) compare solo in inferenza.

Solo libreria standard; i file vengono letti in streaming (RAM costante anche con i 471 MB di train).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import common
import localize
import sources
from localize import extract_function_name, find_calls  # noqa: F401  (ri-esportati: usati dai test)

HF_BASE = "https://huggingface.co/datasets/colin/PrimeVul/resolve/main"
PRIMEVUL_FILES = {"train": "primevul_train.jsonl", "valid": "primevul_valid.jsonl", "test": "primevul_test.jsonl"}
NONE_STRINGS = {"", "none", "null", "nan"}
NO_FIX = "No change needed."
# Lunghezza massima di un esempio: `train.py` tronca oltre `--max-seq-length` token e qui non c'è il tokenizer, quindi si
# stima in caratteri (≈ 3,5 per token su codice e JSON). Gli esempi oltre la stima si scartano e si contano.
MAX_SEQ_LENGTH = 8192
CHARS_PER_TOKEN = 3.5
# Il repository Git e il notebook clonano i file: ognuno resta sotto questa soglia (altrimenti `train.py --train` accetta più file).
MAX_FILE_BYTES = 50 * 1024 * 1024

# --------------------------------------------------------------------------- campi PrimeVul
def is_none(value) -> bool:
    return value is None or str(value).strip().lower() in NONE_STRINGS


def parse_cwes(raw) -> list[str]:
    """`cwe` arriva come lista o come stringa tipo "['CWE-704']"; si tengono solo le etichette `CWE-<n>`."""
    if is_none(raw):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [raw]
    if isinstance(raw, str):
        raw = [raw]
    return [str(c).strip() for c in raw if re.fullmatch(r"CWE-\d+", str(c).strip())]


def guess_lang(file_name, code: str) -> str:
    ext = "" if is_none(file_name) else str(file_name).rsplit(".", 1)[-1].lower()
    if ext == "py":
        return "python"
    if ext in ("cc", "cpp", "cxx", "hpp", "hh", "hxx"):
        return "cpp"
    if ext in ("c", "h"):
        return "c"
    if re.search(r"^\s*(async\s+)?def\s+\w+\s*\(", code, re.M):
        return "python"
    return "cpp" if re.search(r"\w::\w|template\s*<|\bclass\s+\w", code) else "c"


# --------------------------------------------------------------------------- rilevamento chiamate
def detect_sink(kb: dict, code: str, lang: str, preferred_cwe: str | None = None) -> dict | None:
    """Prima chiamata-sink della KB nel corpo (senza usare un diff): la meno affidabile delle localizzazioni."""
    for callee, line in find_calls(code, lang):
        entry = common.match_sink(kb, lang, callee)
        if entry is None or (preferred_cwe and entry["id"] != preferred_cwe):
            continue
        return {"entry": entry, "callee": callee, "line": line}
    return None


# --------------------------------------------------------------------------- costruzione esempi
def _file_label(record: dict, name: str, lang: str) -> str:
    file_name = record.get("file_name")
    if not is_none(file_name):
        return str(file_name)
    ext = {"cpp": "cpp", "python": "py"}.get(lang, "c")
    return f"{record.get('project') or 'unknown'}/{name}.{ext}"


def _reasoning(kind: str, sink: dict, file_label: str, entry: dict) -> str:
    """Frase fissa sul sink: niente id CVE, descrizione CVE, messaggio di commit o parole-etichetta («fixed», «good»…).

    Il modello vedeva l'id e la descrizione del CVE nel target e li riproduceva (inventati) su codice nuovo; qui il verdetto
    si spiega solo con ciò che il prompt mostra (la chiamata, la riga, la CWE). Patch e benigni hanno la stessa frase.
    """
    where = f"The sink `{sink['callee']}` at line {sink['line']} of {file_label}"
    if kind == "vulnerable":
        return f"{where} receives unvalidated data and is vulnerable to {entry['id']} ({entry['name']})."
    return (f"{where} does not receive untrusted data: the value reaching it is constant or checked before use, "
            f"so it is not vulnerable to {entry['id']} ({entry['name']}).")


def _choose_sink(kb: dict, func: str, lang: str, name: str, labels: list[str], record: dict,
                 counterpart: str | None, sinkless: str) -> dict | None:
    """Sceglie il sink dell'esempio, dal più al meno affidabile.

    1. `sink_hint` (commento FLAW di Juliet);
    2. con una controparte (versione corretta/vulnerabile): chiamata-sink sulle righe che la fix tocca; se non c'è, la
       funzione stessa alla prima riga toccata (mai una chiamata qualsiasi altrove: non sarebbe "il" sink);
    3. senza controparte: prima chiamata-sink della KB per la CWE etichettata, altrimenti la funzione alla riga 1.
    """
    entry_by_id = {e["id"]: e for e in kb["entries"]}
    hint = record.get("sink_hint")
    if hint:
        entry = entry_by_id[labels[0]] if labels else common.match_sink(kb, lang, hint["callee"])
        if entry is None:
            return None
        return {"entry": entry, "callee": hint["callee"], "line": hint["line"], "kind": "call", "how": "comment"}

    if counterpart is not None:
        for preferred in labels or [None]:
            found = localize.localize_sink(kb, func, lang, counterpart, preferred)
            if found:
                return {**found, "kind": "call"}
        if sinkless == "function" and labels:
            line = localize.first_changed_line(func, counterpart)
            return {"entry": entry_by_id[labels[0]], "callee": name, "line": line or 1, "kind": "function", "how": "diff" if line else None}
        return None

    for preferred in labels or [None]:
        found = detect_sink(kb, func, lang, preferred)
        if found:
            return {**found, "kind": "call", "how": "scan"}
    if sinkless == "function" and labels:
        return {"entry": entry_by_id[labels[0]], "callee": name, "line": 1, "kind": "function", "how": None}
    return None


def build_example(record: dict, kb: dict, patched: dict, max_chars: int, sinkless: str = "skip",
                  vulnerable_map: dict | None = None) -> tuple[dict | None, str | None]:
    """Restituisce (esempio, None) oppure (None, motivo_dello_scarto).

    `sinkless="function"`: se la funzione ha una CWE nota ma nessuna chiamata-sink localizzabile, la funzione stessa fa da
    sink (alla prima riga toccata dalla fix, se nota). `patched` / `vulnerable_map` indicizzano per (commit, nome) la
    versione corretta / vulnerabile; un record può anche portare direttamente la propria `counterpart`.
    """
    func = (record.get("func") or "").lstrip("\n").rstrip()
    if not func.strip():
        return None, "empty_function"
    if len(func) > max_chars:
        return None, "too_long"

    raw_labels = parse_cwes(record.get("cwe"))
    labels = [c for c in (common.resolve_cwe(kb, l) for l in raw_labels) if c]
    vulnerable = int(record.get("target", 0)) == 1
    explicit = record.get("kind")
    if explicit in ("vulnerable", "patched", "benign"):
        kind = explicit
    elif vulnerable:
        kind = "vulnerable"
    elif raw_labels and (not is_none(record.get("cve")) or record.get("counterpart") is not None):
        kind = "patched"
    else:
        kind = "benign"
    if kind != "benign" and not labels:
        return None, "cwe_not_in_kb"

    lang = record.get("lang") or guess_lang(record.get("file_name"), func)
    name = extract_function_name(func, lang)
    key = (str(record.get("commit_id")), name)
    counterpart = record.get("counterpart")
    if counterpart is None:
        counterpart = patched.get(key) if kind == "vulnerable" else (vulnerable_map or {}).get(key) if kind == "patched" else None
    sink = _choose_sink(kb, func, lang, name, labels, record, counterpart, sinkless)
    if sink is None:
        return None, "no_sink"

    entry = sink["entry"]
    file_label = _file_label(record, name, lang)
    block = {"file": file_label, "name": name, "start_line": 1, "end_line": func.count("\n") + 1, "depth": 0, "lang": lang, "code": func}
    location = {"function": sink["callee"], "file": file_label, "line": sink["line"]}
    messages = common.build_messages(entry["id"], common.render_card(entry), location, [block])

    finding = {
        "vulnerable": kind == "vulnerable",
        "cwe_id": entry["id"],
        "severity": entry["default_severity"] if kind == "vulnerable" else "NONE",
        "sink_function": sink["callee"],
        "sink_file": file_label,
        "sink_line": sink["line"],
        "vex_status": "affected" if kind == "vulnerable" else "not_affected",
        "reasoning": _reasoning(kind, sink, file_label, entry),
        "suggested_fix": entry["fix_hint"] if kind == "vulnerable" else NO_FIX,
    }
    messages.append({"role": "assistant", "content": json.dumps(finding, ensure_ascii=False)})
    meta = {
        "idx": record.get("idx"),
        "project": record.get("project"),
        "commit_id": str(record.get("commit_id")),
        "cve": None if is_none(record.get("cve")) else record.get("cve"),
        "label": int(kind == "vulnerable"),
        "kind": kind,
        "cwe": entry["id"],
        "original_cwe": raw_labels,
        "lang": lang,
        "sink_function": sink["callee"],
        "sink_line": sink["line"],
        "sink_kind": sink["kind"],
        "localized": sink.get("how") if sink.get("how") in ("diff", "comment") else None,
        "source": record.get("source", "primevul"),
        "synthetic": bool(record.get("synthetic", False)),
        "func_hash": func_hash(func),
        "counterpart_hash": func_hash(counterpart) if counterpart else None,
    }
    return {"messages": messages, "meta": meta}, None


# --------------------------------------------------------------------------- split e bilanciamento
def group_key(record: dict) -> str:
    commit = record.get("commit_id")
    return f"idx:{record.get('idx')}" if is_none(commit) else str(commit)


def is_validation(group: str, pct: int, seed: int) -> bool:
    """Split deterministico per commit: funzione vulnerabile e relativa patch finiscono sempre nello stesso insieme."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(digest, 16) % 100 < pct


def negative_plan(n_vulnerable: int, n_patched: int, n_benign: int, ratio: float) -> tuple[int, int]:
    """(patch da tenere, benigni da tenere) per avere `ratio` negativi per ogni vulnerabile.

    Le patch sono i negativi più istruttivi (stesso codice, bug corretto) e hanno la precedenza,
    ma contano nel rapporto: PrimeVul ne ha molte più delle funzioni vulnerabili.
    """
    target = round(ratio * n_vulnerable)
    keep_patched = min(n_patched, target)
    keep_benign = min(n_benign, target - keep_patched)
    return keep_patched, keep_benign


def split_of(group: str, seed: int, val_pct: int, test_pct: int) -> str:
    """"train" | "val" | "test" deterministico per gruppo (commit): le due versioni di una funzione restano insieme."""
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 100
    return "test" if h < test_pct else "val" if h < test_pct + val_pct else "train"


def func_hash(code: str) -> str:
    """Impronta della funzione ignorando gli spazi: serve a non far entrare nel training ciò che sta nel test."""
    return hashlib.sha256(re.sub(r"\s+", "", code).encode()).hexdigest()[:32]


def forbidden_hashes(test_examples: list[dict]) -> set[str]:
    out: set[str] = set()
    for ex in test_examples:
        out.add(ex["meta"]["func_hash"])
        if ex["meta"].get("counterpart_hash"):
            out.add(ex["meta"]["counterpart_hash"])
    return out


def dedupe(examples: list[dict], forbidden: set[str]) -> tuple[list[dict], dict]:
    """Toglie gli esempi presenti nel test (o la cui controparte lo è) e i duplicati esatti."""
    kept, seen, removed = [], set(), Counter()
    for ex in examples:
        key = (ex["meta"]["func_hash"], ex["meta"]["label"])
        if ex["meta"]["func_hash"] in forbidden:
            removed["in_test"] += 1
        elif key in seen:
            removed["duplicate"] += 1
        else:
            seen.add(key)
            kept.append(ex)
    return kept, dict(removed)


def limit_function_level(examples: list[dict], share: float, seed: int) -> tuple[list[dict], int]:
    """Limita gli esempi "function-level" a `share` del totale (tutti i sink-chiamata restano), mantenendo il bilanciamento delle etichette."""
    calls = [e for e in examples if e["meta"]["sink_kind"] == "call"]
    if share >= 1 or not calls:
        return examples, 0
    allowed = int(share / (1 - share) * len(calls)) if share > 0 else 0
    rng = random.Random(seed)
    keep: set[int] = set()
    for label in (0, 1):
        pool = [e for e in examples if e["meta"]["sink_kind"] != "call" and e["meta"]["label"] == label]
        keep.update(id(e) for e in rng.sample(pool, min(allowed // 2, len(pool))))
    kept = [e for e in examples if e["meta"]["sink_kind"] == "call" or id(e) in keep]
    return kept, len(examples) - len(kept)


def limit_label_skew(examples: list[dict], max_ratio: float, seed: int) -> tuple[list[dict], int]:
    """Impedisce che un'etichetta superi `max_ratio` volte l'altra, scartando prima gli esempi sintetici (poi i reali)."""
    if max_ratio <= 0:
        return examples, 0
    by_label = {0: [e for e in examples if e["meta"]["label"] == 0], 1: [e for e in examples if e["meta"]["label"] == 1]}
    major, minor = (1, 0) if len(by_label[1]) >= len(by_label[0]) else (0, 1)
    excess = len(by_label[major]) - int(max_ratio * len(by_label[minor]))
    if excess <= 0:
        return examples, 0
    rng = random.Random(seed)
    synthetic = [e for e in by_label[major] if e["meta"].get("synthetic")]
    real = [e for e in by_label[major] if not e["meta"].get("synthetic")]
    drop = rng.sample(synthetic, min(excess, len(synthetic)))
    if excess > len(drop):
        drop += rng.sample(real, excess - len(drop))
    gone = {id(e) for e in drop}
    return [e for e in examples if id(e) not in gone], len(drop)


def choose_indices(n: int, k: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n), min(k, n)))


# --------------------------------------------------------------------------- esempi con slice, lunghezza, rapporto
def _message(example: dict, role: str) -> str:
    return next((m.get("content") or "" for m in example["messages"] if m.get("role") == role), "")


def _source(example: dict) -> str:
    return str(example["meta"].get("source") or "unknown")


def prompt_hash(example: dict) -> str:
    """Impronta del messaggio utente (senza spazi): negli esempi con slice coincide con `meta.func_hash`."""
    return func_hash(_message(example, "user"))


def _graph_problem(example) -> str | None:
    """Perché una riga di `data/graph/*.jsonl` non è un esempio di chat con metadati (None se va bene)."""
    if not isinstance(example, dict):
        return "non è un oggetto JSON"
    if not isinstance(example.get("meta"), dict):
        return "manca `meta` (un oggetto)"
    messages = example.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages):
        return "manca `messages` (lista di messaggi con testo)"
    if not {"user", "assistant"} <= {m.get("role") for m in messages}:
        return "`messages` deve avere un messaggio `user` e uno `assistant`"
    return None


def load_graph_examples(graph_dir: Path) -> dict[str, list[dict]]:
    """Esempi con slice di `DIR/*_train.jsonl` e `DIR/*_val.jsonl`, per nome di file (ordine deterministico), righe vuote saltate.

    Altri file della cartella (`*_report.json`, file di valutazione) non vengono letti. `ValueError` con `file:riga` se una riga
    non è JSON o non ha `messages` + `meta`: un file di dati rotto va corretto, non saltato in silenzio.
    """
    out: dict[str, list[dict]] = {"train": [], "val": []}
    for split, examples in out.items():
        for path in sorted(Path(graph_dir).glob(f"*_{split}.jsonl")):
            with path.open("r", encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path.name}:{number}: JSON non valido ({exc})") from exc
                    problem = _graph_problem(example)
                    if problem:
                        raise ValueError(f"{path.name}:{number}: {problem}")
                    examples.append(example)
    return out


def merge_graph_examples(existing: list[dict], graph: list[dict], test_prompts: set[str], forbidden: set[str]) -> tuple[list[dict], dict]:
    """`existing` + gli esempi con slice che non sono già presenti né stanno nel test: (elenco, scartati per motivo).

    Scartati come `in_test`: il prompt utente (impronta `prompt_hash`) compare nel test, oppure il loro `func_hash` è fra gli
    hash vietati (funzioni del test). Scartati come `duplicate`: stesso prompt (o stesso `func_hash`) e stessa etichetta di un
    esempio già raccolto. Gli esempi con slice non sono limitati da `limit_label_skew`/`limit_function_level`: il loro
    bilanciamento lo decide chi li ha scritti.
    """
    seen_prompt = {(prompt_hash(e), e["meta"].get("label")) for e in existing}
    seen_func = {(e["meta"]["func_hash"], e["meta"].get("label")) for e in existing if e["meta"].get("func_hash") is not None}
    kept: list[dict] = []
    removed: Counter = Counter()
    for ex in graph:
        meta, label, digest = ex["meta"], ex["meta"].get("label"), prompt_hash(ex)
        fhash = meta.get("func_hash")
        if digest in test_prompts or (fhash is not None and fhash in forbidden):
            removed["in_test"] += 1
        elif (digest, label) in seen_prompt or (fhash is not None and (fhash, label) in seen_func):
            removed["duplicate"] += 1
        else:
            seen_prompt.add((digest, label))
            if fhash is not None:
                seen_func.add((fhash, label))
            kept.append(ex)
    return existing + kept, dict(removed)


def max_example_chars(max_seq_length: int) -> float:
    """Caratteri oltre i quali un esempio supera (a stima) `max_seq_length` token."""
    return max_seq_length * CHARS_PER_TOKEN


def drop_too_long(examples: list[dict], max_chars: float) -> tuple[list[dict], dict]:
    """Toglie gli esempi con `len(user) + len(assistant) > max_chars` (il system prompt è costante e non conta): (rimasti, scartati per fonte)."""
    kept: list[dict] = []
    dropped: Counter = Counter()
    for ex in examples:
        if len(_message(ex, "user")) + len(_message(ex, "assistant")) > max_chars:
            dropped[_source(ex)] += 1
        else:
            kept.append(ex)
    return kept, dict(dropped)


def trace_shares(examples: list[dict]) -> dict:
    """Per ogni `meta.track`: quanti esempi hanno una traccia con almeno 2 blocchi (cioè dei chiamanti) e quanti mostrano il solo sink.

    `sink_only_share` è la quota REALIZZATA: `choose_views` la tiene al massimo a 1 − p_callers e la riequilibra per etichetta.
    Gli esempi senza `meta.track` (le fonti senza slice) non entrano.
    """
    counts: dict[str, Counter] = {}
    for ex in examples:
        track = ex["meta"].get("track")
        if not track:
            continue
        c = counts.setdefault(str(track), Counter())
        c["examples"] += 1
        trace = ex["meta"].get("trace")
        blocks = trace.get("blocks") if isinstance(trace, dict) else None
        if isinstance(blocks, list) and len(blocks) >= 2:
            c["multi_block"] += 1
        if ex["meta"].get("view") == "sink_only":
            c["sink_only"] += 1
    return {track: {"examples": c["examples"], "multi_block": c["multi_block"], "multi_block_share": round(c["multi_block"] / c["examples"], 4),
                    "sink_only": c["sink_only"], "sink_only_share": round(c["sink_only"] / c["examples"], 4)}
            for track, c in sorted(counts.items())}


def size_report(paths: list[Path], limit: int | None = None) -> tuple[dict[str, int], list[str]]:
    """({nome file: byte}, avvisi): un avviso per ogni file più grande di `limit` (default `MAX_FILE_BYTES`)."""
    limit = MAX_FILE_BYTES if limit is None else limit
    sizes = {Path(p).name: Path(p).stat().st_size for p in paths}
    warnings = [f"WARNING: {name} pesa {size:,} byte ({size / 2**20:.1f} MiB), oltre il limite di {limit:,} byte: "
                f"è troppo grande per il repository (ridurre gli esempi o dividere il file)"
                for name, size in sizes.items() if size > limit]
    return sizes, warnings


# --------------------------------------------------------------------------- I/O
def iter_records(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_pairs(path: Path, keep: Callable[[dict], bool], limit: int | None) -> tuple[dict, dict]:
    """Prima passata su PrimeVul: funzioni corrette e vulnerabili con CVE, indicizzate per (commit, nome)."""
    fixed: dict = {}
    vulnerable: dict = {}
    for rec in iter_records(path, limit):
        if is_none(rec.get("cve")) or is_none(rec.get("commit_id")) or not keep(rec):
            continue
        code = (rec.get("func") or "").lstrip("\n").rstrip()
        key = (str(rec["commit_id"]), extract_function_name(code, guess_lang(rec.get("file_name"), code)))
        (vulnerable if int(rec.get("target", 0)) == 1 else fixed)[key] = code
    return fixed, vulnerable


def process_records(records: Iterable[dict], kb: dict, *, ratio: float, seed: int, max_chars: int, tmp_path: Path,
                    sinkless: str = "function", patched: dict | None = None, vulnerable_map: dict | None = None) -> tuple[list[dict], dict]:
    patched, vulnerable_map = patched or {}, vulnerable_map or {}
    vulnerable: list[dict] = []
    patched_examples: list[dict] = []
    skipped: Counter = Counter()
    n_benign = 0
    with tmp_path.open("w", encoding="utf-8") as tmp:
        for rec in records:
            example, reason = build_example(rec, kb, patched, max_chars, sinkless, vulnerable_map)
            if example is None:
                skipped[reason] += 1
            elif example["meta"]["kind"] == "vulnerable":
                vulnerable.append(example)
            elif example["meta"]["kind"] == "patched":
                patched_examples.append(example)
            else:
                tmp.write(json.dumps(example, ensure_ascii=False) + "\n")
                n_benign += 1
    keep_patched, keep_benign = negative_plan(len(vulnerable), len(patched_examples), n_benign, ratio)
    picked_patched = [patched_examples[i] for i in choose_indices(len(patched_examples), keep_patched, seed)]
    wanted = set(choose_indices(n_benign, keep_benign, seed))
    picked_benign = []
    with tmp_path.open("r", encoding="utf-8") as tmp:
        for i, line in enumerate(tmp):
            if i in wanted:
                picked_benign.append(json.loads(line))
    tmp_path.unlink(missing_ok=True)
    examples = vulnerable + picked_patched + picked_benign
    return examples, {"skipped": dict(skipped), "available": {"vulnerable": len(vulnerable), "patched": len(patched_examples), "benign": n_benign}}


def process_file(path: Path, kb: dict, *, ratio: float, seed: int, max_chars: int, tmp_path: Path, sinkless: str = "function",
                 keep: Callable[[dict], bool] = lambda _r: True, limit: int | None = None) -> tuple[list[dict], dict]:
    patched, vulnerable_map = collect_pairs(path, keep, limit)
    records = (r for r in iter_records(path, limit) if keep(r))
    examples, stats = process_records(records, kb, ratio=ratio, seed=seed, max_chars=max_chars, tmp_path=tmp_path, sinkless=sinkless,
                                      patched=patched, vulnerable_map=vulnerable_map)
    random.Random(seed).shuffle(examples)
    stats["kept"] = dict(Counter(e["meta"]["kind"] for e in examples))
    stats["kept_sink_kind"] = dict(Counter(e["meta"]["sink_kind"] for e in examples))
    return examples, stats


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


HF_SOURCES = {  # nome -> (file locale, repo dataset HF, file nel repo)
    "secvuleval": ("secvuleval.parquet", "arag0rn/SecVulEval", "data/train-00000-of-00001.parquet"),
    "python": ("cvefixes_pyvul.parquet", "janhvi-ag/cvefixes_pyvul_merged", "data/train-00000-of-00001.parquet"),
    "juliet": ("juliet_test.csv", "LorenzH/juliet_test_suite_c_1_3", "jts_c_1_3_test.csv"),
}
ALL_SOURCES = ("primevul", "secvuleval", "python", "codeql")  # Juliet entra da `--graph-dir` (con le slice), non più da qui
LEGACY_SOURCES = ("juliet",)  # scaricabile solo se richiesto: il CSV ha le parole-etichetta (FLAW, good, bad…) dentro il codice
CODEQL_URL = "https://github.com/github/codeql.git"


def fetch(url: str, dest: Path) -> Path:
    """Scarica `url` in `dest` (se manca) con la sola libreria standard."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"scarico {dest.name} ...", file=sys.stderr)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        tmp.rename(dest)
    return dest


def download_primevul(raw_dir: Path) -> dict[str, Path]:
    """Scarica gli split originali da Hugging Face (colin/PrimeVul)."""
    return {split: fetch(f"{HF_BASE}/{fname}", raw_dir / fname) for split, fname in PRIMEVUL_FILES.items()}


def download_source(name: str, raw_dir: Path) -> Path:
    fname, repo, remote = HF_SOURCES[name]
    return fetch(f"https://huggingface.co/datasets/{repo}/resolve/main/{remote}", raw_dir / fname)


def download_codeql(raw_dir: Path) -> Path:
    """Sparse checkout (≈15 MB) delle sole cartelle dei test di sicurezza di CodeQL; `git` senza shell, nessun codice eseguito."""
    dest = raw_dir / "codeql-tests"
    if (dest / ".git").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--quiet", "--", CODEQL_URL, str(dest)], check=True, env=env, timeout=900)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *sources.CODEQL_ROOTS], check=True, env=env, timeout=900)
    return dest


def merge_counts(parts: list[dict]) -> dict:
    out: dict = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, dict):
                bucket = out.setdefault(key, {})
                for k, v in value.items():
                    bucket[k] = bucket.get(k, 0) + v
    return out


def format_report(results: dict, warnings: list[str], max_chars: float) -> list[str]:
    """Rapporto finale (righe di testo): dimensione dei file, esempi per fonte, scartati per lunghezza, quote delle tracce nel train."""
    lines = ["", "== rapporto finale =="]
    lines.append("file scritti:")
    for split, res in results.items():
        lines.append(f"  {split}.jsonl: {res['file_bytes']:,} byte ({res['file_bytes'] / 2**20:.1f} MiB)")
    lines += warnings
    lines.append("esempi per fonte:")
    for split, res in results.items():
        counts = ", ".join(f"{source} {n}" for source, n in res["source_counts"].items()) or "nessuno"
        lines.append(f"  {split} ({sum(res['source_counts'].values())}): {counts}")
    lines.append(f"scartati perché più lunghi di {max_chars:,.0f} caratteri (user + assistant, ≈ {max_chars / CHARS_PER_TOKEN:,.0f} token):")
    for split, res in results.items():
        if split != "test":
            dropped = ", ".join(f"{source}: {n}" for source, n in sorted(res["too_long_dropped"].items())) or "nessuno"
            lines.append(f"  {split}: {dropped}")
    shares = results.get("train", {}).get("trace_shares", {})
    lines.append("esempi con slice nel train, per traccia (chiamanti = traccia con almeno 2 blocchi; solo sink = quota realizzata):")
    for track, s in shares.items():
        lines.append(f"  {track}: {s['examples']} esempi, con chiamanti {100 * s['multi_block_share']:.1f}% ({s['multi_block']}), "
                     f"solo sink {100 * s['sink_only_share']:.1f}% ({s['sink_only']})")
    if not shares:
        lines.append("  nessuno")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, help="PrimeVul: primevul_train.jsonl")
    ap.add_argument("--valid", type=Path, help="PrimeVul: primevul_valid.jsonl (se assente, la validazione viene estratta da train per commit)")
    ap.add_argument("--test", type=Path, help="PrimeVul: primevul_test.jsonl (set non visto, usato dal benchmark)")
    ap.add_argument("--secvuleval", type=Path, help="SecVulEval (.parquet/.jsonl): C/C++ con coppie vulnerabile/corretta")
    ap.add_argument("--python-pairs", type=Path, help="CVEfixes+PyVul Python (.parquet/.jsonl): code_before/code_after")
    ap.add_argument("--juliet", type=Path, help="legacy: contiene etichette nel codice. Juliet C/C++ (.csv): sintetico, solo train/val "
                                                "(mai nel test); per Juliet con le slice usare --graph-dir")
    ap.add_argument("--graph-dir", type=Path, help="cartella con gli esempi con slice (*_train.jsonl, *_val.jsonl): si accodano a train e val, mai al test; "
                                                   "i test CodeQL Python di train/val vengono sostituiti da quelli con slice")
    ap.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                    help=f"lunghezza in token di train.py: gli esempi di train/val con più di {CHARS_PER_TOKEN} caratteri per token (user + assistant) vengono scartati e contati")
    ap.add_argument("--codeql", type=Path, help="checkout dei test di sicurezza di CodeQL (cartella con python/ql/test/... e cpp/ql/test/...)")
    ap.add_argument("--download", action="store_true", help="scarica in --raw-dir le fonti scelte con --sources (≈900 MB in tutto)")
    ap.add_argument("--sources", nargs="+", choices=ALL_SOURCES + LEGACY_SOURCES, default=list(ALL_SOURCES), help="fonti da scaricare/usare con --download")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--val-pct", type=int, default=5, help="percentuale di validazione (per commit) dove non c'è uno split ufficiale")
    ap.add_argument("--holdout-pct", type=int, default=10, help="percentuale di commit di SecVulEval/Python destinata al test")
    ap.add_argument("--function-share", type=float, default=0.5,
                    help="quota massima di esempi 'function-level' (sink = la funzione) in train/val; 1 = nessun limite, 0 = solo sink-chiamata")
    ap.add_argument("--label-ratio", type=float, default=1.25,
                    help="rapporto massimo fra l'etichetta più numerosa e l'altra in train/val (si scartano prima i sintetici); 0 = nessun limite")
    ap.add_argument("--juliet-per-cwe", type=int, default=150, help="casi di test Juliet per CWE (il sintetico non deve dominare)")
    ap.add_argument("--max-chars", type=int, default=12000, help="scarta funzioni più lunghe (≈3.5k token)")
    ap.add_argument("--benign-ratio", type=float, default=1.0, help="negativi per ogni esempio vulnerabile (train/val)")
    ap.add_argument("--test-benign-ratio", type=float, default=None, help="come --benign-ratio ma per il test (default: uguale)")
    ap.add_argument("--sinkless", choices=["function", "skip"], default="function",
                    help="funzioni con CWE nota ma senza chiamata-sink localizzabile: usare la funzione come sink (default) o scartarle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="legge solo i primi N record per file (prove rapide)")
    ap.add_argument("--kb", type=Path, default=None, help="cwe_kb.json alternativo")
    args = ap.parse_args(argv)

    if args.download:
        if "primevul" in args.sources:
            got = download_primevul(args.raw_dir)
            args.train, args.valid, args.test = args.train or got["train"], args.valid or got["valid"], args.test or got["test"]
        for flag, name in (("secvuleval", "secvuleval"), ("python_pairs", "python"), ("juliet", "juliet")):
            if name in args.sources and getattr(args, flag) is None:
                setattr(args, flag, download_source(name, args.raw_dir))
        if "codeql" in args.sources and args.codeql is None:
            args.codeql = download_codeql(args.raw_dir)
    if args.graph_dir is not None and not args.graph_dir.is_dir():
        ap.error(f"--graph-dir: {args.graph_dir} non è una cartella")
    if not any([args.train, args.secvuleval, args.python_pairs, args.juliet, args.codeql, args.graph_dir]):
        ap.error("serve almeno una fonte: --train/--secvuleval/--python-pairs/--juliet/--codeql/--graph-dir (oppure --download)")

    kb = common.load_kb(args.kb)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / ".benign_tmp.jsonl"
    test_ratio = args.benign_ratio if args.test_benign_ratio is None else args.test_benign_ratio
    common_kw = dict(seed=args.seed, max_chars=args.max_chars, tmp_path=tmp, sinkless=args.sinkless)
    collected: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    per_source: dict[str, dict[str, list[dict]]] = {"train": {}, "val": {}, "test": {}}  # split -> sorgente -> statistiche dei lavori

    def add(split: str, source: str, examples: list[dict], stats: dict) -> None:
        collected[split].extend(examples)
        per_source[split].setdefault(source, []).append(stats)
        print(f"{source}/{split}: {len(examples)} esempi (scartati: {stats['skipped']})", file=sys.stderr)

    # PrimeVul: split ufficiali (la validazione si ricava da train per commit se manca il file)
    if args.train:
        if args.valid:
            jobs = [("train", args.train, args.benign_ratio, lambda _r: True), ("val", args.valid, args.benign_ratio, lambda _r: True)]
        else:
            def in_val(rec: dict) -> bool:
                return is_validation(group_key(rec), args.val_pct, args.seed)

            jobs = [("train", args.train, args.benign_ratio, lambda r: not in_val(r)), ("val", args.train, args.benign_ratio, in_val)]
        if args.test:
            jobs.append(("test", args.test, test_ratio, lambda _r: True))
        for split, path, ratio, keep in jobs:
            examples, stats = process_file(path, kb, ratio=ratio, keep=keep, limit=args.limit, seed=args.seed, max_chars=args.max_chars,
                                           tmp_path=tmp, sinkless=args.sinkless)
            add(split, "primevul", examples, stats)

    # Altre fonti: split per commit; Juliet (varianti quasi identiche dello stesso caso) non entra mai nel test
    rows_of = lambda path: sources.read_rows(path, args.limit)  # noqa: E731
    with_graph = args.graph_dir is not None
    other = [
        ("secvuleval", args.secvuleval, lambda p: sources.secvuleval_records(rows_of(p)), args.holdout_pct),
        ("cvefixes-python", args.python_pairs, lambda p: sources.python_pair_records(rows_of(p), kb), args.holdout_pct),
        # con --graph-dir i test CodeQL Python di train/val sono quelli con slice (traccia P1): qui restano i C/C++
        ("codeql-tests", args.codeql, lambda p: sources.codeql_records(sources.codeql_files(p), kb, skip_python=with_graph), args.holdout_pct),
        ("juliet", args.juliet, lambda p: sources.juliet_records(rows_of(p), args.juliet_per_cwe, kb), 0),
    ]
    for source, path, to_records, test_pct in other:
        if not path:
            continue
        buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for rec in to_records(path):
            buckets[split_of(group_key(rec), args.seed, args.val_pct, test_pct)].append(rec)
        if source == "codeql-tests" and with_graph:
            # il test è il metro di paragone e la sua composizione non dipende da --graph-dir: i Python di CodeQL restano lì
            # (stesso ordine di prima, quindi stessa scelta dei negativi) anche se in train/val li sostituisce la versione con slice
            buckets["test"] = [r for r in sources.codeql_records(sources.codeql_files(path), kb)
                               if split_of(group_key(r), args.seed, args.val_pct, test_pct) == "test"]
        for split, recs in buckets.items():
            ratio = test_ratio if split == "test" else args.benign_ratio
            examples, stats = process_records(recs, kb, ratio=ratio, **common_kw)
            add(split, source, examples, stats)

    # Il test è il metro di paragone: non deve contenere duplicati, e nulla di ciò che vi compare va usato per addestrare
    results: dict[str, dict] = {}
    collected["test"], removed_test = dedupe(collected["test"], set())
    forbidden = forbidden_hashes(collected["test"])
    removed = {"test": removed_test}
    function_dropped = {"train": 0, "val": 0, "test": 0}
    skew_dropped = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val"):
        collected[split], removed[split] = dedupe(collected[split], forbidden)
        collected[split], skew_dropped[split] = limit_label_skew(collected[split], args.label_ratio, args.seed)
        collected[split], function_dropped[split] = limit_function_level(collected[split], args.function_share, args.seed)
        collected[split], more = limit_label_skew(collected[split], args.label_ratio, args.seed)  # il taglio dei function-level può averla spostata
        skew_dropped[split] += more

    # Esempi con slice: dopo i limiti delle fonti vecchie (che non li toccano), con le stesse garanzie (nessun duplicato esatto,
    # nulla che compaia nel test) e mai nel test
    graph_added = {"train": 0, "val": 0, "test": 0}
    graph_removed: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    if with_graph:
        graph = load_graph_examples(args.graph_dir)
        if not any(graph.values()):
            print(f"ATTENZIONE: {args.graph_dir} non contiene esempi in *_train.jsonl / *_val.jsonl", file=sys.stderr)
        test_prompts = {prompt_hash(e) for e in collected["test"]}
        for split in ("train", "val"):
            before = len(collected[split])
            collected[split], graph_removed[split] = merge_graph_examples(collected[split], graph[split], test_prompts, forbidden)
            graph_added[split] = len(collected[split]) - before
            print(f"graph/{split}: {len(graph[split])} esempi letti, {graph_added[split]} aggiunti (scartati: {graph_removed[split]})", file=sys.stderr)

    # Un esempio oltre `--max-seq-length` verrebbe troncato dal training (via il JSON del verdetto, che sta in fondo): si scarta e si conta.
    # Il test non si tocca: è il metro di paragone e non passa dal tokenizer del training.
    max_chars = max_example_chars(args.max_seq_length)
    too_long: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val"):
        collected[split], too_long[split] = drop_too_long(collected[split], max_chars)

    def kept_counts(examples: list[dict]) -> dict:
        return {  # `.get`: gli esempi con slice non hanno `localized`
            "kept": dict(Counter(str(e["meta"].get("kind")) for e in examples)),
            "kept_sink_kind": dict(Counter(str(e["meta"].get("sink_kind")) for e in examples)),
            "kept_localized": dict(Counter(str(e["meta"].get("localized")) for e in examples)),
        }

    written: dict[str, Path] = {}
    for split, examples in collected.items():
        if not examples and not per_source[split]:
            continue
        random.Random(args.seed).shuffle(examples)
        written[split] = args.out_dir / f"{split}.jsonl"
        write_jsonl(written[split], examples)
        by_source = {}
        for source, parts in per_source[split].items():
            by_source[source] = {**merge_counts(parts), **kept_counts([e for e in examples if _source(e) == source])}
        for source in sorted({_source(e) for e in examples} - set(per_source[split])):  # senza lavoro con questo nome: esempi con slice, `pyvul`
            by_source[source] = kept_counts([e for e in examples if _source(e) == source])
        results[split] = {
            **merge_counts([p for parts in per_source[split].values() for p in parts]),
            **kept_counts(examples),
            "duplicates_removed": removed[split],
            "function_level_dropped": function_dropped[split],
            "label_skew_dropped": skew_dropped[split],
            "by_source": by_source,
            "source_counts": dict(sorted(Counter(_source(e) for e in examples).items())),
            "graph_added": graph_added[split],
            "graph_removed": graph_removed[split],
            "too_long_dropped": too_long[split],
            "trace_shares": trace_shares(examples),
        }
        print(f"{split}: {len(examples)} esempi {results[split]['kept']} sink={results[split]['kept_sink_kind']}", file=sys.stderr)

    sizes, warnings = size_report(list(written.values()), MAX_FILE_BYTES)
    for split, path in written.items():
        results[split]["file_bytes"] = sizes[path.name]
        results[split]["over_size_limit"] = sizes[path.name] > MAX_FILE_BYTES
    (args.out_dir / "stats.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print("\n".join(format_report(results, warnings, max_chars)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
