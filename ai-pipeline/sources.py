"""Adattatori dei dataset pubblici: righe -> record nel formato di prepare_data (campi di PrimeVul + alcuni extra).

Campi extra riconosciuti da `prepare_data.build_example`:
  lang          linguaggio esplicito ("c", "cpp", "python")
  counterpart   codice della versione corretta (per le vulnerabili) o vulnerabile (per le corrette): permette di
                localizzare il sink col diff
  sink_hint     {"callee", "line"} già noto (commenti FLAW/FIX di Juliet)
  kind          "vulnerable" | "patched" | "benign" forzato
  note          riga di commit message (informativa: il target NON la usa, vedi `prepare_data._reasoning`)
  source / synthetic   provenienza dell'esempio

Formati: .parquet (richiede pyarrow, installato da scripts/setup_env.sh), .jsonl, .csv.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import common
import localize

CWE_LABEL = re.compile(r"CWE-\d+")


def read_rows(path: Path, limit: int | None = None) -> Iterator[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    count = 0
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if limit and count >= limit:
                        return
    elif suffix == ".csv":
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                yield row
                count += 1
                if limit and count >= limit:
                    return
    elif suffix == ".parquet":
        import pyarrow.parquet as pq  # import tardivo: serve solo per i file parquet

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                yield row
                count += 1
                if limit and count >= limit:
                    return
    else:
        raise ValueError(f"formato non supportato: {path.suffix} (usa .parquet, .jsonl o .csv)")


def _present(value) -> bool:
    return value is not None and value == value  # esclude None e NaN


def _lang_from_path(file_name: str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower()
    return "cpp" if ext in ("cc", "cpp", "cxx", "hpp", "hh", "hxx") else "python" if ext == "py" else "c"


def secvuleval_records(rows: Iterable[dict]) -> Iterator[dict]:
    """SecVulEval (MIT): funzioni C/C++ di commit di fix, con `fixed_func_idx` che lega la vulnerabile alla corretta."""
    rows = [r for r in rows if (r.get("func_body") or "").strip()]
    by_idx = {int(r["idx"]): r for r in rows}
    reverse = {int(r["fixed_func_idx"]): int(r["idx"]) for r in rows if r["is_vulnerable"] and _present(r.get("fixed_func_idx"))}
    for r in rows:
        idx = int(r["idx"])
        if r["is_vulnerable"]:
            other = by_idx.get(int(r["fixed_func_idx"])) if _present(r.get("fixed_func_idx")) else None
        else:
            other = by_idx.get(reverse[idx]) if idx in reverse else None
        filepath = r.get("filepath") or ""
        yield {
            "idx": f"secvuleval:{idx}",
            "project": r.get("project"),
            "commit_id": r.get("commit_id"),
            "target": int(bool(r["is_vulnerable"])),
            "func": r["func_body"],
            "file_name": filepath,
            "cwe": [c for c in (r.get("cwe_list") or []) if CWE_LABEL.fullmatch(c)],
            "cve": (r.get("cve_list") or [None])[0],
            "cve_desc": None,
            "note": (r.get("commit_message") or "").strip().split("\n")[0] or None,
            "counterpart": other["func_body"] if other else None,
            "lang": _lang_from_path(filepath),
            "source": "secvuleval",
            "synthetic": False,
        }


def juliet_records(rows: Iterable[dict], per_cwe: int | None = None, kb: dict | None = None) -> Iterator[dict]:
    """Juliet C/C++ 1.3 (CC0): funzione `*_bad` col sink marcato da `/* FLAW */` e varianti `good*` col sink dopo `/* FIX */`.

    Con `kb`, una variante buona senza commento FIX ma che usa un sink della stessa CWE diventa comunque un negativo difficile.
    """
    groups_per_cwe: dict[str, set[str]] = {}
    for row in rows:
        filename = row["filename"]
        m = re.search(r"CWE(\d+)", filename)
        if not m:
            continue
        cwe = f"CWE-{m.group(1)}"
        group = "juliet:" + re.sub(r"[a-z]$", "", Path(filename).stem)  # 52a/52b/52c = stesso caso di test
        seen = groups_per_cwe.setdefault(cwe, set())
        if group not in seen:
            if per_cwe and len(seen) >= per_cwe:
                continue
            seen.add(group)
        lang = "cpp" if filename.endswith(".cpp") else "c"
        base = {"project": "juliet", "commit_id": group, "file_name": Path(filename).name, "cwe": [cwe], "cve": "None",
                "cve_desc": "None", "lang": lang, "source": "juliet", "synthetic": True}
        for variant, text, hint_of, target, kind in (("bad", row.get("bad") or "", localize.flaw_sink, 1, "vulnerable"),
                                                    ("good", row.get("good") or "", localize.fix_sink, 0, "patched")):
            for fn in localize.split_c_functions(text):
                hint = hint_of(fn["code"]) if fn["name"] != "main" else None
                if hint is None and variant == "good" and kb is not None:
                    hint = _sink_of_cwe(kb, fn["code"], lang, cwe)
                if hint:
                    yield {**base, "idx": f"juliet:{filename}:{fn['name']}", "target": target, "kind": kind, "func": fn["code"], "sink_hint": hint}


def _sink_of_cwe(kb: dict, code: str, lang: str, cwe: str) -> dict | None:
    """Prima chiamata del corpo che è un sink della KB per la CWE data."""
    wanted = common.resolve_cwe(kb, cwe)
    for callee, line in localize.find_calls(code, lang):
        entry = common.match_sink(kb, lang, callee)
        if entry is not None and entry["id"] == wanted:
            return {"callee": callee, "line": line}
    return None


def python_pair_records(rows: Iterable[dict], kb: dict) -> Iterator[dict]:
    """Coppie prima/dopo la fix di funzioni Python (CVEfixes, PyVul). Se manca la CWE si ricava dal sink toccato dalla fix."""
    for row in rows:
        before, after = row.get("code_before"), row.get("code_after")
        if not before or not before.strip():
            continue
        label = row.get("cwe_id")
        cwe = [label] if label and CWE_LABEL.fullmatch(label) else []
        if after:
            if not cwe:
                found = localize.localize_sink(kb, before, "python", after)
                if found is None:
                    continue  # né etichetta né sink localizzabile: non si può insegnare nulla di affidabile
                cwe = [found["entry"]["id"]]
        elif not cwe:
            continue
        repo = (row.get("repo_url") or "").rstrip("/")
        base = {"project": repo.rsplit("/", 1)[-1] or None, "commit_id": row.get("commit_hash"), "cwe": cwe, "cve": row.get("cve_id"),
                "cve_desc": row.get("cve_description"), "lang": "python", "file_name": None,
                "source": "pyvul" if "PyVul" in (row.get("source_dataset") or "") else "cvefixes-python", "synthetic": False}
        key = f"{base['source']}:{base['commit_id']}:{row.get('function_name')}"
        yield {**base, "idx": key + ":before", "target": 1, "func": before, "counterpart": after or None}
        if after:
            yield {**base, "idx": key + ":after", "target": 0, "func": after, "counterpart": before}


CODEQL_ROOTS = ("python/ql/test/query-tests/Security", "cpp/ql/test/query-tests/Security")
CODEQL_SUFFIXES = {".py": "python", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp"}


def codeql_files(root: Path) -> Iterator[tuple[str, str]]:
    """(percorso relativo, testo) dei file di test di sicurezza di CodeQL (MIT) presenti sotto `root`."""
    root = Path(root)
    for sub in CODEQL_ROOTS:
        for path in sorted((root / sub).rglob("*")):
            if path.is_file() and path.suffix.lower() in CODEQL_SUFFIXES and not path.is_symlink():
                yield path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace")


def codeql_records(files: Iterable[tuple[str, str]], kb: dict, skip_python: bool = False) -> Iterator[dict]:
    """Un record per ogni chiamata-sink di ogni funzione dei test CodeQL: `$ Alert` = vulnerabile, sink senza annotazione = sicuro.

    Le annotazioni e i commenti GOOD/BAD vengono tolti dal codice (non devono finire nel prompt) senza spostare le righe.
    Le righe `MISSING:`/`SPURIOUS:` sono ambigue e vengono escluse. La CWE viene dal nome della directory del test.
    Con `skip_python` i file `.py` non producono record (con `--graph-dir` i test Python hanno la loro versione con slice);
    i record C/C++ restano identici e nello stesso ordine.
    """
    for path, text in files:
        m = re.search(r"CWE-?(\d+)", path)
        suffix = Path(path).suffix.lower()
        if not m or suffix not in CODEQL_SUFFIXES or "py2" in path:
            continue
        if skip_python and CODEQL_SUFFIXES[suffix] == "python":
            continue
        cwe = common.resolve_cwe(kb, f"CWE-{int(m.group(1))}")
        if cwe is None:
            continue
        lang = CODEQL_SUFFIXES[suffix]
        marker = "#" if lang == "python" else "//"
        notes = localize.annotated_lines(text, marker)
        clean = localize.scrub_labels(text, marker)
        functions = localize.split_python_functions(clean) if lang == "python" else localize.split_c_functions(clean)
        for fn in functions:
            start = fn["start_line"]
            seen: set[int] = set()
            calls = localize._calls_with_span(fn["code"], lang)
            # a parità di riga si preferisce la chiamata che è davvero un sink della CWE (poi la prima in ordine)
            calls.sort(key=lambda c: (c[1], 0 if (common.match_sink(kb, lang, c[0]) or {}).get("id") == cwe else 1))
            for callee, first, last in calls:
                span = range(start + first - 1, start + last)
                if any(n in notes["ambiguous"] for n in span) or first in seen:
                    continue
                alert = any(n in notes["alert"] for n in span)
                entry = common.match_sink(kb, lang, callee)
                if not alert and (entry is None or entry["id"] != cwe):
                    continue
                seen.add(first)
                yield {
                    "idx": f"codeql:{path}:{fn['name']}:{start + first - 1}",
                    "project": "codeql",
                    "commit_id": f"codeql:{path}",
                    "target": int(alert),
                    "kind": "vulnerable" if alert else "patched",
                    "func": fn["code"],
                    "file_name": Path(path).name,
                    "cwe": [cwe],
                    "cve": "None",
                    "cve_desc": "None",
                    "lang": lang,
                    "sink_hint": {"callee": callee, "line": first},
                    "source": "codeql-tests",
                    "synthetic": True,
                }
