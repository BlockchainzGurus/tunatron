"""Lista congelata di sink di pygoat per S3 (falsi positivi) e S4 (troncamenti e CVE inventate).

I sink sono scelti a mano, uno per uno, dopo aver letto il sorgente (`candidates` dà l'elenco e 5 righe di contesto):
`expected` è «safe» solo per i casi certi (destinazione costante di `redirect`, `subprocess` con lista costante,
`CREATE TABLE`, `requests.get` con URL dalle impostazioni) e «vulnerable» solo per quelli verificati. I dubbi restano fuori.
Il file è `{"repo", "commit", "entries": [{"file", "line", "callee", "cwe_id", "expected", "evidence"}]}`.

`score_frozen` confronta le risposte del modello con la lista; `benchmark.py frozen` le chiede a vLLM. Solo libreria standard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import slice_io

REQUIRED = ("file", "line", "callee", "cwe_id", "expected", "evidence")
EXPECTED = ("safe", "vulnerable")
CONTEXT = 2  # righe prima e dopo il sink nell'estratto
_CVE = re.compile(r"CVE-\d{4}-\d+")


def candidates(repo: Path, binary: Path | None = None) -> list[dict]:
    """Un candidato per ogni sink del catalogo del repo (slice a profondità 5), ordinati per file, riga e callee.

    `excerpt` sono le 5 righe attorno al sink, numerate, per la revisione a mano; vuoto se il file non si legge o il suo
    percorso esce dal repo.
    """
    repo = Path(repo)
    records, _stderr = slice_io.run_slice(repo, None, depth=5, binary=binary)
    out = []
    for rec in records:
        sink = rec["sink"]
        out.append({"file": sink["file"], "line": sink["line"], "callee": sink["function"], "cwe_id": sink["cwe_id"],
                    "excerpt": _excerpt(repo, sink["file"], sink["line"])})
    return sorted(out, key=lambda c: (c["file"], c["line"], c["callee"]))


def _excerpt(repo: Path, file: str, line: int) -> str:
    path = (repo / file).resolve()
    if not path.is_relative_to(repo.resolve()) or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    first, last = max(1, line - CONTEXT), min(len(lines), line + CONTEXT)
    return "\n".join(f"{n:4d} | {lines[n - 1]}" for n in range(first, last + 1))


def load_frozen(path: Path) -> dict:
    """Legge e valida la lista; `ValueError` se manca un campo, `expected` non è `safe`/`vulnerable` o una voce è doppia."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("la lista congelata deve essere un oggetto JSON")
    for key in ("repo", "commit"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"manca il campo '{key}'")
    if not isinstance(data.get("entries"), list):
        raise ValueError("'entries' deve essere una lista")
    seen = set()
    for n, e in enumerate(data["entries"]):
        if not isinstance(e, dict):
            raise ValueError(f"voce {n}: non è un oggetto")
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            raise ValueError(f"voce {n}: mancano {missing}")
        if not isinstance(e["line"], int) or isinstance(e["line"], bool):
            raise ValueError(f"voce {n}: 'line' deve essere un intero")
        if e["expected"] not in EXPECTED:
            raise ValueError(f"voce {n}: 'expected' deve essere uno di {EXPECTED}, non {e['expected']!r}")
        if not isinstance(e["evidence"], str) or not e["evidence"].strip():
            raise ValueError(f"voce {n}: 'evidence' vuota (una riga che giustifichi il verdetto)")
        key = (e["file"], e["line"], e["callee"])
        if key in seen:
            raise ValueError(f"voce {n}: {key} è già nella lista")
        seen.add(key)
    return data


def score_frozen(frozen: dict, results: list[dict]) -> dict:
    """Punteggi S3/S4 su `results[i] = {"vulnerable": bool | None, "truncated": bool, "reasoning": str}` per `entries[i]`.

    - `fp_still_affected`: quota dei «safe» a cui il modello risponde `vulnerable: true` (S3);
    - `tp_recall`: quota dei «vulnerable» a cui risponde `true`;
    - `truncated_rate`: quota di risposte troncate su tutte (S4);
    - `cve_mentions`: occorrenze di un id CVE nei `reasoning` (S4: devono essere 0).
    Una risposta non valida (`None`) non è né un falso positivo né un successo. `ValueError` se le lunghezze non coincidono.
    """
    entries = frozen["entries"]
    if len(entries) != len(results):
        raise ValueError(f"{len(results)} risultati per {len(entries)} voci")
    safe = [r for e, r in zip(entries, results) if e["expected"] == "safe"]
    vulnerable = [r for e, r in zip(entries, results) if e["expected"] == "vulnerable"]
    ratio = lambda part, whole: round(part / whole, 4) if whole else 0.0  # noqa: E731
    return {
        "n": len(entries),
        "safe": len(safe),
        "vulnerable": len(vulnerable),
        "invalid": sum(r["vulnerable"] is None for r in results),
        "fp_still_affected": ratio(sum(r["vulnerable"] is True for r in safe), len(safe)),
        "tp_recall": ratio(sum(r["vulnerable"] is True for r in vulnerable), len(vulnerable)),
        "truncated_rate": ratio(sum(bool(r["truncated"]) for r in results), len(results)),
        "cve_mentions": sum(len(_CVE.findall(r.get("reasoning") or "")) for r in results),
        "by_cwe": _by_cwe(entries, results),
    }


def _by_cwe(entries: list[dict], results: list[dict]) -> dict:
    """`{cwe: {safe, still_affected, vulnerable, recalled}}`: la lista è dominata da poche classi, S3 va letto anche per classe."""
    out: dict[str, dict] = {}
    for e, r in zip(entries, results):
        row = out.setdefault(e["cwe_id"], {"safe": 0, "still_affected": 0, "vulnerable": 0, "recalled": 0})
        if e["expected"] == "safe":
            row["safe"] += 1
            row["still_affected"] += r["vulnerable"] is True
        else:
            row["vulnerable"] += 1
            row["recalled"] += r["vulnerable"] is True
    return dict(sorted(out.items()))
