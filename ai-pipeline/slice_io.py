"""Wrapper Python del sottocomando `sast-engine slice`: le backward slice reali del grafo di chiamate.

Il motore stampa un record JSON per riga su stdout (`{"sink", "blocks", "truncated"}`) e i messaggi di
diagnostica su stderr; qui li si restituisce entrambi, senza inghiottire nulla. Solo libreria standard.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_ENGINE = Path(__file__).resolve().parents[1] / "sast-engine" / "target" / "release" / "sast-engine"


def engine_bin() -> Path | None:
    """`$SAST_ENGINE_BIN` se impostata (e il file esiste), altrimenti il binario release della repo, se compilato."""
    env = os.environ.get("SAST_ENGINE_BIN")
    candidate = Path(env) if env else DEFAULT_ENGINE
    return candidate if candidate.is_file() else None


def write_targets(path: Path, targets: list[dict]) -> None:
    """Un target `{"file", "line", "callee", "cwe_id"}` per riga (JSONL), come si aspetta `--targets`."""
    lines = (json.dumps(t, ensure_ascii=False) + "\n" for t in targets)
    Path(path).write_text("".join(lines), encoding="utf-8")


def run_slice(
    root: Path,
    targets: list[dict] | None = None,
    *,
    depth: int = 5,
    callers: int = 4,
    max_chars: int = 12000,
    binary: Path | None = None,
) -> tuple[list[dict], str]:
    """Lancia `sast-engine slice` su `root` e restituisce (record JSON, stderr del motore).

    `targets=None` lascia al motore i sink del catalogo della KB; una lista (anche vuota) limita il
    calcolo a quei sink. Esce con `RuntimeError` se il binario manca o il motore termina con codice ≠ 0.
    """
    binary = binary or engine_bin()
    if binary is None:
        raise RuntimeError(
            "binario di sast-engine non trovato: imposta SAST_ENGINE_BIN al suo percorso "
            "oppure compilalo con `cargo build --release` in sast-engine/"
        )
    cmd = [str(binary), "slice", str(root), "--depth", str(depth), "--callers", str(callers), "--max-chars", str(max_chars)]
    targets_file = None
    try:
        if targets is not None:
            fd, name = tempfile.mkstemp(prefix="slice-targets-", suffix=".jsonl")
            os.close(fd)
            targets_file = Path(name)
            write_targets(targets_file, targets)
            cmd += ["--targets", str(targets_file)]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    finally:
        if targets_file is not None:
            targets_file.unlink(missing_ok=True)
    if out.returncode != 0:
        raise RuntimeError(f"sast-engine slice è uscito con codice {out.returncode}: {out.stderr.strip()}")
    records = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    return records, out.stderr
