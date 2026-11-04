#!/usr/bin/env python3
"""Sorgenti dei kernel CUDA di Mamba da PyPI, con `-std=c++17` portato a `-std=c++20`.

    python3 -I ai-pipeline/patched_sdist.py causal-conv1d 1.7.0 DEST

causal-conv1d 1.7.0 e mamba-ssm 2.3.2.post1 fissano `-std=c++17` nel loro setup.py, ma gli header di torch >= 2.14 richiedono
C++20: con il flag originale la compilazione fallisce. Lo script scarica lo sdist, controlla lo sha256 dichiarato da PyPI, lo
estrae in DEST (che diventa la radice dei sorgenti, senza la cartella `nome-versione/` di mezzo) e corregge il flag. Poi si
costruisce con `uv pip install --no-build-isolation DEST` (scripts/setup_env.sh). Solo libreria standard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


def pypi_sdist(name: str, version: str) -> tuple[str, str, str]:
    """(url, nome del file, sha256) dello sdist di name==version secondo PyPI."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as resp:
        meta = json.load(resp)
    sdist = next((u for u in meta["urls"] if u["packagetype"] == "sdist"), None)
    if sdist is None:
        raise RuntimeError(f"{name}=={version}: PyPI non ha uno sdist")
    return sdist["url"], sdist["filename"], sdist["digests"]["sha256"]


def extract_sdist(archive: Path, dest: Path) -> None:
    """Estrae l'archivio in dest togliendo la cartella di primo livello. Rifiuta i percorsi che escono da dest."""
    dest = Path(dest)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if len({m.name.split("/", 1)[0] for m in members}) != 1:
            raise RuntimeError(f"{archive}: ci si aspetta una sola cartella di primo livello")
        todo = []
        for m in members:
            parts = PurePosixPath(m.name).parts[1:]
            if not parts:
                continue
            if ".." in parts or PurePosixPath(m.name).is_absolute():
                raise RuntimeError(f"{archive}: il percorso {m.name!r} esce dalla cartella di destinazione")
            m.name = "/".join(parts)
            todo.append(m)
        dest.mkdir(parents=True, exist_ok=True)
        for m in todo:
            try:
                tf.extract(m, dest, filter="data")
            except TypeError:  # Python senza i filtri di tarfile: i percorsi sono già stati controllati sopra
                tf.extract(m, dest)


def fetch_sdist(name: str, version: str, dest: Path, *, lookup=pypi_sdist, download=urllib.request.urlretrieve) -> Path:
    """Scarica e verifica lo sdist, poi lo estrae in dest. Quello che c'era prima in dest sparisce (niente oggetti di un altro torch)."""
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    url, filename, sha256 = lookup(name, version)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / filename
        download(url, archive)
        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        if got != sha256:
            raise RuntimeError(f"{filename}: sha256 {got} diverso da quello di PyPI ({sha256})")
        extract_sdist(archive, dest)
    return dest


def patch_cxx20(src: Path) -> int:
    """Sostituisce -std=c++17 con -std=c++20 in setup.py; ritorna le occorrenze sostituite (0 se già corretto)."""
    setup = Path(src) / "setup.py"
    text = setup.read_text(encoding="utf-8")
    n = text.count("-std=c++17")
    if n == 0 and "-std=c++20" not in text:
        raise RuntimeError(f"{setup}: non trovo -std=c++17 da correggere, la struttura del pacchetto è cambiata")
    setup.write_text(text.replace("-std=c++17", "-std=c++20"), encoding="utf-8")
    return n


def fetch_and_patch(name: str, version: str, dest: Path, **kwargs) -> tuple[Path, int]:
    root = fetch_sdist(name, version, dest, **kwargs)
    return root, patch_cxx20(root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("version")
    ap.add_argument("dest", type=Path)
    args = ap.parse_args(argv)
    root, n = fetch_and_patch(args.name, args.version, args.dest)
    print(f"{args.name}=={args.version}: -std=c++17 -> -std=c++20 in {n} punti di {root}/setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
