"""Sorgenti, sink ed etichette Juliet dall'archivio NIST (SARD, test suite 112, dominio pubblico / CC0).

Il modulo scarica e verifica lo zip, estrae solo i `.c` delle CWE usate, elenca i sink (le chiamate del catalogo dentro le
funzioni `bad*`/`good*`, con le macro in maiuscolo di Juliet riportate al nome vero) come `--targets` di
`sast-engine slice`, e ricava etichetta e metadati dai nomi ORIGINALI di funzioni e file, cioè prima dell'anonimizzazione
(`c_anonymize`). Solo libreria standard.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import common
import localize

JULIET_URL = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"
JULIET_SHA256 = "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb"
JULIET_CWES = ("CWE-120", "CWE-78", "CWE-190", "CWE-416")

# Macro di `std_testcase.h` che in Juliet fanno da sink: il catalogo di produzione conosce solo il nome vero, e le macro
# NON vanno aggiunte lì (vedi spec). Chiave = nome nel sorgente, valore = funzione vera.
MACROS: dict[str, str] = {
    "SYSTEM": "system",
    "EXECL": "execl",
    "EXECLP": "execlp",
    "EXECV": "execv",
    "EXECVP": "execvp",
    "POPEN": "popen",
    "SNPRINTF": "snprintf",
}

# Membro dell'archivio che ci interessa: un `.c` dentro la directory di una CWE (con o senza sottodirectory `sNN`).
_MEMBER_RE = re.compile(r"^C/testcases/(CWE(\d+)_[^/]+)/(.+\.c)$")
_FILE_CWE_RE = re.compile(r"^CWE(\d+)_")
# Prefisso di una funzione globale di Juliet: `CWE78_…__char_environment_system_54e_` + nome locale (`badSink`, …).
_LOCAL_NAME_RE = re.compile(r"^CWE\d+_.*_\d{2}[a-z]?_(.+)$")
_FLOW_SUFFIX_RE = re.compile(r"_(\d{2})([a-z]?)$")
# Da dove arriva il dato nel nome del file (`…_char_environment_system_54e.c`), come parole intere separate da `_`.
_SOURCE_RE = re.compile(
    r"(?:^|_)(connect_socket|listen_socket|environment|console|file|fgets|fscanf|rand|max|min|negative|large|fixed)(?:_|$)"
)


# --------------------------------------------------------------------------- archivio
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# NIST (Cloudflare) risponde 403 allo user-agent predefinito di urllib: ne serve uno che dica chi scarica.
USER_AGENT = "tunatron-data-pipeline (Juliet download for SAST fine-tuning data)"


def fetch_juliet(dest_dir: Path) -> Path:
    """Percorso dello zip di Juliet in `dest_dir`: lo scarica se manca, poi ne verifica lo sha256 (`ValueError` se diverso).

    Un file già presente col nome atteso non viene riscaricato né cancellato, anche se ha lo sha256 sbagliato. Il download
    va in un file temporaneo che prende il nome definitivo solo dopo il controllo.
    """
    dest_dir = Path(dest_dir)
    zip_path = dest_dir / JULIET_URL.rsplit("/", 1)[-1]
    if not zip_path.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        part = zip_path.with_name(zip_path.name + ".part")
        try:
            request = urllib.request.Request(JULIET_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out)
            found = _sha256(part)
            if found != JULIET_SHA256:
                raise ValueError(f"sha256 dello zip scaricato {found}, atteso {JULIET_SHA256}")
            part.replace(zip_path)
        finally:
            part.unlink(missing_ok=True)
        return zip_path
    found = _sha256(zip_path)
    if found != JULIET_SHA256:
        raise ValueError(f"sha256 di {zip_path} {found}, atteso {JULIET_SHA256}")
    return zip_path


def extract_c_sources(zip_path: Path, out_dir: Path, kb: dict, cwes: tuple[str, ...] = JULIET_CWES) -> list[Path]:
    """Estrae in `out_dir` solo i `.c` di `C/testcases/CWE<n>_*/` la cui CWE (alias inclusi) è una di `cwes`.

    I file si scrivono in UTF-8 (byte non validi sostituiti) con `\\n` al posto di `\\r\\n` e `\\r`, così righe e offset
    coincidono per il motore Rust e per i moduli Python. Un membro il cui percorso esce dalla propria directory di CWE
    dà `ValueError`, controllato su tutti i membri prima di scrivere il primo file. Restituisce una radice per directory
    di CWE Juliet (es. `out_dir/CWE121_Stack_Based_Buffer_Overflow`), che contiene le eventuali `sNN/`.
    """
    out_dir = Path(out_dir)
    with zipfile.ZipFile(zip_path) as archive:
        chosen: list[tuple[zipfile.ZipInfo, Path]] = []
        roots: set[Path] = set()
        for info in archive.infolist():
            m = _MEMBER_RE.match(info.filename)
            if not m or common.resolve_cwe(kb, f"CWE-{m.group(2)}") not in cwes:
                continue
            root = out_dir / m.group(1)
            dest = root / m.group(3)
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"il membro {info.filename!r} esce dalla cartella di destinazione")
            chosen.append((info, dest))
            roots.add(root)
        for info, dest in chosen:
            text = archive.read(info).decode("utf-8", errors="replace")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(re.sub(r"\r\n?", "\n", text).encode("utf-8"))
    return sorted(roots)


# --------------------------------------------------------------------------- sink
def _file_cwe(file_name: str, kb: dict) -> str | None:
    """CWE primaria del catalogo per il test case, dal prefisso `CWE<n>_` del nome del file."""
    m = _FILE_CWE_RE.match(file_name.rsplit("/", 1)[-1])
    return common.resolve_cwe(kb, f"CWE-{m.group(1)}") if m else None


def sink_targets(root: Path, kb: dict) -> list[dict]:
    """Chiamate-sink di tutti i `.c` sotto `root`, come `--targets` di `sast-engine slice`, ordinate per (file, riga).

    Conta ogni chiamata, dentro una funzione il cui nome locale comincia per `bad` o `good`, il cui callee (con le macro
    di `MACROS` riportate al nome vero) è un sink del catalogo per la CWE del test case. `line` è la riga assoluta nel
    file; `file` è relativo a `root`, con `/`. Non dipende dall'etichetta: quella è `sink_label` sul nome della funzione.
    """
    root = Path(root)
    targets: list[dict] = []
    for path in sorted(root.rglob("*.c")):
        cwe_id = _file_cwe(path.name, kb)
        if cwe_id is None:
            continue
        rel = path.relative_to(root).as_posix()
        source = path.read_bytes().decode("utf-8", errors="replace")
        for function in localize.split_c_functions(source):
            if sink_label(function["name"]) is None:
                continue
            for callee, line in localize.find_calls(function["code"], "c"):
                callee = MACROS.get(callee, callee)
                entry = common.match_sink(kb, "c", callee)
                if entry is not None and entry["id"] == cwe_id:
                    targets.append({"file": rel, "line": function["start_line"] + line - 1, "callee": callee, "cwe_id": cwe_id})
    return sorted(targets, key=lambda t: (t["file"], t["line"]))


# --------------------------------------------------------------------------- nomi originali
def _local_name(function_name: str) -> str:
    """Nome senza il prefisso del test case: `CWE78_…_54e_badSink` → `badSink`; `goodG2B` resta com'è."""
    m = _LOCAL_NAME_RE.match(function_name)
    return m.group(1) if m else function_name


def variant_kind(function_name: str) -> str | None:
    """`bad`, `g2b` (`goodG2B*`), `b2g` (`goodB2G*`), `good` (ogni altra `good*`); None per le altre funzioni."""
    local = _local_name(function_name)
    if local.startswith("bad"):
        return "bad"
    if local.startswith("goodG2B"):
        return "g2b"
    if local.startswith("goodB2G"):
        return "b2g"
    if local.startswith("good"):
        return "good"
    return None


def sink_label(function_name: str) -> int | None:
    """Etichetta di un sink dal nome della funzione che lo contiene: `bad*` → 1, `good*` → 0, altro → None."""
    kind = variant_kind(function_name)
    return None if kind is None else int(kind == "bad")


def testcase_group(file_name: str) -> str:
    """Nome del test case: il file senza estensione né lettera finale dopo le cifre (`…_54e.c` → `…_54`)."""
    stem = Path(file_name).stem
    m = _FLOW_SUFFIX_RE.search(stem)
    return stem[: m.end(1)] if m else stem


def flow_variant(file_name: str) -> str | None:
    """Le due cifre della variante di flusso (`…_54e.c` → `"54"`), None se il nome non ne ha."""
    m = _FLOW_SUFFIX_RE.search(Path(file_name).stem)
    return m.group(1) if m else None


def source_token(file_name: str) -> str | None:
    """Da dove arriva il dato secondo il nome (`…_char_environment_system_54e.c` → `"environment"`), None se non c'è."""
    stem = Path(file_name).stem
    m = _SOURCE_RE.search(stem.split("__", 1)[-1])
    return m.group(1) if m else None
