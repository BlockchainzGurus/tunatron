"""Traccia P1: i test CodeQL Python con le backward slice del grafo e l'etichetta dei `$ Alert`.

Dall'albero dei test di sicurezza di CodeQL (MIT, `python/ql/test/query-tests/Security`) agli esempi con chiamanti:

1. `prepare_tree` copia i `.py` togliendo le annotazioni-soluzione (`# $ Alert`, `# GOOD`…) senza spostare le righe, e
   ricorda per ogni file le righe con `$ Alert`/`MISSING`/`SPURIOUS` e la CWE della directory della query;
2. `sast-engine slice` (una radice per directory di query, così funzioni omonime di test diversi non si collegano)
   dà una slice per ogni sink del catalogo;
3. `label_slices` unisce la riga del sink con le annotazioni: `$ Alert` = vulnerabile, sink della stessa CWE senza
   annotazione = sicuro, riga ambigua o CWE diversa = scartato;
4. `scrub_record` e `rename_leaky_names` tolgono dal codice ciò che direbbe l'etichetta (commenti, stringhe e nomi come
   `bad_cmd`, `unsafe_filter`, `good1`), così il modello deve guardare il flusso e i chiamanti.

Solo CWE a flusso di dati (`TAINT_CWES`): per 327/377/611 il motivo generico del reasoning sarebbe falso (R23). Solo
libreria standard.
"""
from __future__ import annotations

import ast
import copy
import io
import re
import textwrap
import tokenize
from collections import Counter
from pathlib import Path

import c_anonymize
import common
import localize

SECURITY_ROOT = "python/ql/test/query-tests/Security"
# Parole-spia in più per il Python dei test, oltre a quelle di `c_anonymize.leaky_identifier` (`bad`, `good`, `fix`…).
PY_EXTRA_PARTS = frozenset({"safe", "unsafe", "vuln", "vulnerable", "secure", "insecure"})
# Identificatori di librerie la cui parola-spia È la semantica dell'API, non un nome scelto dall'autore del test: nei test di
# CWE-502 `yaml.load(x, Loader=yaml.SafeLoader)` è il caso sicuro. Come i nomi dei sink del catalogo sono esentati dal
# controllo finale (R26); ogni altro nome esterno con una parola-spia fa scartare l'esempio e finisce nel rapporto.
API_LEAKY_NAMES = frozenset({"SafeLoader", "CSafeLoader", "UnsafeLoader", "safe_load", "safe_load_all", "unsafe_load_all"})
_WORD_RE = re.compile(r"[^\W\d]\w*")
_NEUTRAL_STRING = '"text"'
# CWE del catalogo il cui verdetto dipende dal flusso di dati verso il sink (R23): «dato non fidato arriva senza
# validazione» vale per il vulnerabile, «costante o controllato» per il sicuro. Fuori: 327 (algoritmo debole), 377 (file
# temporaneo), 611 (il sicuro è una configurazione del parser).
TAINT_CWES = ("CWE-22", "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-502", "CWE-601", "CWE-918")

_QUERY_DIR_RE = re.compile(r"CWE-?(\d+)")


def _has_inline_annotation(text: str) -> bool:
    """Vero se il file ha almeno un commento `# $ …`: i test con le aspettative inline."""
    for line in text.split("\n"):
        at = localize._comment_start(line, "#")
        if at >= 0 and line[at + 1:].strip().startswith("$"):
            return True
    return False


# --------------------------------------------------------------------------- albero e annotazioni
def prepare_tree(codeql_root: Path, dest: Path, kb: dict | None = None) -> dict[str, dict]:
    """Copia in `dest` i `.py` dei test di sicurezza con le annotazioni tolte e le righe invariate.

    Ritorna `{percorso relativo a codeql_root: {"alert", "ambiguous", "cwe", "annotated", "root"}}`: `alert` e `ambiguous`
    sono insiemi di righe (1-based) dalle annotazioni originali; `cwe` viene dal nome della directory della query
    (`CWE-078-…` → `CWE-78`, via `common.resolve_cwe`); `annotated` dice se il file ha aspettative inline (`# $ …`);
    `root` è la directory della query (la radice da dare al motore). Restano fuori i percorsi con `py2`, i link simbolici,
    le directory che non sono una `CWE-…` del catalogo e le CWE fuori da `TAINT_CWES`. `\\r\\n` diventa `\\n`, così le
    righe del motore coincidono con quelle delle annotazioni.
    """
    kb = common.load_kb() if kb is None else kb
    codeql_root, dest = Path(codeql_root), Path(dest)
    base = codeql_root / SECURITY_ROOT
    notes: dict[str, dict] = {}
    if not base.is_dir():
        return notes
    depth = len(SECURITY_ROOT.split("/"))
    for path in sorted(base.rglob("*.py")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(codeql_root).as_posix()
        if "py2" in rel:
            continue
        query_dir = rel.split("/")[depth]
        m = _QUERY_DIR_RE.match(query_dir)
        if not m or not query_dir.startswith("CWE-"):
            continue
        cwe = common.resolve_cwe(kb, f"CWE-{int(m.group(1))}")
        if cwe not in TAINT_CWES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        found = localize.annotated_lines(text, "#")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(localize.scrub_labels(text, "#"), encoding="utf-8", newline="\n")
        notes[rel] = {"alert": found["alert"], "ambiguous": found["ambiguous"], "cwe": cwe,
                      "annotated": _has_inline_annotation(text), "root": f"{SECURITY_ROOT}/{query_dir}"}
    return notes


def _call_span(rec: dict) -> range:
    """Righe (assolute) della chiamata-sink, dalla prima all'ultima: l'annotazione può stare su una riga qualsiasi.

    Si cerca nel blocco a profondità 0 la chiamata che comincia alla riga del sink e ha il suo nome; se non c'è (blocco
    mancante, nome diverso) si ricade sulla sola riga del sink.
    """
    sink = rec["sink"]
    line = sink["line"]
    block = next((b for b in rec["blocks"] if b["depth"] == 0), None)
    if block is not None:
        relative = line - block["start_line"] + 1
        name = sink["function"]
        for callee, first, last in localize._calls_with_span(block["code"], "python"):
            if first == relative and (callee == name or callee.endswith("." + name) or name.endswith("." + callee)):
                return range(line, line + (last - first) + 1)
    return range(line, line + 1)


def label_slices(slices: list[dict], notes: dict[str, dict], prefix: str = "",
                 stats: Counter | None = None) -> list[tuple[dict, int]]:
    """`[(slice, etichetta)]`: 1 se la chiamata-sink ha una riga `$ Alert`, 0 se è un sink della CWE della directory senza.

    `notes` è il risultato di `prepare_tree`; `prefix` è la directory della query (`notes[prefix + "/" + sink.file]`) quando
    le slice vengono da una radice che è sotto la radice dell'albero. Si scartano (e si contano in `stats`) i sink in
    file sconosciuti (`skipped_unknown_file`) o senza aspettative inline (`skipped_not_annotated`: lì «non annotato» non
    vuol dire «sicuro», R22), di una CWE diversa da quella della directory (`skipped_other_cwe`) e le chiamate con una
    riga `MISSING`/`SPURIOUS` (`skipped_ambiguous`).
    """
    stats = Counter() if stats is None else stats
    out: list[tuple[dict, int]] = []
    for rec in slices:
        sink = rec["sink"]
        note = notes.get(f"{prefix}/{sink['file']}" if prefix else sink["file"])
        if note is None:
            stats["skipped_unknown_file"] += 1
        elif not note["annotated"]:
            stats["skipped_not_annotated"] += 1
        elif sink["cwe_id"] != note["cwe"]:
            stats["skipped_other_cwe"] += 1
        else:
            span = _call_span(rec)
            if any(n in note["ambiguous"] for n in span):
                stats["skipped_ambiguous"] += 1
            else:
                out.append((rec, int(any(n in note["alert"] for n in span))))
    return out


# --------------------------------------------------------------------------- commenti, stringhe e righe
def has_leaky_word(text: str) -> bool:
    """Vero se una parola di `text` ha una parte-spia (`unsafe value`, `Good`, `bad_cmd`; non `prefix` né `fixture`)."""
    return any(c_anonymize.leaky_identifier(w, PY_EXTRA_PARTS) for w in _WORD_RE.findall(text))


def scrub_python(code: str) -> tuple[str, list[int]]:
    """Toglie i commenti e rende neutre le stringhe con una parola-spia: (codice, mappa delle righe).

    Un commento sparisce (anche quello a fine riga); una stringa, docstring comprese, con una parola-spia diventa
    `"text"` (il prefisso `b`/`r` si perde). Le f-string non si toccano: il testo fra le graffe è codice, e se contiene
    una parola-spia lo dirà il controllo finale. Le righe rimaste vuote per un commento o per una stringa ridotta a una
    riga spariscono, le righe vuote consecutive si riducono a una, quelle in testa e in coda si tolgono, e gli spazi a
    fine riga pure; dentro una stringa su più righe che resta non si tocca niente. La mappa dà, per ogni riga nuova
    (0-based), la riga originale (0-based). `ValueError` se il codice non si lascia tokenizzare.
    """
    code = code.replace("\r\n", "\n")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"codice non tokenizzabile: {exc}") from exc
    starts = [0]
    for line in code.split("\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    def offset(pos: tuple[int, int]) -> int:
        return starts[pos[0] - 1] + pos[1]

    edits: list[tuple[int, int, str]] = []
    touched: set[int] = set()
    protected: set[int] = set()
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            edits.append((offset(tok.start), offset(tok.end), ""))
            touched.add(tok.start[0] - 1)
        elif tok.type == tokenize.STRING:
            if has_leaky_word(tok.string):
                # stesso numero di a capo: le righe non si spostano finché non si ripulisce riga per riga
                edits.append((offset(tok.start), offset(tok.end), _NEUTRAL_STRING + "\n" * tok.string.count("\n")))
                touched.update(range(tok.start[0] - 1, tok.end[0]))
            else:
                protected.update(range(tok.start[0], tok.end[0] - 1))  # righe interne: l'ultima ha la chiusura e il codice dopo
    text = code
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]

    lines: list[str] = []
    line_map: list[int] = []
    for i, line in enumerate(text.split("\n")):
        if i not in protected:
            line = line.rstrip()
            if not line and (i in touched or not lines or not lines[-1]):
                continue
        lines.append(line)
        line_map.append(i)
    while lines and not lines[-1]:
        lines.pop()
        line_map.pop()
    return "\n".join(lines), line_map


def scrub_record(rec: dict, seed: str) -> dict:
    """Copia del record di slice con commenti e stringhe ripulite (`scrub_python`) e righe sintetiche.

    Come in Juliet (R9) le posizioni originali non arrivano al modello: nei test CodeQL i casi vulnerabili stanno spesso
    prima di quelli sicuri, e una riga bassa diventerebbe una scorciatoia. `start_line` di ogni blocco è
    `1 + int(sha256(seed:line:profondità:rango)[:4], 16) % 900` (rango = blocchi precedenti con la stessa profondità),
    `end_line = start_line + righe − 1`, `sink.line` = `start_line` del blocco a profondità 0 + la riga del sink nel codice
    ripulito (mappa di `scrub_python`) e `sink.file` = file di quel blocco. `ValueError` se manca il blocco a profondità 0
    o se la riga del sink non è più nel codice ripulito. `rec` non cambia.
    """
    out = copy.deepcopy(rec)
    blocks, sink = out["blocks"], out["sink"]
    at_sink = next((i for i, b in enumerate(blocks) if b["depth"] == 0), None)
    if at_sink is None:
        raise ValueError("record senza blocco a profondità 0")
    sink_block = blocks[at_sink]
    maps = []
    for b in blocks:
        b["code"], line_map = scrub_python(b["code"])
        maps.append(line_map)
    old_row = sink["line"] - sink_block["start_line"]
    if old_row not in maps[at_sink]:
        raise ValueError(f"la riga del sink ({sink['line']}) non è nel codice ripulito del blocco a profondità 0")
    sink_row = maps[at_sink].index(old_row)
    seen: dict[int, int] = {}
    for b in blocks:
        rank = seen.get(b["depth"], 0)
        seen[b["depth"]] = rank + 1
        b["start_line"] = 1 + int(c_anonymize._digest(seed, f"line:{b['depth']}:{rank}")[:4], 16) % 900
        b["end_line"] = b["start_line"] + b["code"].count("\n")
    sink["line"] = sink_block["start_line"] + sink_row
    sink["file"] = sink_block["file"]
    return out


# --------------------------------------------------------------------------- nomi
def catalog_leaky_names(kb: dict) -> frozenset[str]:
    """Componenti dei nomi dei sink Python del catalogo che hanno una parte-spia (`mark_safe`, `unsafe_load`).

    Sono vocabolario del catalogo, uguale per i due verdetti, non un difetto dei test: il controllo finale
    (`leak_audit.forbidden_hits(…, allow=…)`) le esenta.
    """
    names: set[str] = set()
    for entry in kb["entries"]:
        for sink in entry.get("sinks", {}).get("python", []):
            names.update(p for p in sink.split(".") if p and c_anonymize.leaky_identifier(p, PY_EXTRA_PARTS))
    return frozenset(names)


def _line_offsets(code: str) -> list[int]:
    starts = [0]
    for line in code.split("\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    return starts


def _name_tokens(code: str) -> list[tuple[str, tuple[int, int], tuple[int, int], bool]]:
    """`(nome, inizio, fine, è un attributo)` di ogni NAME del codice, in ordine; `ValueError` se non si tokenizza."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"codice non tokenizzabile: {exc}") from exc
    out = []
    previous = None
    for tok in tokens:
        if tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        if tok.type == tokenize.NAME:
            out.append((tok.string, tok.start, tok.end, previous is not None and previous.string == "."))
        previous = tok
    return out


def _defined_names(code: str) -> tuple[set[str], set[str]]:
    """`(funzioni, variabili)` definite nel codice: `def`/`class`, bersagli di assegnazione, parametri, `except … as`.

    Gli `import` non contano: un nome d'API (`safe_load`) non si riscrive, e se è una spia lo scarta il controllo finale.
    `ValueError` se il codice non è Python valido (anche dopo averlo spostato a sinistra: i metodi sono indentati).
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"blocco non analizzabile: {exc}") from exc
    functions: set[str] = set()
    variables: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            functions.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            variables.add(node.id)
        elif isinstance(node, ast.arg):
            variables.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            variables.add(node.name)
    return functions, variables


def rename_leaky_names(rec: dict, seed: str) -> dict:
    """Copia del record con i nomi d'autore DEFINITI nella slice sostituiti e i file nascosti (`rec` non cambia).

    Ogni funzione che è un blocco della slice (`ok8`, `good_caller`, `command_injection1`) e ogni altra funzione definita
    con una parte-spia (`unsafe_filter`) diventano `fn_<6 esadecimali di sha256(seme:fn:nome)>`, le variabili e i parametri
    con una parte-spia `v_<6 esadecimali>`, in ogni blocco, in modo coerente fra definizione e chiamate (anche gli argomenti per nome e le espressioni delle f-string, da Python 3.12). Non
    si toccano gli attributi (`obj.unsafe`), i nomi che la slice non definisce (importati, globali di modulo), le
    stringhe e i commenti, né gli identificatori che contengono una parola-spia solo come sottostringa (`prefix`,
    `fixture`). I file diventano `src/<8 esadecimali>.py` (uno per file originale) e `sink.file` segue il blocco a profondità
    0. Le righe non cambiano. `ValueError` se un blocco non si tokenizza o non è Python valido.
    """
    out = copy.deepcopy(rec)
    blocks, sink = out["blocks"], out["sink"]
    functions: set[str] = {b["name"] for b in blocks}
    variables: set[str] = set()
    appearing: dict[str, None] = {}
    for b in blocks:
        defined_functions, defined_variables = _defined_names(b["code"])
        functions |= defined_functions
        variables |= defined_variables
        for name, _start, _end, is_attribute in _name_tokens(b["code"]):
            if not is_attribute:
                appearing.setdefault(name)
    # Ogni funzione che è un blocco della slice si rinomina, anche se il nome non contiene parole-spia (R25): `ok8`, `not_ok7`,
    # `full_ssrf` sono metadati d'autore che dicono il verdetto e nessuna lista di parole li copre. Le altre funzioni e le
    # variabili solo se hanno una parte-spia.
    block_names = list(dict.fromkeys(b["name"] for b in blocks))
    leaky = [n for n in appearing if n not in block_names and (n in functions or n in variables)
             and c_anonymize.leaky_identifier(n, PY_EXTRA_PARTS)]
    taken = set(appearing) - set(leaky) - set(block_names)
    names = c_anonymize._assign([(n, f"fn:{n}") for n in [*block_names, *(n for n in leaky if n in functions)]],
                                seed, "fn_", 6, "", taken)
    names |= c_anonymize._assign([(n, f"v:{n}") for n in leaky if n not in functions], seed, "v_", 6, "", taken)
    files = c_anonymize._assign([(b["file"], b["file"]) for b in blocks], seed, "src/", 8, ".py", set())

    for b in blocks:
        edits = []
        starts = _line_offsets(b["code"])
        for name, start, end, is_attribute in _name_tokens(b["code"]):
            if name in names and not is_attribute:
                edits.append((starts[start[0] - 1] + start[1], starts[end[0] - 1] + end[1], names[name]))
        code = b["code"]
        for a, z, new in sorted(edits, reverse=True):
            code = code[:a] + new + code[z:]
        b["code"] = code
        b["name"] = names.get(b["name"], b["name"])
        b["file"] = files[b["file"]]
    sink["file"] = next(b["file"] for b in blocks if b["depth"] == 0)
    return out


def anonymize_python(rec: dict, seed: str) -> dict:
    """`scrub_record` (commenti, stringhe, righe) e poi `rename_leaky_names` (nomi, file) di un record di slice Python."""
    return rename_leaky_names(scrub_record(rec, seed), seed)
