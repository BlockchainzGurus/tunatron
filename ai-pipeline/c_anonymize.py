"""Anonimizzazione del C delle slice Juliet: l'etichetta non deve arrivare al modello per nomi, commenti o stringhe.

In Juliet l'etichetta è scritta nel codice (`…_badSink`, `…_goodG2BSink`, file `CWE78_…`, commenti `/* FLAW */`,
stringhe come "Benign, fixed string"). Qui si toglie ogni traccia da un record di `sast-engine slice`, così il modello
deve seguire il dato attraverso i chiamanti. Un tokenizzatore a regex separa commenti, stringhe, caratteri, numeri e
identificatori: nulla viene riscritto dentro stringhe, caratteri o commenti. Solo libreria standard.
"""
from __future__ import annotations

import copy
import hashlib
import re

# Controllo finale a sottostringa (il cancello di `leak_audit` usa la stessa regex).
LEAK_RE = re.compile(r"(?i)bad|good|g2b|b2g|flaw|fix|benign|cwe\d")

_LEAK_PARTS = frozenset({"bad", "good", "g2b", "b2g", "flaw", "fix", "benign", "cwe"})
# Parti di un identificatore: `g2b`/`b2g` come parte intera, poi parole camelCase, sigle in maiuscolo e cifre;
# `_` e ogni altro carattere fanno da separatore.
_PART_RE = re.compile(r"(?i:g2b|b2g)|[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

# Un solo tokenizzatore per commenti e riscrittura. Stringhe e caratteri si fermano a fine riga se non chiusi
# (blocchi troncati, apostrofi nelle direttive); `\\.` con DOTALL copre anche la continuazione `\`+a capo.
_TOKEN_RE = re.compile(
    r"""
      (?P<comment>/\*.*?(?:\*/|\Z)|//(?:\\\n|[^\n])*)
    | (?P<string>"(?:\\.|[^"\\\n])*(?:"|$))
    | (?P<char>'(?:\\.|[^'\\\n])*(?:'|$))
    | (?P<number>\.?\d(?:[eEpP][+-]|[\w.])*)
    | (?P<ident>[^\W\d]\w*)
    """,
    re.S | re.M | re.X,
)

_NEUTRAL_LITERAL = {"string": '"text"', "char": "'t'"}


def strip_comments(code: str) -> tuple[str, list[int]]:
    """Toglie `/* */` e `//` (non dentro stringhe o caratteri) e restituisce (codice, mappa delle righe).

    Un commento in mezzo alla riga diventa uno spazio; uno su più righe lascia i suoi a capo, così il codice dopo `*/`
    resta sulla riga originale. Le righe toccate da un commento e rimaste vuote spariscono, le righe vuote consecutive
    si riducono a una e quelle in testa e in coda si tolgono; gli spazi a fine riga si tolgono. `\\r\\n` diventa `\\n`.
    La mappa dà, per ogni riga nuova (0-based), la riga originale (0-based).
    """
    code = code.replace("\r\n", "\n")
    touched: set[int] = set()

    def drop(m: re.Match) -> str:
        if m.lastgroup != "comment":
            return m.group(0)
        first = code.count("\n", 0, m.start())
        newlines = m.group(0).count("\n")
        touched.update(range(first, first + newlines + 1))
        return "\n" * newlines if newlines else " "

    lines: list[str] = []
    line_map: list[int] = []
    for i, line in enumerate(_TOKEN_RE.sub(drop, code).split("\n")):
        line = line.rstrip()
        if not line and (i in touched or not lines or not lines[-1]):
            continue
        lines.append(line)
        line_map.append(i)
    while lines and not lines[-1]:
        lines.pop()
        line_map.pop()
    return "\n".join(lines), line_map


def leaky_identifier(name: str, extra_parts: frozenset[str] = frozenset()) -> bool:
    """Vero se una *parte* del nome (spezzato su `_`, cifre e camelCase) è una parola-spia.

    `badSink`, `dataGoodBuffer`, `goodG2BSource` sì; `badge_count`, `fixture`, `prefix` no. `extra_parts` aggiunge
    parole (minuscole), per esempio quelle delle tracce Python.
    """
    parts = _LEAK_PARTS | extra_parts
    return any(p.lower() in parts for p in _PART_RE.findall(name))


def _digest(seed: str, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def _assign(names, seed: str, prefix: str, width: int, suffix: str, taken: set[str]) -> dict[str, str]:
    """Nome neutro `prefix<width esadecimali di sha256(seed:nome)>suffix` per ciascun nome.

    I nomi si visitano in ordine, così il risultato dipende solo dall'insieme; se un nome nuovo è già preso (collisione
    dei primi esadecimali o identificatore già presente) o contiene esso stesso una parola-spia (`bad` è fatto di cifre
    esadecimali: `src/d99bbad2.c`) si rifà l'hash con `seed:nome:k`, k = 1, 2, …
    """
    out: dict[str, str] = {}
    for name in sorted(names):
        key, k = name, 0
        while (new := f"{prefix}{_digest(seed, key)[:width]}{suffix}") in taken or LEAK_RE.search(new):
            k += 1
            key = f"{name}:{k}"
        taken.add(new)
        out[name] = new
    return out


def anonymize_slice(rec: dict, seed: str, macros: dict[str, str]) -> dict:
    """Copia anonimizzata di un record di `sast-engine slice` (`rec` non viene modificato).

    Commenti tolti; macro riscritte col nome vero (`SYSTEM` → `system`, anche nel callee del sink); funzioni definite
    nei blocchi → `fn_<6 hex>`; altri identificatori con una parola-spia (per parti o, in C, anche per sottostringa di
    `LEAK_RE`, per esempio `prefix`) → `v_<6 hex>`; stringhe e caratteri che soddisfano `LEAK_RE` → `"text"`/`'t'`;
    file → `src/<8 hex>.c`. Le righe sono quelle del codice ripulito: `end_line = start_line + righe − 1` e `sink.line`
    segue la mappa di `strip_comments` nel blocco a profondità 0. `ValueError` se manca quel blocco o se la riga del
    sink non è più nel codice ripulito.
    """
    out = copy.deepcopy(rec)
    blocks = out["blocks"]
    sink = out["sink"]
    at_sink = next((i for i, b in enumerate(blocks) if b["depth"] == 0), None)
    if at_sink is None:
        raise ValueError("record senza blocco a profondità 0")
    sink_block = blocks[at_sink]

    line_maps = []
    for b in blocks:
        b["code"], line_map = strip_comments(b["code"])
        line_maps.append(line_map)

    # Identificatori del record dopo le macro (codice e callee del sink): da qui i nomi da cambiare e quelli già presi.
    texts = [b["code"] for b in blocks] + [sink["function"]]
    idents = {macros.get(m.group(0), m.group(0)) for t in texts for m in _TOKEN_RE.finditer(t) if m.lastgroup == "ident"}
    defined = {b["name"] for b in blocks}
    leaky = {n for n in idents - defined if leaky_identifier(n) or LEAK_RE.search(n)}
    taken = idents - defined - leaky
    names = _assign(defined, seed, "fn_", 6, "", taken)
    names |= _assign(leaky, seed, "v_", 6, "", taken)
    files = _assign({b["file"] for b in blocks}, seed, "src/", 8, ".c", set())

    def rewrite(m: re.Match) -> str:
        tok = m.group(0)
        if m.lastgroup == "ident":
            tok = macros.get(tok, tok)
            return names.get(tok, tok)
        if m.lastgroup in _NEUTRAL_LITERAL and LEAK_RE.search(tok):
            return _NEUTRAL_LITERAL[m.lastgroup]
        return tok

    for b in blocks:
        b["code"] = _TOKEN_RE.sub(rewrite, b["code"])
        b["name"] = names[b["name"]]
        b["file"] = files[b["file"]]
        b["end_line"] = b["start_line"] + len(b["code"].split("\n")) - 1

    old = sink["line"] - sink_block["start_line"]
    line_map = line_maps[at_sink]
    if old not in line_map:
        raise ValueError(f"la riga del sink ({sink['line']}) non è nel codice ripulito del blocco a profondità 0")
    sink["line"] = sink_block["start_line"] + line_map.index(old)
    sink["file"] = sink_block["file"]
    sink["function"] = _TOKEN_RE.sub(rewrite, sink["function"])
    return out
