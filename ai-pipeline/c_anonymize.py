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
# Classe di memorizzazione in testa alla definizione: in Juliet `bad()` è globale e `goodG2B()` è `static`, quindi la
# sola parola chiave direbbe l'etichetta (misurato nel Task 6: 646 sicuri contro 21 vulnerabili nel train).
_STATIC_DEF_RE = re.compile(r"\Astatic[ \t]+")


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


def _digest(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _assign(items: list[tuple[str, str]], seed: str, prefix: str, width: int, suffix: str,
            taken: set[str]) -> dict[str, str]:
    """Nome neutro `prefix<width esadecimali di sha256(seed:chiave)>suffix` per ogni coppia (nome, chiave).

    Le coppie si visitano nell'ordine dato (che il chiamante sceglie senza etichette); un nome già visto tiene il primo
    nome nuovo. Se il nome nuovo è già preso (collisione dei primi esadecimali o identificatore già presente) o contiene
    esso stesso una parola-spia (`bad` è fatto di cifre esadecimali: `src/d99bbad2.c`) si rifà l'hash con
    `seed:chiave:j`, j = 1, 2, …
    """
    out: dict[str, str] = {}
    for name, key in items:
        if name in out:
            continue
        attempt, j = key, 0
        while (new := f"{prefix}{_digest(seed, attempt)[:width]}{suffix}") in taken or LEAK_RE.search(new):
            j += 1
            attempt = f"{key}:{j}"
        taken.add(new)
        out[name] = new
    return out


def anonymize_slice(rec: dict, seed: str, macros: dict[str, str]) -> dict:
    """Copia anonimizzata di un record di `sast-engine slice` (`rec` non viene modificato).

    La ridenominazione è canonica (alpha-renaming), così due sink gemelli (`…_54e_badSink` e `…_54e_goodG2BSink`, stesso
    corpo, righe diverse dello stesso file) danno lo stesso record:

    - commenti tolti; macro riscritte col nome vero (`SYSTEM` → `system`, anche nel callee del sink);
    - la `static` in testa alla definizione di ogni blocco tolta (le `static` nel corpo restano): in Juliet `bad()` è
      globale e `goodG2B()` statica;
    - funzione definita nel blocco di profondità `d` e rango `k` fra i blocchi di quella profondità (nell'ordine di
      `rec["blocks"]`) → `fn_<6 hex di sha256(seed:fn:d:k)>`, anche nelle chiamate;
    - l'n-esimo identificatore distinto con una parola-spia (per parti o, in C, anche per sottostringa di `LEAK_RE`, per
      esempio `prefix`), in ordine di prima comparsa nei blocchi e poi nel callee del sink → `v_<6 hex di sha256(seed:v:n)>`;
    - stringhe e caratteri che soddisfano `LEAK_RE` → `"text"`/`'t'`; file → `src/<8 hex di sha256(seed:percorso)>.c`;
    - righe sintetiche: `start_line = 1 + int(sha256(seed:line:d:k)[:4], 16) % 900`, `end_line = start_line + righe − 1`
      e `sink.line` = `start_line` del blocco a profondità 0 + riga del sink nel codice ripulito (mappa di
      `strip_comments`); `sink.file` = file del blocco a profondità 0.

    Il risultato dipende solo dal testo senza commenti, da `seed` e da `macros`. I nomi ripetuti fra test case diversi
    (`v_…`, file) dipendono comunque dal seme: chi costruisce i dati passa un seme per test case (`f"{seed}:{gruppo}"`).
    `ValueError` se manca il blocco a profondità 0 o se la riga del sink non è più nel codice ripulito.
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
        code, line_map = strip_comments(b["code"])
        b["code"] = _STATIC_DEF_RE.sub("", code, count=1)
        line_maps.append(line_map)
    old = sink["line"] - sink_block["start_line"]
    if old not in line_maps[at_sink]:
        raise ValueError(f"la riga del sink ({sink['line']}) non è nel codice ripulito del blocco a profondità 0")
    sink_row = line_maps[at_sink].index(old)

    # Rango (profondità, k) di ogni blocco: k conta i blocchi precedenti con la stessa profondità.
    seen_depths: dict[int, int] = {}
    ranks = []
    for b in blocks:
        k = seen_depths.get(b["depth"], 0)
        seen_depths[b["depth"]] = k + 1
        ranks.append((b["depth"], k))

    # Identificatori del record dopo le macro, in ordine di prima comparsa (codice dei blocchi, poi callee del sink).
    texts = [b["code"] for b in blocks] + [sink["function"]]
    idents = list(dict.fromkeys(
        macros.get(m.group(0), m.group(0)) for t in texts for m in _TOKEN_RE.finditer(t) if m.lastgroup == "ident"))
    defined = {b["name"] for b in blocks}
    leaky = [n for n in idents if n not in defined and (leaky_identifier(n) or LEAK_RE.search(n))]
    taken = set(idents) - defined - set(leaky)
    by_rank = sorted(range(len(blocks)), key=lambda i: ranks[i])
    names = _assign([(blocks[i]["name"], "fn:{}:{}".format(*ranks[i])) for i in by_rank], seed, "fn_", 6, "", taken)
    names |= _assign([(n, f"v:{i}") for i, n in enumerate(leaky)], seed, "v_", 6, "", taken)
    files = _assign([(b["file"], b["file"]) for b in blocks], seed, "src/", 8, ".c", set())

    def rewrite(m: re.Match) -> str:
        tok = m.group(0)
        if m.lastgroup == "ident":
            tok = macros.get(tok, tok)
            return names.get(tok, tok)
        if m.lastgroup in _NEUTRAL_LITERAL and LEAK_RE.search(tok):
            return _NEUTRAL_LITERAL[m.lastgroup]
        return tok

    for b, (depth, k) in zip(blocks, ranks):
        b["code"] = _TOKEN_RE.sub(rewrite, b["code"])
        b["name"] = names[b["name"]]
        b["file"] = files[b["file"]]
        b["start_line"] = 1 + int(_digest(seed, f"line:{depth}:{k}")[:4], 16) % 900
        b["end_line"] = b["start_line"] + len(b["code"].split("\n")) - 1

    sink["line"] = sink_block["start_line"] + sink_row
    sink["file"] = sink_block["file"]
    sink["function"] = _TOKEN_RE.sub(rewrite, sink["function"])
    return out
