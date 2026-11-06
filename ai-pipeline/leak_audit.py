"""Verifica delle perdite di etichetta negli esempi con slice (cancello bloccante prima di ogni run di training).

Un'etichetta che trapela dal prompt (un nome `goodG2B`, un commento FIX, un blocco in più nelle catene sicure) fa
imparare la scorciatoia invece del ragionamento e butta via un run intero. Qui si misura:

- `forbidden_hits`: parole-spia rimaste nei blocchi di `meta.trace` (nomi, file, codice);
- `top_tokens`: i token più sbilanciati fra le due etichette (log-odds), da leggere a mano;
- `bow_accuracy`: accuratezza di un naive Bayes a sacco di token addestrato su un insieme e valutato su un altro;
- `structure_accuracy` / `structure_table`: quanto la sola forma della traccia (numero di blocchi, profondità
  massima) predice l'etichetta.

Tutte le misure sui token e sulla struttura leggono i blocchi MOSTRATI nel prompt (il solo blocco a profondità 0 per
la view `sink_only`), cioè ciò che vede il modello. La CLI esce con 1 se ci sono parole-spia o se la sola struttura
predice l'etichetta (`structure_accuracy >= STRUCTURE_LIMIT`). Solo libreria standard.

Uso: `python3 leak_audit.py --train data/graph/juliet_train.jsonl --heldout data/eval/juliet_test.jsonl`
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from c_anonymize import LEAK_RE, leaky_identifier

# Soglia del cancello sulla struttura: con etichette bilanciate il caso dà 0,5.
STRUCTURE_LIMIT = 0.65
MODES = ("substring", "parts")
SINK_MARKER = "## Candidate sink"
# Soglia oltre la quale una caratteristica a buon mercato si segnala (senza far fallire il cancello).
SHORTCUT_LIMIT = 0.65
STUMP_FEATURES = ("prompt_chars", "code_chars", "code_lines", "braces", "numbers", "n_blocks", "max_depth",
                  "sink_block_lines", "fn_tokens", "v_tokens", "text_placeholder")
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\w.]*")
_CWE_ID_RE = re.compile(r"\bCWE-\d+\b")
_FN_RE = re.compile(r"\bfn_[0-9a-f]{6}\b")
_V_RE = re.compile(r"\bv_[0-9a-f]{6}\b")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_IDENT_RE = re.compile(r"[^\W\d]\w*")
_LEAK_WORD_RE = re.compile(r"\w*(?:" + LEAK_RE.pattern.removeprefix("(?i)") + r")\w*", re.I)


# --------------------------------------------------------------------------- blocchi
def _trace(example: dict) -> dict:
    return example["meta"]["trace"]


def shown_blocks(example: dict) -> list[dict]:
    """Blocchi che il prompt mostra: tutti, o il solo blocco a profondità 0 per la view `sink_only`."""
    blocks = _trace(example)["blocks"]
    if example["meta"].get("view") == "sink_only":
        return [b for b in blocks if b["depth"] == 0]
    return list(blocks)


def _tokens(example: dict) -> list[str]:
    return [t for b in shown_blocks(example) for t in _TOKEN_RE.findall(b["code"])]


# --------------------------------------------------------------------------- parole-spia
def _prompt_tail(example: dict) -> str:
    """Prompt utente reso dal marcatore `## Candidate sink` in poi (codice, intestazioni e file come li vede il modello).

    La scheda della CWE sopra il marcatore è costante e resta fuori. Senza marcatore si guarda tutto il prompt.
    """
    user = next((m["content"] for m in example.get("messages", []) if m.get("role") == "user"), "")
    at = user.find(SINK_MARKER)
    return user[at:] if at >= 0 else user


def _reasoning_text(example: dict) -> str:
    """Il `reasoning` del target (tutto il messaggio dell'assistente se non è un JSON con quel campo)."""
    content = next((m["content"] for m in example.get("messages", []) if m.get("role") == "assistant"), "")
    try:
        return str(json.loads(content)["reasoning"])
    except (ValueError, KeyError, TypeError):
        return content


def forbidden_hits(examples: list[dict], mode: str = "substring", extra_parts: frozenset[str] = frozenset()) -> list[str]:
    """Parole-spia in `meta.trace`, nel prompt reso dopo `## Candidate sink` e nel `reasoning` del target.

    Nella traccia si guardano `code`, `file`, `name` di ogni blocco e `file`/`function` del sink. `"substring"` cerca
    `LEAK_RE` ovunque (il C anonimizzato di Juliet non deve contenerne nemmeno dentro una parola); `"parts"` controlla
    gli identificatori con `leaky_identifier(…, extra_parts)`, per il codice Python vero, dove `prefix` o `fixture` non
    sono perdite. Ogni occorrenza è `"esempio <i> <dove>: <parola>"` (`blocco <j> <campo>`, `sink <campo>`, `prompt`,
    `reasoning`).
    """
    if mode not in MODES:
        raise ValueError(f"modo sconosciuto: {mode!r} (attesi {MODES})")
    hits: list[str] = []

    def scan(where: str, text: str) -> None:
        if mode == "substring":
            found = _LEAK_WORD_RE.findall(text)
        else:
            found = [w for w in _IDENT_RE.findall(text) if leaky_identifier(w, extra_parts)]
        hits.extend(f"{where}: {w}" for w in dict.fromkeys(found))

    for i, ex in enumerate(examples):
        trace = _trace(ex)
        for j, b in enumerate(trace["blocks"]):
            for field in ("code", "file", "name"):
                scan(f"esempio {i} blocco {j} {field}", str(b[field]))
        for field in ("file", "function"):
            scan(f"esempio {i} sink {field}", str(trace["sink"][field]))
        # L'id del catalogo (`CWE-78` nel compito) non è una perdita: in modo `parts` sarebbe la parte `cwe`.
        scan(f"esempio {i} prompt", _CWE_ID_RE.sub(" ", _prompt_tail(ex)))
        scan(f"esempio {i} reasoning", _CWE_ID_RE.sub(" ", _reasoning_text(ex)))
    return hits


# --------------------------------------------------------------------------- token
def top_tokens(examples: list[dict], k: int = 30) -> list[tuple[str, float, int, int]]:
    """I `k` token con log-odds più alto in modulo fra etichetta 1 e 0: (token, log-odds, n_vulnerabili, n_sicuri).

    `n_*` conta gli esempi che contengono il token; log-odds = log((n1+1)/(N1−n1+1)) − log((n0+1)/(N0−n0+1)), con N1 e
    N0 gli esempi per etichetta. Positivo = sbilanciato verso i vulnerabili. Ordine: modulo decrescente, poi token.
    """
    totals = Counter()
    seen: dict[int, Counter] = {0: Counter(), 1: Counter()}
    for ex in examples:
        label = ex["meta"]["label"]
        totals[label] += 1
        seen[label].update(set(_tokens(ex)))
    rows = []
    for token in set(seen[0]) | set(seen[1]):
        n1, n0 = seen[1][token], seen[0][token]
        lo = math.log((n1 + 1) / (totals[1] - n1 + 1)) - math.log((n0 + 1) / (totals[0] - n0 + 1))
        rows.append((token, round(lo, 4), n1, n0))
    rows.sort(key=lambda r: (-abs(r[1]), r[0]))
    return rows[:k]


def bow_accuracy(train: list[dict], heldout: list[dict]) -> float:
    """Accuratezza su `heldout` di un naive Bayes multinomiale (smoothing +1) sui token dei blocchi mostrati di `train`.

    I token mai visti in `train` si ignorano; a parità di punteggio vince l'etichetta più frequente in `train` (0 se
    pari). 0.0 se `heldout` è vuoto.
    """
    if not heldout:
        return 0.0
    counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    docs = Counter()
    for ex in train:
        label = ex["meta"]["label"]
        docs[label] += 1
        counts[label].update(_tokens(ex))
    vocab = set(counts[0]) | set(counts[1])
    totals = {c: sum(counts[c].values()) for c in (0, 1)}
    n_docs = sum(docs.values())
    prior_winner = 1 if docs[1] > docs[0] else 0

    def score(label: int, tokens: list[str]) -> float:
        s = math.log((docs[label] + 1) / (n_docs + 2))
        denom = totals[label] + len(vocab)
        for t in tokens:
            if t in vocab:
                s += math.log((counts[label][t] + 1) / denom)
        return s

    right = 0
    for ex in heldout:
        tokens = _tokens(ex)
        s0, s1 = score(0, tokens), score(1, tokens)
        guess = prior_winner if s0 == s1 else int(s1 > s0)
        right += guess == ex["meta"]["label"]
    return right / len(heldout)


# --------------------------------------------------------------------------- struttura
def _shape(example: dict) -> tuple[int, int]:
    blocks = shown_blocks(example)
    return len(blocks), max((b["depth"] for b in blocks), default=0)


def structure_accuracy(train: list[dict], heldout: list[dict]) -> float:
    """Accuratezza su `heldout` di un classificatore che vede solo (numero di blocchi mostrati, profondità massima).

    Impara su `train` l'etichetta più frequente per ogni coppia; per una coppia mai vista, o a parità, usa l'etichetta
    più frequente di `train` (0 se pari). 0.0 se `heldout` è vuoto.
    """
    if not heldout:
        return 0.0
    by_shape: dict[tuple[int, int], Counter] = defaultdict(Counter)
    overall = Counter()
    for ex in train:
        by_shape[_shape(ex)][ex["meta"]["label"]] += 1
        overall[ex["meta"]["label"]] += 1
    default = 1 if overall[1] > overall[0] else 0

    def guess(shape: tuple[int, int]) -> int:
        c = by_shape.get(shape)
        if not c or c[0] == c[1]:
            return default
        return int(c[1] > c[0])

    return sum(guess(_shape(ex)) == ex["meta"]["label"] for ex in heldout) / len(heldout)


def structure_table(examples: list[dict]) -> dict[str, dict[str, int]]:
    """`{numero di blocchi mostrati: {"0": sicuri, "1": vulnerabili}}`, chiavi stringa (JSON) in ordine numerico."""
    table: dict[int, Counter] = defaultdict(Counter)
    for ex in examples:
        table[_shape(ex)[0]][ex["meta"]["label"]] += 1
    return {str(n): {"0": table[n][0], "1": table[n][1]} for n in sorted(table)}


# --------------------------------------------------------------------------- scorciatoie a buon mercato
def cheap_features(example: dict) -> dict[str, int]:
    """Caratteristiche che non richiedono di capire il codice, sui blocchi mostrati (e sul prompt reso).

    `prompt_chars` è la lunghezza del messaggio utente; `code_chars`, `code_lines`, `braces` (`{`), `numbers`
    (letterali numerici), `fn_tokens`/`v_tokens` (nomi anonimi) e `text_placeholder` (presenza di `"text"`) contano sul
    codice mostrato; `n_blocks`/`max_depth` sono la forma; `sink_block_lines` le righe del blocco a profondità 0.
    """
    blocks = shown_blocks(example)
    code = "\n".join(b["code"] for b in blocks)
    user = next((m["content"] for m in example.get("messages", []) if m.get("role") == "user"), "")
    sink = next((b for b in blocks if b["depth"] == 0), blocks[0] if blocks else {"code": ""})
    return {
        "prompt_chars": len(user),
        "code_chars": sum(len(b["code"]) for b in blocks),
        "code_lines": sum(b["code"].count("\n") + 1 for b in blocks),
        "braces": code.count("{"),
        "numbers": len(_NUMBER_RE.findall(code)),
        "n_blocks": len(blocks),
        "max_depth": max((b["depth"] for b in blocks), default=0),
        "sink_block_lines": sink["code"].count("\n") + 1,
        "fn_tokens": len(_FN_RE.findall(code)),
        "v_tokens": len(_V_RE.findall(code)),
        "text_placeholder": int('"text"' in code),
    }


def _fit_stump(values: list[int], labels: list[int]) -> tuple[float, int]:
    """Soglia e verso migliori sul train: si predice 1 se `x > soglia` (verso +1) o se `x <= soglia` (verso −1).

    A parità vince la soglia più bassa e poi il verso +1: la scelta è deterministica.
    """
    pos_total = sum(labels)
    n = len(labels)
    order = sorted(set(values))
    by_value = Counter()
    pos_by_value = Counter()
    for x, y in zip(values, labels):
        by_value[x] += 1
        pos_by_value[x] += y
    best = (-1.0, float("-inf"), 1)
    seen = seen_pos = 0  # esempi (e positivi) con x <= soglia
    for t in [float("-inf"), *order]:
        if t != float("-inf"):
            seen += by_value[t]
            seen_pos += pos_by_value[t]
        right_plus = (pos_total - seen_pos) + (seen - seen_pos)  # positivi sopra + negativi sotto
        for acc, direction in ((right_plus / n, 1), ((n - right_plus) / n, -1)):
            if acc > best[0]:
                best = (acc, t, direction)
    return best[1], best[2]


def stump_accuracies(train: list[dict], heldout: list[dict]) -> dict[str, float]:
    """Per ogni caratteristica di `STUMP_FEATURES`: accuratezza su `heldout` della soglia migliore imparata su `train`.

    Misura quanto una scorciatoia (per esempio «più lungo ⇒ vulnerabile») predice l'etichetta. 0.0 se uno dei due
    insiemi è vuoto.
    """
    if not train or not heldout:
        return {name: 0.0 for name in STUMP_FEATURES}
    tr = [cheap_features(ex) for ex in train]
    he = [cheap_features(ex) for ex in heldout]
    y_tr = [ex["meta"]["label"] for ex in train]
    y_he = [ex["meta"]["label"] for ex in heldout]
    out = {}
    for name in STUMP_FEATURES:
        t, direction = _fit_stump([f[name] for f in tr], y_tr)
        guesses = [int((f[name] > t) == (direction == 1)) for f in he]
        out[name] = round(sum(g == y for g, y in zip(guesses, y_he)) / len(he), 4)
    return out


# --------------------------------------------------------------------------- CLI
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True, nargs="+", help="uno o più file tenuti fuori")
    ap.add_argument("--mode", choices=MODES, default="substring", help="parts per il codice Python vero")
    ap.add_argument("-k", type=int, default=30, help="quanti token stampare")
    args = ap.parse_args(argv)

    train = read_jsonl(args.train)
    heldouts = {path: read_jsonl(path) for path in args.heldout}
    hits = forbidden_hits(train + [ex for exs in heldouts.values() for ex in exs], mode=args.mode)
    print(f"esempi: train {len(train)}, " + ", ".join(f"{p} {len(exs)}" for p, exs in heldouts.items()))
    print(f"forbidden_hits: {len(hits)}")
    for h in hits[:50]:
        print(f"  {h}")
    print(f"top_tokens (train, k={args.k}): token, log-odds (>0 = vulnerabili), n_vulnerabili, n_sicuri")
    for token, lo, n1, n0 in top_tokens(train, args.k):
        print(f"  {token!r:24} {lo:+8.3f} {n1:6d} {n0:6d}")
    print(f"structure_table (train): {json.dumps(structure_table(train))}")

    failed = False
    for path, heldout in heldouts.items():
        bow = bow_accuracy(train, heldout)
        structure = structure_accuracy(train, heldout)
        print(f"== heldout {path}")
        print(f"bow_accuracy: {bow:.4f}")
        print(f"structure_accuracy: {structure:.4f} (soglia {STRUCTURE_LIMIT})")
        print(f"structure_table (heldout): {json.dumps(structure_table(heldout))}")
        stumps = stump_accuracies(train, heldout)
        print("stump_accuracies: " + ", ".join(f"{name} {acc:.4f}" for name, acc in stumps.items()))
        for name, acc in stumps.items():
            if acc >= SHORTCUT_LIMIT:
                print(f"SHORTCUT WARNING: {name} {acc:.4f} ({path})")
        if structure >= STRUCTURE_LIMIT:
            print(f"PERDITA: la sola struttura (blocchi, profondità) predice l'etichetta su {path}: "
                  f"structure_accuracy {structure:.4f} >= {STRUCTURE_LIMIT}", file=sys.stderr)
            failed = True
    if hits:
        print(f"PERDITA: {len(hits)} parole-spia (vedi sopra)", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
