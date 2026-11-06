"""Esempi di training dalle slice del grafo delle chiamate (funzioni pure, solo libreria standard).

Un record di slice (`sast-engine slice`, già anonimizzato da `c_anonymize` o rinominato) diventa un esempio di chat
`{"messages": [system, user, assistant], "meta": {…}}`. Il prompt si ottiene SEMPRE da `messages_from_trace` sulla
forma strutturata salvata in `meta.trace`: è l'unico punto che rende una traccia, così la view «solo sink» di questo
modulo e quella dei benchmark coincidono per costruzione.

L'idea centrale (S1): due sink gemelli (`…54e_badSink` e `…54e_goodG2BSink`) hanno lo stesso corpo, quindi dopo
l'anonimizzazione i loro prompt «solo sink» sono identici con etichette opposte e decidono solo i chiamanti. Qui:

- `drop_conflicts` toglie gli esempi il cui prompt compare con entrambe le etichette (non decidibili);
- `make_candidates` / `choose_views` mescolano la slice completa (0,7) e il solo sink (0,3), quest'ultimo solo se il
  suo prompt non è in conflitto;
- `link_pairs` riconosce i gemelli (stesso prompt «solo sink», etichette opposte): sono il test causale «il modello
  usa i chiamanti?».

Ogni scelta pseudo-casuale passa da `hashlib` (mai `hash()`): l'uscita non dipende da `PYTHONHASHSEED` né dall'ordine
di esecuzione.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict

import common
import prepare_data

VARIANT_KINDS = ("bad", "g2b", "b2g", "good")
VIEWS = ("callers", "sink_only")
# Valore di `meta.track` per gli esempi Juliet (lo passa Task 6); le tracce Python passano `"py_codeql"`, `"py_synth"`,
# `"py_pyvul"`. Solo in Juliet un sink `g2b`/`good` ha lo stesso corpo del gemello `bad` (vedi `make_candidates`).
JULIET_TRACK = "juliet"
NO_FIX = prepare_data.NO_FIX

# --------------------------------------------------------------------------- vocabolario chiuso del reasoning
# Motivo del verdetto per (CWE, variante Juliet). Le frasi valgono per tutte le CWE del catalogo che Juliet porta sotto
# lo stesso id (per esempio CWE-415 sotto CWE-416, CWE-680/191/195 sotto CWE-190, CWE-121/122/124/127 sotto CWE-120)
# e per i sink che il catalogo riconosce (CWE-416: `free`; CWE-190: le allocazioni; CWE-120: le copie).
PHRASES: dict[tuple[str, str], str] = {
    ("CWE-78", "bad"): "the value enters the command without validation or neutralization, so it can inject shell commands",
    ("CWE-78", "g2b"): "the command cannot be influenced from outside, so executing it is not exploitable",
    ("CWE-78", "b2g"): "the value is validated before use, so it cannot inject shell commands",
    ("CWE-78", "good"): "on the path to the sink the command contains no externally controllable data",
    # Vale anche per CWE-170 (sotto CWE-126): la copia sta nei limiti ma lascia il buffer senza terminatore.
    ("CWE-120", "bad"): "sizes, offsets or string termination around the copy are not kept inside the buffers, so the "
                        "copy or a later read of the buffer can go out of bounds",
    ("CWE-120", "g2b"): "the sizes and positions involved are compatible with the buffers, so the copy stays in bounds",
    # In Juliet il b2g di CWE-120 è lo stile CWE-135: destinazione dimensionata con `wcslen` del dato, nessun confronto.
    ("CWE-120", "b2g"): "the size used for the copy is derived from the data's own length or checked against the buffer "
                        "capacity before use, so the copy stays within bounds",
    ("CWE-120", "good"): "on the path to the sink the copy stays within the buffer bounds",
    # CWE-194/195 hanno `if (data < 100)`: un controllo c'è, ma non esclude i negativi; CWE-680 non ne ha.
    ("CWE-190", "bad"): "the value reaches the sink without a check that rules out negative or too large values, so the "
                        "size computation can overflow or change sign",
    ("CWE-190", "g2b"): "the value is small enough to stay within the range of the type, so the computation does not overflow",
    ("CWE-190", "b2g"): "the value is compared with a limit before use, so the computation stays within the range of the type",
    ("CWE-190", "good"): "on the path to the sink the computation stays within the range of the type",
    ("CWE-416", "bad"): "on this path the memory is released and then used or released again, so memory that is no longer valid is accessed",
    ("CWE-416", "g2b"): "the memory is released only once and is not used after the release",
    ("CWE-416", "b2g"): "after the release the memory is no longer used or released again",
    ("CWE-416", "good"): "the memory is released only once and is not used after the release",
}

# Motivo per ogni CWE fuori da `PHRASES` (tracce Python), per etichetta.
GENERIC: dict[int, str] = {
    1: "untrusted data reaches the sink without validation or sanitization",
    0: "the value reaching the sink is constant or is checked before use",
}

# Origine del dato dal `source_token` del NOME DEL FILE Juliet (`juliet_graph.source_token`), come frase completa (nessuna
# preposizione condivisa incollata davanti). Vale solo per le varianti `bad` e `b2g`, che usano davvero la sorgente del
# nome.
ORIGINS: dict[str, str] = {
    "environment": "the data is read from an environment variable",
    "console": "the data is a line read from the console",
    "fgets": "the data is a line read with fgets",
    "fscanf": "the data is a number read with fscanf",
    "file": "the data is read from the content of a file",
    "connect_socket": "the data is received on a network socket opened by the program",
    "listen_socket": "the data is received on a listening network socket",
    "rand": "the data is a random value",
    "max": "the data is the largest value the type can hold",
    "min": "the data is the smallest value the type can hold",
    "negative": "the data is a negative number",
    "large": "the data is a large number",
    "fixed": "the data is a constant value hard-coded in the program",
}
ORIGIN_FALLBACK = "the data comes from an external source"

# Variante `g2b`: la sorgente è quella buona (una costante), anche se il nome del file dice `environment`.
G2B_ORIGINS: dict[str, str] = {
    "CWE-78": "the command is built only from constant strings in the code",
    # Vale anche per CWE-124/127 (conta dove punta il dato) e CWE-135 (destinazione dimensionata sulla stringa costante).
    "CWE-120": "the data copied is constant in the code and the buffers involved are sized and positioned for it",
    "CWE-190": "the value is a small constant in the code",
    "CWE-416": "the pointer refers to memory that was just allocated and not yet released",
}
G2B_FALLBACK = "the value is a constant in the code"

# Nessuna origine affidabile (`source_token` assente, o variante `good`): una frase vera per ogni variante della CWE.
NEUTRAL_ORIGINS: dict[str, str] = {
    "CWE-78": "the data becomes part of the executed command",
    "CWE-120": "the copy depends on how the amount of data to copy compares with the capacity of the buffers involved",
    "CWE-190": "the data is an integer used to compute a size or a result",
    "CWE-416": "the data is a pointer to dynamically allocated memory",
}
NEUTRAL_FALLBACK = "the data reaches the sink through the blocks shown"

# View `sink_only` di una traccia CON chiamanti (solo il blocco a profondità 0 nel prompt): il reasoning afferma solo ciò che
# il prompt mostra, quindi niente origine e niente «comes from …». Il motivo resta `PHRASES`/`GENERIC`: un esempio
# solo-sink esiste solo se il suo prompt non è gemello di uno con l'etichetta opposta.
SINK_ONLY_REASONING = "`{sink}` in `{fn}` (depth 0, only block shown): the origin of the data is not visible in this block; {reason}."

# Slice senza chiamanti (la traccia ha un solo blocco, qualunque sia la view): un esempio così è decidibile solo se la
# sorgente del dato sta nella funzione stessa, quindi si dice da dove viene (`_origin_clause`) invece di «non visibile» (R15).
NO_CALLERS_REASONING = "`{sink}` in `{fn}` (depth 0, no callers in the slice): {origin}; {reason}."

# Chiavi di `meta` calcolate qui: chi chiama passa solo le altre (`source`, `synthetic`, `track`, `group`, `variant`…).
_COMPUTED = frozenset({"label", "kind", "cwe", "lang", "sink_function", "sink_line", "sink_kind", "func_hash", "view",
                       "variant_kind", "origin", "pair_id", "trace"})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(text: str) -> float:
    """Numero in [0, 1) ricavato da sha256: il «caso» deterministico del modulo."""
    return int(_sha(text)[:16], 16) / 16**16


def _user_content(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


# --------------------------------------------------------------------------- tracce e view
def _sink_block(blocks: list[dict]) -> dict:
    """Il blocco a profondità 0 (la funzione che contiene il sink); `ValueError` se manca o non è unico."""
    found = [b for b in blocks if b["depth"] == 0]
    if len(found) != 1:
        raise ValueError(f"la slice deve avere esattamente un blocco a profondità 0, ne ha {len(found)}")
    return found[0]


def _view_blocks(blocks: list[dict], *, max_depth: int | None = None, sink_only: bool = False) -> list[dict]:
    """Blocchi mostrati nel prompt: tutti, quelli con `depth <= max_depth`, oppure il solo blocco a profondità 0."""
    sink_block = _sink_block(blocks)
    if sink_only:
        return [sink_block]
    if max_depth is None:
        return list(blocks)
    if max_depth < 0:
        raise ValueError(f"max_depth negativo: {max_depth}")
    return [b for b in blocks if b["depth"] <= max_depth]


def _entry(kb: dict, cwe_id: str) -> dict:
    for entry in kb["entries"]:
        if entry["id"] == cwe_id:
            return entry
    raise ValueError(f"{cwe_id} non è nel catalogo")


def messages_from_trace(trace: dict, kb: dict, *, max_depth: int | None = None, sink_only: bool = False) -> list[dict]:
    """System + user ricostruiti da un `meta.trace` (`{"cwe_id", "sink", "blocks"}`) con `common.build_messages`.

    È l'unica resa di una traccia: la usano `build_example`, `make_candidates`, `link_pairs` e i benchmark.
    `max_depth` tiene i blocchi con `depth <= max_depth`; `sink_only` il solo blocco a profondità 0.
    """
    entry = _entry(kb, trace["cwe_id"])
    blocks = _view_blocks(trace["blocks"], max_depth=max_depth, sink_only=sink_only)
    return common.build_messages(trace["cwe_id"], common.render_card(entry), trace["sink"], blocks)


def sink_only(rec: dict) -> dict:
    """Copia del record di slice con il solo blocco a profondità 0 (`ValueError` se manca)."""
    out = copy.deepcopy(rec)
    out["blocks"] = [_sink_block(out["blocks"])]
    return out


# --------------------------------------------------------------------------- esempio
def _origin_clause(cwe_id: str, variant_kind: str, origin: str | None) -> str:
    """Da dove arriva il dato, detto solo quando è vero.

    `origin` è il `source_token` del nome del file: per `g2b` è falso (la sorgente buona è una costante, il nome dice
    ancora `environment`) e per `good` non è garantito che quella sorgente arrivi al sink; lì e quando manca si usa una
    frase neutra della CWE. `ORIGIN_FALLBACK` resta solo per un token sconosciuto.
    """
    if variant_kind == "g2b":
        return G2B_ORIGINS.get(cwe_id, G2B_FALLBACK)
    if origin is None or variant_kind == "good":
        return NEUTRAL_ORIGINS.get(cwe_id, NEUTRAL_FALLBACK)
    return ORIGINS.get(origin, ORIGIN_FALLBACK)


def _reasoning(cwe_id: str, label: int, variant_kind: str, origin: str | None, sink_function: str, shown: list[dict],
               trace_blocks: int) -> str:
    """Sink, funzione del blocco mostrato più profondo (il primo, a parità) con la sua profondità, origine e motivo.

    Se la traccia ha un solo blocco (nessun chiamante, qualunque view) si usa `NO_CALLERS_REASONING`, con l'origine nel
    blocco; se ha chiamanti ma il prompt mostra solo il blocco a profondità 0 si usa `SINK_ONLY_REASONING`: nessuna
    affermazione su blocchi non mostrati.
    """
    sink_block = _sink_block(shown)
    reason = PHRASES.get((cwe_id, variant_kind), GENERIC[label])
    if trace_blocks == 1:
        return NO_CALLERS_REASONING.format(sink=sink_function, fn=sink_block["name"],
                                           origin=_origin_clause(cwe_id, variant_kind, origin), reason=reason)
    if len(shown) == 1:
        return SINK_ONLY_REASONING.format(sink=sink_function, fn=sink_block["name"], reason=reason)
    root = max(shown, key=lambda b: b["depth"])
    return (f"`{sink_function}` in `{sink_block['name']}` receives data that comes from `{root['name']}` "
            f"(depth {root['depth']}): {_origin_clause(cwe_id, variant_kind, origin)}; {reason}.")


def build_example(rec: dict, kb: dict, *, label: int, variant_kind: str, origin: str | None, meta: dict,
                  view: str = "callers") -> dict:
    """Esempio di chat da un record di slice.

    `variant_kind` è uno di `VARIANT_KINDS` e deve concordare con l'etichetta (`bad` ⇔ 1); fuori da Juliet si passa
    `"bad"` per 1 e `"good"` per 0. `origin` è il `source_token` del file (o None). `meta` porta le chiavi di chi chiama
    (`source`, `synthetic`, `track`, `group`, `variant`…), che non possono coincidere con quelle calcolate qui.
    `view="sink_only"` mostra nel prompt il solo blocco a profondità 0, ma `meta.trace` conserva sempre tutti i blocchi,
    così i benchmark possono ricostruire ogni profondità. `meta.func_hash` è l'impronta del prompt utente reso (non del
    codice), quindi le due view dello stesso sink hanno impronte diverse. `ValueError` per una slice troncata, una view o una variante
    sconosciuta, un'etichetta incoerente o un record senza blocco a profondità 0.
    """
    if rec.get("truncated"):
        raise ValueError("slice troncata: la catena dei chiamanti è incompleta")
    if view not in VIEWS:
        raise ValueError(f"view sconosciuta: {view!r}")
    if variant_kind not in VARIANT_KINDS:
        raise ValueError(f"variant_kind sconosciuto: {variant_kind!r}")
    if label not in (0, 1) or (variant_kind == "bad") != (label == 1):
        raise ValueError(f"etichetta {label!r} incoerente con variant_kind {variant_kind!r}")
    clash = sorted(_COMPUTED & set(meta))
    if clash:
        raise ValueError(f"chiavi di meta calcolate da build_example: {clash}")

    sink = rec["sink"]
    cwe_id = sink["cwe_id"]
    entry = _entry(kb, cwe_id)
    trace = {"cwe_id": cwe_id, "sink": copy.deepcopy(sink), "blocks": copy.deepcopy(rec["blocks"])}
    is_sink_only = view == "sink_only"
    messages = messages_from_trace(trace, kb, sink_only=is_sink_only)
    shown = _view_blocks(trace["blocks"], sink_only=is_sink_only)

    vulnerable = label == 1
    finding = {
        "vulnerable": vulnerable,
        "cwe_id": cwe_id,
        "severity": entry["default_severity"] if vulnerable else "NONE",
        "sink_function": sink["function"],
        "sink_file": sink["file"],
        "sink_line": sink["line"],
        "vex_status": "affected" if vulnerable else "not_affected",
        "reasoning": _reasoning(cwe_id, label, variant_kind, origin, sink["function"], shown, len(trace["blocks"])),
        "suggested_fix": entry["fix_hint"] if vulnerable else NO_FIX,
    }
    messages.append({"role": "assistant", "content": json.dumps(finding, ensure_ascii=False)})
    return {
        "messages": messages,
        "meta": {
            **meta,
            "label": int(vulnerable),
            "kind": "vulnerable" if vulnerable else "patched",
            "cwe": cwe_id,
            "lang": _sink_block(trace["blocks"])["lang"],
            "sink_function": sink["function"],
            "sink_line": sink["line"],
            "sink_kind": "call",
            "func_hash": prepare_data.func_hash(_user_content(messages)),
            "view": view,
            "variant_kind": variant_kind,
            "origin": origin,
            "pair_id": None,
            "trace": trace,
        },
    }


# --------------------------------------------------------------------------- conflitti e view
def prompt_key(example: dict) -> str:
    """sha256 esadecimale del messaggio utente (il system prompt è costante)."""
    return _sha(_user_content(example["messages"]))


def drop_conflicts(examples: list[dict]) -> tuple[list[dict], int]:
    """Toglie (da entrambi i lati) gli esempi il cui `prompt_key` compare con l'etichetta opposta: (rimasti, scartati)."""
    keys = [prompt_key(ex) for ex in examples]
    labels: dict[str, set[int]] = defaultdict(set)
    for ex, key in zip(examples, keys):
        labels[key].add(ex["meta"]["label"])
    kept = [ex for ex, key in zip(examples, keys) if len(labels[key]) == 1]
    return kept, len(examples) - len(kept)


def make_candidates(fulls: list[dict], kb: dict) -> list[dict]:
    """`[{"full": esempio, "sink_only": esempio | None}, …]` nell'ordine di `fulls`.

    La view «solo sink» si costruisce con `build_example(..., view="sink_only")` dalla stessa traccia, con etichetta,
    variante, origine e meta dell'esempio completo. È None quando la resa solo-sink della sua traccia compare fra i
    `fulls` con entrambe le etichette (i gemelli: lì decidono i chiamanti), e quando la traccia non ha chiamanti (la
    slice completa È già il solo sink, e un doppione con `view="sink_only"` sarebbe solo rumore). Solo per
    `meta.track == JULIET_TRACK` è None anche per `variant_kind` `g2b` e `good` con chiamanti: in Juliet il loro sink ha
    lo stesso corpo del gemello `bad`, quindi il solo sink non è decidibile anche quando il gemello manca dai dati. Le
    altre tracce (o `track` assente) seguono solo la regola dei conflitti: fuori da Juliet ogni sicuro è `good` e il suo
    solo sink può essere decidibile (per esempio un argomento costante).
    """
    keys = [_sha(_user_content(messages_from_trace(ex["meta"]["trace"], kb, sink_only=True))) for ex in fulls]
    labels: dict[str, set[int]] = defaultdict(set)
    for ex, key in zip(fulls, keys):
        labels[key].add(ex["meta"]["label"])
    out = []
    for ex, key in zip(fulls, keys):
        meta = ex["meta"]
        trace = meta["trace"]
        alone = None
        has_callers = any(b["depth"] > 0 for b in trace["blocks"])
        juliet_twin_body = meta.get("track") == JULIET_TRACK and meta["variant_kind"] in ("g2b", "good")
        if len(labels[key]) == 1 and has_callers and not juliet_twin_body:
            alone = build_example(
                {"sink": trace["sink"], "blocks": trace["blocks"], "truncated": False}, kb,
                label=meta["label"], variant_kind=meta["variant_kind"], origin=meta["origin"],
                meta={k: v for k, v in meta.items() if k not in _COMPUTED}, view="sink_only",
            )
        out.append({"full": ex, "sink_only": alone})
    return out


def choose_views(candidates: list[dict], seed: int, p_callers: float = 0.7) -> list[dict]:
    """Per ogni candidato la view completa con probabilità `p_callers`, altrimenti il solo sink se esiste.

    Le scelte solo-sink restano bilanciate per etichetta (|vulnerabili − sicuri| ≤ 1): se non lo sono, l'eccedenza
    dell'etichetta più numerosa torna alla view completa, a partire dal più piccolo sha256(seme, "rebalance", prompt
    completo). Così «si vede solo il sink» non diventa una scorciatoia per un'etichetta; la quota solo-sink
    `1 − p_callers` è quindi un MASSIMO, non il valore atteso. Tutto dipende da sha256 del prompt completo: stabile fra
    esecuzioni e indipendente dall'ordine dei candidati.
    """
    keys = [prompt_key(cand["full"]) for cand in candidates]
    picked: dict[int, list[int]] = {0: [], 1: []}
    for i, (cand, key) in enumerate(zip(candidates, keys)):
        alone = cand.get("sink_only")
        if alone is not None and _unit(f"{seed}:view:{key}") >= p_callers:
            picked[alone["meta"]["label"]].append(i)
    major, minor = (1, 0) if len(picked[1]) > len(picked[0]) else (0, 1)
    surplus = len(picked[major]) - len(picked[minor]) - 1
    if surplus > 0:
        back = sorted(picked[major], key=lambda i: (_sha(f"{seed}:rebalance:{keys[i]}"), keys[i]))[:surplus]
        picked[major] = sorted(set(picked[major]) - set(back))
    use_alone = set(picked[0]) | set(picked[1])
    return [cand["sink_only"] if i in use_alone else cand["full"] for i, cand in enumerate(candidates)]


# --------------------------------------------------------------------------- coppie
def _own_key(example: dict) -> str:
    try:
        return prompt_key(example)
    except (KeyError, StopIteration, TypeError):
        return ""


def link_pairs(examples: list[dict], kb: dict | None = None) -> list[dict]:
    """Coppie S1: esempi con lo stesso prompt «solo sink» (ricostruito da `meta.trace`) ed etichette opposte.

    La view dell'esempio non conta. Restituisce solo gli esempi accoppiati, nell'ordine d'ingresso, come copie con
    `meta.pair_id` (sha256 della chiave solo-sink e del numero della coppia: uguale per i due, deterministico); gli
    esempi in ingresso non cambiano. Se più vulnerabili e più sicuri condividono la chiave, si accoppiano nell'ordine
    del loro `prompt_key` (poi dell'indice): il primo vulnerabile col primo sicuro e così via; gli esclusi restano
    senza coppia. Restano fuori anche gli esempi senza `meta.trace` o con una traccia non ricostruibile, e quelli la
    cui slice COMPLETA compare con entrambe le etichette: se anche i chiamanti coincidono la coppia non è decidibile.
    Si accoppiano solo esempi con lo stesso `meta.group` (test case; chi non ce l'ha solo con chi non ce l'ha): due
    test case diversi non sono gemelli anche se il sink coincide. `kb` di default è il catalogo del progetto.
    """
    kb = common.load_kb() if kb is None else kb
    sink_keys: dict[int, str] = {}
    full_keys: dict[int, str] = {}
    full_labels: dict[str, set[int]] = defaultdict(set)
    for i, ex in enumerate(examples):
        meta = ex.get("meta") or {}
        trace, label = meta.get("trace"), meta.get("label")
        if not isinstance(trace, dict) or label not in (0, 1):
            continue
        try:
            sink_keys[i] = _sha(_user_content(messages_from_trace(trace, kb, sink_only=True)))
            full_keys[i] = _sha(_user_content(messages_from_trace(trace, kb)))
        except (KeyError, TypeError, ValueError):
            sink_keys.pop(i, None)
            continue
        full_labels[full_keys[i]].add(label)

    groups: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for i, key in sink_keys.items():
        if len(full_labels[full_keys[i]]) == 1:
            meta = examples[i]["meta"]
            groups[f"{key}:{json.dumps(meta.get('group'))}"][meta["label"]].append(i)

    pair_of: dict[int, str] = {}
    for key, by_label in groups.items():
        order = lambda i: (_own_key(examples[i]), i)  # noqa: E731
        for n, (bad, safe) in enumerate(zip(sorted(by_label[1], key=order), sorted(by_label[0], key=order))):
            pair_of[bad] = pair_of[safe] = _sha(f"{key}:{n}")[:16]
    return [{**examples[i], "meta": {**examples[i]["meta"], "pair_id": pair_of[i]}} for i in sorted(pair_of)]


# --------------------------------------------------------------------------- split e tetti
def split_of_juliet(group: str, variant: str, seed: int, val_pct: int, test_pct: int, held_variants: set[str]) -> str:
    """`"unseen_variants"` per una variante tenuta fuori, altrimenti `prepare_data.split_of` sul gruppo (test case)."""
    if variant in held_variants:
        return "unseen_variants"
    return prepare_data.split_of(group, seed, val_pct, test_pct)


def holdout_variants(variants: list[str], seed: int, pct: int = 12) -> set[str]:
    """`round(pct/100 · n)` varianti distinte, le prime per sha256(seme, variante): 6 su 48 con il 12%."""
    distinct = sorted(set(variants))
    k = round(pct / 100 * len(distinct))
    return set(sorted(distinct, key=lambda v: _sha(f"{seed}:holdout:{v}"))[:k])


def _round_robin(indices: list[int], examples: list[dict], n: int, seed: int) -> list[int]:
    """`n` indici presi a turno dai gruppi (cwe, variant): ogni gruppo dà uno per giro finché ne ha."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i in indices:
        meta = examples[i]["meta"]
        buckets[(str(meta.get("cwe")), str(meta.get("variant")))].append(i)
    queues = [sorted(buckets[b], key=lambda i: (_sha(f"{seed}:cap:{prompt_key(examples[i])}"), i))
              for b in sorted(buckets, key=lambda b: (_sha(f"{seed}:bucket:{b[0]}:{b[1]}"), b))]
    chosen: list[int] = []
    for r in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if len(chosen) == n:
                return chosen
            if r < len(q):
                chosen.append(q[r])
    return chosen


def cap_balanced(examples: list[dict], max_total: int, seed: int) -> list[dict]:
    """Al più `max_total` esempi, con etichette bilanciate (differenza ≤ 1) e uniformi per `(cwe, variant)`.

    Ogni etichetta ha `min(max_total // 2, vulnerabili, sicuri)` posti (uno in più all'etichetta più numerosa se resta
    spazio); dentro un'etichetta i gruppi `(cwe, variant)` danno un esempio a turno, così un gruppo piccolo entra tutto e
    i grandi si dividono il resto. Dentro un gruppo l'ordine è sha256(seme, `prompt_key`), così l'insieme scelto non
    dipende dall'ordine della lista; l'uscita conserva l'ordine d'ingresso.
    """
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, ex in enumerate(examples):
        by_label[ex["meta"]["label"]].append(i)
    k = min(max_total // 2, len(by_label[0]), len(by_label[1]))
    quota = {0: k, 1: k}
    if max_total - 2 * k >= 1:
        richer = max((1, 0), key=lambda label: len(by_label[label]) - k)
        if len(by_label[richer]) > k:
            quota[richer] += 1
    chosen = [i for label in (0, 1) for i in _round_robin(by_label[label], examples, quota[label], seed)]
    return [examples[i] for i in sorted(chosen)]
