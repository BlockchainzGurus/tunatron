"""Traccia P3: piccoli pacchetti Python sintetici a coppie di gemelli, perché decidano i chiamanti.

Ogni coppia è fatta di due pacchetti (una cartella con un solo `app.py`) con lo stesso scheletro: gli stessi nomi, l'ordine
delle funzioni, la funzione del sink e le funzioni intermedie. Cambia solo il valore che la funzione in cima alla catena
passa (o, a catena lunga zero, che la funzione stessa usa): input non fidato per il vulnerabile, per il sicuro una
costante, una costante composta in più passi o un input controllato (lista consentita, `shlex.quote`, `int`). Così la
funzione del sink è identica nei due gemelli e solo i chiamanti dicono il verdetto.

Contro le scorciatoie misurate su Juliet (R19) e in questo generatore: i negativi `validated` contengono la sorgente non
fidata come i positivi, i `decoy_source` la leggono ma non la passano al sink (basta cercare la parola «request» per
sbagliare), i `long_constant` sono lunghi quanto i positivi, e i positivi hanno operazioni-esca (`strip`, `lower`,
troncamento, valore di ripiego) e a volte il dato passa per un contenitore. Nessun nome contiene parole che dicano il verdetto. Solo libreria standard: template
composti qui, nessun modello, nessuna rete.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import c_anonymize

SYNTH_CWES = ("CWE-601", "CWE-78", "CWE-89", "CWE-22")
CHAIN_LENGTHS = (0, 1, 2, 3)
NEGATIVE_KINDS = ("constant", "validated", "long_constant", "decoy_source")
# Segni di una sorgente non fidata nel testo di un pacchetto.
SOURCE_MARKERS = ("request.", "input(", "sys.argv", "os.environ")
# Nessun nome del vocabolario contiene una di queste parti (stesso controllo di `c_anonymize.leaky_identifier`).
_AVOID = frozenset({"safe", "unsafe", "tainted", "vuln", "vulnerable", "clean", "const", "bad", "good", "secure", "insecure"})

VERBS = ["fetch", "build", "prepare", "render", "handle", "process", "load", "run", "dispatch", "collect", "compose",
         "resolve", "apply", "forward", "relay", "wrap", "deliver", "assemble", "route", "submit"]
NOUNS = ["report", "user", "order", "item", "profile", "archive", "session", "invoice", "ticket", "record", "account",
         "catalog", "export", "summary", "notice", "document", "bundle", "entry", "listing", "channel"]
PARAMS = ["value", "item", "name", "target", "text", "arg", "data", "token", "field", "param"]
KEYS = ["q", "name", "path", "url", "next", "id", "file", "target", "page", "ref"]
# Tipo di pacchetto, deciso nello scheletro (uguale nei due gemelli: il decoratore sta dentro il blocco che il modello vede):
# "web" = vista Flask con `@app.route`, "script" = funzione che legge argv, l'ambiente o la tastiera. Ogni sorgente è
# (righe di import, espressione con {key}/{KEY}).
FLAVORS: dict[str, dict] = {
    "web": {"imports": ["from flask import Flask, request"], "decorator": True,
            "sources": [((), 'request.args.get("{key}", "")'), ((), 'request.form["{key}"]')]},
    "script": {"imports": [], "decorator": False,
               "sources": [(("import sys",), "sys.argv[1]"), (("import os",), 'os.environ.get("{KEY}", "")'), ((), "input()")]},
}
# Operazioni sul valore: le prime tre non cambiano ciò che è (una costante resta tale, un valore controllato resta controllato),
# il troncamento non sanifica un dato non fidato. Stesso vocabolario per tutte le forme, così il lessico non dice il verdetto.
PASS_OPS = ["{x}", "{x}.strip()", "{x}.lower()", "str({x})"]
SLICE_OP = "{x}[:64]"
# Nomi della prima variabile, uguali per tutte le forme (distinti da PARAMS) e righe accessorie con cui ogni forma si allunga.
NAMES = ["raw", "incoming", "spare", "extra", "given", "entry", "other_input", "origin"]
FILLERS = ["size = len({src})", "count = len({v})", "label = \"item\"", "step = 1", "print(len({v}))",
           "if not {v}:\n    {v} = {fallback}"]
NOISE = [
    "def format_{n}(text):\n    return text.title()",
    "def count_{n}(items):\n    return len(items)",
    "def join_{n}(left, right):\n    return left + \"-\" + right",
]

# Per CWE: import dello stile del sink, righe del sink (con {arg}), costanti, controlli che rendono sicuro il valore.
SPECS: dict[str, dict] = {
    "CWE-78": {
        "styles": [
            ("os.system", ["import os"], ['os.system("{prefix}" + {arg})']),
            ("subprocess.call", ["import subprocess"], ['subprocess.call("{prefix}" + {arg}, shell=True)']),
            ("os.popen", ["import os"], ['return os.popen("{prefix}" + {arg}).read()']),
        ],
        "prefixes": ["ls ", "cat ", "ping -c 1 ", "echo ", "du -sh "],
        "constants": ["-l", "-a", "/tmp", "/var/log", "localhost"],
        "validators": [(["import shlex"], "shlex.quote({v})"),
                       ([], '{v} if {v} in ("-l", "-a", "/tmp", "localhost") else "-l"')],
    },
    "CWE-89": {
        "styles": [
            ("{db}.execute", ["import sqlite3"], ['{db}.execute("SELECT * FROM {table} WHERE {col} = \'" + {arg} + "\'")']),
            ("{db}.execute", ["import sqlite3"], ['{db}.execute(f"SELECT * FROM {table} WHERE {col} = \'{{{arg}}}\'")']),
            ("{db}.execute", ["import sqlite3"], ['{db}.execute("SELECT * FROM {table} WHERE {col} = \'%s\'" % {arg})']),
        ],
        "tables": ["users", "orders", "items", "tickets"],
        "cols": ["name", "owner", "code", "label"],
        "constants": ["alice", "bob", "admin", "guest"],
        "validators": [([], "str(int({v}))"), ([], '{v} if {v} in ("alice", "bob", "admin") else "alice"')],
    },
    "CWE-601": {
        "styles": [
            ("redirect", ["from flask import redirect"], ["return redirect({arg})"]),
            ("HttpResponseRedirect", ["from django.http import HttpResponseRedirect"], ["return HttpResponseRedirect({arg})"]),
        ],
        "constants": ["/home", "/dashboard", "/profile", "/settings"],
        "validators": [([], '{v} if {v} in ("/home", "/dashboard", "/profile") else "/home"'),
                       ([], '{v} if {v}.startswith("/") and not {v}.startswith("//") else "/home"')],
    },
    "CWE-22": {
        "styles": [
            ("send_file", ["from flask import send_file"], ["return send_file({arg})"]),
            ("send_file", ["from flask import send_file"], ['return send_file("data/" + {arg})']),
            ("send_file", ["from flask import send_file", "import os"], ['return send_file(os.path.join("data", {arg}))']),
        ],
        "constants": ["report.txt", "notes.txt", "summary.txt"],
        "validators": [([], '{v} if {v} in ("report.txt", "notes.txt", "summary.txt") else "report.txt"')],
    },
}


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random(":".join(str(p) for p in (seed, *parts)))


def _name_pool(rng: random.Random, count: int) -> list[str]:
    """`count` nomi di funzione distinti `verbo_nome`, senza parole-spia."""
    pairs = [f"{v}_{n}" for v in VERBS for n in NOUNS if not c_anonymize.leaky_identifier(f"{v}_{n}", _AVOID)]
    return rng.sample(pairs, count)


def _combos(seed: int, pairs: int) -> list[tuple[str, int, str]]:
    """(CWE, lunghezza della catena, forma del negativo) per ogni coppia: a cicli di 48, mescolati col seme."""
    combos = [(cwe, chain, kind) for kind in NEGATIVE_KINDS for chain in CHAIN_LENGTHS for cwe in SYNTH_CWES]
    out: list[tuple[str, int, str]] = []
    cycle = 0
    while len(out) < pairs:
        batch = list(combos)
        _rng(seed, "cycle", cycle).shuffle(batch)
        out.extend(batch)
        cycle += 1
    return out[:pairs]


def _indent(lines: list[str], by: str = "    ") -> list[str]:
    return [by + line for ln in lines for line in ln.split("\n")]


def _value_statements(label: int, kind: str, cwe: str, rng: random.Random, var: str, source: str) -> tuple[list[str], list[str]]:
    """(righe che producono `var`, import necessari) per un positivo o per la forma `kind` di negativo.

    Tutte le forme hanno la stessa struttura: `src = …` (la sorgente non fidata, una costante, o un contenitore con dentro la
    sorgente) e poi `var = …`, più 0–2 righe accessorie dallo stesso insieme. Cambia solo ciò che conta davvero:
    - positivo: `src` è la sorgente e `var` ne deriva con un'operazione che non sanifica (anche il troncamento);
    - `validated`: `src` è la sorgente e `var` esce da un controllo vero;
    - `decoy_source`: `src` è la sorgente ma `var` è una costante: la sorgente non arriva al sink;
    - `constant`: `src` è una costante e `var` ne deriva; `long_constant`: la costante è composta da due pezzi.
    """
    spec = SPECS[cwe]
    const = rng.choice(spec["constants"])
    src = rng.choice(NAMES)
    imports: list[str] = []

    def read() -> str:  # la sorgente nella variabile `src`, direttamente o attraverso un contenitore (30%)
        if rng.random() < 0.3:
            return f'payload = {{"k": {source}}}\n{src} = payload["k"]'
        return f"{src} = {source}"

    if label == 1:
        lines = [read(), f"{var} = {rng.choice([*PASS_OPS, SLICE_OP]).format(x=src)}"]
    elif kind == "validated":
        needs, check = rng.choice(spec["validators"])
        imports.extend(needs)
        lines = [read(), f"{var} = {check.format(v=src)}"]
    elif kind == "decoy_source":
        lines = [read(), f'{var} = "{const}"']
    elif kind == "long_constant":
        cut = max(1, len(const) // 2)
        lines = [f'{src} = "{const[:cut]}"', f'{var} = {rng.choice(PASS_OPS).format(x=f"{src} + {chr(34)}{const[cut:]}{chr(34)}")}']
    else:  # constant
        lines = [f'{src} = "{const}"', f"{var} = {rng.choice([*PASS_OPS, SLICE_OP]).format(x=src)}"]
    extras = [f.format(src=src, v=var, fallback=f'"{const}"') for f in rng.sample(FILLERS, rng.choice([0, 1, 1, 2]))]
    return [*lines, *extras], imports


def _package(pair: int, label: int, cwe: str, chain_len: int, kind: str, seed: int) -> tuple[str, int, str]:
    """(testo di `app.py`, riga del sink, callee) del pacchetto `label` della coppia `pair`.

    Tutte le scelte dello scheletro vengono da `rng` (uguale per i due gemelli); quelle del valore da `rng_value`, che
    dipende dall'etichetta.
    """
    rng = _rng(seed, "pair", pair)
    rng_value = _rng(seed, "pair", pair, "label", label)
    spec = SPECS[cwe]
    names = _name_pool(rng, chain_len + 3)
    fns = names[: chain_len + 1]  # fns[0] contiene il sink, fns[-1] sta in cima
    noise_names = names[chain_len + 1:]
    params = rng.sample(PARAMS, chain_len + 1)
    callee_template, style_imports, sink_lines = rng.choice(spec["styles"])
    fmt = {"prefix": rng.choice(spec.get("prefixes", [""])), "table": rng.choice(spec.get("tables", [""])),
           "col": rng.choice(spec.get("cols", [""])), "db": rng.choice(["cursor", "cur", "db"])}
    callee = callee_template.format(**fmt)
    imports = list(style_imports)
    module_lines: list[str] = []
    if cwe == "CWE-89":
        module_lines = ["conn = sqlite3.connect(\"app.db\")", f"{fmt['db']} = conn.cursor()"]
    # Tipo di pacchetto e sorgente sono dello scheletro: i due gemelli hanno lo stesso decoratore e la stessa sorgente nei
    # positivi e nei `validated`. `HttpResponseRedirect` è di Django, quindi mai nel tipo `web` (Flask).
    flavor_name = "script" if callee == "HttpResponseRedirect" else rng.choice(sorted(FLAVORS))
    flavor = FLAVORS[flavor_name]
    source_imports, source_template = rng.choice(flavor["sources"])
    key = rng.choice(KEYS)
    source = source_template.format(key=key, KEY=key.upper())
    imports.extend([*flavor["imports"], *source_imports])
    decorator = f'@app.route("/{key}")' if flavor["decorator"] else ""
    if decorator:
        module_lines.append("app = Flask(__name__)")

    value_var = params[-1]
    value_lines, value_imports = _value_statements(label, kind, cwe, rng_value, value_var, source)
    imports.extend(value_imports)

    def sink_body(arg: str) -> list[str]:
        return [line.format(arg=arg, callee=callee, **fmt) for line in sink_lines]

    blocks: dict[str, list[str]] = {}  # nome -> righe della funzione
    marker = "\0sink\0"
    if chain_len == 0:
        body = [*value_lines, *[marker + s for s in sink_body(value_var)]]
        blocks[fns[0]] = [*([decorator] if decorator else []), f"def {fns[0]}():", *_indent(body)]
    else:
        sink_marked = [marker + s for s in sink_body(params[0])]
        blocks[fns[0]] = [f"def {fns[0]}({params[0]}):", *_indent(sink_marked)]
        for depth in range(1, chain_len):
            op = rng.choice(PASS_OPS).format(x=params[depth])
            arg = op if rng.random() < 0.5 else params[depth]
            call = f"{fns[depth - 1]}({arg})" if rng.random() < 0.7 else f"{fns[depth - 1]}({params[depth - 1]}={arg})"
            body = [f"return {call}"] if rng.random() < 0.6 else [f"result = {call}", "return result"]
            blocks[fns[depth]] = [f"def {fns[depth]}({params[depth]}):", *_indent(body)]
        top = fns[chain_len]
        call = f"{fns[chain_len - 1]}({value_var})"
        top_body = [*value_lines, f"return {call}"]
        blocks[top] = [*([decorator] if decorator else []), f"def {top}():", *_indent(top_body)]
    for n in noise_names[: rng.choice([0, 1, 2])]:
        blocks[n] = rng.choice(NOISE).format(n=n.split("_")[0]).split("\n")
    order = list(blocks)
    rng.shuffle(order)

    header = sorted(set(imports))
    lines = [*header, "", *module_lines, "", ""] if module_lines else [*header, "", ""]
    for name in order:
        lines.extend([*blocks[name], "", ""])
    text = "\n".join(lines).rstrip("\n") + "\n"
    out_lines = text.split("\n")
    sink_line = next(i for i, ln in enumerate(out_lines, 1) if marker in ln)
    text = text.replace(marker, "")
    return text, sink_line, callee


def generate(out_dir: Path, n: int, seed: int) -> list[dict]:
    """Scrive `n` pacchetti in `out_dir/pkg_<NNNN>/app.py` e restituisce il manifesto, uno per pacchetto.

    Voce: `{"file" (relativo a out_dir), "line" (del sink), "callee", "cwe_id", "label", "chain_len", "group" (la coppia di
    gemelli), "kind" ("tainted" per i positivi, altrimenti la forma del negativo), "package"}`. I pacchetti vengono a
    coppie (vulnerabile, sicuro) con lo stesso scheletro; un `n` dispari aggiunge un solo vulnerabile. CWE e lunghezza
    della catena ruotano su tutte le combinazioni, mescolate col seme. Deterministico per `(n, seed)`.
    """
    out_dir = Path(out_dir)
    manifest: list[dict] = []
    pairs = (n + 1) // 2
    for pair, (cwe, chain_len, negative) in enumerate(_combos(seed, pairs)):
        for label in (1, 0):
            if len(manifest) >= n:
                break
            kind = "tainted" if label else negative
            text, line, callee = _package(pair, label, cwe, chain_len, negative, seed)
            package = f"pkg_{len(manifest):04d}"
            (out_dir / package).mkdir(parents=True, exist_ok=True)
            (out_dir / package / "app.py").write_text(text, encoding="utf-8")
            manifest.append({"file": f"{package}/app.py", "line": line, "callee": callee, "cwe_id": cwe, "label": label,
                             "chain_len": chain_len, "group": f"synth:{pair:04d}", "kind": kind, "package": package})
    return manifest


def cap_for(real_count: int, share: float = 0.15) -> int:
    """Massimo di sintetici perché siano al più `share` del totale con `real_count` esempi veri: `floor(share/(1−share)·real)`."""
    if share >= 1:
        raise ValueError("con share >= 1 non c'è un massimo")
    return math.floor(share * real_count / (1 - share) + 1e-9) if share > 0 else 0


def take_groups(examples: list[dict], cap: int) -> list[dict]:
    """Fino a `cap` esempi a gruppi interi (`meta.group`, o l'esempio da solo), nell'ordine originale.

    I gruppi si prendono nell'ordine di sha256 del gruppo finché ci stanno: la scelta è stabile e i gemelli restano insieme.
    """
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(str((ex.get("meta") or {}).get("group", f"#{i}")), []).append(i)
    taken: set[int] = set()
    for key in sorted(groups, key=lambda k: (hashlib.sha256(k.encode()).hexdigest(), k)):
        if len(taken) + len(groups[key]) <= cap:
            taken.update(groups[key])
    return [ex for i, ex in enumerate(examples) if i in taken]


def cap_share(synthetic: list[dict], real: list[dict], share: float = 0.15) -> list[dict]:
    """Al più `share` del totale `real + synthetic`: gruppi interi (i gemelli restano insieme), ordine originale.

    Con `share >= 1` non si toglie nulla; senza esempi veri non resta nessun sintetico (`cap_for`).
    """
    if share >= 1:
        return list(synthetic)
    return take_groups(synthetic, cap_for(len(real), share))
