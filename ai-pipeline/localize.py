"""Localizzazione del sink *vero* in una funzione, senza tree-sitter (solo libreria standard).

Tre fonti di verità, dalla più forte alla più debole:
  1. i commenti `/* FLAW */` dei casi di test Juliet (`flaw_sink`);
  2. il diff fra la funzione vulnerabile e la sua versione corretta (`localize_sink`): il sink è una chiamata che sta
     sulle righe toccate dalla fix, o subito accanto a un controllo aggiunto dalla fix;
  3. (fuori da questo modulo) qualunque chiamata-sink della KB nella funzione.
"""
from __future__ import annotations

import difflib
import re

import common

CALL_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "alignof", "typeof", "typeid", "defined", "catch",
    "do", "else", "case", "static_assert", "decltype", "__attribute__", "__declspec", "throw", "new", "delete",
    # Python
    "elif", "not", "and", "or", "in", "is", "assert", "with", "yield", "await", "except", "del", "raise", "lambda",
}
HEADER_SKIP = {"__attribute__", "__declspec", "__attribute", "alignas"}

_CALL_RE = re.compile(r"(?<!\w)([A-Za-z_~]\w*(?:\s*(?:::|->|\.)\s*[A-Za-z_~]\w*)*)\s*\(")
_HEADER_NAME_RE = re.compile(r"([A-Za-z_~]\w*(?:::[A-Za-z_~]\w*)*)\s*\(")
_PY_DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)[ \t]*\(", re.M)
_FLAW_RE = re.compile(r"/\*\s*(POTENTIAL\s+)?FLAW")
_NOT_A_FUNCTION = re.compile(r"(typedef|struct|union|enum|class|namespace|extern)\b")


# --------------------------------------------------------------------------- testo senza commenti/stringhe
def blank_non_code(code: str, lang: str = "c") -> str:
    """Sostituisce commenti e letterali con spazi mantenendo posizioni e a-capo."""
    out = list(code)
    i, n = 0, len(code)
    python = lang == "python"
    while i < n:
        c = code[i]
        if python and c == "#" or not python and code.startswith("//", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
        elif not python and code.startswith("/*", i):
            j = code.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif python and code[i : i + 3] in ('"""', "'''"):
            j = code.find(code[i : i + 3], i + 3)
            j = n if j < 0 else j + 3
        elif c in "\"'":
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            i += 1
            continue
        for k in range(i, j):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def _python_body_start(blanked: str) -> int:
    """Indice subito dopo i due punti dell'intestazione `def` (decoratori e parametri esclusi)."""
    m = _PY_DEF_RE.search(blanked)
    if not m:
        return 0
    depth, i = 1, m.end()
    while i < len(blanked) and depth:
        depth += {"(": 1, ")": -1}.get(blanked[i], 0)
        i += 1
    colon = blanked.find(":", i)
    return colon + 1 if colon >= 0 else i


def _calls_with_span(code: str, lang: str) -> list[tuple[str, int, int]]:
    """(callee, riga iniziale, riga finale) di ogni chiamata; la riga finale è quella della parentesi che la chiude."""
    blanked = blank_non_code(code, lang)
    if lang == "python":
        body = _python_body_start(blanked)
    else:
        body = blanked.find("{")
        if body < 0:
            return []
    calls = []
    for m in _CALL_RE.finditer(blanked, body):
        callee = re.sub(r"\s+", "", m.group(1))
        if callee in CALL_KEYWORDS:
            continue
        depth, i = 0, m.end() - 1  # m.end() - 1 è la '(' di apertura
        while i < len(blanked):
            depth += {"(": 1, ")": -1}.get(blanked[i], 0)
            if depth == 0:
                break
            i += 1
        calls.append((callee, blanked.count("\n", 0, m.start()) + 1, blanked.count("\n", 0, i) + 1))
    return calls


def find_calls(code: str, lang: str = "c") -> list[tuple[str, int]]:
    """(callee, riga 1-based) delle chiamate nel corpo della funzione, in ordine di apparizione."""
    return [(callee, start) for callee, start, _end in _calls_with_span(code, lang)]


def extract_function_name(code: str, lang: str = "c") -> str:
    if lang == "python":
        m = _PY_DEF_RE.search(code)
        return m.group(1) if m else "unknown"
    header = code.split("{", 1)[0]
    for m in _HEADER_NAME_RE.finditer(header):
        name = m.group(1).split("::")[-1]
        if name not in HEADER_SKIP:
            return name
    return "unknown"


# --------------------------------------------------------------------------- diff vulnerabile / corretta
def _opcodes(func: str, other: str):
    a = [line.strip() for line in func.splitlines()]
    b = [line.strip() for line in other.splitlines()]
    return len(a), difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def changed_lines(func: str, other: str | None) -> dict | None:
    """Righe (1-based) di `func` che differiscono da `other`, ignorando gli spazi.

    `changed`: righe sostituite o cancellate. `boundary`: le due righe attorno a un punto in cui `other` ha AGGIUNTO
    del codice (tipicamente il controllo che la fix introduce prima della chiamata pericolosa).
    """
    if other is None:
        return None
    total, ops = _opcodes(func, other)
    changed: set[int] = set()
    boundary: set[int] = set()
    for tag, i1, i2, _j1, _j2 in ops:
        if tag in ("replace", "delete"):
            changed.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            boundary.update(n for n in (i1, i1 + 1) if 1 <= n <= total)
    return {"changed": changed, "boundary": boundary - changed}


def first_changed_line(func: str, other: str | None) -> int | None:
    """Prima riga di `func` toccata dal diff (la riga che segue, se la fix si limita ad aggiungere codice)."""
    if other is None:
        return None
    total, ops = _opcodes(func, other)
    for tag, i1, _i2, _j1, _j2 in ops:
        if tag in ("replace", "delete"):
            return i1 + 1
        if tag == "insert":
            return max(1, min(i1 + 1, total))
    return None


def localize_sink(kb: dict, code: str, lang: str, other: str | None, preferred_cwe: str | None = None) -> dict | None:
    """Chiamata-sink della KB sulle righe modificate dalla fix (o accanto a un controllo aggiunto), altrimenti None."""
    diff = changed_lines(code, other)
    if not diff or not (diff["changed"] or diff["boundary"]):
        return None
    calls = _calls_with_span(code, lang)
    for pool in (diff["changed"], diff["boundary"]):
        for callee, start, end in calls:
            # la chiamata conta se una riga toccata cade nella sua istruzione (anche su più righe), non in quelle dopo
            if not any(n in pool for n in range(start, end + 1)):
                continue
            entry = common.match_sink(kb, lang, callee)
            if entry is None or (preferred_cwe and entry["id"] != preferred_cwe):
                continue
            return {"entry": entry, "callee": callee, "line": start, "how": "diff"}
    return None


# --------------------------------------------------------------------------- Juliet
def flaw_sink(code: str) -> dict | None:
    """Chiamata che segue il commento `/* FLAW */` (o, se manca, `/* POTENTIAL FLAW */`) di un caso Juliet."""
    plain = potential = None
    for number, line in enumerate(code.splitlines(), start=1):
        m = _FLAW_RE.search(line)
        if not m:
            continue
        if m.group(1):
            potential = potential or number
        else:
            plain = plain or number
    marker = plain or potential
    if marker is None:
        return None
    for callee, line in find_calls(code, "c"):
        if line > marker:
            return {"callee": callee, "line": line}
    return None


def fix_sink(code: str) -> dict | None:
    """Chiamata che segue il commento `/* FIX */` della variante buona di un caso Juliet."""
    marker = next((number for number, line in enumerate(code.splitlines(), start=1) if re.search(r"/\*\s*FIX", line)), None)
    if marker is None:
        return None
    for callee, line in find_calls(code, "c"):
        if line > marker:
            return {"callee": callee, "line": line}
    return None


def split_c_functions(source: str) -> list[dict]:
    """Funzioni definite al livello più esterno di un file C/C++: nome, codice, riga di inizio (1-based)."""
    blanked = blank_non_code(source, "c")
    functions: list[dict] = []
    depth = 0
    boundary = 0  # primo carattere dopo l'ultimo ';', '}' o direttiva del preprocessore a livello esterno
    header = ""
    i, n = 0, len(blanked)
    while i < n:
        c = blanked[i]
        if depth == 0 and c == "#" and blanked[blanked.rfind("\n", 0, i) + 1 : i].strip() == "":
            while True:  # direttiva (con eventuali continuazioni "\")
                end = blanked.find("\n", i)
                end = n if end < 0 else end
                if end > i and blanked[end - 1] == "\\" and end < n:
                    i = end + 1
                    continue
                i = end
                break
            boundary = i
            continue
        if c == "{":
            if depth == 0:
                header = blanked[boundary:i]
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                h = header.strip()
                if h.endswith(")") and not _NOT_A_FUNCTION.match(h):
                    start = boundary + (len(header) - len(header.lstrip()))
                    functions.append({
                        "name": extract_function_name(h, "c"),
                        "code": source[start : i + 1],
                        "start_line": source.count("\n", 0, start) + 1,
                    })
                boundary = i + 1
        elif c == ";" and depth == 0:
            boundary = i + 1
        i += 1
    return functions


# --------------------------------------------------------------------------- Python e annotazioni dei test (CodeQL)
_PY_FUNC_LINE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")
_PY_CLASS_LINE = re.compile(r"^(\s*)class\s+\w+")
_LABEL_WORDS = re.compile(r"\b(GOOD|BAD|SAFE|UNSAFE|VULNERABLE|SECURE|INSECURE|FLAW|FIX|OK|NOT OK)\b")


def split_python_functions(source: str) -> list[dict]:
    """Funzioni e metodi (con i decoratori) a livello di modulo o di classe; le def annidate restano nel genitore."""
    lines = source.split("\n")
    stack: list[dict] = []  # blocchi aperti: {"indent", "kind", "start", "name", "record"}
    found: list[dict] = []

    def close(block: dict, upto: int) -> None:
        if block["kind"] == "def" and block["record"]:
            end = upto
            while end > block["start"] and not lines[end].strip():
                end -= 1
            found.append({"name": block["name"], "code": "\n".join(lines[block["start"] : end + 1]), "start_line": block["start"] + 1})

    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        while stack and indent <= stack[-1]["indent"]:
            close(stack.pop(), i - 1)
        fm, cm = _PY_FUNC_LINE.match(line), _PY_CLASS_LINE.match(line)
        if fm:
            start = i
            while start > 0 and lines[start - 1].lstrip().startswith("@") and len(lines[start - 1]) - len(lines[start - 1].lstrip(" \t")) == indent:
                start -= 1
            top_level = all(b["kind"] == "class" for b in stack)
            stack.append({"indent": indent, "kind": "def", "start": start, "name": fm.group(2), "record": top_level})
        elif cm:
            stack.append({"indent": indent, "kind": "class", "start": i, "name": "", "record": False})
    while stack:
        close(stack.pop(), len(lines) - 1)
    return sorted(found, key=lambda f: f["start_line"])


def _comment_start(line: str, marker: str) -> int:
    """Indice dell'inizio del commento di fine riga (fuori dalle stringhe), oppure -1."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif line.startswith(marker, i):
            return i
        i += 1
    return -1


def scrub_labels(code: str, marker: str) -> str:
    """Toglie dai test le annotazioni-soluzione (`# $ Alert`, `# GOOD`, `// BAD`...) conservando il numero delle righe."""
    out = []
    for line in code.split("\n"):
        at = _comment_start(line, marker)
        if at >= 0:
            comment = line[at + len(marker) :].strip()
            if comment.startswith("$") or _LABEL_WORDS.search(comment):
                line = line[:at].rstrip()
        out.append(line)
    return "\n".join(out)


def annotated_lines(code: str, marker: str) -> dict:
    """Righe (1-based) con `$ Alert` (sink vulnerabile) e righe ambigue (`MISSING:`/`SPURIOUS:`) dei test di CodeQL."""
    alert: set[int] = set()
    ambiguous: set[int] = set()
    for number, line in enumerate(code.split("\n"), start=1):
        at = _comment_start(line, marker)
        if at < 0:
            continue
        comment = line[at + len(marker) :].strip()
        if not comment.startswith("$"):
            continue
        if "MISSING" in comment or "SPURIOUS" in comment:
            ambiguous.add(number)
        elif re.search(r"\bAlert\b", comment):
            alert.add(number)
    return {"alert": alert, "ambiguous": ambiguous}
