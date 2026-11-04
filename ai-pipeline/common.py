"""Contratto condiviso fra la pipeline AI e sast-engine.

Tutto ciò che il modello vede (prompt, card CWE, schema di output, matching dei sink)
è definito una volta sola in `shared/` e `sast-engine/cwe_kb.json` e viene riprodotto
byte-per-byte sia qui sia in Rust: i golden test di `shared/golden.json` lo garantiscono.
Solo libreria standard, così i test non richiedono dipendenze pesanti.
"""
from __future__ import annotations

import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "sast-engine" / "cwe_kb.json"
TEMPLATE_PATH = ROOT / "shared" / "prompt_template.json"
SCHEMA_PATH = ROOT / "shared" / "vex_schema.json"

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


# --------------------------------------------------------------------------- file condivisi
def load_kb(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or KB_PATH).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_template() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- matching sink
def normalize_callee(callee: str) -> str:
    """Toglie gli spazi e porta `::` e `->` al separatore `.`."""
    return re.sub(r"\s+", "", callee).replace("::", ".").replace("->", ".")


def callee_matches(pattern: str, callee: str) -> bool:
    """`execute` combacia con `cursor.execute`; `os.system` solo con `os.system` (o `x.os.system`)."""
    c = normalize_callee(callee)
    p = normalize_callee(pattern)
    return c == p or c.endswith("." + p)


def kb_lang_key(lang: str) -> str:
    return "c" if lang == "cpp" else lang


def match_sink(kb: dict, lang: str, callee: str) -> dict | None:
    """Prima entry della KB (ordine del file) che dichiara un sink combaciante."""
    key = kb_lang_key(lang)
    for entry in kb["entries"]:
        for pattern in entry["sinks"].get(key, []):
            if callee_matches(pattern, callee):
                return entry
    return None


def resolve_cwe(kb: dict, label: str) -> str | None:
    """Riporta una label di dataset (es. CWE-787) all'id primario presente nella KB."""
    label = label.strip().upper()
    for entry in kb["entries"]:
        if entry["id"] == label or label in entry.get("aliases", []):
            return entry["id"]
    return None


# --------------------------------------------------------------------------- prompt
def render_card(entry: dict) -> str:
    lines = [
        f"{entry['id']} — {entry['name']}",
        f"Descrizione: {entry['summary']}",
        "Indicatori di vulnerabilità:",
        *[f"- {item}" for item in entry["indicators"]],
        "Pattern sicuri:",
        *[f"- {item}" for item in entry["safe_patterns"]],
    ]
    return "\n".join(lines)


def _fill(template: str, values: dict) -> str:
    """Sostituzione a passata singola: il testo inserito non viene mai riscansionato."""
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


def render_block(block: dict) -> str:
    header = _fill(
        load_template()["block_header"],
        {k: block[k] for k in ("file", "name", "start_line", "end_line", "depth")},
    )
    code = block["code"].replace("\r\n", "\n").rstrip("\n")
    numbered = "\n".join(f"{block['start_line'] + i:>5} | {line}" for i, line in enumerate(code.split("\n")))
    return f"{header}\n```{block['lang']}\n{numbered}\n```"


def system_prompt() -> str:
    return load_template()["system"]


def render_user_prompt(cwe_id: str, cwe_card: str, sink: dict, blocks: list[dict]) -> str:
    return _fill(
        load_template()["user"],
        {
            "cwe_card": cwe_card,
            "cwe_id": cwe_id,
            "sink_function": sink["function"],
            "sink_file": sink["file"],
            "sink_line": sink["line"],
            "trace": "\n\n".join(render_block(b) for b in blocks),
        },
    )


def build_messages(cwe_id: str, cwe_card: str, sink: dict, blocks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": render_user_prompt(cwe_id, cwe_card, sink, blocks)},
    ]


# --------------------------------------------------------------------------- validazione output
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
}


def validate_finding(obj) -> list[str]:
    """Valida un output del modello contro vex_schema.json (+ coerenza vulnerable/vex_status)."""
    schema = load_schema()
    if not isinstance(obj, dict):
        return ["il risultato non è un oggetto JSON"]
    errors: list[str] = []
    for name in schema["required"]:
        if name not in obj:
            errors.append(f"campo mancante: {name}")
    for name, value in obj.items():
        spec = schema["properties"].get(name)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"proprietà non ammessa: {name}")
            continue
        if not _TYPE_CHECKS[spec["type"]](value):
            errors.append(f"{name}: tipo errato (atteso {spec['type']})")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"{name}: valore {value!r} fuori da {spec['enum']}")
    if not errors:
        expected = "affected" if obj["vulnerable"] else "not_affected"
        if obj["vex_status"] != expected:
            errors.append(f"vex_status incoerente con vulnerable (atteso {expected})")
        if obj["vulnerable"] == (obj["severity"] == "NONE"):
            errors.append(f"severity {obj['severity']} incoerente con vulnerable={obj['vulnerable']}")
    return errors


# --------------------------------------------------------------------------- formato chat (train / quantize / benchmark)
# Verificati renderizzando il chat_template.jinja di Nemotron con le impostazioni jinja di HuggingFace:
# con enable_thinking=False il prompt di inferenza termina con "<|im_start|>assistant\n<think></think>" e il testo
# di training è esattamente quel prompt + risposta + "<|im_end|>\n".
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think></think>"


def chat_text(tokenizer, messages: list[dict], generation_prompt: bool = False) -> str:
    """Testo del template con il thinking SEMPRE spento (come fa il motore con chat_template_kwargs)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False)


def copy_remote_code(src: Path, dst: Path) -> list[str]:
    """Copia i file .py del codice remoto del modello (config/modeling) senza sovrascrivere quelli presenti."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        return []
    copied = []
    for f in sorted(src.glob("*.py")):
        target = dst / f.name
        if not target.exists():
            shutil.copy2(f, target)
            copied.append(f.name)
    return copied
