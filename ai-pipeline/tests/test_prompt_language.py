"""Il contratto del prompt è in inglese: prompt di sistema, template, schede CWE, hint e testi del target.

Nemotron è un modello addestrato soprattutto in inglese: prompt e target (reasoning, suggested_fix) non devono
contenere italiano. Il controllo è euristico (parole funzionali e lettere accentate), quindi nei testi in inglese si
evitano i composti che un `\\b` spezza (es. «non-null» contiene la parola «non», «un-sanitized» la parola «un»).
"""
import json
import re
import unittest
from pathlib import Path

import benchmark
import common
import prepare_data

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "shared" / "golden.json").read_text(encoding="utf-8"))
KB = common.load_kb()

# Parole funzionali italiane (quelle del brief) più alcune preposizioni e articoli che l'inglese non usa.
ITALIAN_WORDS = re.compile(
    r"\b(il|lo|la|gli|della|delle|degli|dei|nel|nella|che|non|viene|vengono|senza|dati|usare|evitare|una|sono"
    r"|del|dal|dalla|dall|alla|alle|sul|sulla|più|ogni|uno|ed|dalle|dagli|un|di|con|sei|le|si|ha|oppure|anche|dopo"
    r"|quando|tutti|tutte)\b",
    re.I,
)
ITALIAN_ACCENTS = re.compile("[àèéìòù]", re.I)


def italian_hits(text: str) -> list[str]:
    """Parole funzionali e lettere accentate italiane trovate nel testo (lista vuota se è inglese)."""
    return [m.group(0) for m in ITALIAN_WORDS.finditer(text)] + [m.group(0) for m in ITALIAN_ACCENTS.finditer(text)]


class DetectorTests(unittest.TestCase):
    """Il rilevatore deve riconoscere l'italiano, altrimenti i test sotto sarebbero tautologie."""

    def test_flags_italian_sentences(self):
        for text in (
            "Sei un analizzatore di sicurezza SAST. Rispondi ESCLUSIVAMENTE con un oggetto JSON.",
            "Descrizione: Una query SQL viene costruita con input non fidato.",
            "Usare query parametrizzate invece di comporre la stringa SQL.",
            "Più percorsi di errore liberano la stessa risorsa",
        ):
            with self.subTest(text):
                self.assertTrue(italian_hits(text), text)

    def test_accepts_english_sentences(self):
        for text in (
            "You are a SAST security analyzer. Answer ONLY with a JSON object.",
            "Description: A SQL query is built from untrusted input.",
            "Use parameterized queries instead of building the SQL string.",
            "Pointer set to NULL right after free and checked before use",
        ):
            with self.subTest(text):
                self.assertEqual(italian_hits(text), [], text)


class PromptTemplateTests(unittest.TestCase):
    def test_template_version_and_system_language(self):
        template = common.load_template()
        self.assertEqual(template["version"], 2)
        self.assertIn("in English", template["system"])

    def test_template_texts_are_english(self):
        template = common.load_template()
        for key in ("system", "user", "block_header"):
            with self.subTest(key):
                self.assertEqual(italian_hits(template[key]), [], template[key])

    def test_golden_prompts_and_card_are_english(self):
        lines = list(GOLDEN["card"]["expected_lines"])
        for case in GOLDEN["prompts"]:
            lines += case["expected_user_lines"] + case["cwe_card_lines"]
        for line in lines:
            with self.subTest(line):
                self.assertEqual(italian_hits(line), [], line)


class CatalogTests(unittest.TestCase):
    def test_every_card_and_fix_hint_is_english(self):
        for entry in KB["entries"]:
            with self.subTest(entry["id"]):
                self.assertEqual(italian_hits(common.render_card(entry)), [], common.render_card(entry))
                self.assertEqual(italian_hits(entry["fix_hint"]), [], entry["fix_hint"])


class TargetTextTests(unittest.TestCase):
    def test_no_fix_is_english(self):
        self.assertEqual(prepare_data.NO_FIX, "No change needed.")
        self.assertEqual(italian_hits(prepare_data.FIX_PREFIX), [], prepare_data.FIX_PREFIX)

    def test_generated_reasoning_is_english(self):
        entry = next(e for e in KB["entries"] if e["id"] == "CWE-120")
        sink = {"callee": "strcpy", "line": 3}
        record = {"cve": "CVE-2020-1", "cve_desc": "Buffer overflow in copy_name", "note": "Use snprintf instead."}
        cases = {
            "vulnerable (cve_desc)": ("vulnerable", record),
            "vulnerable (note)": ("vulnerable", {"note": "Use snprintf instead."}),
            "vulnerable (bare)": ("vulnerable", {}),
            "patched": ("patched", record),
            "benign": ("benign", {}),
        }
        for name, (kind, rec) in cases.items():
            with self.subTest(name):
                text = prepare_data._reasoning(kind, sink, "a.c", entry, rec)
                self.assertEqual(italian_hits(text), [], text)


class BenchmarkPromptTests(unittest.TestCase):
    def test_noslice_baseline_prompt_is_english(self):
        self.assertIn("in English", benchmark.NOSLICE_SYSTEM)
        self.assertEqual(italian_hits(benchmark.NOSLICE_SYSTEM), [], benchmark.NOSLICE_SYSTEM)
        self.assertEqual(italian_hits(benchmark.NOSLICE_USER), [], benchmark.NOSLICE_USER)


if __name__ == "__main__":
    unittest.main()
