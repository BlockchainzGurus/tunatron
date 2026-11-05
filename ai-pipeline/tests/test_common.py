"""Test del contratto condiviso (prompt, card, matching sink, schema).

I golden in shared/golden.json sono gli stessi usati dai test Rust di sast-engine:
se questi test passano in entrambi i linguaggi, training e inferenza vedono lo stesso prompt.
"""
import json
import tempfile
import unittest
from pathlib import Path

import common

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "shared" / "golden.json").read_text(encoding="utf-8"))
KB = common.load_kb()


def _block(raw: dict) -> dict:
    block = {k: v for k, v in raw.items() if k not in ("code_lines", "code_crlf")}
    if "code_lines" in raw:
        block["code"] = "\n".join(raw["code_lines"])
    else:
        block["code"] = raw["code_crlf"]
    return block


class NormalizeTests(unittest.TestCase):
    def test_golden_cases(self):
        for case in GOLDEN["normalize"]:
            with self.subTest(case["input"]):
                self.assertEqual(common.normalize_callee(case["input"]), case["expected"])


class CalleeMatchTests(unittest.TestCase):
    def test_golden_cases(self):
        for case in GOLDEN["callee_matches"]:
            with self.subTest(f'{case["pattern"]} vs {case["callee"]}'):
                self.assertEqual(common.callee_matches(case["pattern"], case["callee"]), case["expected"])


class KbTests(unittest.TestCase):
    def test_match_sink_golden(self):
        for case in GOLDEN["kb_match"]:
            with self.subTest(f'{case["lang"]}:{case["callee"]}'):
                entry = common.match_sink(KB, case["lang"], case["callee"])
                self.assertEqual(entry["id"] if entry else None, case["cwe"])

    def test_alias_resolution_golden(self):
        for case in GOLDEN["alias"]:
            with self.subTest(case["label"]):
                self.assertEqual(common.resolve_cwe(KB, case["label"]), case["cwe"])

    def test_every_entry_has_the_fields_the_prompt_needs(self):
        for entry in KB["entries"]:
            with self.subTest(entry["id"]):
                for key in ("id", "name", "summary", "indicators", "safe_patterns", "fix_hint", "default_severity", "sinks"):
                    self.assertIn(key, entry)
                self.assertIn(entry["default_severity"], ("LOW", "MEDIUM", "HIGH", "CRITICAL"))

    def test_no_sink_pattern_is_claimed_by_two_cwes(self):
        # Un sink deve avere una CWE primaria, altrimenti il matching dipenderebbe dall'ordine.
        seen = {}
        for entry in KB["entries"]:
            for lang, patterns in entry["sinks"].items():
                for p in patterns:
                    key = (lang, p)
                    self.assertNotIn(key, seen, f"{key} in {seen.get(key)} e {entry['id']}")
                    seen[key] = entry["id"]


class CardTests(unittest.TestCase):
    def test_golden_card(self):
        card = GOLDEN["card"]
        self.assertEqual(common.render_card(card["entry"]), "\n".join(card["expected_lines"]))


class PromptTests(unittest.TestCase):
    def test_golden_user_prompts(self):
        for case in GOLDEN["prompts"]:
            with self.subTest(case["name"]):
                got = common.render_user_prompt(
                    cwe_id=case["cwe_id"],
                    cwe_card="\n".join(case["cwe_card_lines"]),
                    sink=case["sink"],
                    blocks=[_block(b) for b in case["blocks"]],
                )
                self.assertEqual(got, "\n".join(case["expected_user_lines"]))

    def test_system_prompt_comes_from_the_shared_template(self):
        template = json.loads((ROOT / "shared" / "prompt_template.json").read_text(encoding="utf-8"))
        self.assertEqual(common.system_prompt(), template["system"])

    def test_build_messages_has_system_and_user_roles(self):
        case = GOLDEN["prompts"][0]
        msgs = common.build_messages(
            cwe_id=case["cwe_id"],
            cwe_card="\n".join(case["cwe_card_lines"]),
            sink=case["sink"],
            blocks=[_block(b) for b in case["blocks"]],
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class ChatFormatTests(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "RENDERED"

    def test_thinking_is_always_disabled_so_training_matches_inference(self):
        tok = self.FakeTokenizer()
        msgs = [{"role": "user", "content": "x"}]
        self.assertEqual(common.chat_text(tok, msgs), "RENDERED")
        self.assertEqual(tok.calls[0][1], {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False})
        common.chat_text(tok, msgs, generation_prompt=True)
        self.assertEqual(tok.calls[1][1], {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})

    def test_response_markers_match_the_nemotron_template(self):
        # Verificati renderizzando chat_template.jinja con le impostazioni jinja di HF:
        # il testo di training = prompt di inferenza (enable_thinking=False) + risposta + <|im_end|>.
        self.assertEqual(common.INSTRUCTION_PART, "<|im_start|>user\n")
        self.assertEqual(common.RESPONSE_PART, "<|im_start|>assistant\n<think></think>")


class CopyRemoteCodeTests(unittest.TestCase):
    def test_copies_only_python_files_and_never_overwrites(self):
        import tempfile

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            src, dst = Path(a), Path(b)
            (src / "modeling_x.py").write_text("NEW", encoding="utf-8")
            (src / "configuration_x.py").write_text("CFG", encoding="utf-8")
            (src / "model.safetensors").write_bytes(b"weights")
            (dst / "modeling_x.py").write_text("KEEP", encoding="utf-8")
            copied = common.copy_remote_code(src, dst)
            self.assertEqual(copied, ["configuration_x.py"])
            self.assertEqual((dst / "modeling_x.py").read_text(encoding="utf-8"), "KEEP")
            self.assertFalse((dst / "model.safetensors").exists())

    def test_missing_source_is_a_noop(self):
        import tempfile

        with tempfile.TemporaryDirectory() as b:
            self.assertEqual(common.copy_remote_code(Path("/definitely/not/here"), Path(b)), [])


class FindingValidationTests(unittest.TestCase):
    VALID = {
        "vulnerable": True,
        "cwe_id": "CWE-89",
        "severity": "HIGH",
        "sink_function": "cursor.execute",
        "sink_file": "src/db/orders.py",
        "sink_line": 45,
        "vex_status": "affected",
        "reasoning": "Il parametro user_id arriva da request.args.",
        "suggested_fix": "Usare una query parametrizzata.",
    }

    def test_plan_example_is_valid(self):
        self.assertEqual(common.validate_finding(self.VALID), [])

    def test_missing_field_is_reported(self):
        bad = dict(self.VALID)
        del bad["reasoning"]
        errors = common.validate_finding(bad)
        self.assertTrue(any("reasoning" in e for e in errors), errors)

    def test_bad_enum_and_type_are_reported(self):
        bad = dict(self.VALID, severity="SEVERE", sink_line="45")
        errors = common.validate_finding(bad)
        self.assertTrue(any("severity" in e for e in errors), errors)
        self.assertTrue(any("sink_line" in e for e in errors), errors)

    def test_extra_property_is_rejected(self):
        errors = common.validate_finding(dict(self.VALID, extra=1))
        self.assertTrue(any("extra" in e for e in errors), errors)

    def test_bool_is_not_an_integer(self):
        errors = common.validate_finding(dict(self.VALID, sink_line=True))
        self.assertTrue(any("sink_line" in e for e in errors), errors)

    def test_severity_must_agree_with_the_vulnerable_flag_like_the_rust_engine(self):
        not_vuln_but_high = dict(self.VALID, vulnerable=False, vex_status="not_affected", severity="HIGH")
        self.assertTrue(any("severity" in e for e in common.validate_finding(not_vuln_but_high)))
        vuln_but_none = dict(self.VALID, severity="NONE")
        self.assertTrue(any("severity" in e for e in common.validate_finding(vuln_but_none)))
        self.assertEqual(common.validate_finding(dict(self.VALID, vulnerable=False, vex_status="not_affected", severity="NONE")), [])

    def test_inconsistent_vex_status_is_reported(self):
        errors = common.validate_finding(dict(self.VALID, vex_status="not_affected"))
        self.assertTrue(any("vex_status" in e for e in errors), errors)


class NormalizeNemotronConfigTests(unittest.TestCase):
    """`save_pretrained_merged` di Unsloth scrive il config NATIVO di transformers 5.x (`layers_block_type`, senza
    `hybrid_override_pattern` né `num_hidden_layers`) accanto ai pesi `backbone.*` e al codice remoto NVIDIA, la cui classe
    config ha `layers_block_type` read-only e, senza pattern, costruisce 52 layer invece dei 42 del 4B.
    Il fixture è un config reale: generato da NemotronHConfig di transformers 5.18.0 dal config del 4B."""

    FIXTURE = Path(__file__).parent / "fixtures" / "unsloth_merged_nemotron_4b_config.json"
    PATTERN_4B = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"

    def model_dir(self, config=None):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        cfg = json.loads(self.FIXTURE.read_text(encoding="utf-8")) if config is None else config
        (Path(d.name) / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return Path(d.name)

    def read(self, d):
        return json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))

    def test_fixture_is_the_broken_native_shape(self):
        cfg = json.loads(self.FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("layers_block_type", cfg)
        self.assertNotIn("hybrid_override_pattern", cfg)
        self.assertNotIn("num_hidden_layers", cfg)

    def test_native_config_becomes_the_nvidia_shape(self):
        d = self.model_dir()
        self.assertTrue(common.normalize_nemotron_config(d))
        cfg = self.read(d)
        self.assertEqual(cfg["hybrid_override_pattern"], self.PATTERN_4B)
        self.assertEqual(cfg["num_hidden_layers"], 42)
        self.assertNotIn("layers_block_type", cfg)  # la property read-only della classe remota fa fallire il caricamento

    def test_everything_else_is_left_alone(self):
        before = json.loads(self.FIXTURE.read_text(encoding="utf-8"))
        d = self.model_dir()
        common.normalize_nemotron_config(d)
        after = self.read(d)
        for key, value in before.items():
            if key != "layers_block_type":
                self.assertEqual(after[key], value, key)
        self.assertEqual(set(after) - set(before), {"hybrid_override_pattern", "num_hidden_layers"})

    def test_second_call_changes_nothing(self):
        d = self.model_dir()
        common.normalize_nemotron_config(d)
        written = (d / "config.json").read_text(encoding="utf-8")
        self.assertFalse(common.normalize_nemotron_config(d))
        self.assertEqual((d / "config.json").read_text(encoding="utf-8"), written)

    def test_nvidia_shape_is_not_touched(self):
        cfg = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-*-", "num_hidden_layers": 4}
        d = self.model_dir(cfg)
        before = (d / "config.json").read_text(encoding="utf-8")
        self.assertFalse(common.normalize_nemotron_config(d))
        self.assertEqual((d / "config.json").read_text(encoding="utf-8"), before)

    def test_both_shapes_present_and_consistent_just_drops_the_list(self):
        cfg = {"hybrid_override_pattern": "M-*-", "num_hidden_layers": 4,
               "layers_block_type": ["linear_attention", "mlp", "full_attention", "mlp"]}
        d = self.model_dir(cfg)
        self.assertTrue(common.normalize_nemotron_config(d))
        self.assertEqual(self.read(d), {"hybrid_override_pattern": "M-*-", "num_hidden_layers": 4})

    def test_conflicting_pattern_is_an_error_and_the_file_is_untouched(self):
        cfg = {"hybrid_override_pattern": "M-M-", "layers_block_type": ["linear_attention", "mlp", "full_attention", "mlp"]}
        d = self.model_dir(cfg)
        before = (d / "config.json").read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            common.normalize_nemotron_config(d)
        self.assertEqual((d / "config.json").read_text(encoding="utf-8"), before)

    def test_wrong_num_hidden_layers_is_an_error(self):
        cfg = {"num_hidden_layers": 52, "layers_block_type": ["linear_attention", "mlp"]}
        with self.assertRaises(ValueError):
            common.normalize_nemotron_config(self.model_dir(cfg))

    def test_layer_types_the_4b_remote_code_cannot_build_are_an_error(self):
        # "moe" è dei Nemotron-3 grandi: la classe remota del 4B conosce solo M, * e -
        with self.assertRaises(ValueError):
            common.normalize_nemotron_config(self.model_dir({"layers_block_type": ["linear_attention", "moe"]}))

    def test_a_directory_without_config_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(common.normalize_nemotron_config(Path(d)))

    def test_a_config_with_no_layer_description_is_a_noop(self):
        d = self.model_dir({"model_type": "llama"})
        self.assertFalse(common.normalize_nemotron_config(d))
        self.assertEqual(self.read(d), {"model_type": "llama"})


if __name__ == "__main__":
    unittest.main()
