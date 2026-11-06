"""Test delle parti pure di quantize.py (nessuna GPU, nessun import di torch o di modelopt)."""
import fnmatch
import json
import sys
import tempfile
import unittest
from pathlib import Path

import quantize


def write_examples(path: Path, n: int):
    rows = [{"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": "{}"}], "meta": {"label": i % 2}} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "nvidia_quant_recipes.json").read_text(encoding="utf-8"))
NANO_4B = FIXTURES["nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"]
NANO_9B_FP8 = FIXTURES["nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"]


class ArgsTests(unittest.TestCase):
    def test_defaults_follow_the_plan(self):
        a = quantize.parse_args([])
        self.assertEqual(a.model, Path("model_bf16"))
        self.assertEqual(a.num_samples, 512)
        self.assertEqual(a.out_fp8, Path("model_fp8"))
        self.assertEqual(a.out_nvfp4, Path("model_nvfp4"))
        self.assertEqual(a.formats, ["fp8", "nvfp4"])

    def test_only_selects_one_format(self):
        self.assertEqual(quantize.parse_args(["--only", "nvfp4"]).formats, ["nvfp4"])

    def test_nvfp4_uses_nvidias_recipe_unless_told_otherwise(self):
        a = quantize.parse_args([])
        self.assertFalse(a.bf16_embeddings)
        self.assertTrue(quantize.parse_args(["--bf16-embeddings"]).bf16_embeddings)
        self.assertFalse(a.bf16_lm_head)
        self.assertTrue(quantize.parse_args(["--bf16-lm-head"]).bf16_lm_head)


class Fp8ExclusionTests(unittest.TestCase):
    """La regola di esclusione deve riprodurre esattamente le liste `exclude_modules` pubblicate da NVIDIA."""

    def test_4b_fp8_matches_the_published_checkpoint(self):
        self.assertEqual(sorted(quantize.exclusion_modules(NANO_4B["pattern"])), NANO_4B["exclude_modules"])

    def test_the_9b_fp8_checkpoint_confirms_the_attention_and_conv1d_part(self):
        got = quantize.exclusion_modules(NANO_9B_FP8["pattern"], preceding_mamba=False)
        self.assertEqual(sorted(got), NANO_9B_FP8["exclude_modules"])

    def test_disable_rules_match_exactly_the_excluded_modules_whatever_the_checkpoint_prefix(self):
        pattern = NANO_4B["pattern"]
        excluded = {name.split(".layers.", 1)[-1] for name in quantize.exclusion_modules(pattern) if ".layers." in name}
        rules = [e["quantizer_name"] for e in quantize.disable_entries(quantize.exclusion_modules(pattern))]
        for prefix in ("backbone", "model"):  # nomi originali NVIDIA / NemotronH nativo di transformers 5
            hit = set()
            for i, kind in enumerate(pattern):
                leaves = quantize.LAYER_LINEARS[kind] + (["conv1d"] if kind == "M" else [])
                for leaf in leaves:
                    module = f"{prefix}.layers.{i}.mixer.{leaf}"
                    quantizers = [f"{module}.weight_quantizer", f"{module}.input_quantizer"]
                    matched = [any(fnmatch.fnmatchcase(q, r) for r in rules) for q in quantizers]
                    self.assertEqual(matched[0], matched[1], module)  # pesi e attivazioni vanno insieme
                    if matched[0]:
                        hit.add(f"{i}.mixer.{leaf}")
            self.assertEqual(hit, {m.replace("layers.", "", 1) if m.startswith("layers.") else m for m in excluded if "mixer" in m})
        self.assertTrue(any(fnmatch.fnmatchcase("lm_head.weight_quantizer", r) for r in rules))

    def test_layer_one_does_not_swallow_layer_eleven(self):
        rules = [e["quantizer_name"] for e in quantize.disable_entries(["backbone.layers.1.mixer.in_proj"])]
        self.assertTrue(any(fnmatch.fnmatchcase("backbone.layers.1.mixer.in_proj.weight_quantizer", r) for r in rules))
        for other in ("backbone.layers.11.mixer.in_proj.weight_quantizer", "backbone.layers.21.mixer.in_proj.weight_quantizer"):
            self.assertFalse(any(fnmatch.fnmatchcase(other, r) for r in rules), other)

    def test_pattern_is_read_from_the_model_config(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({"hybrid_override_pattern": "M-*-"}), encoding="utf-8")
            self.assertEqual(quantize.read_hybrid_pattern(Path(d)), "M-*-")
            (Path(d) / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                quantize.read_hybrid_pattern(Path(d))


BASE_FP8 = {"algorithm": "max", "quant_cfg": [{"quantizer_name": "*", "enable": False}, {"quantizer_name": "*lm_head*", "enable": False}]}
BASE_NVFP4 = {"algorithm": "max", "quant_cfg": [{"quantizer_name": "*embeddings*weight_quantizer", "enable": True, "cfg": {"num_bits": (2, 1)}}]}


class QuantConfigTests(unittest.TestCase):
    def test_recipes_come_from_modelopt_not_from_this_repo(self):
        self.assertEqual(quantize.NVFP4_RECIPE, "models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16")
        self.assertEqual(quantize.FP8_RECIPE, "general/ptq/fp8_default-kv_fp8")

    def test_fp8_config_appends_the_nvidia_exclusions_after_the_base_rules(self):
        cfg = quantize.fp8_quant_cfg(BASE_FP8, NANO_4B["pattern"])
        self.assertEqual(cfg["quant_cfg"][: len(BASE_FP8["quant_cfg"])], BASE_FP8["quant_cfg"])
        extra = cfg["quant_cfg"][len(BASE_FP8["quant_cfg"]):]
        self.assertEqual(extra, quantize.disable_entries(quantize.exclusion_modules(NANO_4B["pattern"])))
        self.assertTrue(all(e["enable"] is False for e in extra))
        self.assertEqual(len(BASE_FP8["quant_cfg"]), 2, "la base non va modificata")

    def test_nvfp4_config_is_nvidias_recipe_untouched_by_default(self):
        cfg = quantize.nvfp4_quant_cfg(BASE_NVFP4)
        self.assertEqual(cfg, BASE_NVFP4)
        self.assertIsNot(cfg, BASE_NVFP4)

    def test_bf16_embeddings_is_one_last_rule_that_switches_the_embedding_quantizer_off(self):
        cfg = quantize.nvfp4_quant_cfg(BASE_NVFP4, bf16_embeddings=True)
        self.assertEqual(cfg["quant_cfg"][:-1], BASE_NVFP4["quant_cfg"])
        self.assertEqual(cfg["quant_cfg"][-1], {"quantizer_name": "*embeddings*weight_quantizer", "enable": False})
        self.assertEqual(len(BASE_NVFP4["quant_cfg"]), 1, "la base non va modificata")

    def test_bf16_lm_head_is_one_last_rule_that_switches_off_both_lm_head_quantizers(self):
        cfg = quantize.nvfp4_quant_cfg(BASE_NVFP4, bf16_lm_head=True)
        self.assertEqual(cfg["quant_cfg"][:-1], BASE_NVFP4["quant_cfg"])
        rule = cfg["quant_cfg"][-1]
        self.assertEqual(rule, {"quantizer_name": "*lm_head*", "enable": False})
        # pesi E attivazioni: la lm_head W8A16 della ricetta esporta un FP8 senza input_scale, che vLLM tratta come W8A8
        for name in ("lm_head.weight_quantizer", "lm_head.input_quantizer"):
            self.assertTrue(fnmatch.fnmatchcase(name, rule["quantizer_name"]), name)
        for other in ("backbone.layers.0.mixer.in_proj.weight_quantizer", "backbone.embeddings.weight_quantizer"):
            self.assertFalse(fnmatch.fnmatchcase(other, rule["quantizer_name"]), other)
        self.assertEqual(len(BASE_NVFP4["quant_cfg"]), 1, "la base non va modificata")

    def test_both_bf16_options_add_both_rules_in_order(self):
        cfg = quantize.nvfp4_quant_cfg(BASE_NVFP4, bf16_embeddings=True, bf16_lm_head=True)
        self.assertEqual(cfg["quant_cfg"][:-2], BASE_NVFP4["quant_cfg"])
        self.assertEqual([r["quantizer_name"] for r in cfg["quant_cfg"][-2:]], ["*embeddings*weight_quantizer", "*lm_head*"])
        self.assertTrue(all(r["enable"] is False for r in cfg["quant_cfg"][-2:]))

    def test_modelopt_must_ship_the_4b_recipe(self):
        quantize.check_modelopt_version("0.48.0")
        quantize.check_modelopt_version("0.49.1rc1")
        quantize.check_modelopt_version("1.0.0")
        for old in ("0.47.0", "0.46.1", "0.9"):
            with self.assertRaises(RuntimeError, msg=old):
                quantize.check_modelopt_version(old)


class CalibrationTests(unittest.TestCase):
    def test_selection_is_seeded_and_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.jsonl"
            write_examples(p, 100)
            a = quantize.select_calibration(p, 20, seed=1)
            self.assertEqual(len(a), 20)
            self.assertEqual(a, quantize.select_calibration(p, 20, seed=1))
            self.assertNotEqual(a, quantize.select_calibration(p, 20, seed=2))
            self.assertEqual(len(quantize.select_calibration(p, 500, seed=1)), 100)

    def test_calibration_texts_use_the_whole_conversation_like_real_generation(self):
        class Tok:
            def apply_chat_template(self, messages, **kw):
                return "|".join(m["role"] for m in messages) + f"|gen={kw['add_generation_prompt']}|think={kw['enable_thinking']}"

        ex = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}]
        self.assertEqual(quantize.calibration_texts(ex, Tok()), ["system|user|assistant|gen=False|think=False"])


class DtypeKwargTests(unittest.TestCase):
    def test_new_transformers_use_dtype_and_old_ones_torch_dtype(self):
        self.assertEqual(quantize.dtype_kwarg("5.0.0", "BF16"), {"dtype": "BF16"})
        self.assertEqual(quantize.dtype_kwarg("4.56.0", "BF16"), {"dtype": "BF16"})
        self.assertEqual(quantize.dtype_kwarg("4.55.2", "BF16"), {"torch_dtype": "BF16"})
        self.assertEqual(quantize.dtype_kwarg("4.57.0.dev0", "BF16"), {"dtype": "BF16"})


class CheckpointVerificationTests(unittest.TestCase):
    def write(self, d, quantization):
        (Path(d) / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": quantization}), encoding="utf-8")

    def test_fp8_checkpoint_is_described(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8", "exclude_modules": ["lm_head", "a", "b"]})
            info = quantize.verify_checkpoint(Path(d))
        self.assertEqual(info, {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8", "excluded": 3, "layers_by_algo": {}})

    def test_mixed_precision_checkpoint_counts_layers_per_format(self):
        layers = {"l1": {"quant_algo": "NVFP4"}, "l2": {"quant_algo": "NVFP4"}, "l3": {"quant_algo": "FP8"}}
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers})
            info = quantize.verify_checkpoint(Path(d))
        self.assertEqual(info["quant_algo"], "MIXED_PRECISION")
        self.assertEqual(info["layers_by_algo"], {"NVFP4": 2, "FP8": 1})
        self.assertEqual(info["excluded"], 0)

    def test_checkpoint_without_modelopt_metadata_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                quantize.verify_checkpoint(Path(d))
            self.write(d, {})
            with self.assertRaises(RuntimeError):
                quantize.verify_checkpoint(Path(d))


class LoadingInfoTests(unittest.TestCase):
    """Con pesi `backbone.*` e una classe che si aspetta altri nomi transformers carica lo stesso, re-inizializza a caso e avvisa soltanto:
    la calibrazione "riuscirebbe" su un modello rotto. Per questo i `missing_keys` fermano la quantizzazione."""

    def test_clean_load_passes(self):
        quantize.check_loading_info({"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []})

    def test_missing_keys_stop_the_run_and_are_named(self):
        info = {"missing_keys": {"backbone.layers.0.mixer.in_proj.weight", "lm_head.weight"}, "unexpected_keys": set()}
        with self.assertRaises(RuntimeError) as ctx:
            quantize.check_loading_info(info)
        self.assertIn("lm_head.weight", str(ctx.exception))

    def test_message_lists_a_bounded_number_of_keys(self):
        info = {"missing_keys": {f"backbone.layers.{i}.x.weight" for i in range(500)}}
        with self.assertRaises(RuntimeError) as ctx:
            quantize.check_loading_info(info)
        self.assertIn("500", str(ctx.exception))
        self.assertLess(len(str(ctx.exception)), 2000)

    def test_unexpected_keys_alone_do_not_stop_the_run(self):
        quantize.check_loading_info({"missing_keys": set(), "unexpected_keys": {"extra.weight"}})


class ConfigMatchesPatternTests(unittest.TestCase):
    def test_matching_layer_count_passes(self):
        quantize.check_layer_count(42, "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-")

    def test_wrong_layer_count_is_an_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            quantize.check_layer_count(52, "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-")
        self.assertIn("52", str(ctx.exception))
        self.assertIn("42", str(ctx.exception))


class LegacyTiedWeightsKeysTests(unittest.TestCase):
    """Il modeling remoto NVIDIA dichiara `_tied_weights_keys = ["lm_head.weight"]` (lista, transformers 4); transformers 5 vuole un dict
    `{target: sorgente}` e in `save_pretrained` (cioè nell'export di ModelOpt) fa `.keys()` sulla lista: AttributeError."""

    def mod(self, **attrs):
        return type("NemotronHForCausalLM", (), attrs)()

    def test_a_list_becomes_the_empty_dict_when_nothing_is_tied(self):
        m = self.mod(_tied_weights_keys=["lm_head.weight"])
        self.assertEqual(quantize.fix_legacy_tied_weights_keys([m], tie_word_embeddings=False), ["NemotronHForCausalLM"])
        self.assertEqual(m._tied_weights_keys, {})

    def test_dicts_and_modules_without_the_attribute_are_untouched(self):
        ok = self.mod(_tied_weights_keys={"lm_head.weight": "model.embed.weight"})
        none = self.mod(_tied_weights_keys=None)
        plain = self.mod()
        self.assertEqual(quantize.fix_legacy_tied_weights_keys([ok, none, plain], tie_word_embeddings=True), [])
        self.assertEqual(ok._tied_weights_keys, {"lm_head.weight": "model.embed.weight"})
        self.assertIsNone(none._tied_weights_keys)
        self.assertFalse(hasattr(plain, "_tied_weights_keys"))

    def test_second_call_changes_nothing(self):
        m = self.mod(_tied_weights_keys=["lm_head.weight"])
        quantize.fix_legacy_tied_weights_keys([m], tie_word_embeddings=False)
        self.assertEqual(quantize.fix_legacy_tied_weights_keys([m], tie_word_embeddings=False), [])

    def test_really_tied_embeddings_are_not_guessed(self):
        # la lista non dice a QUALE peso è legata: con tie_word_embeddings=True un dict inventato esporterebbe pesi sbagliati
        with self.assertRaises(RuntimeError):
            quantize.fix_legacy_tied_weights_keys([self.mod(_tied_weights_keys=["lm_head.weight"])], tie_word_embeddings=True)


class NoHeavyImportsTests(unittest.TestCase):
    def test_importing_the_module_does_not_load_ml_libraries(self):
        self.assertNotIn("modelopt", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
