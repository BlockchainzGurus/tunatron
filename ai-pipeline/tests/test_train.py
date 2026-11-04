"""Test delle parti pure di train.py (nessuna GPU, nessun import di unsloth/torch)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import common
import train


class ArgsTests(unittest.TestCase):
    def test_defaults_follow_the_plan(self):
        a = train.parse_args([])
        self.assertEqual(a.model, "unsloth/NVIDIA-Nemotron-3-Nano-4B")
        self.assertEqual((a.rank, a.alpha), (64, 64))
        self.assertEqual((a.batch_size, a.grad_accum), (2, 4))
        self.assertEqual((a.optim, a.max_seq_length), ("adamw_8bit", 8192))
        self.assertEqual(a.out_dir, Path("model_bf16"))
        self.assertEqual(a.train, Path("data/train.jsonl"))

    def test_attention_implementation_follows_the_unsloth_nemotron_notebook(self):
        self.assertEqual(train.parse_args([]).attn_implementation, "eager")

    def test_targets_are_the_nemotron_h_linears_that_peft_accepts(self):
        self.assertEqual(train.TARGET_MODULES, ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj"])
        # PEFT >= 0.18 rifiuta questi due nei modelli Mamba (i kernel fusi usano direttamente i loro pesi: l'adapter sarebbe ignorato)
        self.assertFalse(set(train.TARGET_MODULES) & {"out_proj", "conv1d"})


class ModuleMatchingTests(unittest.TestCase):
    # Nomi reali di NemotronH: Mamba (in_proj/out_proj/conv1d), attention (q/k/v/o), MLP relu² (up/down, niente gate_proj)
    NAMES = [
        "backbone.embeddings",
        "backbone.layers.0.mixer.in_proj",
        "backbone.layers.0.mixer.conv1d",
        "backbone.layers.0.mixer.out_proj",
        "backbone.layers.1.mixer.up_proj",
        "backbone.layers.1.mixer.down_proj",
        "backbone.layers.2.mixer.q_proj",
        "backbone.layers.2.mixer.k_proj",
        "backbone.layers.2.mixer.v_proj",
        "backbone.layers.2.mixer.o_proj",
        "lm_head",
    ]

    def test_every_target_exists_in_nemotron_h(self):
        self.assertEqual(train.unmatched_targets(self.NAMES, train.TARGET_MODULES), [])

    def test_lora_targets_never_include_the_incompatible_mamba_modules(self):
        got = train.lora_targets(self.NAMES)
        self.assertEqual(got, train.TARGET_MODULES)
        self.assertNotIn("out_proj", got)

    def test_lora_targets_keep_only_modules_the_model_has(self):
        self.assertEqual(train.lora_targets(["x.mixer.in_proj", "x.mixer.q_proj"]), ["q_proj", "in_proj"])

    def test_a_model_without_any_target_is_an_error(self):
        with self.assertRaises(RuntimeError):
            train.lora_targets(["lm_head", "embed"])

    def test_partial_names_do_not_count_as_matches(self):
        self.assertEqual(train.unmatched_targets(["x.my_q_proj"], ["q_proj"]), ["q_proj"])


class DoraCheckTests(unittest.TestCase):
    def model(self, use_dora, with_magnitude=True):
        params = [("layers.0.q_proj.lora_A.default.weight", None)]
        if with_magnitude:
            params.append(("layers.0.q_proj.lora_magnitude_vector.default.weight", None))
        return SimpleNamespace(peft_config={"default": SimpleNamespace(use_dora=use_dora)}, named_parameters=lambda: params)

    def test_active_only_if_config_and_parameters_agree(self):
        self.assertTrue(train.dora_is_active(self.model(True)))
        self.assertFalse(train.dora_is_active(self.model(False)))
        self.assertFalse(train.dora_is_active(self.model(True, with_magnitude=False)), "config dice DoRA ma mancano i vettori di magnitudine")
        self.assertFalse(train.dora_is_active(SimpleNamespace()))


class FirstSupportedTests(unittest.TestCase):
    @staticmethod
    def factory(a, max_length=0, eval_strategy="no"):
        return None

    def test_picks_the_first_option_whose_keys_exist(self):
        got = train.first_supported(self.factory, [{"max_seq_length": 1}, {"max_length": 2}])
        self.assertEqual(got, {"max_length": 2})

    def test_raises_when_nothing_fits(self):
        with self.assertRaises(ValueError):
            train.first_supported(self.factory, [{"nope": 1}])


class LengthGroupingTests(unittest.TestCase):
    """Senza raggruppamento ogni micro-batch è riempito fino alla sequenza più lunga che contiene: ~22% di calcolo sprecato a 2 x 4."""

    def test_transformers_5_names_the_option_train_sampling_strategy(self):
        def config(*, output_dir, train_sampling_strategy="random"):
            return None

        self.assertEqual(train.length_grouping_kwargs(config), {"train_sampling_strategy": "group_by_length"})

    def test_older_transformers_use_the_boolean_flag(self):
        def config(*, output_dir, group_by_length=False):
            return None

        self.assertEqual(train.length_grouping_kwargs(config), {"group_by_length": True})

    def test_a_config_that_cannot_group_is_an_error_not_a_silent_random_order(self):
        def config(*, output_dir):
            return None

        with self.assertRaises(ValueError):
            train.length_grouping_kwargs(config)

    @unittest.skipUnless(importlib.util.find_spec("trl"), "serve trl: eseguire con .venv/bin/python")
    def test_the_sft_config_asks_for_length_grouping(self):
        from trl import SFTConfig

        kwargs = train.sft_config_kwargs(train.parse_args([]), SFTConfig)
        self.assertTrue(kwargs.get("train_sampling_strategy") == "group_by_length" or kwargs.get("group_by_length") is True)

    @unittest.skipUnless(importlib.util.find_spec("trl"), "serve trl: eseguire con .venv/bin/python")
    def test_the_sft_config_keeps_batch_and_accumulation_from_the_arguments(self):
        from trl import SFTConfig

        kwargs = train.sft_config_kwargs(train.parse_args(["--batch-size", "1", "--grad-accum", "8"]), SFTConfig)
        self.assertEqual((kwargs["per_device_train_batch_size"], kwargs["gradient_accumulation_steps"]), (1, 8))

    @unittest.skipUnless(
        all(importlib.util.find_spec(m) for m in ("torch", "transformers", "datasets")),
        "servono torch, transformers e datasets: eseguire con .venv/bin/python",
    )
    def test_the_trainer_samples_by_length_so_micro_batches_have_almost_no_padding(self):
        import random

        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_pt_utils import LengthGroupedSampler

        rng = random.Random(0)
        lengths = [rng.randint(100, 3000) for _ in range(400)]
        ds = Dataset.from_list([{"input_ids": [1] * n, "attention_mask": [1] * n} for n in lengths])  # stessa forma di tokenized_dataset
        with tempfile.TemporaryDirectory() as d:
            args = TrainingArguments(
                output_dir=d, per_device_train_batch_size=2, gradient_accumulation_steps=4, use_cpu=True, report_to="none",
                **train.length_grouping_kwargs(TrainingArguments),
            )
            sampler = Trainer(model=torch.nn.Linear(1, 1), args=args, train_dataset=ds)._get_train_sampler()
        self.assertIsInstance(sampler, LengthGroupedSampler)
        order = list(sampler)
        self.assertEqual(lengths[order[0]], max(lengths))  # il caso peggiore per primo: un OOM si vedrebbe al passo 1, non a metà corsa
        padded = used = 0
        for i in range(0, len(order), 2):  # micro-batch da 2
            pair = [lengths[j] for j in order[i : i + 2]]
            padded += 2 * max(pair)
            used += sum(pair)
        self.assertLess(1 - used / padded, 0.05)  # con ordine casuale qui sarebbe ~30%


class TokenLengthTests(unittest.TestCase):
    @staticmethod
    def tokenizer(text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def test_fits_uses_the_real_token_count(self):
        self.assertTrue(train.fits("a b c", self.tokenizer, 3))
        self.assertFalse(train.fits("a b c d", self.tokenizer, 3))


def write_examples(path: Path, n=3):
    rows = []
    for i in range(n):
        rows.append({
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u" * (100 * (i + 1))},
                {"role": "assistant", "content": json.dumps({"vulnerable": bool(i % 2)})},
            ],
            "meta": {"label": i % 2, "kind": "vulnerable" if i % 2 else "patched"},
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("torch"), "serve torch: eseguire con .venv/bin/python")
class FusedLossTests(unittest.TestCase):
    """Loss solo sulle posizioni con etichetta (testa del modello fusa di TRL): con 131k parole di vocabolario, i logit di
    tutta la sequenza mandano in OOM i 16 GB già al primo passo (successo: 2 x ~2.700 token)."""

    def setUp(self):
        import torch

        class Out:  # stesso contratto di trl FusedCausalLMOutput
            log_probs = torch.tensor([[0.0, -1.0, -3.0]])
            label_mask = torch.tensor([[False, True, True]])

        self.calls = []
        calls = self.calls

        class Model:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return Out()

        self.model = Model()
        self.out = Out

    def test_loss_is_the_mean_nll_over_the_labelled_tokens_only(self):
        loss = train.fused_compute_loss(None, self.model, {"input_ids": 1, "labels": 2})
        self.assertAlmostEqual(loss.item(), 2.0)  # (1 + 3) / 2: la posizione mascherata non conta

    def test_the_fused_head_is_requested_and_the_cache_is_off(self):
        train.fused_compute_loss(None, self.model, {"input_ids": 1, "labels": 2})
        self.assertIs(self.calls[0]["fused_lm_head"], True)
        self.assertIs(self.calls[0]["use_cache"], False)
        self.assertEqual((self.calls[0]["input_ids"], self.calls[0]["labels"]), (1, 2))

    def test_gradient_accumulation_normalizes_by_the_tokens_of_the_whole_batch(self):
        loss = train.fused_compute_loss(None, self.model, {"labels": 2}, num_items_in_batch=8)
        self.assertAlmostEqual(loss.item(), 0.5)  # 4 / 8

    def test_outputs_are_returned_on_request_and_inputs_are_not_mutated(self):
        inputs = {"input_ids": 1, "labels": 2}
        loss, outputs = train.fused_compute_loss(None, self.model, inputs, return_outputs=True)
        self.assertIsInstance(outputs, self.out)
        self.assertAlmostEqual(loss.item(), 2.0)
        self.assertEqual(inputs, {"input_ids": 1, "labels": 2})

    def test_a_batch_without_labelled_tokens_gives_a_finite_zero(self):
        import torch

        self.out.label_mask = torch.zeros(1, 3, dtype=torch.bool)
        self.out.log_probs = torch.zeros(1, 3)
        self.assertEqual(train.fused_compute_loss(None, self.model, {"labels": 2}).item(), 0.0)


@unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("torch"), "servono torch e peft: eseguire con .venv/bin/python")
class DoraDeadRowsTests(unittest.TestCase):
    """In 8 layer Mamba di NemotronH `in_proj` ha righe di pesi interamente a zero (es. 25 su 17.504 nel layer 0). DoRA divide la
    magnitudine per la norma della riga: 0/0 = NaN, che avvelena tutto il modello (loss e gradienti NaN al primo passo)."""

    def setUp(self):
        import torch
        from peft.tuners.lora import dora

        self.torch = torch
        self.cls = dora.DoraLinearLayer
        self.original = self.cls.get_weight_norm
        self.addCleanup(setattr, self.cls, "get_weight_norm", self.original)

    def model(self):
        import torch
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        base = torch.nn.Sequential()
        base.add_module("proj", torch.nn.Linear(8, 6, bias=False))
        with torch.no_grad():
            base.proj.weight[2] = 0.0  # riga morta
        cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"], use_dora=True, lora_dropout=0.0)
        return get_peft_model(base, cfg), torch.randn(4, 8)

    def test_unpatched_peft_gives_nan(self):
        m, x = self.model()
        if not self.torch.isnan(m(x)).any():
            self.skipTest("questo PEFT non produce più NaN: la patch di train.py si può togliere")

    def test_patched_dora_is_finite_and_leaves_the_model_unchanged_at_init(self):
        train.patch_dora_dead_rows()
        m, x = self.model()
        out = m(x)
        self.assertTrue(self.torch.isfinite(out).all())
        self.assertTrue(self.torch.allclose(out, m.base_model.model.proj.base_layer(x), atol=1e-6))  # B = 0: nessuna modifica

    def test_patched_gradients_are_finite(self):
        train.patch_dora_dead_rows()
        m, x = self.model()
        m(x).pow(2).sum().backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(self.torch.isfinite(g).all() for g in grads))

    def test_merged_weights_match_the_adapter_forward_and_the_dead_row_stays_dead(self):
        train.patch_dora_dead_rows()
        m, x = self.model()
        with self.torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.copy_(self.torch.randn_like(p) * 0.1)
        out = m(x)
        merged = m.merge_and_unload()
        self.assertTrue(self.torch.isfinite(merged.proj.weight).all())
        self.assertTrue(self.torch.allclose(out, merged(x), atol=1e-4))
        self.assertLess(merged.proj.weight[2].abs().max().item(), 1e-3)

    def test_patch_is_applied_once(self):
        train.patch_dora_dead_rows()
        first = self.cls.get_weight_norm
        train.patch_dora_dead_rows()
        self.assertIs(self.cls.get_weight_norm, first)


class DryRunTests(unittest.TestCase):
    def test_summary_without_importing_heavy_libraries(self):
        with tempfile.TemporaryDirectory() as d:
            tr, va = Path(d) / "train.jsonl", Path(d) / "val.jsonl"
            write_examples(tr, 3)
            write_examples(va, 1)
            args = train.parse_args(["--train", str(tr), "--val", str(va)])
            summary = train.dry_run(args)
        self.assertEqual((summary["train_examples"], summary["val_examples"]), (3, 1))
        self.assertEqual(summary["train_labels"], {"1": 1, "0": 2})
        self.assertEqual(summary["effective_batch_size"], 8)

    def test_dry_run_does_not_import_heavy_libraries(self):
        # in un processo a parte: altri test (loss fusa, DoRA) importano torch, e sys.modules è condiviso
        code = (
            f"import sys; sys.path.insert(0, {str(Path(train.__file__).parent)!r})\n"
            "import tempfile, pathlib, json, train\n"
            "d = pathlib.Path(tempfile.mkdtemp()); p = d / 'x.jsonl'\n"
            "p.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'u'}], 'meta': {'label': 1}}) + '\\n')\n"
            "train.dry_run(train.parse_args(['--train', str(p), '--val', str(p)]))\n"
            "print(sorted(m for m in ('unsloth', 'torch', 'peft', 'trl') if m in sys.modules))\n"
        )
        import subprocess

        out = subprocess.run([sys.executable, "-I", "-c", code], cwd=Path(train.__file__).parent, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")

    def test_missing_files_are_reported(self):
        args = train.parse_args(["--train", "/nope/train.jsonl"])
        with self.assertRaises(FileNotFoundError):
            train.dry_run(args)


if __name__ == "__main__":
    unittest.main()
