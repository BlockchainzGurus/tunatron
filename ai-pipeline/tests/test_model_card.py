"""Test di model_card.py: lettura degli input, testo della card e grafici (nessuna GPU)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import model_card
import train

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

NOISY_LOG = (
    "Unsloth: Will smartly offload gradients to save VRAM!\n"
    "\x1b[A\x1b[A  3%|▎         | 10/330 [01:00<30:00,  6.0s/it]\r"
    "{'loss': '0.5321', 'grad_norm': '1.2', 'learning_rate': '0.0002', 'epoch': '0.05'}\n"
    "{'loss': 0.31, 'epoch': 0.1}\n"
    "{'loss': 'nan', 'epoch': '0.2'}\n"  # training divergente: il punto si scarta
    "{'loss': '0.2'}\n"  # senza epoch: non si può collocare
    "\x1b[A{'eval_loss': '0.28', 'epoch': '0.1'}\r  4%|▍         | 11/330 [01:06<30:00,  6.0s/it]\n"
    "{'train_runtime': '5432.1', 'train_loss': '0.3', 'epoch': '2.0'}\n"
)


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ParseTrainLogTests(unittest.TestCase):
    def test_curves_and_runtime_survive_ansi_and_progress_bars(self):
        parsed = model_card.parse_train_log(NOISY_LOG)
        self.assertEqual(parsed["train"], [(0.05, 0.5321), (0.1, 0.31)], "nan e righe senza epoch non compaiono")
        self.assertEqual(parsed["eval"], [(0.1, 0.28)])
        self.assertEqual(parsed["runtime_s"], 5432.1)

    def test_text_without_metrics_gives_empty_curves(self):
        self.assertEqual(model_card.parse_train_log("garbage"), {"train": [], "eval": [], "runtime_s": None})


class ParseFreezeTests(unittest.TestCase):
    def test_only_key_packages_with_normalised_names(self):
        text = "torch==2.14.1\nUnsloth_Zoo==2026.11.2\nnumpy==2.0\n"
        self.assertEqual(model_card.parse_freeze(text), {"torch": "2.14.1", "unsloth-zoo": "2026.11.2"})

    def test_kernels_built_from_a_local_folder_get_the_version_of_the_folder(self):
        text = "mamba-ssm @ file:///content/mamba_src/train/mamba-ssm/mamba_ssm-2.3.2.post1\ncausal-conv1d @ file:///x/unknown\n"
        self.assertEqual(model_card.parse_freeze(text), {"mamba-ssm": "2.3.2.post1", "causal-conv1d": "built from source"})


class LoadResultsTests(unittest.TestCase):
    def test_rows_are_merged_in_format_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "nvfp4" / "hardware.json", [{"format": "nvfp4", "f1": 0.7}])  # creata per prima
            write_json(root / "bf16" / "hardware.json", [{"format": "bf16", "f1": 0.8}])
            self.assertEqual([r["format"] for r in model_card.load_results(root)], ["bf16", "nvfp4"])

    def test_missing_folder_or_broken_file_gives_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(model_card.load_results(Path(d) / "nessuno"), [])
            (Path(d) / "fp8").mkdir()
            (Path(d) / "fp8" / "hardware.json").write_text("not json")
            self.assertEqual(model_card.load_results(Path(d)), [])


class DatasetSummaryTests(unittest.TestCase):
    def test_counts_per_label_source_and_language_and_missing_meta_is_question_mark(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [
                {"meta": {"label": 1, "source": "primevul", "lang": "c"}},
                {"meta": {"label": 0, "source": "primevul", "lang": "c"}},
                {"meta": {}},  # Review Focus 5
            ]
            (Path(d) / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
            summary = model_card.dataset_summary(Path(d))
        self.assertEqual(set(summary), {"train"}, "solo gli split presenti")
        train_split = summary["train"]
        self.assertEqual((train_split["examples"], train_split["vulnerable"], train_split["not_vulnerable"]), (3, 1, 1))
        self.assertEqual(train_split["sources"], {"primevul": 2, "?": 1})
        self.assertEqual(train_split["languages"], {"c": 2, "?": 1})

    def test_folder_without_jsonl_gives_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(model_card.dataset_summary(Path(d)))
            self.assertIsNone(model_card.dataset_summary(Path(d) / "nessuno"))


class CollectContextTests(unittest.TestCase):
    def build(self, root: Path) -> dict:
        write_json(root / "adapter" / "adapter_config.json", {
            "base_model_name_or_path": "unsloth/NVIDIA-Nemotron-3-Nano-4B", "r": 64, "lora_alpha": 64,
            "target_modules": ["q_proj", "in_proj"], "use_dora": True})
        (root / "train.log").write_text(NOISY_LOG, encoding="utf-8")
        write_json(root / "results" / "bf16" / "hardware.json", [{"format": "bf16", "f1": 0.8}])
        (root / "model_bf16").mkdir()
        write_json(root / "model_fp8" / "hf_quant_config.json", {"quantization": {"quant_algo": "FP8"}})
        (root / "data").mkdir()
        (root / "data" / "train.jsonl").write_text(json.dumps({"meta": {"label": 1, "source": "s", "lang": "c"}}) + "\n")
        (root / "commit.txt").write_text("https://example/repo master\nabc123 2026-11-04 fix\n")
        (root / "freeze.txt").write_text("torch==2.14.1\npeft==0.21.2\n")
        return model_card.collect_context(
            log=root / "train.log", results=root / "results", data=root / "data", adapter=root / "adapter",
            models={"fp8": root / "model_fp8", "bf16": root / "model_bf16", "nvfp4": root / "model_nvfp4"},  # nvfp4 non esiste
            commit_file=root / "commit.txt", freeze_file=root / "freeze.txt", gpu="NVIDIA RTX PRO 6000, 97887 MiB",
            train_args="--batch-size 2 --grad-accum 4 --epochs 1", vllm_version="0.11.0")

    def test_everything_is_collected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.build(Path(d))
        self.assertEqual(ctx["base_model"], "unsloth/NVIDIA-Nemotron-3-Nano-4B")
        self.assertEqual((ctx["hyper"]["batch_size"], ctx["hyper"]["grad_accum"], ctx["hyper"]["epochs"]), (2, 4, 1.0))
        self.assertEqual(ctx["adapter"], {"r": 64, "lora_alpha": 64, "target_modules": ["q_proj", "in_proj"], "use_dora": True})
        self.assertEqual(ctx["curves"]["train"], [(0.05, 0.5321), (0.1, 0.31)])
        self.assertEqual(ctx["runtime_s"], 5432.1)
        self.assertIn("abc123", ctx["commit"])
        self.assertEqual(ctx["gpu"], "NVIDIA RTX PRO 6000, 97887 MiB")
        self.assertEqual(ctx["vllm_version"], "0.11.0")
        self.assertEqual(ctx["versions"], {"torch": "2.14.1", "peft": "0.21.2"})
        self.assertEqual(ctx["data"]["train"]["examples"], 1)
        self.assertEqual(ctx["formats"], ["bf16", "fp8"], "in ordine, e solo le cartelle che esistono")
        self.assertEqual([r["format"] for r in ctx["results"]], ["bf16"])
        self.assertEqual(set(ctx["quant"]), {"fp8"})
        self.assertEqual(ctx["charts"], {})

    def test_invalid_train_args_do_not_exit(self):  # Review Focus 3
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            ctx = model_card.collect_context(log=None, results=Path(d), data=Path(d), adapter=None, models={}, commit_file=None,
                                             freeze_file=None, gpu=None, train_args="--non-esiste", vllm_version=None)
        self.assertIsNone(ctx["hyper"])

    def test_broken_adapter_config_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "adapter_config.json").write_text("not json")
            ctx = model_card.collect_context(log=None, results=Path(d), data=Path(d), adapter=Path(d), models={}, commit_file=None,
                                             freeze_file=None, gpu=None, train_args=None, vllm_version=None)
        self.assertIsNone(ctx["adapter"])
        self.assertEqual(ctx["base_model"], train.DEFAULT_MODEL)

    def test_no_inputs_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = model_card.collect_context(log=None, results=Path(d) / "x", data=Path(d) / "y", adapter=None, models={}, commit_file=None,
                                             freeze_file=None, gpu=None, train_args=None, vllm_version=None)
        self.assertEqual(ctx["formats"], [])
        self.assertEqual(ctx["results"], [])
        self.assertIsNone(ctx["data"])
        self.assertIsNone(ctx["hyper"])
        self.assertEqual(ctx["curves"], {"train": [], "eval": []})
        self.assertEqual(ctx["versions"], {})


def result_row(fmt: str, f1: float, **extra) -> dict:
    """Riga di hardware.json come la produce benchmark.hardware_row."""
    counts = {"tp": 75, "fp": 8, "tn": 92, "fn": 25}
    metrics = {"precision": 0.9, "recall": 0.75, "f1": f1, "fpr": 0.1, "accuracy": 0.85, "counts": counts}
    row = {"format": fmt, "weights_gb": 7.5, "kv_tokens": 90000, "vram_gb": 11.2, "disk_gb": 7.9, "tok_s": 40.0, "latency_s": 2.5,
           "latency_p50_s": 2.2, "latency_p95_s": 4.1, "scan_s": None, "f1": f1, "f1_call_sinks": f1, "invalid": 2, "n": 200,
           "concurrency": 8, "vllm_version": "0.11.0",
           "report": {"n": 200, "invalid": 2, "all": metrics, "call_sinks": metrics, "by_source": {"primevul": metrics}}}
    return {**row, **extra}


def full_ctx() -> dict:
    split = {"examples": 1000, "vulnerable": 400, "not_vulnerable": 600, "sources": {"primevul": 700, "secvuleval": 300}, "languages": {"c": 800, "python": 200}}
    return {
        "base_model": "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "hyper": {"rank": 64, "alpha": 64, "batch_size": 2, "grad_accum": 4, "optim": "adamw_8bit", "max_seq_length": 8192, "epochs": 2.0,
                  "max_steps": -1, "lr": 2e-4, "warmup_ratio": 0.03, "seed": 3407},
        "adapter": {"r": 64, "lora_alpha": 64, "target_modules": ["q_proj", "in_proj"], "use_dora": True},
        "curves": {"train": [(0.1, 0.5), (0.2, 0.4)], "eval": [(0.1, 0.45)]},
        "runtime_s": 5432.1, "commit": "https://example/repo master\nabc123 2026-11-04 fix", "gpu": "NVIDIA RTX PRO 6000, 97887 MiB",
        "vllm_version": None, "versions": {"torch": "2.14.1", "peft": "0.21.2"},
        "data": {"train": split, "val": split, "test": split},
        "formats": ["bf16", "fp8", "nvfp4"],
        "results": [result_row("bf16", 0.8123), result_row("fp8", 0.80), result_row("nvfp4", 0.78)],
        "quant": {"fp8": {"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}},
                  "nvfp4": {"quantization": {"quant_algo": "NVFP4", "group_size": 16}}},
        "charts": {"loss_curves": "model_card_assets/loss_curves.png", "quality": "model_card_assets/quality.png",
                   "latency_throughput": "model_card_assets/latency_throughput.png", "memory": "model_card_assets/memory.png"},
    }


def empty_ctx() -> dict:
    nothing = Path("/nonexistent-model-card-input")
    return model_card.collect_context(log=None, results=nothing, data=nothing, adapter=None, models={}, commit_file=None, freeze_file=None,
                                      gpu=None, train_args=None, vllm_version=None)


def section(text: str, title: str) -> str:
    """Il testo di una sezione `## title`, fino alla successiva."""
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


class RenderCardTests(unittest.TestCase):
    TITLES = ["Summary", "Intended use and limitations", "Training", "Data", "Evaluation", "Efficiency", "Files and reproduction"]

    def test_front_matter_and_sections_in_order(self):
        text = model_card.render_card(full_ctx())
        self.assertTrue(text.startswith("---\n"))
        for line in ("license: other", "library_name: transformers", "pipeline_tag: text-generation", "base_model: unsloth/NVIDIA-Nemotron-3-Nano-4B"):
            self.assertIn(line, text)
        positions = [text.index(f"\n## {t}\n") for t in self.TITLES]
        self.assertEqual(positions, sorted(positions))

    def test_tables_have_one_row_per_format_and_the_load_caveat(self):
        text = model_card.render_card(full_ctx())
        evaluation, efficiency = section(text, "Evaluation"), section(text, "Efficiency")
        for fmt in ("bf16", "fp8", "nvfp4"):
            self.assertIn(f"| {fmt} |", evaluation)
            self.assertIn(f"| {fmt} |", efficiency)
        self.assertIn("0.812", evaluation, "F1 a tre decimali")
        self.assertIn("per-request latency under load (concurrency 8)", efficiency)
        self.assertIn("NVFP4", efficiency, "algoritmo di quantizzazione da hf_quant_config.json")

    def test_missing_inputs_leave_the_headings_and_say_why(self):
        text = model_card.render_card(empty_ctx())
        for title in self.TITLES:
            self.assertIn(f"\n## {title}\n", text)
        self.assertGreaterEqual(text.count("not available:"), 4)

    def test_a_format_without_results_is_not_measured(self):
        ctx = {**full_ctx(), "formats": ["bf16", "nvfp4"], "results": [result_row("bf16", 0.8)]}
        evaluation = section(model_card.render_card(ctx), "Evaluation")
        line = next(l for l in evaluation.splitlines() if l.startswith("| nvfp4 |"))
        self.assertIn("not measured", line)

    def test_old_result_rows_render_with_n_d(self):  # Review Focus 2: hardware.json di un benchmark.py precedente
        old = {"format": "bf16", "weights_gb": 7.6, "kv_tokens": 90000, "vram_gb": 11.2, "disk_gb": 7.9, "tok_s": 40.0, "latency_s": 2.5,
               "scan_s": None, "f1": 0.8, "f1_call_sinks": 0.8, "invalid": 0}
        text = model_card.render_card({**full_ctx(), "formats": ["bf16"], "results": [old]})
        self.assertIn("n/d", section(text, "Evaluation"))
        self.assertIn("0.800", section(text, "Evaluation"))
        self.assertIn("n/d", section(text, "Efficiency"))

    def test_vllm_version_falls_back_to_the_context(self):
        no_version = [result_row("bf16", 0.8, vllm_version=None)]
        text = model_card.render_card({**full_ctx(), "formats": ["bf16"], "results": no_version, "vllm_version": "0.12.0"})
        self.assertIn("0.12.0", section(text, "Efficiency"))
        with_version = [result_row("bf16", 0.8, vllm_version="0.11.0")]
        text = model_card.render_card({**full_ctx(), "formats": ["bf16"], "results": with_version, "vllm_version": "0.12.0"})
        self.assertIn("0.11.0", section(text, "Efficiency"))
        self.assertNotIn("0.12.0", text)

    def test_training_section_reports_what_was_really_used(self):
        training = section(model_card.render_card(full_ctx()), "Training")
        for fragment in ("DoRA", "rank 64", "adamw_8bit", "effective batch size 8", "8192", "3407", "1 h 30 min", "torch 2.14.1", "abc123"):
            self.assertIn(fragment, training)

    def test_the_card_is_english_only(self):
        accents = "àèéìòùÀÈÉÌÒÙ"
        for ctx in (full_ctx(), empty_ctx()):
            text = model_card.render_card(ctx)
            self.assertEqual([c for c in text if c in accents], [])

    def test_images_only_for_existing_charts(self):
        text = model_card.render_card({**full_ctx(), "charts": {"loss_curves": "model_card_assets/loss_curves.png"}})
        self.assertIn("](model_card_assets/loss_curves.png)", text)
        self.assertNotIn("quality.png", text)


class MalformedResultsTests(unittest.TestCase):
    """Un valore strano in hardware.json non deve far perdere la card (finding della revisione finale)."""

    @staticmethod
    def ctx_with(**bad) -> dict:
        return {**full_ctx(), "formats": ["bf16"], "results": [result_row("bf16", 0.8, **bad)]}

    @staticmethod
    def cells(text: str, title: str) -> list[str]:
        row = next(l for l in section(text, title).splitlines() if l.startswith("| bf16 |"))
        return [c.strip() for c in row.split("|")[1:-1]]

    def test_text_and_non_finite_numbers_render_as_n_d(self):
        text = model_card.render_card(self.ctx_with(latency_s="1.5x", tok_s=float("nan"), disk_gb=float("inf"), invalid="many", n="?"))
        efficiency, evaluation = self.cells(text, "Efficiency"), self.cells(text, "Evaluation")
        self.assertEqual((efficiency[1], efficiency[4], efficiency[8]), ("n/d", "n/d", "n/d"))  # latenza media, tok/s, disco
        self.assertEqual((evaluation[5], evaluation[6]), ("n/d", "n/d"))                         # output non validi, esempi
        self.assertEqual(efficiency[2], "2.20")                                                 # il resto della riga resta

    def test_a_failing_chart_is_skipped_and_the_others_are_kept(self):
        figure = mock.MagicMock()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(model_card, "_pyplot", return_value=mock.MagicMock()), \
                mock.patch.object(model_card, "_chart_quality", side_effect=ValueError("boom")), \
                mock.patch.object(model_card, "_chart_loss", return_value=figure), mock.patch.object(model_card, "_chart_latency", return_value=figure), \
                mock.patch.object(model_card, "_chart_memory", return_value=figure), contextlib.redirect_stderr(io.StringIO()) as err:
            charts = model_card.make_charts(full_ctx(), Path(d) / "model_card_assets")
        self.assertEqual(set(charts), {"loss_curves", "latency_throughput", "memory"})
        self.assertIn("quality", err.getvalue())

    @unittest.skipUnless(HAVE_MPL, "matplotlib non installato")
    def test_non_finite_values_do_not_stop_the_other_charts(self):
        ctx = self.ctx_with(latency_s=float("nan"), latency_p50_s="x", tok_s=float("inf"))
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            charts = model_card.make_charts(ctx, Path(d) / "assets")
        self.assertIn("quality", charts)
        self.assertIn("loss_curves", charts)


class MakeChartsTests(unittest.TestCase):
    def test_without_matplotlib_returns_no_charts_and_does_not_fail(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(model_card, "_pyplot", side_effect=ImportError):
            self.assertEqual(model_card.make_charts(full_ctx(), Path(d) / "model_card_assets"), {})

    @unittest.skipUnless(HAVE_MPL, "matplotlib non installato")
    def test_charts_are_written(self):
        with tempfile.TemporaryDirectory() as d:
            assets = Path(d) / "model_card_assets"
            charts = model_card.make_charts(full_ctx(), assets)
            self.assertEqual(set(charts), {"loss_curves", "quality", "latency_throughput", "memory"})
            for name, relative in charts.items():
                self.assertEqual(relative, f"model_card_assets/{name}.png")
                self.assertGreater((assets / f"{name}.png").stat().st_size, 1000)

    @unittest.skipUnless(HAVE_MPL, "matplotlib non installato")
    def test_single_format_and_no_curves(self):  # Review Focus 4
        with tempfile.TemporaryDirectory() as d:
            only_bf16 = {**full_ctx(), "formats": ["bf16"], "results": [result_row("bf16", 0.8)], "curves": {"train": [], "eval": []}}
            self.assertEqual(set(model_card.make_charts(only_bf16, Path(d) / "a")), {"quality", "latency_throughput", "memory"})
            only_curves = {**full_ctx(), "results": []}
            self.assertEqual(set(model_card.make_charts(only_curves, Path(d) / "b")), {"loss_curves"})

    @unittest.skipUnless(HAVE_MPL, "matplotlib non installato")
    def test_old_rows_and_unknown_formats_still_draw(self):
        old = {"format": "bf16", "weights_gb": None, "kv_tokens": None, "vram_gb": 11.2, "disk_gb": 7.9, "tok_s": 40.0, "latency_s": 2.5,
               "scan_s": None, "f1": 0.8, "f1_call_sinks": 0.8, "invalid": 0}
        ctx = {**full_ctx(), "results": [old, {**old, "format": "weird"}]}
        with tempfile.TemporaryDirectory() as d:
            charts = model_card.make_charts(ctx, Path(d) / "assets")
        self.assertIn("quality", charts)
        self.assertIn("latency_throughput", charts)


class ChartTitlesTests(unittest.TestCase):
    def test_titles_are_english(self):
        accents = "àèéìòùÀÈÉÌÒÙ"
        self.assertTrue(all(c not in accents for text in model_card.CHART_TITLES.values() for c in text))
        self.assertTrue(any("Latency" in text for text in model_card.CHART_TITLES.values()))


class MainTests(unittest.TestCase):
    def run_main(self, root: Path, *extra: str) -> int:
        argv = ["--out-dir", str(root / "out"), "--results", str(root / "nessuno"), "--data", str(root / "nessuno"), *extra]
        return model_card.main(argv)

    def test_writes_the_card_with_no_inputs(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(model_card, "_pyplot", side_effect=ImportError), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_main(Path(d)), 0)
            text = (Path(d) / "out" / "MODEL_CARD.md").read_text(encoding="utf-8")
        self.assertIn("not available:", text)

    def test_stale_charts_are_removed(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(model_card, "_pyplot", side_effect=ImportError):
            stale = Path(d) / "out" / "model_card_assets" / "quality.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"png di una corsa precedente")
            self.run_main(Path(d))
            self.assertFalse(stale.exists())
            self.assertNotIn("quality.png", (Path(d) / "out" / "MODEL_CARD.md").read_text(encoding="utf-8"))

    def test_train_args_in_equals_form_are_accepted(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(model_card, "_pyplot", side_effect=ImportError):
            self.run_main(Path(d), "--train-args=--epochs 3")
            self.assertIn("3 epochs", (Path(d) / "out" / "MODEL_CARD.md").read_text(encoding="utf-8"))

    def test_unknown_model_format_is_a_cli_error(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_main(Path(d), "--models", "weird=somewhere")

    @unittest.skipUnless(HAVE_MPL, "matplotlib non installato")
    def test_links_charts_that_exist(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "results" / "bf16" / "hardware.json", [result_row("bf16", 0.8)])
            (root / "model_bf16").mkdir()
            argv = ["--out-dir", str(root / "out"), "--results", str(root / "results"), "--data", str(root / "nessuno"),
                    "--models", f"bf16={root / 'model_bf16'}"]
            self.assertEqual(model_card.main(argv), 0)
            text = (root / "out" / "MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn("](model_card_assets/quality.png)", text)
            self.assertTrue((root / "out" / "model_card_assets" / "quality.png").exists())


if __name__ == "__main__":
    unittest.main()
