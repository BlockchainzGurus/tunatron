"""Struttura di notebooks/train_colab.ipynb: la sezione 11 (benchmark vLLM e model card) non si può eseguire sul PC, quindi si controlla
ciò che si può: celle, ordine, impostazioni e i comandi che le celle lanciano."""
import ast
import contextlib
import io
import json
import os
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import model_card
from tests.test_model_card import result_row

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"
SECTION_11 = ["hmc", "hmc1", "vllmenv", "hmc2", "bench", "hmc3", "card"]


def load() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(cell_id: str) -> str:
    return "".join(next(c for c in load() if c["id"] == cell_id)["source"])


class NotebookStructureTests(unittest.TestCase):
    def test_ids_are_unique_and_code_cells_parse(self):
        cells = load()
        ids = [c["id"] for c in cells]
        self.assertEqual(len(ids), len(set(ids)))
        for c in cells:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]), filename=c["id"])

    def test_new_code_cells_are_clean(self):
        for c in load():
            if c["id"] in SECTION_11 and c["cell_type"] == "code":
                self.assertEqual((c["metadata"], c["outputs"], c["execution_count"]), ({}, [], None), c["id"])

    def test_model_card_settings_and_defaults(self):
        ns: dict = {}
        exec(source("settings"), ns)  # solo assegnazioni di costanti
        self.assertIs(ns["MODEL_CARD"], True)
        self.assertIs(ns["RUN_BENCHMARK"], True)
        self.assertEqual(ns["VENV_VLLM"], "/content/venv-vllm")
        self.assertEqual(ns["BENCH_SAMPLES"], 200)
        self.assertEqual(ns["BENCH_CONCURRENCY"], 8)
        self.assertEqual(ns["BENCH_EXTRA_ARGS"], "")

    def test_section_order(self):
        ids = [c["id"] for c in load()]
        order = ["qcopy", *SECTION_11, "h11", "release"]
        self.assertEqual([i for i in ids if i in order], order)
        self.assertEqual(ids[ids.index("qcopy") + 1:ids.index("h11")], SECTION_11, "la sezione 11 sta fra qcopy e h11, senza altre celle")
        self.assertTrue(source("h11").startswith("## 12."))
        self.assertTrue(source("hmc").startswith("## 11."))
        self.assertTrue(source("hmc1").startswith("### 11.1"))
        self.assertTrue(source("hmc2").startswith("### 11.2"))
        self.assertTrue(source("hmc3").startswith("### 11.3"))

    def test_benchmark_runs_one_format_per_call(self):
        src = source("bench")
        for needle in ("benchmark.py hardware", "--vllm-bin", "--out-dir results/{fmt}", "--models {fmt}=model_{fmt}", "for fmt in"):
            self.assertIn(needle, src)

    def test_bench_clears_stale_results(self):
        src = source("bench")
        self.assertIn("rmtree", src)
        self.assertIn("results/{fmt}", src)

    def test_bench_does_not_need_vllm_to_have_run(self):
        src = source("bench")
        self.assertIn('globals().get("VLLM_OK")', src)  # runtime riavviato: la cella 11.1 può non essere stata eseguita
        self.assertIn("RuntimeError", src)              # GPU occupata, come qrun
        self.assertIn("running()", src)

    def test_vllm_is_never_pinned(self):
        src = source("vllmenv")
        install = next(line for line in src.splitlines() if "uv pip install" in line and "vllm" in line)
        self.assertIn("--torch-backend=auto", install)
        for c in load():
            text = "".join(c["source"])
            self.assertNotIn("vllm==", text, c["id"])
            self.assertNotIn("vllm>=", text, c["id"])
        ns: dict = {}
        exec(source("settings"), ns)
        self.assertNotIn("vllm", {name.lower() for name in ns["PINS"]})  # nemmeno via PINS

    def test_vllm_cell_never_raises(self):
        src = source("vllmenv")
        self.assertIn("try:", src)
        self.assertIn("except Exception", src)
        self.assertIn("VLLM_OK", src)
        self.assertNotIn("raise ", src)

    def test_vllm_version_is_recorded(self):
        self.assertIn("VLLM_VERSION", source("vllmenv"))
        self.assertIn("--vllm-version", source("card"))

    def test_card_cell_passes_what_the_card_needs(self):
        src = source("card")
        for needle in ("model_card.py", "--train-args", "--gpu", "--commit-file", "--freeze-file", "{DRIVE_DIR}/adapter", "--out-dir"):
            self.assertIn(needle, src)

    def test_card_is_copied_to_drive(self):
        src = source("card")
        self.assertIn("copytree", src)
        self.assertIn("DRIVE_DIR", src)

    def test_intro_mentions_the_new_outputs(self):
        intro = source("intro")
        for needle in ("MODEL_CARD.md", "model_card_assets/", "benchmark_results/"):
            self.assertIn(needle, intro)
        self.assertNotIn("Blackwell", intro)


class Section11RunTests(unittest.TestCase):
    """Esegue le celle della sezione 11 con le impostazioni e le funzioni di supporto vere, ma `sh`, `out` e `running` finti, in una cartella
    temporanea. Il comando della cella `card` passa dalla CLI vera di model_card.py: se i due non si capiscono, il test lo vede."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.calls, self.failing, self.install_fails, self.is_running = [], set(), False, False
        self.ns: dict = {}
        exec(source("settings"), self.ns)
        self.ns.update(WORK=str(root / "work"), DRIVE_DIR=str(root / "drive"), VENV=str(root / "venv"), VENV_QUANT=str(root / "venv-quant"),
                       VENV_VLLM=str(root / "venv-vllm"))
        exec(source("helpers"), self.ns)
        self.ai, self.drive = Path(self.ns["AI"]), Path(self.ns["DRIVE_DIR"])
        self.ai.mkdir(parents=True)
        self.drive.mkdir()
        self.ns.update(sh=self.fake_sh, out=lambda cmd: "WARNING rumore\n0.99.0", running=lambda: self.is_running,
                       GPU_NAME="NVIDIA Test GPU", VRAM="1000 MiB")

    def fake_sh(self, cmd, env=None, cwd=None, check=True):
        self.calls.append(cmd)
        if self.install_fails and "uv pip install" in cmd and "vllm" in cmd:
            raise RuntimeError("comando fallito (exit 1)")
        if "benchmark.py hardware" in cmd:
            fmt = re.search(r"--out-dir results/(\w+)", cmd).group(1)
            if fmt in self.failing:
                return 1
            (Path(cwd) / "results" / fmt).mkdir(parents=True, exist_ok=True)
            (Path(cwd) / "results" / fmt / "hardware.json").write_text(json.dumps([result_row(fmt, 0.8, vllm_version=None)]), encoding="utf-8")
            return 0
        if "model_card.py" in cmd:
            old = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.object(model_card, "_pyplot", side_effect=ImportError), contextlib.redirect_stderr(io.StringIO()):
                    model_card.main(shlex.split(cmd)[2:])
            finally:
                os.chdir(old)
        return 0

    def run_cell(self, cell_id: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()):  # le celle stampano per l'utente di Colab
            exec(source(cell_id), self.ns)

    def weights(self, base: Path, fmt: str) -> None:
        (base / f"model_{fmt}").mkdir(parents=True, exist_ok=True)
        (base / f"model_{fmt}" / "model.safetensors").write_bytes(b"pesi")

    def benchmarks(self) -> list[str]:
        return [c for c in self.calls if "benchmark.py hardware" in c]

    # --- 11.1
    def test_vllm_cell_records_the_installed_version(self):
        self.run_cell("vllmenv")
        self.assertIs(self.ns["VLLM_OK"], True)
        self.assertEqual(self.ns["VLLM_VERSION"], "0.99.0")
        install = next(c for c in self.calls if "uv pip install" in c)
        self.assertIn(" vllm --torch-backend=auto", install)
        self.assertNotIn("==", install)

    def test_vllm_cell_survives_a_failed_install(self):
        self.install_fails = True
        self.run_cell("vllmenv")  # non deve sollevare
        self.assertIs(self.ns["VLLM_OK"], False)
        self.assertIsNone(self.ns["VLLM_VERSION"])

    def test_vllm_cell_is_skipped_when_disabled(self):
        self.ns["RUN_BENCHMARK"] = False
        self.run_cell("vllmenv")
        self.assertEqual(self.calls, [])
        self.assertIs(self.ns["VLLM_OK"], False)

    # --- 11.2
    def test_bench_measures_one_format_per_call_and_a_failure_does_not_lose_the_others(self):
        self.weights(self.ai, "bf16")
        self.weights(self.drive, "fp8")      # solo su Drive (runtime nuovo): va copiato in locale
        self.weights(self.drive, "nvfp4")
        stale = self.ai / "results" / "fp8" / "hardware.json"
        stale.parent.mkdir(parents=True)
        stale.write_text("[]")               # risultato di una corsa precedente: fp8 ora fallisce e non deve sopravvivere
        self.failing = {"fp8"}
        self.ns["VLLM_OK"] = True
        self.run_cell("bench")
        self.assertEqual([re.search(r"--models (\w+)=model_\w+", c).group(1) for c in self.benchmarks()], ["bf16", "fp8", "nvfp4"])
        self.assertTrue((self.ai / "results" / "bf16" / "hardware.json").exists())
        self.assertTrue((self.ai / "results" / "nvfp4" / "hardware.json").exists())
        self.assertFalse((self.ai / "results" / "fp8").exists())
        self.assertTrue((self.ai / "model_fp8" / "model.safetensors").exists())
        self.assertTrue((self.drive / "benchmark_results" / "bf16" / "hardware.json").exists())
        self.assertFalse((self.drive / "benchmark_results" / "fp8").exists())

    def test_bench_removes_the_old_drive_results_of_a_format_it_measures_again(self):
        self.weights(self.ai, "bf16")
        self.weights(self.ai, "fp8")
        for fmt in ("bf16", "fp8"):  # esito di una corsa precedente su Drive
            old = self.drive / "benchmark_results" / fmt / "vllm_old.log"
            old.parent.mkdir(parents=True)
            old.write_text("vecchio")
        self.failing = {"fp8"}
        self.ns["VLLM_OK"] = True
        self.run_cell("bench")
        self.assertFalse((self.drive / "benchmark_results" / "fp8").exists(), "fp8 ha fallito: su Drive non deve restare il suo esito vecchio")
        self.assertFalse((self.drive / "benchmark_results" / "bf16" / "vllm_old.log").exists())
        self.assertTrue((self.drive / "benchmark_results" / "bf16" / "hardware.json").exists())

    def test_bench_passes_the_settings_to_benchmark_py(self):
        self.weights(self.ai, "bf16")
        self.ns.update(VLLM_OK=True, BENCH_SAMPLES=17, BENCH_CONCURRENCY=3, BENCH_EXTRA_ARGS="--gpu-util 0.5")
        self.run_cell("bench")
        (cmd,) = self.benchmarks()
        for part in ("--samples 17", "--concurrency 3", "--gpu-util 0.5", f"--vllm-bin {self.ns['VENV_VLLM']}/bin/vllm", "--models bf16=model_bf16"):
            self.assertIn(part, cmd)

    def test_bench_without_vllm_runs_nothing_and_leaves_no_old_numbers(self):
        self.weights(self.ai, "bf16")
        old = self.ai / "results" / "bf16" / "hardware.json"
        old.parent.mkdir(parents=True)
        old.write_text("[]")
        self.run_cell("bench")  # VLLM_OK non è nemmeno definito: runtime riavviato
        self.assertEqual(self.calls, [])
        self.assertFalse(old.exists())

    def test_bench_refuses_while_training_runs(self):
        self.is_running = True
        self.ns["VLLM_OK"] = True
        with self.assertRaises(RuntimeError):
            self.run_cell("bench")

    def test_bench_is_skipped_when_disabled(self):
        self.weights(self.ai, "bf16")
        self.ns.update(RUN_BENCHMARK=False, VLLM_OK=True)
        self.run_cell("bench")
        self.assertEqual(self.calls, [])

    def test_bench_with_no_model_prints_and_stops(self):
        self.ns["VLLM_OK"] = True
        self.run_cell("bench")
        self.assertEqual(self.calls, [])

    # --- 11.3
    def prepare_card_inputs(self) -> None:
        for fmt in ("bf16", "fp8"):
            self.weights(self.ai, fmt)
            (self.ai / "results" / fmt).mkdir(parents=True, exist_ok=True)
            (self.ai / "results" / fmt / "hardware.json").write_text(json.dumps([result_row(fmt, 0.8, vllm_version=None)]), encoding="utf-8")
        self.weights(self.ai, "nvfp4")    # il modello c'è, ma la sua misura è fallita: la card dice "not measured"
        (self.drive / "adapter").mkdir()  # runtime nuovo: l'adapter e il log sono solo su Drive
        (self.drive / "adapter" / "adapter_config.json").write_text(
            json.dumps({"r": 64, "lora_alpha": 64, "use_dora": True, "target_modules": ["q_proj"]}), encoding="utf-8")
        (self.drive / "train_colab.log").write_text("{'loss': 0.5, 'epoch': 0.1}\n{'train_runtime': 5400.0}\n", encoding="utf-8")
        (self.drive / "colab_commit.txt").write_text("https://example/repo master\nabc123\n", encoding="utf-8")
        (self.drive / "colab_freeze.txt").write_text("torch==2.14.1\n", encoding="utf-8")

    def test_card_command_is_understood_by_the_cli_and_lands_on_drive(self):
        self.prepare_card_inputs()
        self.ns["VLLM_VERSION"] = "0.99.0"
        self.run_cell("card")
        text = (self.drive / "MODEL_CARD.md").read_text(encoding="utf-8")
        for part in ("DoRA", "rank 64", "effective batch size 8", "NVIDIA Test GPU, 1000 MiB", "abc123", "torch 2.14.1", "Last logged: training loss 0.5000",
                     "1 h 30 min", "vLLM 0.99.0", "| fp8 |", "| nvfp4 | not measured"):
            self.assertIn(part, text)

    def test_card_extra_train_args_with_spaces_reach_the_card(self):
        self.prepare_card_inputs()
        self.ns["EXTRA_TRAIN_ARGS"] = "--epochs 1 --lr 0.0001"
        self.run_cell("card")
        text = (self.drive / "MODEL_CARD.md").read_text(encoding="utf-8")
        self.assertIn("1 epochs", text)
        self.assertIn("learning rate 0.0001", text)

    def test_card_does_not_leave_old_charts_on_drive(self):
        self.prepare_card_inputs()
        old = self.drive / "model_card_assets" / "old.png"
        old.parent.mkdir()
        old.write_bytes(b"png")
        self.run_cell("card")
        self.assertFalse(old.exists())

    def test_card_is_skipped_when_disabled(self):
        self.ns["MODEL_CARD"] = False
        self.run_cell("card")
        self.assertEqual(self.calls, [])
        self.assertFalse((self.drive / "MODEL_CARD.md").exists())

    def test_card_still_comes_out_with_nothing_but_the_repo(self):
        self.run_cell("card")  # nessun modello, nessun log, nessun risultato: "not available", non un errore
        self.assertIn("not available:", (self.drive / "MODEL_CARD.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
