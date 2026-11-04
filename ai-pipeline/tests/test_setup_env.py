"""Test di scripts/setup_env.sh con DRY_RUN=1 (stampa i comandi senza eseguirli): kernel patchati e costruiti uno alla volta, versioni
fissate uguali a quelle del notebook Colab."""
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup_env.sh"


def dry_run(**env) -> list[str]:
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root"), "DRY_RUN": "1"}
    out = subprocess.run(["bash", str(SCRIPT)], env={**base, **env}, capture_output=True, text=True, check=True).stdout
    return [line[2:] for line in out.splitlines() if line.startswith("+ ")]


def notebook_settings() -> dict:
    nb = json.loads((ROOT / "notebooks" / "train_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(next(c for c in nb["cells"] if c["id"] == "settings")["source"])
    ns: dict = {}
    exec(source, ns)  # solo assegnazioni di costanti
    return ns


def fetches(lines):
    return [(i, l.split()) for i, l in enumerate(lines) if "patched_sdist.py" in l]


def builds(lines):
    return [(i, l) for i, l in enumerate(lines) if "--no-build-isolation" in l]


class KernelBuildTests(unittest.TestCase):
    def test_each_kernel_is_fetched_patched_and_built_on_its_own(self):
        lines = dry_run(ENVS="train")
        f, b = fetches(lines), builds(lines)
        self.assertEqual(len(f), 2)
        self.assertEqual(len(b), 2, "un'installazione per kernel: uv costruirebbe più sdist in parallelo e la RAM non basta")
        for (fi, ftoks), (bi, bline) in zip(f, b):
            self.assertLess(fi, bi)
            self.assertTrue(bline.endswith(ftoks[-1]), "si costruisce la cartella appena scaricata e patchata")
        names = [t[3] for _, t in f]
        self.assertEqual(names, ["causal-conv1d", "mamba-ssm"])
        for _, bline in b:
            self.assertFalse("causal-conv1d" in bline and "mamba-ssm" in bline)

    def test_kernel_versions_are_the_ones_of_the_notebook(self):
        lines = dry_run(ENVS="train")
        self.assertEqual([(t[3], t[4]) for _, t in fetches(lines)], list(notebook_settings()["KERNELS"]))

    def test_builds_force_a_local_compile_with_two_jobs_by_default(self):
        for _, line in builds(dry_run(ENVS="train")):
            for expected in ("MAX_JOBS=2", "NVCC_THREADS=2", "TORCH_CUDA_ARCH_LIST=12.0", "CAUSAL_CONV1D_FORCE_BUILD=TRUE", "MAMBA_FORCE_BUILD=TRUE"):
                self.assertIn(expected, line)

    def test_max_jobs_can_be_raised(self):
        for _, line in builds(dry_run(ENVS="train", MAX_JOBS="5")):
            self.assertIn("MAX_JOBS=5", line)

    @unittest.skipUnless(shutil.which("systemd-run"), "senza systemd-run lo script non applica il tetto di RAM")
    def test_builds_run_under_the_memory_cap(self):
        for _, line in builds(dry_run(ENVS="train")):
            self.assertTrue(line.startswith("systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 env "), line)

    def test_sources_are_never_shared_between_environments(self):
        lines = dry_run(ENVS="train quant")
        dirs = [t[-1] for _, t in fetches(lines)]
        self.assertEqual(len(dirs), 4)
        self.assertEqual(len(set(dirs)), 4, "ogni ambiente compila contro il proprio torch: niente oggetti compilati in comune")

    def test_sources_go_to_the_user_cache_not_to_tmp_which_is_ram(self):
        dirs = [t[-1] for _, t in fetches(dry_run(ENVS="train", HOME="/home/pippo"))]
        self.assertEqual(len(dirs), 2)
        self.assertTrue(all(d.startswith("/home/pippo/.cache/tunatron/mamba-src/") for d in dirs), dirs)
        dirs = [t[-1] for _, t in fetches(dry_run(ENVS="train", MAMBA_SRC="/dati/src"))]
        self.assertEqual(len(dirs), 2)
        self.assertTrue(all(d.startswith("/dati/src/") for d in dirs), dirs)

    def test_skip_mamba_build_skips_everything_about_the_kernels(self):
        lines = dry_run(SKIP_MAMBA_BUILD="1")
        self.assertEqual(fetches(lines) + builds(lines), [])

    def test_the_serving_environment_has_no_kernels(self):
        lines = dry_run(ENVS="vllm")
        self.assertEqual(fetches(lines) + builds(lines), [])
        self.assertTrue(any("vllm" in l for l in lines))


class PinTests(unittest.TestCase):
    @staticmethod
    def install_line(lines, python, contains):
        return next(l.split() for l in lines if f"--python {python}" in l and contains in l and "--no-build-isolation" not in l)

    def test_training_stack_is_pinned_like_the_notebook(self):
        nb = notebook_settings()
        lines = dry_run(ENVS="train")
        torch_line = self.install_line(lines, ".venv/bin/python", "torch")
        self.assertIn(f"torch=={nb['TORCH_PIN']}", torch_line)
        self.assertIn("--torch-backend=auto", torch_line)
        stack = self.install_line(lines, ".venv/bin/python", "unsloth")
        self.assertEqual({t for t in stack if "==" in t}, {f"{k}=={v}" for k, v in nb["PINS"].items()})
        for extra in nb["EXTRA_PACKAGES"]:
            self.assertIn(extra, stack)

    def test_pins_can_be_switched_off(self):
        lines = dry_run(ENVS="train", PIN_VERSIONS="0")
        self.assertNotIn("torch==", " ".join(self.install_line(lines, ".venv/bin/python", "torch")))
        self.assertEqual([t for t in self.install_line(lines, ".venv/bin/python", "unsloth") if "==" in t], [])

    def test_quantization_env_uses_the_same_torch_and_the_notebooks_modelopt(self):
        nb = notebook_settings()
        lines = dry_run(ENVS="quant")
        self.assertIn(f"torch=={nb['TORCH_PIN']}", self.install_line(lines, ".venv-quant/bin/python", "torch"))
        self.assertIn(nb["MODELOPT_SPEC"], self.install_line(lines, ".venv-quant/bin/python", "modelopt"))
        self.assertEqual(len(builds(lines)), 2, "anche .venv-quant ha i propri kernel Mamba")


if __name__ == "__main__":
    unittest.main()
