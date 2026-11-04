import json
import tempfile
import unittest
from pathlib import Path

import vllm_flags


def model_dir(quantization_config=None, extra_files=None):
    d = tempfile.TemporaryDirectory()
    cfg = {"architectures": ["NemotronHForCausalLM"]}
    if quantization_config is not None:
        cfg["quantization_config"] = quantization_config
    (Path(d.name) / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for name, content in (extra_files or {}).items():
        (Path(d.name) / name).write_text(json.dumps(content), encoding="utf-8")
    return d


class QuantizationFlagTests(unittest.TestCase):
    """vLLM riconosce da solo il formato (config.json / hf_quant_config.json): come nelle model card NVIDIA, nessun --quantization."""

    def test_no_quantization_flag_for_any_checkpoint_of_the_project(self):
        cases = {
            "bf16": model_dir(),
            "compressed-tensors": model_dir({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}),
            "modelopt fp8": model_dir({"quant_method": "modelopt"}, {"hf_quant_config.json": {"quantization": {"quant_algo": "FP8"}}}),
            "modelopt mixed nvfp4+fp8": model_dir(None, {"hf_quant_config.json": {"quantization": {"quant_algo": "MIXED_PRECISION"}}}),
        }
        for name, d in cases.items():
            with d:
                self.assertNotIn("--quantization", vllm_flags.serve_args(Path(d.name)), name)


class ServeArgsTests(unittest.TestCase):
    def test_arguments_follow_the_plan_and_the_model_card(self):
        with model_dir() as d:
            args = vllm_flags.serve_args(Path(d))
        self.assertEqual(args[0], "serve")
        joined = " ".join(args)
        for expected in [
            "--served-model-name sast-nemotron",
            "--max-model-len 8192",
            "--gpu-memory-utilization 0.80",
            "--port 8000",
            "--trust-remote-code",
            "--mamba_ssm_cache_dtype float32",
            "--max-num-seqs 8",
        ]:
            self.assertIn(expected, joined)

    def test_overrides_are_included(self):
        with model_dir() as d:
            args = vllm_flags.serve_args(Path(d), port=9001, gpu_util=0.5, max_len=4096, served_name="m")
        joined = " ".join(args)
        self.assertIn("--port 9001", joined)
        self.assertIn("--gpu-memory-utilization 0.50", joined)
        self.assertIn("--max-model-len 4096", joined)
        self.assertIn("--served-model-name m", joined)


if __name__ == "__main__":
    unittest.main()
