#!/usr/bin/env python3
"""Argomenti di `vllm serve` per i checkpoint del progetto (usato da scripts/serve_vllm.sh e da benchmark.py).

    python3 vllm_flags.py ./model_nvfp4 [--port 8000] [--gpu-util 0.80] [--max-len 8192]   # un argomento per riga

Il flag --quantization NON viene mai aggiunto: vLLM riconosce da solo i checkpoint ModelOpt (FP8 -> modelopt, NVFP4 ->
modelopt_fp4, mix FP8/NVFP4 -> modelopt_mixed) leggendo config.json / hf_quant_config.json, come nelle model card NVIDIA.
Per forzarlo: VLLM_QUANT_FLAG=<metodo> scripts/serve_vllm.sh ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_NAME = "sast-nemotron"


def serve_args(model_dir: Path, port: int = 8000, gpu_util: float = 0.80, max_len: int = 8192,
               served_name: str = DEFAULT_NAME, max_num_seqs: int = 8) -> list[str]:
    return [
        "serve", str(model_dir),
        "--served-model-name", served_name,
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", f"{gpu_util:.2f}",
        "--max-num-seqs", str(max_num_seqs),
        "--port", str(port),
        "--trust-remote-code",
        "--mamba_ssm_cache_dtype", "float32",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--name", default=DEFAULT_NAME)
    args = ap.parse_args(argv)
    print("\n".join(serve_args(args.model_dir, args.port, args.gpu_util, args.max_len, args.name)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
