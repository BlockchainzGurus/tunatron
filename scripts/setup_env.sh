#!/usr/bin/env bash
# Fase 1.2 — ambienti Python con uv. NON è stato eseguito durante lo sviluppo del progetto (compila kernel CUDA).
#
#   DRY_RUN=1 scripts/setup_env.sh      # stampa solo i comandi
#   ENVS=train scripts/setup_env.sh     # solo alcuni ambienti: train | quant | vllm (default: tutti e tre)
#   scripts/setup_env.sh                # .venv (train + benchmark), .venv-quant (quantizzazione ModelOpt) e .venv-vllm (serving)
#
# Tre ambienti: vLLM fissa la propria versione di torch e confligge con Unsloth; nvidia-modelopt[hf] fissa transformers < 5.19.
# La compilazione di causal-conv1d / mamba-ssm per sm_120 è l'unico passo davvero pesante, e si fa in .venv e in .venv-quant
# (due volte: ogni ambiente compila contro il proprio torch). Gira in uno scope systemd con tetto di RAM (BUILD_MEM_MAX,
# default 14G) e MAX_JOBS limitato. SKIP_MAMBA_BUILD=1 la salta (il modello funziona comunque con i kernel di fallback in
# PyTorch, molto più lenti: sconsigliato per training e calibrazione).
set -euo pipefail
cd "$(dirname "$0")/.."

DRY="${DRY_RUN:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-3}"          # ogni job nvcc può usare 3-4 GB di RAM
export NVCC_THREADS="${NVCC_THREADS:-2}"
BUILD_MEM_MAX="${BUILD_MEM_MAX:-14G}"

want() { [[ " ${ENVS:-train quant vllm} " == *" $1 "* ]]; }
run() { echo "+ $*"; [[ "$DRY" == 1 ]] || "$@"; }
guarded() {
  if command -v systemd-run >/dev/null; then
    run systemd-run --user --scope -q -p "MemoryMax=$BUILD_MEM_MAX" -p MemorySwapMax=0 "$@"
  else
    run "$@"
  fi
}
# Kernel CUDA di Mamba-2 per l'ambiente $1 (torch deve essere già installato: --no-build-isolation lo usa per compilare).
install_mamba_kernels() {
  local py="$1"
  [[ "${SKIP_MAMBA_BUILD:-0}" != 1 ]] || return 0
  run uv pip install --python "$py" packaging ninja wheel setuptools
  guarded env CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --python "$py" --no-build-isolation causal-conv1d mamba-ssm
}

# --- ambiente di training / quantizzazione / benchmark --------------------------------------------------
if want train; then
run uv venv --python 3.12 .venv
PY=.venv/bin/python
# --torch-backend=auto sceglie da solo la build CUDA adatta al driver installato (oggi il default di PyPI è già cu132)
run uv pip install --python "$PY" torch torchvision --torch-backend=auto
run uv pip install --python "$PY" unsloth trl peft accelerate datasets pyarrow bitsandbytes huggingface_hub semgrep
install_mamba_kernels "$PY"
fi

# --- ambiente di quantizzazione (NVIDIA Model Optimizer; la ricetta NVFP4 del 4B c'è dalla 0.48.0) ----------------
# Con gli stessi kernel Mamba del training: la calibrazione fa 512 forward da fino a 4096 token.
if want quant; then
run uv venv --python 3.12 .venv-quant
run uv pip install --python .venv-quant/bin/python torch --torch-backend=auto
run uv pip install --python .venv-quant/bin/python "nvidia-modelopt[hf]>=0.48.0"
install_mamba_kernels .venv-quant/bin/python
fi

# --- ambiente di serving ------------------------------------------------------------------------------
if want vllm; then
run uv venv --python 3.12 .venv-vllm
run uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
fi

echo
echo "Fatto. Verifica:  .venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'"
