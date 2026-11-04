#!/usr/bin/env bash
# Fase 1.2 — ambienti Python con uv, per il PC (RTX 5070 Ti, sm_120). Su Google Colab NON serve: notebooks/train_colab.ipynb ha le
# sue celle (stesse versioni, compilazione con tutti i core, nessun systemd).
#
#   DRY_RUN=1 scripts/setup_env.sh      # stampa solo i comandi
#   ENVS=train scripts/setup_env.sh     # solo alcuni ambienti: train | quant | vllm (default: tutti e tre)
#   scripts/setup_env.sh                # .venv (train + benchmark), .venv-quant (quantizzazione ModelOpt) e .venv-vllm (serving)
#   PIN_VERSIONS=0 scripts/setup_env.sh # ultime versioni invece di quelle provate (sotto)
#
# Stato: i test (ai-pipeline/tests/test_setup_env.py) controllano i comandi stampati da DRY_RUN=1; i passi dei kernel sono quelli
# con cui è stato costruito .venv sul PC. Lo script intero non è mai stato eseguito end to end e .venv-quant / .venv-vllm non
# sono mai stati creati.
#
# Tre ambienti: vLLM fissa la propria versione di torch e confligge con Unsloth; nvidia-modelopt[hf] fissa transformers < 5.19.
# Versioni: torch e lo stack di training sono fissati a quelli con cui si è addestrato (ultime versioni = torch 2.15, mai provato
# con questi kernel e con Unsloth). Sono le stesse del notebook Colab: un test confronta i due elenchi.
#
# La compilazione di causal-conv1d / mamba-ssm per sm_120 è l'unico passo davvero pesante, e si fa in .venv e in .venv-quant
# (due volte: ogni ambiente compila contro il proprio torch). I loro setup.py fissano -std=c++17 ma torch >= 2.14 vuole C++20:
# ai-pipeline/patched_sdist.py scarica gli sorgenti da PyPI (sha256 controllato) e corregge il flag. I due kernel si costruiscono
# uno alla volta (uv, con più sdist insieme, li costruisce in parallelo e la RAM non basta), in uno scope systemd con tetto di RAM
# (BUILD_MEM_MAX, default 14G) e MAX_JOBS=2 (ogni job nvcc usa 3-4 GB). SKIP_MAMBA_BUILD=1 la salta (il modello funziona comunque
# con i kernel di fallback in PyTorch, molto più lenti: sconsigliato per training e calibrazione).
set -euo pipefail
cd "$(dirname "$0")/.."

DRY="${DRY_RUN:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-2}"          # ogni job nvcc può usare 3-4 GB di RAM: con il tetto di 14G, 2 job
export NVCC_THREADS="${NVCC_THREADS:-2}"
BUILD_MEM_MAX="${BUILD_MEM_MAX:-14G}"
# Sorgenti dei kernel: su disco, non in /tmp (di solito è un tmpfs, cioè RAM) e fuori dal repo.
MAMBA_SRC="${MAMBA_SRC:-$HOME/.cache/tunatron/mamba-src}"

# Versioni provate (le stesse di notebooks/train_colab.ipynb, che le ha da .venv). I kernel sono sempre fissi: la patch C++20 vale per loro.
PIN_VERSIONS="${PIN_VERSIONS:-1}"
TORCH_PIN="2.14.1"
TRAIN_PINS=(unsloth==2026.11.2 unsloth_zoo==2026.11.2 trl==1.15.0 peft==0.21.2 transformers==5.17.0 accelerate==1.15.0 datasets==4.8.5 bitsandbytes==0.50.2)
MODELOPT_SPEC="nvidia-modelopt[hf]>=0.48.0"   # la ricetta NVFP4 del 4B c'è dalla 0.48.0
KERNELS=("causal-conv1d 1.7.0" "mamba-ssm 2.3.2.post1")

want() { [[ " ${ENVS:-train quant vllm} " == *" $1 "* ]]; }
run() { echo "+ $*"; [[ "$DRY" == 1 ]] || "$@"; }
guarded() {
  if command -v systemd-run >/dev/null; then
    run systemd-run --user --scope -q -p "MemoryMax=$BUILD_MEM_MAX" -p MemorySwapMax=0 "$@"
  else
    run "$@"
  fi
}
# "pacchetto==versione" com'è, oppure solo "pacchetto" con PIN_VERSIONS=0.
pin() { if [[ "$PIN_VERSIONS" == 1 ]]; then echo "$1"; else echo "${1%%==*}"; fi; }

# Kernel CUDA di Mamba-2 per l'ambiente $1 (torch deve essere già installato: --no-build-isolation lo usa per compilare). $2 = nome
# dell'ambiente: i sorgenti sono separati, perché ogni ambiente compila contro il proprio torch. Un kernel alla volta.
install_mamba_kernels() {
  local py="$1" tag="$2" k name version src
  [[ "${SKIP_MAMBA_BUILD:-0}" != 1 ]] || return 0
  run uv pip install --python "$py" packaging ninja wheel setuptools
  for k in "${KERNELS[@]}"; do
    read -r name version <<<"$k"
    src="$MAMBA_SRC/$tag/$name"
    run python3 -I ai-pipeline/patched_sdist.py "$name" "$version" "$src"
    guarded env MAX_JOBS="$MAX_JOBS" NVCC_THREADS="$NVCC_THREADS" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
      CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --python "$py" --no-build-isolation "$src"
  done
}

# --- ambiente di training / quantizzazione / benchmark --------------------------------------------------
if want train; then
run uv venv --python 3.12 .venv
PY=.venv/bin/python
# --torch-backend=auto sceglie da solo la build CUDA adatta al driver installato (oggi il default di PyPI è già cu132)
run uv pip install --python "$PY" "$(pin "torch==$TORCH_PIN")" torchvision --torch-backend=auto
mapfile -t TRAIN_PKGS < <(for p in "${TRAIN_PINS[@]}"; do pin "$p"; done)
run uv pip install --python "$PY" "${TRAIN_PKGS[@]}" pyarrow huggingface_hub semgrep
install_mamba_kernels "$PY" train
fi

# --- ambiente di quantizzazione (NVIDIA Model Optimizer; la ricetta NVFP4 del 4B c'è dalla 0.48.0) ----------------
# Con gli stessi kernel Mamba del training: la calibrazione fa 512 forward da fino a 4096 token.
if want quant; then
run uv venv --python 3.12 .venv-quant
# Stesso torch del training (ModelOpt chiede solo torch >= 2.8 e non lo cambia): così i kernel provati con questa versione valgono anche qui.
run uv pip install --python .venv-quant/bin/python "$(pin "torch==$TORCH_PIN")" --torch-backend=auto
run uv pip install --python .venv-quant/bin/python "$MODELOPT_SPEC" --torch-backend=auto
install_mamba_kernels .venv-quant/bin/python quant
fi

# --- ambiente di serving ------------------------------------------------------------------------------
if want vllm; then
run uv venv --python 3.12 .venv-vllm
run uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
fi

echo
echo "Fatto. Verifica:  .venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'"
