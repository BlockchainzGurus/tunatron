#!/usr/bin/env bash
# Fase 1.1 — verifica (SOLA LETTURA) di GPU, driver e toolkit CUDA 13.5. Non usa sudo e non modifica nulla.
set -uo pipefail
CUDA_HOME_WANTED="${CUDA_HOME_WANTED:-/usr/local/cuda-13.5}"
ok=1
say() { printf '%-28s %s\n' "$1" "$2"; }

if command -v nvidia-smi >/dev/null; then
  say "GPU" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  say "Driver" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  say "Compute capability" "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) (Blackwell = 12.0)"
else
  say "nvidia-smi" "NON TROVATO"; ok=0
fi

if [[ -x "$CUDA_HOME_WANTED/bin/nvcc" ]]; then
  say "nvcc ($CUDA_HOME_WANTED)" "$("$CUDA_HOME_WANTED/bin/nvcc" --version | grep -o 'release [0-9.]*' )"
else
  say "nvcc" "assente in $CUDA_HOME_WANTED"; ok=0
  echo "  -> installa il toolkit: sudo apt install cuda-toolkit-13-5   (e, se serve, cuda-compat-13-5)"
fi
[[ -d "$CUDA_HOME_WANTED/compat" ]] && say "cuda-compat" "presente" || say "cuda-compat" "assente (serve solo se il driver è più vecchio del toolkit)"

missing=()
[[ ":$PATH:" == *":$CUDA_HOME_WANTED/bin:"* ]] || missing+=("export PATH=$CUDA_HOME_WANTED/bin:\$PATH")
[[ ":${LD_LIBRARY_PATH:-}:" == *":$CUDA_HOME_WANTED/lib64:"* ]] || missing+=("export LD_LIBRARY_PATH=$CUDA_HOME_WANTED/compat:$CUDA_HOME_WANTED/lib64:\${LD_LIBRARY_PATH:-}")
[[ "${TORCH_CUDA_ARCH_LIST:-}" == "12.0" ]] || missing+=('export TORCH_CUDA_ARCH_LIST="12.0"')
if ((${#missing[@]})); then
  ok=0
  echo; echo "Variabili mancanti nella shell corrente — aggiungi a ~/.bashrc:"
  printf '  %s\n' "${missing[@]}"
fi
command -v uv >/dev/null && say "uv" "$(uv --version)" || { say "uv" "NON TROVATO"; ok=0; }
command -v git >/dev/null && say "git" "$(git --version)" || { say "git" "NON TROVATO"; ok=0; }
command -v cargo >/dev/null && say "cargo" "$(cargo --version)" || { say "cargo" "NON TROVATO"; ok=0; }
echo; [[ $ok == 1 ]] && echo "Ambiente OK" || echo "Ambiente da completare (vedi sopra)"
exit $((1 - ok))
