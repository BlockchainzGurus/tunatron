#!/usr/bin/env bash
# Fase 3.1 — avvia vLLM sulla porta 8000 con il checkpoint indicato (default: ./model_nvfp4).
#
#   scripts/serve_vllm.sh [DIR_MODELLO] [ARGOMENTI EXTRA PER vLLM...]   # PORT, GPU_UTIL, MAX_LEN, VLLM_BIN via ambiente
#   es. FP8 come nella model card NVIDIA:  scripts/serve_vllm.sh model_fp8 --kv-cache-dtype fp8
#
# --quantization NON è fissato: vLLM riconosce da solo i checkpoint ModelOpt (come nelle model card NVIDIA).
# Per forzarlo:  VLLM_QUANT_FLAG=modelopt_mixed scripts/serve_vllm.sh ...   (modelopt | modelopt_fp4 | modelopt_mixed)
set -euo pipefail
CALLER_DIR="$PWD"
cd "$(dirname "$0")/.."

MODEL_DIR="${1:-model_nvfp4}"
[[ $# -gt 0 ]] && shift
# train.py e quantize.py (lanciati da ai-pipeline/) scrivono lì le cartelle dei modelli: un percorso relativo si cerca
# dove sei, poi nella radice del progetto, poi in ai-pipeline/.
if [[ "$MODEL_DIR" != /* ]]; then
  for base in "$CALLER_DIR" "$PWD" "$PWD/ai-pipeline"; do
    if [[ -d "$base/$MODEL_DIR" ]]; then MODEL_DIR="$base/$MODEL_DIR"; break; fi
  done
fi
[[ -d "$MODEL_DIR" ]] || { echo "Cartella del modello non trovata: $MODEL_DIR (la crea train.py / quantize.py in ai-pipeline/)" >&2; exit 1; }
PORT="${PORT:-8000}"; GPU_UTIL="${GPU_UTIL:-0.80}"; MAX_LEN="${MAX_LEN:-8192}"
if [[ -n "${VLLM_BIN:-}" ]]; then VLLM="$VLLM_BIN"
elif [[ -x .venv-vllm/bin/vllm ]]; then VLLM=.venv-vllm/bin/vllm
else VLLM=vllm; fi

mapfile -t ARGS < <(python3 ai-pipeline/vllm_flags.py "$MODEL_DIR" --port "$PORT" --gpu-util "$GPU_UTIL" --max-len "$MAX_LEN")
[[ -z "${VLLM_QUANT_FLAG:-}" ]] || ARGS+=(--quantization "$VLLM_QUANT_FLAG")
ARGS+=("$@")
echo "+ $VLLM ${ARGS[*]}" >&2
# Il venv non è attivato: senza la sua bin/ nel PATH FlashInfer non trova ninja per compilare i kernel al volo (FileNotFoundError: 'ninja').
if [[ "$VLLM" == */* ]]; then VLLM_BIN_DIR="$(cd "$(dirname "$VLLM")" && pwd)"; export PATH="$VLLM_BIN_DIR:$PATH"; fi
exec "$VLLM" "${ARGS[@]}"
