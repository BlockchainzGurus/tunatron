#!/usr/bin/env bash
# Impacchetta il minimo per addestrare su Google Colab (notebooks/train_colab.ipynb): train.py e i moduli che usa, i test di
# train/common (il notebook li esegue sul suo ambiente), il dataset di train/validazione e i file condivisi letti da common.py.
#
#   scripts/pack_for_colab.sh [FILE_ZIP]      # default: notebooks/tunatron_code.zip  (poi caricalo in Drive/Tunatron/)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-notebooks/tunatron_code.zip}"
for f in ai-pipeline/train.py ai-pipeline/common.py ai-pipeline/data/train.jsonl ai-pipeline/data/val.jsonl \
         shared/prompt_template.json shared/vex_schema.json sast-engine/cwe_kb.json; do
  [[ -f "$f" ]] || { echo "manca $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
zip -q -r "$OUT" ai-pipeline/*.py ai-pipeline/tests ai-pipeline/data/train.jsonl ai-pipeline/data/val.jsonl \
    shared sast-engine/cwe_kb.json -x '*/__pycache__/*' '*.pyc'
echo "creato $OUT ($(du -h "$OUT" | cut -f1)):"
unzip -l "$OUT" | tail -n 1
