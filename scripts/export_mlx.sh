#!/usr/bin/env bash
# Esportazione del modello Tunatron (Nemotron-3-Nano-4B / nemotron_h) in formato Apple MLX
# a 4, 6 e 8 bit con pubblicazione automatica su Hugging Face Hub.
#
# Prerequisiti:
#   - macOS con chip Apple Silicon (M1/M2/M3/M4 o successivi)
#   - Python >= 3.10 con mlx-lm installato (pip install -U mlx-lm huggingface_hub)
#   - Login a Hugging Face effettuato (huggingface-cli login) oppure variabile d'ambiente HF_TOKEN
#
# Utilizzo:
#   scripts/export_mlx.sh [OPZIONI]
#
# Esempi:
#   # Scarica da HF e pubblica le versioni a 4, 6 e 8 bit sotto la tua organizzazione:
#   scripts/export_mlx.sh -s BlockchainzGurus/tunatron-sast-nano-4b-bf16 -u BlockchainzGurus
#
#   # Usa la cartella locale model_bf16 esportando solo 4 e 8 bit:
#   scripts/export_mlx.sh -s ai-pipeline/model_bf16 -b 4,8 -u MioUsername
#
#   # Solo conversione locale (senza upload su Hugging Face):
#   scripts/export_mlx.sh -s BlockchainzGurus/tunatron-sast-nano-4b-bf16 --no-upload
#
set -euo pipefail

CALLER_DIR="$PWD"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Valori di default
SOURCE_MODEL=""
UPLOAD_ORG=""
REPO_NAME_PREFIX="tunatron-nano-4b"
BITS_LIST="4,6,8"
OUT_BASE_DIR="$ROOT_DIR/outputs/mlx"
NO_UPLOAD=0
DRY_RUN=0
HF_TOKEN_VAL="${HF_TOKEN:-}"

usage() {
  cat << 'EOF'
Uso: scripts/export_mlx.sh -s <MODELLO_SORGENTE> [OPZIONI]

Opzioni obbligatorie (se non presente un modello locale di default):
  -s, --source <PATH_O_HF_REPO>  Repository Hugging Face sorgente (es. 'Org/modello-bf16')
                                 oppure percorso locale (es. 'ai-pipeline/model_bf16').

Opzioni generali:
  -u, --upload-org <ORG_O_USER>  Account/Organizzazione Hugging Face dove pubblicare
                                 (es. 'BlockchainzGurus'). Se omesso e non è impostato
                                 --no-upload, viene ricavato dal repo sorgente o richiesto.
  -b, --bits <LISTA_BIT>         Lista di precisioni separate da virgola (default: "4,6,8").
  -p, --prefix <NOME_BASE>       Prefisso per i nuovi repo MLX (default: "tunatron-nano-4b").
                                 Il nome su HF sarà: <ORG>/<NOME_BASE>-mlx-<BIT>bit
  -o, --out-dir <DIR>            Cartella locale dove salvare i modelli (default: outputs/mlx).
  -t, --token <TOKEN>            Hugging Face Access Token (con permessi di scrittura).
      --no-upload                Esegui solo la conversione locale senza caricare su HF.
      --dry-run                  Mostra solo i comandi che verrebbero eseguiti.
  -h, --help                     Mostra questo messaggio di aiuto.
EOF
  exit 0
}

# Parsing argomenti
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source)
      SOURCE_MODEL="$2"; shift 2 ;;
    -u|--upload-org)
      UPLOAD_ORG="$2"; shift 2 ;;
    -b|--bits)
      BITS_LIST="$2"; shift 2 ;;
    -p|--prefix)
      REPO_NAME_PREFIX="$2"; shift 2 ;;
    -o|--out-dir)
      OUT_BASE_DIR="$2"; shift 2 ;;
    -t|--token)
      HF_TOKEN_VAL="$2"; shift 2 ;;
    --no-upload)
      NO_UPLOAD=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "Errore: opzione non riconosciuta: $1" >&2
      usage ;;
  esac
done

echo "======================================================================"
echo "    Tunatron — Esportazione MLX (Apple Silicon) a 4, 6 e 8 bit"
echo "======================================================================"

# 1. Verifica piattaforma
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" != "Darwin" ]]; then
  echo "⚠️  ATTENZIONE: Stai eseguendo questo script su $OS ($ARCH)."
  echo "    Apple MLX è progettato nativamente per macOS con processori Apple Silicon."
  echo "    Se intendi convertire per Mac, assicurati di eseguire questo script da macOS."
fi

# 2. Identificazione Python con mlx-lm
PYTHON_BIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import mlx_lm" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -f "$ROOT_DIR/.venv-mlx/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv-mlx/bin/python"
  elif [[ -f "$ROOT_DIR/.venv/bin/python" ]] && "$ROOT_DIR/.venv/bin/python" -c "import mlx_lm" >/dev/null 2>&1; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    PYTHON_BIN="python3"
    echo "ℹ [DRY-RUN] mlx-lm non presente in locale, uso '$PYTHON_BIN' a scopo dimostrativo"
  else
    echo "❌ Errore: mlx-lm non trovato nel sistema o nel virtualenv." >&2
    echo "   Per installarlo su macOS (Apple Silicon):" >&2
    echo "     python3 -m venv .venv-mlx" >&2
    echo "     source .venv-mlx/bin/activate" >&2
    echo "     pip install -U mlx-lm huggingface_hub" >&2
    exit 1
  fi
fi
echo "✓ Python con mlx-lm trovato: $PYTHON_BIN"

# 3. Risoluzione sorgente modello
if [[ -z "$SOURCE_MODEL" ]]; then
  # Cerca se esiste un modello locale in ai-pipeline/model_bf16 o model_bf16
  for cand in "$ROOT_DIR/ai-pipeline/model_bf16" "$ROOT_DIR/model_bf16"; do
    if [[ -d "$cand" ]]; then
      SOURCE_MODEL="$cand"
      echo "ℹ Nessun sorgente specificato: trovato modello BF16 locale in $SOURCE_MODEL"
      break
    fi
  done
fi

if [[ -z "$SOURCE_MODEL" ]]; then
  echo "❌ Errore: specifica il modello sorgente con -s / --source" >&2
  echo "   Esempio: scripts/export_mlx.sh -s BlockchainzGurus/tunatron-sast-nano-4b-bf16 -u BlockchainzGurus" >&2
  exit 1
fi

# 4. Verifica credenziali Hugging Face se upload richiesto
if [[ "$NO_UPLOAD" -eq 0 ]]; then
  if [[ -n "$HF_TOKEN_VAL" ]]; then
    export HF_TOKEN="$HF_TOKEN_VAL"
  fi

  # Se UPLOAD_ORG non è stato passato, proviamo a ricavarlo dalla sorgente se è nel formato "org/repo"
  if [[ -z "$UPLOAD_ORG" && "$SOURCE_MODEL" == *"/"* && ! -d "$SOURCE_MODEL" ]]; then
    UPLOAD_ORG="${SOURCE_MODEL%%/*}"
    echo "ℹ Organizzazione dedotta dal repository sorgente: $UPLOAD_ORG"
  fi

  if [[ -z "$UPLOAD_ORG" ]]; then
    echo "❌ Errore: per pubblicare su Hugging Face specifica l'utente o l'organizzazione con -u / --upload-org" >&2
    echo "   Oppure usa il flag --no-upload se desideri solo i file locali." >&2
    exit 1
  fi
fi

# Estrai i bit da processare
IFS=',' read -r -a BITS_ARRAY <<< "$BITS_LIST"
mkdir -p "$OUT_BASE_DIR"

echo "----------------------------------------------------------------------"
echo " Modello sorgente : $SOURCE_MODEL"
echo " Precisioni bit   : ${BITS_ARRAY[*]}"
echo " Cartella output  : $OUT_BASE_DIR"
if [[ "$NO_UPLOAD" -eq 0 ]]; then
  echo " Upload HuggingFace: SÌ (Destinazione: $UPLOAD_ORG/$REPO_NAME_PREFIX-mlx-<bit>bit)"
else
  echo " Upload HuggingFace: NO (--no-upload attivo)"
fi
echo "----------------------------------------------------------------------"

# 5. Esecuzione conversioni per ciascuna precisione
for bit in "${BITS_ARRAY[@]}"; do
  # Pulisci spazi
  bit="$(echo "$bit" | tr -d ' ')"
  [[ -z "$bit" ]] && continue

  TARGET_LOCAL_DIR="$OUT_BASE_DIR/${REPO_NAME_PREFIX}-mlx-${bit}bit"
  TARGET_HF_REPO="${UPLOAD_ORG}/${REPO_NAME_PREFIX}-mlx-${bit}bit"

  echo ""
  echo "▶ [1/2] Elaborazione conversione a ${bit}-bit..."
  echo "        Destinazione locale: $TARGET_LOCAL_DIR"

  CONVERT_CMD=(
    "$PYTHON_BIN" -m mlx_lm.convert
    --hf-path "$SOURCE_MODEL"
    --mlx-path "$TARGET_LOCAL_DIR"
    -q
    --q-bits "$bit"
  )

  if [[ "$NO_UPLOAD" -eq 0 ]]; then
    echo "        Repository HF destinazione: $TARGET_HF_REPO"
    CONVERT_CMD+=(--upload-repo "$TARGET_HF_REPO")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [DRY-RUN] Comand: ${CONVERT_CMD[*]}"
    continue
  fi

  # Esecuzione
  echo "  Esecuzione mlx_lm.convert in corso (download, quantizzazione e salvataggio)..."
  "${CONVERT_CMD[@]}"

  echo "  ✓ Conversione ${bit}-bit completata con successo!"
  if [[ "$NO_UPLOAD" -eq 0 ]]; then
    echo "  ✓ Modello pubblicato su: https://huggingface.co/$TARGET_HF_REPO"
  fi
done

echo ""
echo "======================================================================"
echo "🎉 Tutte le conversioni richieste sono state completate!"
echo "   File locali salvati in: $OUT_BASE_DIR"
if [[ "$NO_UPLOAD" -eq 0 ]]; then
  echo "   I modelli sono disponibili su Hugging Face Hub pronti per essere"
  echo "   usati su qualsiasi dispositivo Apple Silicon con:"
  echo ""
  for bit in "${BITS_ARRAY[@]}"; do
    bit="$(echo "$bit" | tr -d ' ')"
    [[ -z "$bit" ]] && continue
    echo "   mlx_lm.generate --model ${UPLOAD_ORG}/${REPO_NAME_PREFIX}-mlx-${bit}bit --prompt '...'"
  done
fi
echo "======================================================================"
