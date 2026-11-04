# Piano di Implementazione: SAST GraphRAG + Nemotron-3-Nano-4B (NVFP4) + Rust

## Architettura Hardware & Software Target
- **OS:** Ubuntu 26.04 LTS
- **CPU / RAM:** AMD Ryzen 9 9900X / 32 GB DDR5
- **GPU:** NVIDIA RTX 5070 Ti (16 GB VRAM, Architettura Blackwell `sm_120`)
- **CUDA:** Toolkit 13.5 / Driver API 13.2 (`TORCH_CUDA_ARCH_LIST="12.0"`)
- **Stack AI:** Unsloth (DoRA 16-bit) -> NVIDIA ModelOpt / LLM-Compressor (NVFP4 & FP8) -> vLLM
- **Stack Backend:** Rust (`axum`, `tree-sitter`, `petgraph`, `reqwest`, `tokio`)

---

## Checklist Operativa

> Legenda: `[x]` fatto e verificato · `[ ]` codice pronto ma **non eseguito** (richiede GPU, `sudo` o tempo macchina) — vedi README.md, sezione *Stato*.

### Fase 1: Setup Ambiente e Dataset
- [ ] **1.1 Allineamento CUDA 13.5 / Driver 13.2 su Ubuntu 26.04:**
  - Verifica forward compatibility: `sudo apt install cuda-compat-13-5` oppure aggiorna i driver NVIDIA per allinearli a CUDA 13.5.
  - *Stato:* toolkit 13.5 e driver 595 già presenti; mancano `cuda-compat-13-5` (sudo) e gli export. `scripts/check_env.sh` li stampa; `~/.bashrc` **non è stato modificato**.
  - Esporta nel `.bashrc`:
    ```bash
    export PATH=/usr/local/cuda-13.5/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.5/compat:/usr/local/cuda-13.5/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="12.0"
    ```
- [ ] **1.2 Creazione Virtualenv Python (uv) e compilazione kernel Mamba-2 per Blackwell:**
  - Crea ambiente: `uv venv --python 3.12 && source .venv/bin/activate`
  - Installa PyTorch compatibile CUDA 13+, Unsloth, TRL, `mamba-ssm`, `causal-conv1d`, `llm-compressor` e `vllm`.
  - *Stato:* `scripts/setup_env.sh` pronto (provato solo con `DRY_RUN=1`; due venv; compilazione kernel con tetto di RAM).
- [x] **1.3 Creazione Dataset (`prepare_data.py`):**
  - Scarica il dataset base (*PrimeVul* o *Juliet Test Suite*).
  - Estrai e formatta gli esempi nel file `train.jsonl` e `val.jsonl` seguendo lo schema `Slicing + CWE -> JSON VEX`.
  - *Stato:* 5 fonti (PrimeVul, SecVulEval, CVEfixes+PyVul, Juliet, test CodeQL); `train.jsonl` 5.985 (41% con vera chiamata-sink) · `val.jsonl` 402 · `test.jsonl` 3.266 esempi. Dettagli e fonti scartate nel README.

### Fase 2: Fine-Tuning (DoRA 16-bit) e Quantizzazione (NVFP4 / FP8)
- [ ] **2.1 Training con Unsloth (`train.py`):**
  - Carica `unsloth/Nemotron-3-Nano-4B` in BF16 (`load_in_16bit=True`, `full_finetuning=False`, `max_seq_length=8192`).
  - Applica DoRA (`r=64`, `lora_alpha=64`, `use_dora=True`) sui target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj, out_proj`.
  - Avvia `SFTTrainer` con `optim="adamw_8bit"`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `use_gradient_checkpointing="unsloth"`.
  - *Stato:* `train.py` pronto (`--dry-run` provato sui dati reali); non addestrato. *Deviazione:* target senza `out_proj`/`conv1d` (PEFT li rifiuta nei modelli Mamba) e senza `gate_proj` (non esiste in NemotronH); DoRA verificato su un NemotronH minuscolo su CPU.
  - Esporta il modello fuso in BF16: `model.save_pretrained_merged("./model_bf16", tokenizer, save_method="merged_16bit")`.
- [ ] **2.2 Quantizzazione Hardware Blackwell (`quantize.py`):**
  - Usa `llm-compressor` (o `nvidia-modelopt`) con 512 campioni di calibrazione presi da `train.jsonl`.
  - Esporta il checkpoint `./model_fp8` (formato FP8 dinamico/statico).
  - *Stato:* `quantize.py` pronto; non eseguito. Usa **nvidia-modelopt ≥ 0.48** (non llm-compressor): per NVFP4 la ricetta ufficiale NVIDIA per il 4B (`models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16`), per FP8 la ricetta generica + le esclusioni del checkpoint NVIDIA 4B-FP8 (verificate contro la lista pubblicata). Produce `model_fp8/` e `model_nvfp4/` dal `model_bf16/`.
  - Esporta il checkpoint `./model_nvfp4` (formato NVFP4 nativo per Tensor Core di 5a generazione).

### Fase 3: Motore di Inferenza Locale (vLLM)
- [ ] **3.1 Avvio Server vLLM:**
  - *Stato:* `scripts/serve_vllm.sh` (nessun flag `--quantization`: vLLM riconosce i checkpoint ModelOpt, vedi README); non avviato.
  - Lancia il server locale sulla porta 8000:
    ```bash
    vllm serve ./model_nvfp4 \
      --quantization modelopt \
      --gpu-memory-utilization 0.80 \
      --max-model-len 8192 \
      --port 8000
    ```
- [ ] **3.2 Test Guided Decoding:**
  - *Stato:* `scripts/test_guided_decoding.sh` pronto e provato contro un server finto (OK/KO); da lanciare con vLLM reale.
  - Verifica con una chiamata `curl` che il parametro `response_format` (JSON Schema) forzi Nemotron a restituire esclusivamente il JSON valido della vulnerabilità.

### Fase 4: Microservizio Core in Rust (`sast-engine`)
- [x] **4.1 Inizializzazione Progetto Cargo:**
  - `cargo new sast-engine`
  - Dipendenze in `Cargo.toml`: `axum`, `tokio`, `serde`, `serde_json`, `reqwest`, `tree-sitter`, `petgraph`, `git2` (o command `git`), `sha2`.
- [x] **4.2 Modulo `ingestor.rs` (Git Clone + SBOM Base):**
  - Riceve l'URL della repository GitHub via POST `/analyze`.
  - Clona la repository in `/tmp/repo_<uuid>`.
  - Genera l'SBOM base in formato CycloneDX JSON (tramite parser dei manifest o wrapper `syft`).
- [x] **4.3 Modulo `graph_slicer.rs` (`tree-sitter` + `petgraph`):**
  - Esegue il parsing AST di tutti i file sorgente della repository.
  - Costruisce in RAM il Call Graph direzionato con `petgraph`.
  - Identifica i nodi **Sink** (query SQL, `exec`/`system`, `memcpy`/`strcpy`, deserializzazioni, API di librerie esterne vulnerabili).
  - Per ogni Sink, esegue una visita all'indietro (`petgraph::Direction::Incoming`, max profondità 3) per estrarre la catena di funzioni chiamanti (Backward Slice).
- [x] **4.4 Modulo `rag_cwe.rs` (Arricchimento Contesto):**
  - Associa al tipo di Sink individuato la scheda tecnica della relativa CWE da un dizionario JSON locale in-memory.
- [x] **4.5 Modulo `vllm_client.rs` (Batching Asincrono):**
  - Costruisce i prompt unendo `Regola CWE + Traccia Codice Estratta`.
  - Invia tutte le richieste in parallelo a `http://localhost:8000/v1/chat/completions` con `temperature: 0.0` e schema JSON forzato.
  - Filtra i risultati e aggiorna l'SBOM generando il report **VEX** (`affected` / `not_affected`).
- [x] **4.6 Modulo `notarizer.rs` (Opzionale per Paper - Blockchain & IPFS):**
  - Carica il file JSON SBOM+VEX su IPFS ottenendo il `CID`.
  - Calcola l'hash `SHA-256` del report e l'hash di configurazione del modello (`Model Provenance`).
  - Registra l'attestazione su uno Smart Contract Solidity locale (Anvil/Hardhat).
  - *Stato:* contratto compilato e calldata Rust verificato in EVM; non provato su Anvil/IPFS reali.

### Fase 5: Benchmark e Valutazione Sperimentale (`benchmark.py`)
- [ ] **5.1 Valutazione Accuratezza SAST:**
  - *Stato:* `benchmark.py accuracy` pronto (Semgrep + due modalità LLM, sottoinsieme `call_sinks`); non eseguito.
  - Esegui la pipeline su un test set di repository/funzioni non viste durante il training.
  - Calcola **Precision, Recall, F1-Score e False Positive Rate (FPR)** confrontando:
    1. Scanner SAST tradizionale (Semgrep / Bandit)
    2. Nemotron-3-Nano-4B senza Graph Slicing
    3. Nemotron-3-Nano-4B con Rust Graph Slicing + Fine-Tuning
- [ ] **5.2 Benchmark Hardware Blackwell (RTX 5070 Ti):**
  - *Stato:* `benchmark.py hardware` pronto; non eseguito.
  - Misura e tabula per **BF16 vs FP8 vs NVFP4**:
    - VRAM allocata (GB)
    - Throughput di generazione (tokens/sec)
    - Tempo medio di scansione per repository (secondi)
    - Variazione di F1-Score dovuta alla quantizzazione

---

## Struttura delle Directory del Progetto

```text
sast-nemotron-project/
├── ai-pipeline/
│   ├── data/
│   │   ├── raw/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   ├── prepare_data.py      # Conversione dataset in formato Slicing -> JSON VEX
│   ├── train.py             # Fine-tuning Unsloth DoRA 16-bit + Merge BF16
│   ├── quantize.py          # Export in FP8 e NVFP4 per RTX 5070 Ti
│   └── benchmark.py         # Script per metriche F1, VRAM e tok/s
└── sast-engine/             # Microservizio Rust
    ├── Cargo.toml
    ├── cwe_kb.json          # Knowledge base locale delle CWE
    └── src/
        ├── main.rs          # Server HTTP Axum
        ├── ingestor.rs      # Clone GitHub repo + generazione SBOM CycloneDX
        ├── graph_slicer.rs  # Parsing tree-sitter + slicing su petgraph
        ├── rag_cwe.rs       # Matching Sink -> Regola CWE
        ├── vllm_client.rs   # Chiamate parallele asincrone a vLLM
        └── notarizer.rs     # (Opzionale) IPFS + Smart Contract attestation
```

---

## Schema JSON di Output del Modello (Guided Decoding)

```json
{
  "vulnerable": true,
  "cwe_id": "CWE-89",
  "severity": "HIGH",
  "sink_function": "cursor.execute",
  "sink_file": "src/db/orders.py",
  "sink_line": 45,
  "vex_status": "affected",
  "reasoning": "Il parametro user_id proviene da request.args in routes.py (linea 12) e viene concatenato tramite f-string in utils.py (linea 88) senza parametrizzazione prima di raggiungere cursor.execute.",
  "suggested_fix": "Sostituire la f-string con una query parametrizzata: cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))"
}
```
