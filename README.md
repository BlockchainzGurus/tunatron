# Tunatron — SAST GraphRAG con Nemotron-3-Nano-4B (NVFP4) e motore in Rust

Pipeline SAST (Static Application Security Testing) locale ad alte prestazioni che combina analisi statica del codice e modelli di linguaggio: un microservizio Rust clona una repository, ne costruisce il **call graph** (tree-sitter + petgraph), individua i **sink** a rischio ed estrae per ciascuno il **backward slice** contestuale (profondità configurabile). Il contesto viene arricchito tramite GraphRAG con la tassonomia **CWE** e inviato a un modello **Nemotron-3-Nano-4B fine-tuned (DoRA) e quantizzato (FP8 / NVFP4)** servito tramite vLLM con *guided decoding*. L'output finale è un **SBOM CycloneDX con VEX** (`affected` / `not_affected`), notarizzabile su IPFS e smart contract EVM.

```
POST /analyze ─► ingestor ─► graph_slicer ─► rag_cwe ─► prompt ─► vllm_client ─► vex ─► (notarizer)
 (axum)          git clone    tree-sitter     scheda     Regola     N richieste    SBOM +    IPFS + EVM
                 + SBOM       + petgraph      CWE        + traccia  parallele,     VEX
                                                                    JSON Schema
```

---

## Indice

- [Requisiti di Sistema](#requisiti-di-sistema)
- [Struttura del Progetto](#struttura-del-progetto)
- [Guida Rapida](#guida-rapida)
- [Modelli e Quantizzazione](#modelli-e-quantizzazione-bf16-fp8-nvfp4)
- [Configurazione del Motore SAST](#configurazione-del-motore-sast)
- [Dataset e Metodologia](#dataset-e-metodologia)
- [Scelte Architetturali e Note Tecniche](#scelte-architetturali-e-note-tecniche)
- [Notarizzazione e Attestazione Verificabile](#notarizzazione-e-attestazione-verificabile)
- [Test Suite](#test-suite)
- [Limiti Noti](#limiti-noti)
- [Licenza](#licenza)

---

## Requisiti di Sistema

- **Sistema Operativo:** Linux (testato su Ubuntu 22.04 / 24.04 / 26.04 LTS).
- **GPU:** NVIDIA con supporto Tensor Core e almeno 16 GB di VRAM per training locale (es. RTX 4080/4090, RTX 5070 Ti architettura Blackwell `sm_120`, o GPU datacenter A100/H100/L40/RTX PRO 6000). Per ambienti con VRAM limitata è fornito un notebook per Google Colab.
- **CUDA & Driver:** Driver NVIDIA ≥ 550, CUDA Toolkit 12.8+ o 13.x.
- **Linguaggi & Toolchain:**
  - Python 3.12 (gestito con `uv` o `venv`)
  - Rust stable con `cargo` (≥ 1.80)
  - Git CLI
  - (Opzionale) Foundry / `solc` per il deploy dei contratti Solidity e nodo IPFS Kubo per la notarizzazione.

---

## Struttura del Progetto

```
Tunatron/
├── ai-pipeline/       # Pipeline Python: dataset, training DoRA, quantizzazione ModelOpt, benchmark
│   ├── data/          # Dataset train/val/test in formato JSONL
│   ├── tests/         # Suite di test unitari e di calibrazione
│   ├── prepare_data.py
│   ├── train.py
│   ├── quantize.py
│   └── benchmark.py
├── sast-engine/       # Motore SAST ad alte prestazioni in Rust (Axum, Tree-sitter, Petgraph)
│   ├── src/           # Ingestor, Graph Slicer, RAG CWE, VLLM client, CycloneDX/VEX, Notarizer
│   ├── tests/         # Test unitari e di integrazione end-to-end
│   └── cwe_kb.json    # Knowledge Base CWE integrata
├── contracts/         # Smart contract Solidity (AttestationRegistry.sol)
├── shared/            # Contratti e schemi condivisi (prompt_template.json, vex_schema.json)
├── notebooks/         # Notebook per esecuzione su ambienti cloud GPU (Google Colab)
└── scripts/           # Script di automazione per verifica, setup, serving ed esecuzione
```

---

## Guida Rapida

### 1. Verifica dell'Ambiente e Installazione

```bash
# Verifica compatibilità driver e configurazione CUDA
scripts/check_env.sh

# Installazione dei virtualenv Python (.venv, .venv-quant, .venv-vllm) sul PC; ENVS=quant per crearne uno solo.
# Su Google Colab non serve: il notebook ha già i suoi passi.
scripts/setup_env.sh
```

### 2. Preparazione Dataset e Addestramento (DoRA)

```bash
cd ai-pipeline

# Download e formattazione dataset da fonti pubbliche
python3 prepare_data.py --download --test-benign-ratio 3

# Verifica preliminare di dati e iperparametri
../.venv/bin/python train.py --dry-run

# Avvio fine-tuning DoRA a 16-bit (genera model_bf16/)
../.venv/bin/python train.py

cd ..
```

> **Esecuzione alternativa su cloud GPU:**
> Se non si dispone di 16 GB di VRAM in locale, eseguire `notebooks/train_colab.ipynb` su Google Colab (es. istanza con RTX PRO 6000 o A100): [aprilo in Colab](https://colab.research.google.com/github/BlockchainzGurus/tunatron/blob/master/notebooks/train_colab.ipynb). Il notebook clona questa repo da GitHub (impostazioni `REPO_URL`/`REPO_REF`), quindi addestra il codice già pubblicato con `git push`; i risultati (`model_bf16/`, `adapter/`, log) finiscono in `Il mio Drive/Tunatron/`. La sezione 10 del notebook esegue anche `quantize.py` (ModelOpt) e salva lì `model_fp8/` e `model_nvfp4/`.

### 3. Quantizzazione con NVIDIA ModelOpt (FP8 & NVFP4)

```bash
cd ai-pipeline

# Genera model_fp8/ e model_nvfp4/ a partire da model_bf16/
../.venv-quant/bin/python quantize.py

cd ..
```

### 4. Serving con vLLM e Avvio Motore SAST

```bash
# Avvio di vLLM con il modello quantizzato (porta :8000)
scripts/serve_vllm.sh model_nvfp4

# Verifica del guided decoding (la risposta deve validare lo schema JSON VEX)
scripts/test_guided_decoding.sh

# Avvio del microservizio Rust (porta :8080)
scripts/run_engine.sh

# Esecuzione di un'analisi di test
curl -X POST localhost:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{"repo_url":"https://github.com/we45/Vulnerable-Flask-App"}'
```

### 5. Benchmark e Valutazione

```bash
cd ai-pipeline

# Valutazione accuratezza (Semgrep vs LLM senza slice vs LLM con slice)
python3 benchmark.py accuracy --systems semgrep llm-noslice llm-slice

# Benchmark prestazionale hardware (confronto automatico BF16 / FP8 / NVFP4)
python3 benchmark.py hardware --models bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4

cd ..
```

---

## Modelli e Quantizzazione: BF16, FP8, NVFP4

La pipeline produce tre varianti dello stesso checkpoint, tutte compatibili e caricabili direttamente da vLLM:

| Directory | Precisione / Formato | Descrizione | Generato da |
|---|---|---|---|
| `model_bf16/` | BF16 (16-bit) | Modello fine-tuned con adapter DoRA fuso: riferimento qualitativo a piena precisione. | `train.py` |
| `model_fp8/` | FP8 W8A8 (E4M3) | Quantizzazione FP8 su pesi e attivazioni con KV cache FP8, allineata al checkpoint ufficiale NVIDIA. | `quantize.py` |
| `model_nvfp4/` | NVFP4 (W4A4) mixed | Pesi e attivazioni lineari in FP4 con scale a blocchi da 16; layer MLP critici preservati in FP8 per accelerazione su Tensor Core Blackwell. | `quantize.py` |

### Ricette di Quantizzazione ModelOpt

La quantizzazione è affidata a **NVIDIA Model Optimizer (ModelOpt)**, garantendo fedeltà rispetto alle configurazioni ufficiali Nemotron:

| Formato | Ricetta ModelOpt | Dettagli Architetturali |
|---|---|---|
| **NVFP4** | `models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16` | Moduli lineari (`in_proj`, `out_proj`, `up_proj`, `q/k/v/o_proj`) in **NVFP4 W4A4** (incluse attention); layer `down_proj` di 9 blocchi MLP (1, 3, 5, 8, 10, 18, 25, 33, 41) in **FP8 W8A8**; embedding in **NVFP4 W4A16**; `lm_head` in **FP8 W8A16**; `conv1d`, LayerNorm e parametri SSM in BF16. KV cache non quantizzata. |
| **FP8** | `general/ptq/fp8_default-kv_fp8` + moduli esclusi | FP8 W8A8 per i layer standard e KV cache in FP8. Esclusi in BF16: `lm_head`, `conv1d`, proiezioni Q/K/V/O dei 4 blocchi di attention e proiezioni `in_proj`/`out_proj` Mamba contigue. |

### Note di Esecuzione e Profilazione della Memoria

- **Embedding in BF16 per vLLM:** Qualora la versione di vLLM in uso presenti incompatibilità con `VocabParallelEmbedding` quantizzato in FP4, è possibile preservare la tabella di embedding in BF16 tramite il flag:
  ```bash
  ../.venv-quant/bin/python quantize.py --only nvfp4 --bf16-embeddings
  ```
- **KV Cache FP8:** Abilitabile facoltativamente in fase di serving con `scripts/serve_vllm.sh model_fp8 --kv-cache-dtype fp8`.
- **Budget VRAM in Training:** Su schede da 16 GB, la configurazione predefinita impiega batch size 1 e gradient accumulation 8 (batch effettivo 8), mantenendo l'allocazione entro ~13-14 GB. Per contesti con limiti più stringenti è possibile limitare la lunghezza del contesto tramite `--max-seq-length 4096`.

---

## Configurazione del Motore SAST

Le impostazioni del microservizio Rust sono configurabili tramite variabili d'ambiente:

| Variabile | Valore Predefinito | Descrizione |
|---|---|---|
| `SAST_BIND` | `127.0.0.1:8080` | Indirizzo di ascolto del server HTTP (Axum). |
| `VLLM_URL` | `http://localhost:8000` | Endpoint del server di inferenza vLLM. |
| `VLLM_MODEL` | `sast-nemotron` | Nome del modello registrato su vLLM (`--served-model-name`). |
| `VLLM_CONCURRENCY` | `8` | Numero massimo di richieste parallele inviate a vLLM. |
| `SLICE_DEPTH` | `3` | Profondità massima di backward slicing nel call graph. |
| `SLICE_MAX_CALLERS` | `4` | Numero massimo di chiamanti per nodo del grafo. |
| `SLICE_MAX_CHARS` | `12000` | Budget massimo di caratteri per la traccia di contesto nel prompt. |
| `MAX_SINKS` | `300` | Tetto massimo di sink analizzati per singola scansione. |
| `MAX_FILES` | `20000` | Numero massimo di file scansionati nel repository. |
| `MAX_FILE_BYTES` | `1000000` | Dimensione massima ammessa per singolo file sorgente (1 MB). |
| `ALLOWED_GIT_HOSTS` | `github.com` | Domini consentiti per l'ingestione Git (`https://<host>/<owner>/<repo>`). |
| `MAX_REPO_MB` | `500` | Limite massimo di spazio su disco per il repository clonato. |
| `GIT_CLONE_TIMEOUT_SECS` | `120` | Timeout per l'operazione di `git clone`. |
| `MAX_CONCURRENT_ANALYSES`| `1` | Numero massimo di analisi concorrenti attive. |
| `ANALYSIS_TIMEOUT_SECS` | `900` | Timeout totale dell'analisi (15 minuti). |
| `SYFT_BIN` | *(nessuno)* | Percorso facoltativo del binario `syft` per generare l'SBOM; in assenza, usa il parser di manifest integrato. |
| `CWE_KB_PATH` | *(incorporata)* | Percorso di una Knowledge Base CWE personalizzata in formato JSON. |
| `IPFS_API` | *(nessuno)* | Endpoint HTTP dell'API Kubo IPFS per la notarizzazione. |
| `ETH_RPC_URL` | *(nessuno)* | RPC Ethereum/EVM per la registrazione on-chain delle attestazioni. |
| `ATTEST_CONTRACT` | *(nessuno)* | Indirizzo dello smart contract `AttestationRegistry`. |
| `ETH_FROM` | *(nessuno)* | Indirizzo mittente per l'invio delle transazioni di attestazione. |
| `MODEL_HASH_WEIGHTS` | `0` | Se impostato a `1`, include l'hash completo dei pesi nella registrazione della *Model Provenance*. |

### Endpoint API REST

- `POST /analyze`
  - **Payload:** `{"repo_url": "https://github.com/owner/repo", "notarize": false}`
  - **Risposta:** `{ "summary": { ... }, "bom": { ... }, "errors": [ ... ], "attestation": { ... } }`
- `GET /health`
  - Controllo dello stato del servizio e connettività verso vLLM.

**Codici di errore HTTP:**
- `400 Bad Request`: URL del repository non valido o host non consentito.
- `429 Too Many Requests`: Limite di analisi concorrenti superato.
- `502 Bad Gateway`: Errore durante il clone o parsing del repository.
- `503 Service Unavailable`: Server vLLM non raggiungibile.
- `504 Gateway Timeout`: Scadenza del timeout globale di analisi.

---

## Dataset e Metodologia

Il modulo `prepare_data.py` genera esempi di addestramento strutturati nel formato chat `system` / `user` (regola CWE + sink localizzato + backward slice) / `assistant` (verdetto JSON VEX conforme a schema), garantendo perfetta parità con il template di inferenza del motore (`shared/prompt_template.json` e `shared/vex_schema.json`).

### Fonti Dati

| Fonte | Licenza | Contenuto | Mappatura Sink |
|---|---|---|---|
| **PrimeVul** | MIT | CVE reali in C/C++, coppie vulnerabile/corretta. | Sink di chiamata o livello di funzione. |
| **SecVulEval** | MIT | C/C++, ~10k coppie con commit di fix. | Localizzazione tramite diff del commit. |
| **CVEfixes + PyVul** | CC-BY-4.0 | Repository Python storici con `code_before` e `code_after`. | Derivato dalle modifiche apportate dalla fix. |
| **Juliet C/C++ 1.3** | CC0 | Casi sintetici mirati (SARD), impiegati esclusivamente in train/validation con etichetta esplicita `/* FLAW */`. | Annotazione esplicita nel sorgente. |
| **CodeQL Security Suite**| MIT | Test suite manuali in Python e C/C++ con tag `# $ Alert`. | Annotazione di riga. |

### Pulizia e Bilanciamento

1. **Prevenzione del Data Leakage:** Qualsiasi funzione presente nel test set (o la sua controparte corretta/vulnerabile) viene rimossa da training e validation set tramite hash crittografico del corpo della funzione.
2. **Deduplicazione e Ratio delle Classi:** Eliminazione dei duplicati esatti e bilanciamento del rapporto vulnerabile/benigno (`--label-ratio 1.25`), con priorità agli esempi estratti da repository reali rispetto a quelli sintetici.
3. **Controllo dei Function-Level Sink:** La quota di esempi senza sink esplicito è limitata al 50% (`--function-share 0.5`).
4. **Esclusione di Dataset Non Certificati:** Il dataset `CyberNative/Code_Vulnerability_Security_DPO` è stato escluso a causa dell'assenza di mappature CWE native e per problemi di validità delle annotazioni riscontrati negli audit ufficiali del dataset.

---

## Scelte Architetturali e Note Tecniche

- **Disattivazione Thinking Overhead:** Il template di default di Nemotron abilita il blocco di reasoning (`<think>`). Per azzerare la latenza superflua e garantire un output deterministico, il blocco viene impostato a vuoto (`<think></think>`) sia in fase di training sia in inferenza, e la loss è calcolata esclusivamente sul token stream del verdetto JSON (`train_on_responses_only`).
- **Target DoRA e Moduli Mamba:** L'adapter DoRA è applicato a `q/k/v/o_proj`, `up/down_proj` e al modulo Mamba `in_proj`. I layer `out_proj` e `conv1d` sono esclusi poiché i kernel fusi di Mamba accedono direttamente ai pesi sottostanti, ignorando l'adapter in PEFT.
- **Loss con LM Head Fusa:** Con il vocabolario esteso di Nemotron, il calcolo della loss sull'intero vocabolario genererebbe picchi di memoria incompatibili con 16 GB di VRAM. Il training impiega una testa LM fusa (`fused_compute_loss`) che calcola i logit e la cross-entropy unicamente per i token supervisionati.
- **Stabilità Numerica di DoRA:** È stata integrata una soglia minima (`patch_dora_dead_rows`) nel calcolo di `magnitude / weight_norm` per gestire tensori a norma zero presenti in alcuni layer `in_proj` di Mamba, prevenendo gradienti NaN durante l'addestramento.
- **Isolamento degli Ambienti Virtuali:** Per gestire requisiti divergenti di PyTorch, Transformers e kernel CUDA compilati (`mamba-ssm`, `causal-conv1d`), la pipeline prevede tre ambienti virtuali dedicati: `.venv` (training con Unsloth), `.venv-quant` (quantizzazione con NVIDIA ModelOpt) e `.venv-vllm` (serving ad alte prestazioni). I due kernel si scaricano da PyPI con `-std=c++17` portato a `c++20` (`ai-pipeline/patched_sdist.py`: torch ≥ 2.14 non compila con C++17) e si costruiscono uno alla volta, con tetto di RAM; torch e lo stack di training sono fissati alle versioni provate (`PIN_VERSIONS=0` per le ultime).
- **Architettura Rust Senza Dipendenze Monolitiche:** Il microservizio Rust interagisce con Git tramite chiamate dirette al binario di sistema (con timeout e quote di dimensione) e implementa un encoder ABI / JSON-RPC nativo per le chiamate EVM, eliminando dipendenze ingombranti quali `libgit2` o interi SDK Web3.
- **Specifica CycloneDX VEX:** Le segnalazioni sono mappate secondo lo standard CycloneDX (`analysis.state = "exploitable"` per sink vulnerabili, `"not_affected"` per sink sicuri, `"in_triage"` per casi ambigui), mantenendo lo stato dettagliato nelle properties del componente.
- **Determinismo delle Coordinate del Sink:** File, riga e funzione del sink riportati nell'output finale sono ancorati ai metadati estratti staticamente da Tree-sitter, evitando possibili discrepanze o allucinazioni di coordinate da parte dell'LLM.

---

## Notarizzazione e Attestazione Verificabile

Il microservizio include un modulo opzionale per la notarizzazione dei report su rete decentralizzata:

1. **Storage Decentrato su IPFS:** Il report CycloneDX JSON (depurato dei metadati di attestazione successivi) viene caricato su IPFS tramite l'API Kubo. L'hash SHA-256 del payload IPFS corrisponde esattamente al parametro `report_sha256` salvato on-chain.
2. **Model Provenance:** Lo smart contract registra un hash crittografico che vincola l'identità del modello utilizzato: configurazione (`config.json`), tokenizer, parametri di generazione (`generation_config.json`), ricetta di quantizzazione (`hf_quant_config.json`) e opzionalmente i pesi (`MODEL_HASH_WEIGHTS=1`).
3. **Registro On-Chain:** Il contratto Solidity `AttestationRegistry` memorizza l'attestazione impedendo registrazioni duplicate per lo stesso report.

---

## Test Suite

La correttezza dei componenti statici e della pipeline AI è validata tramite suite di test automatizzate:

```bash
# Esecuzione dei test unitari e di integrazione del motore SAST (Rust)
cargo test --manifest-path sast-engine/Cargo.toml

# Verifica della pipeline Python (eseguibile con sola stdlib o nel venv)
cd ai-pipeline
python3 -m unittest discover -s tests -t .
cd ..
```

---

## Limiti Noti

- **Risoluzione Call Graph:** L'analisi del grafo delle chiamate si basa su euristiche di corrispondenza per nome (senza analisi approfondita dei tipi o dei puntatori per C/C++); chiamate a livello di modulo globale non vengono attualmente tracciate come sink.
- **Linguaggi Supportati:** C, C++ e Python.
- **Generazione SBOM:** La generazione SBOM da manifest analizza file standard di dipendenze (`requirements.txt`, `Cargo.lock`, ecc.); per scansioni binarie o alberi complessi di dipendenze è raccomandato configurare `syft` tramite `SYFT_BIN`.
- **Requisiti Notarizzazione:** La funzione di notarizzazione richiede un nodo IPFS locale o remoto raggiungibile e un RPC EVM con account sbloccato per la firma delle transazioni.

---

## Licenza

Questo progetto è rilasciato sotto licenza **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**.

La licenza AGPLv3 è una licenza open source a forte copyleft progettata specificamente per software di rete e servizi: chiunque modifichi, distribuisca o eroghi Tunatron (o sue opere derivate) come servizio accessibile via rete è tenuto a rendere disponibile il codice sorgente completo delle modifiche con la medesima licenza.

Consulta il file [LICENSE](file:///home/gabs/Documenti/Università/Software%20Security%20&%20Blockchain/Progetto/Tunatron/LICENSE) per il testo completo.
