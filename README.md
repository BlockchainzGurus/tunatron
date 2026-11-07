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
│   ├── model_card.py  # MODEL_CARD.md con grafici (in inglese) da log di training e risultati del benchmark
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

# Esempi con slice reali del grafo (Juliet C, test CodeQL Python, pilota CVEfixes/PyVul, sintetici Python):
# vedi "Dati con slice reali" sotto. Il motore va compilato (cargo build --release) e i comandi girano sotto un tetto di memoria.
# python3 build_graph_data.py juliet --max-train 1080 && python3 build_graph_data.py codeql
# python3 pyvul_pilot.py --parquet data/raw/cvefixes_pyvul.parquet --repos 20 --work-dir outputs/pyvul-pilot
# python3 build_graph_data.py synth     # per ultimo: il tetto dipende dagli esempi Python veri già scritti
# python3 prepare_data.py ... --graph-dir data/graph --out-dir data

# Verifica preliminare di dati e iperparametri
../.venv/bin/python train.py --dry-run

# Avvio fine-tuning DoRA a 16-bit (genera model_bf16/)
../.venv/bin/python train.py

cd ..
```

> **Esecuzione alternativa su cloud GPU:**
> Se non si dispone di 16 GB di VRAM in locale, eseguire `notebooks/train_colab.ipynb` su Google Colab (es. istanza con RTX PRO 6000 o A100): [aprilo in Colab](https://colab.research.google.com/github/BlockchainzGurus/tunatron/blob/master/notebooks/train_colab.ipynb). Il notebook clona questa repo da GitHub (impostazioni `REPO_URL`/`REPO_REF`), quindi addestra il codice già pubblicato con `git push`; i risultati (`model_bf16/`, `adapter/`, log) finiscono in `Il mio Drive/Tunatron/`. La sezione 10 del notebook esegue anche `quantize.py` (ModelOpt) e salva lì `model_fp8/` e `model_nvfp4/`. La sezione 11 esegue il benchmark vLLM (sempre l'ultima versione, in un ambiente a parte) e salva in `Il mio Drive/Tunatron/` il file `MODEL_CARD.md` con i grafici (`model_card_assets/`) e i risultati grezzi (`benchmark_results/`).

### 3. Quantizzazione con NVIDIA ModelOpt (FP8 & NVFP4)

```bash
cd ai-pipeline

# Genera model_fp8/ e model_nvfp4/ a partire da model_bf16/ (--bf16-embeddings e --bf16-lm-head: con vLLM 0.33 l'NVFP4 funziona solo così, vedi "Note di Esecuzione")
../.venv-quant/bin/python quantize.py --bf16-embeddings --bf16-lm-head

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

# I chiamanti servono? Lo stesso modello con la catena fino a --depth e con il solo blocco del sink, sulle coppie di Juliet
python3 benchmark.py accuracy --test data/eval/juliet_pairs.jsonl --systems llm-graph llm-sink-only --depth 5

# Falsi positivi su pygoat (lista congelata di 80 sink), risposte troncate e CVE inventate
python3 benchmark.py frozen --repo outputs/repos/pygoat/repo --depth 5

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
| **NVFP4** | `models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16` | Moduli lineari (`in_proj`, `out_proj`, `up_proj`, `q/k/v/o_proj`) in **NVFP4 W4A4** (incluse attention); layer `down_proj` di 9 blocchi MLP (1, 3, 5, 8, 10, 18, 25, 33, 41) in **FP8 W8A8**; embedding in **NVFP4 W4A16** (in **BF16** con `--bf16-embeddings`, vedi sotto); `lm_head` in **FP8 W8A16** (in **BF16** con `--bf16-lm-head`, vedi sotto); `conv1d`, LayerNorm e parametri SSM in BF16. KV cache non quantizzata. |
| **FP8** | `general/ptq/fp8_default-kv_fp8` + moduli esclusi | FP8 W8A8 per i layer standard e KV cache in FP8. Esclusi in BF16: `lm_head`, `conv1d`, proiezioni Q/K/V/O dei 4 blocchi di attention e proiezioni `in_proj`/`out_proj` Mamba contigue. |

### Note di Esecuzione e Profilazione della Memoria

- **Embedding in BF16 per vLLM:** vLLM 0.33 non carica l'NVFP4 con gli embedding quantizzati della ricetta NVIDIA: il suo `VocabParallelEmbedding` è sempre non quantizzato (3136 colonne) e fallisce sui pesi impacchettati (1568 colonne) con `RuntimeError: The size of tensor a (3136) must match the size of tensor b (1568)`. Per questo il notebook ha `QUANT_BF16_EMBEDDINGS = True`, che tiene la tabella di embedding in BF16 (circa +0,6 GB su disco); da riga di comando:
  ```bash
  ../.venv-quant/bin/python quantize.py --only nvfp4 --bf16-embeddings
  ```
  Senza il flag `quantize.py` produce la ricetta NVIDIA invariata, utile solo con un vLLM che supporti gli embedding quantizzati.
- **`lm_head` in BF16 per vLLM:** nel checkpoint della ricetta NVIDIA la `lm_head` è in FP8 W8A16 e si esporta con `weight` e `weight_scale` ma **senza `input_scale`** (gli altri 9 layer FP8, i `down_proj`, ce l'hanno). Il loader ModelOpt di vLLM 0.33.0 legge ogni layer `FP8` come W8A8 statico: registra `input_scale` inizializzato a `torch.finfo(torch.float32).min` e, non trovandolo nel file, lo lascia a quel valore. Il checkpoint si carica ma la misura dava 200 risposte su 200 non valide (JSON mai chiuso, ogni richiesta fermata a `max_tokens`, F1 0,000), mentre l'FP8 (che ha la `lm_head` in BF16) risponde normalmente. Per questo il notebook ha `QUANT_BF16_LM_HEAD = True` (circa +0,4 GB su disco); da riga di comando:
  ```bash
  ../.venv-quant/bin/python quantize.py --only nvfp4 --bf16-embeddings --bf16-lm-head
  ```
  Senza il flag `quantize.py` produce la ricetta NVIDIA invariata. Confermato dalla misura: con la `lm_head` in BF16 l'NVFP4 risponde normalmente (F1 0,689, latenza media 1,37 s, 3,4 GB di pesi e 3,6 GB su disco, contro 0,667 del BF16 e 0,694 dell'FP8 nella stessa corsa su 200 esempi).
- **Pubblicazione su Hugging Face:** il modello base è sotto la *NVIDIA Nemotron Open Model License*, che chiede di dare ai destinatari una copia della licenza: in ogni repo va caricato il file `LICENSE` del modello base (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) accanto al `README.md` (la model card, con `license_name` e `license_link` nell'intestazione). Un repo per formato: `model_bf16`, `model_fp8`, `model_nvfp4`.
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
| **CVEfixes + PyVul** | CVEfixes: dati CC-BY-4.0, codice MIT. **PyVul: nessuna licenza dichiarata** (nel repo non c'è un file LICENSE; la CC BY-NC-ND vista su arXiv è del paper, non dei dati) | Repository Python storici con `code_before` e `code_after`. | Derivato dalle modifiche apportate dalla fix. |
| **Juliet C/C++ 1.3** | Non coperto da copyright negli USA; CC0 1.0 dove NIST ha diritti all'estero | Casi sintetici mirati (SARD) per le CWE 120, 78, 190, 416, usati con le slice reali del grafo (vedi sotto); il commento `FLAW` non si usa e il codice è anonimizzato. | Chiamate del catalogo; etichetta dal nome originale della funzione (`bad` / `good*`). |
| **CodeQL Security Suite**| MIT | Test suite manuali in Python e C/C++ con tag `# $ Alert`. I Python hanno anche la versione con slice (traccia P1). | Annotazione di riga. |

Licenze verificate alla fonte il 2026-11-06 (le dichiarano i metadati o i repository dei dataset); sono riprese, con i riferimenti da citare, nella sezione *License and attributions* della model card (`DATA_LICENSES` in `ai-pipeline/model_card.py`). Il codice dei dataset viene da progetti open source e ogni frammento resta sotto la licenza del progetto d'origine.

### Pulizia e Bilanciamento

1. **Prevenzione del Data Leakage:** Qualsiasi funzione presente nel test set (o la sua controparte corretta/vulnerabile) viene rimossa da training e validation set tramite hash crittografico del corpo della funzione.
2. **Deduplicazione e Ratio delle Classi:** Eliminazione dei duplicati esatti e bilanciamento del rapporto vulnerabile/benigno (`--label-ratio 1.25`), con priorità agli esempi estratti da repository reali rispetto a quelli sintetici.
3. **Controllo dei Function-Level Sink:** La quota di esempi senza sink esplicito è limitata al 50% (`--function-share 0.5`); gli esempi con slice sono tutti a sink-chiamata e contano per questa quota, ma non vengono mai tagliati né ribilanciati da `prepare_data.py`.
4. **Target senza CVE:** il `reasoning` e il `suggested_fix` non citano id CVE, descrizioni CVE né messaggi di commit (nelle prime versioni il modello imparava a inventare CVE); la correzione suggerita per i vulnerabili è il `fix_hint` del catalogo.
5. **Esclusione di Dataset Non Certificati:** Il dataset `CyberNative/Code_Vulnerability_Security_DPO` è stato escluso a causa dell'assenza di mappature CWE native e per problemi di validità delle annotazioni riscontrati negli audit ufficiali del dataset.

### Dati con slice reali (chiamanti)

Gli esempi con slice usano le backward slice vere del motore (`sast-engine slice`, lo stesso `build_graph` / `CodeGraph::slice` del server): il prompt mostra la catena dei chiamanti dal più esterno al sink, fino a profondità 5, e il verdetto dipende da ciò che i chiamanti passano. Le tracce:

| Traccia (`meta.track`) | Origine | Cosa insegna |
|---|---|---|
| `juliet` | Juliet 1.3, CWE 120/78/190/416, codice anonimizzato (nomi, righe, file, commenti, stringhe) | Il verdetto sta nei chiamanti: gemelli `bad` / `good*` con lo stesso corpo del sink. |
| `py_codeql` | Test CodeQL Python (CWE a flusso di dati), etichetta dai `$ Alert` | Funzioni singole, senza chiamanti: ragionare dentro la funzione. |
| `py_pyvul` | Pilota CVEfixes/PyVul su 20 repo, checkout dei commit | Genitore della fix = vulnerabile, commit di fix = corretto, chiamanti veri. |
| `py_synth` | Pacchetti Python sintetici a coppie di gemelli (catena di 0-3 funzioni) | Il chiamante in cima cambia, il sink no; negativi costanti, controllati, con sorgente non usata. Al più il 15% degli esempi Python con slice, mai nella valutazione. |

Conteggi dei dati generati (`ai-pipeline/data/`, seme 42):

| Insieme | Esempi | Per fonte (`meta.source`) |
|---|---|---|
| `train.jsonl` | 5.261 | primevul 1.190, secvuleval 2.021, cvefixes-python 373, pyvul 149, codeql-tests 274, **juliet-nist 1.080** (20,5%), **codeql-python 94**, **pyvul-graph 54**, **py-synth 26** |
| `val.jsonl` | 369 | primevul 134, secvuleval 120, cvefixes-python 13, pyvul 11, codeql-tests 23, juliet-nist 64, codeql-python 2, pyvul-graph 2 |
| `test.jsonl` | 3.266 | invariato nella composizione: secvuleval 1.622, primevul 1.464, cvefixes-python 116, pyvul 50, codeql-tests 14 |

Ogni esempio con slice ha in `meta.trace` la traccia completa: `{"cwe_id", "sink": {"function", "file", "line", "cwe_id"}, "blocks": [{"file", "name", "start_line", "end_line", "depth", "lang", "code"}, ...]}` (blocchi dal chiamante più esterno al sink). `meta.view` è `callers` (tutti i blocchi) o `sink_only` (il solo blocco a profondità 0, ammesso solo se quel prompt non compare con l'etichetta opposta), `meta.pair_id` lega i gemelli che hanno lo stesso prompt solo-sink con etichette opposte. `benchmark.py` rifà il prompt da `meta.trace` a qualsiasi profondità.

File di valutazione in `ai-pipeline/data/eval/`:

| File | Contenuto |
|---|---|
| `juliet_test.jsonl` (600) | Test case di Juliet mai visti (divisione per test case). |
| `juliet_unseen_variants.jsonl` (600) | Varianti di flusso intere tenute fuori (12, 14, 15, 52): forme di catena mai viste. |
| `juliet_pairs.jsonl` (600 = 300 coppie) | Gemelli con lo stesso prompt solo-sink e verdetti opposti: misurano se i chiamanti contano (S1). |
| `py_codeql.jsonl` (17) | File CodeQL Python che `prepare_data.py` tiene nel test; sottile per costruzione (2 file, CWE 78/94). |
| `py_cvefixes.jsonl` (16) | Coppie dei commit che stanno nel test di `prepare_data.py` (divisione per commit). |
| `pygoat_frozen.json` (80 sink) | Lista letta a mano su pygoat `2aa4cc1`: 69 sicuri certi (62 `redirect` con destinazione costante, query e `subprocess` costanti, URL da impostazioni) e 11 vulnerabili verificati; serve per i falsi positivi (S3) e per le risposte troncate o con CVE inventate (S4). |

Rigenerazione completa (dalla cartella `ai-pipeline`, motore compilato, tutto sotto `systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0`):

```bash
python3 build_graph_data.py juliet --raw-dir data/raw --out-dir data/graph --eval-dir data/eval --max-train 1080
python3 build_graph_data.py codeql --raw-dir data/raw --out-dir data/graph --eval-dir data/eval
uv run --no-project --with pyarrow python pyvul_pilot.py --parquet data/raw/cvefixes_pyvul.parquet --repos 20 \
    --work-dir outputs/pyvul-pilot --out-dir data/graph --eval-dir data/eval
python3 build_graph_data.py synth --out-dir data/graph --n 400
uv run --no-project --with pyarrow python prepare_data.py --train data/raw/primevul_train.jsonl \
    --valid data/raw/primevul_valid.jsonl --test data/raw/primevul_test.jsonl --secvuleval data/raw/secvuleval.parquet \
    --python-pairs data/raw/cvefixes_pyvul.parquet --codeql data/raw/codeql-tests --test-benign-ratio 3 \
    --graph-dir data/graph --out-dir data
python3 leak_audit.py --train data/graph/juliet_train.jsonl --heldout data/eval/juliet_test.jsonl   # esce con 1 se ci sono perdite
```

I prompt, le schede delle CWE e i suggerimenti di correzione sono in inglese (`shared/prompt_template.json` versione 2): un modello addestrato sui dati precedenti ha visto solo prompt italiani, quindi va riaddestrato dal modello base.

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
- **Dati con slice, limiti misurati:** (a) la quota di esempi «solo sink» è lo 0,7% del train con slice (Juliet 0,2%, pilota PyVul 13%) contro il 30% voluto: quasi tutti i prompt solo-sink sono gemelli di uno con l'etichetta opposta e quindi non decidibili; (b) in Juliet il vulnerabile è quasi sempre il più lungo (un classificatore sulla sola lunghezza arriva a ~0,68): `benchmark.py` riporta S1 anche a lunghezza controllata e la base «il più lungo è il vulnerabile», e senza quelle due righe S1 non prova il ragionamento sul flusso; (c) le slice delle funzioni Flask/Django non hanno chiamanti (il chiamante è il framework): `py_codeql` ha 0% di slice con chiamanti; (d) il pilota CVEfixes/PyVul ha reso 38 coppie da 167 righe (22,75%) su 20 repo, ed è fermo lì in attesa di una decisione; (e) la valutazione `py_codeql` ha solo 17 esempi in 2 file.
- **Generazione SBOM:** La generazione SBOM da manifest analizza file standard di dipendenze (`requirements.txt`, `Cargo.lock`, ecc.); per scansioni binarie o alberi complessi di dipendenze è raccomandato configurare `syft` tramite `SYFT_BIN`.
- **Requisiti Notarizzazione:** La funzione di notarizzazione richiede un nodo IPFS locale o remoto raggiungibile e un RPC EVM con account sbloccato per la firma delle transazioni.

---

## Licenza

Questo progetto è rilasciato sotto licenza **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**.

La licenza AGPLv3 è una licenza open source a forte copyleft progettata specificamente per software di rete e servizi: chiunque modifichi, distribuisca o eroghi Tunatron (o sue opere derivate) come servizio accessibile via rete è tenuto a rendere disponibile il codice sorgente completo delle modifiche con la medesima licenza.

Consulta il file [LICENSE](file:///home/gabs/Documenti/Università/Software%20Security%20&%20Blockchain/Progetto/Tunatron/LICENSE) per il testo completo.
