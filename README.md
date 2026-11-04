# Tunatron — SAST GraphRAG con Nemotron-3-Nano-4B (NVFP4) e motore in Rust

Pipeline SAST locale: un microservizio Rust clona una repository, ne costruisce il **call graph** (tree-sitter + petgraph),
individua i **sink** pericolosi, estrae per ciascuno il **backward slice** (profondità 3), arricchisce il prompt con la scheda
della **CWE** e chiede il verdetto a un **Nemotron-3-Nano-4B fine-tuned (DoRA) e quantizzato (FP8 / NVFP4)** servito da vLLM
con *guided decoding*. L'output è un **SBOM CycloneDX con VEX** (`affected` / `not_affected`), notarizzabile su IPFS + smart contract.

```
POST /analyze ─► ingestor ─► graph_slicer ─► rag_cwe ─► prompt ─► vllm_client ─► vex ─► (notarizer)
 (axum)          git clone    tree-sitter     scheda     Regola     N richieste    SBOM +    IPFS + Anvil
                 + SBOM       + petgraph      CWE        + traccia  parallele,     VEX
                                                                    JSON Schema
```

## Stato: cosa è stato verificato e cosa no

| Parte | Stato |
|---|---|
| `sast-engine` (Rust) | **102 test verdi**, clippy senza avvisi. Provato su repo reali via `https` (cJSON in C, Vulnerable-Flask-App) con un vLLM finto: ha trovato `render_template_string`, `hashlib.md5`, `db.engine.execute`. Picco RAM del motore: ~19 MB |
| `ai-pipeline` (Python) | **190 test verdi** con torch/PEFT in `.venv` (con la sola stdlib 190 passano e 10 si saltano da soli). Dataset reale da 5 fonti: `train` 5.985 / `val` 402 / `test` 3.266 esempi (vedi *Dataset*) |
| DoRA su NemotronH | Verificato su CPU con un NemotronH minuscolo (transformers 5.22 nativo, PEFT 0.21): DoRA si applica ai 7 tipi di modulo, backward con gradienti finiti anche sui vettori di magnitudine, merge senza residui. Poi provato su GPU con Unsloth sul modello vero (RTX 5070 Ti, 16 GB): ha richiesto la loss fusa e una correzione al DoRA di PEFT (vedi *Deviazioni*) |
| Ricetta di quantizzazione | **NVFP4**: la ricetta che NVIDIA pubblica in ModelOpt (≥ 0.48.0) per questo modello, caricata così com'è. **FP8**: ricetta generica di ModelOpt + le esclusioni di `exclude_modules` del checkpoint NVIDIA 4B-FP8, riprodotte esattamente (e quelle del 9B-FP8); test in `tests/test_quantize.py`. Non eseguita: richiede GPU |
| Contratto `AttestationRegistry` | Compilato con solc 0.8.24; il calldata prodotto dall'encoder **Rust** è identico alla codifica ABI ufficiale ed è accettato dal contratto in una EVM (py-evm). **Non provato su Anvil** (non installato) |
| `train.py` | Eseguito su GPU: smoke test, prova del caso peggiore (picco 13,3 GB di VRAM) e training completo lanciato con `--batch-size 1 --grad-accum 8` (vedi *Deviazioni*). L'esito finale va letto in `ai-pipeline/outputs/train.log` |
| `quantize.py` | Scritto e con helper testati, ma **mai eseguito** (richiede GPU/RAM e `.venv-quant`) |
| `benchmark.py` | Metriche, client, parsing e gestione vLLM testati con server finti; **mai eseguito su GPU né con Semgrep reale** |
| `scripts/setup_env.sh` | `.venv` (training) installato con torch/Unsloth e kernel `mamba-ssm`/`causal-conv1d` compilati. Per compilarli ho dovuto passare a C++20 (torch 2.14 lo richiede, i `setup.py` hardcodano C++17) e costruire un pacchetto alla volta con `MAX_JOBS=2`: lo script **non** riflette ancora queste due correzioni. `.venv-quant` e `.venv-vllm` non ancora installati |
| Fine-tuning → quantizzazione → vLLM reale | Il fine-tuning è stato avviato (vedi sopra). Quantizzazione e vLLM **non eseguiti**: nessun numero di F1/VRAM/tok/s è stato misurato. La Fase 5 produce la tabella quando i passi sono stati lanciati |

Per non saturare la macchina, ogni build/test pesante è stato lanciato in `systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0`
e con `CARGO_BUILD_JOBS=4`: un eventuale OOM uccide solo quel processo. Conviene fare lo stesso (vedi `scripts/`).

## Avvio rapido (in ordine)

```bash
scripts/check_env.sh                  # 1.1 verifica CUDA 13.5 / driver (sola lettura; stampa gli export da mettere in ~/.bashrc)
DRY_RUN=1 scripts/setup_env.sh        # 1.2 controlla i comandi...
scripts/setup_env.sh                  #     ...poi installa .venv (train), .venv-quant (ModelOpt) e .venv-vllm (serving). La build di mamba-ssm/causal-conv1d si fa in .venv e in .venv-quant (due volte) ed è capped (BUILD_MEM_MAX=14G, MAX_JOBS=3)

cd ai-pipeline
python3 -m unittest discover -s tests -t .          # test (stdlib, nessuna GPU)
python3 prepare_data.py --download --test-benign-ratio 3   # 1.3 (già eseguito: data/*.jsonl esistono; richiede pyarrow, presente in .venv)
../.venv/bin/python train.py --dry-run             # controlla dati e iperparametri
../.venv/bin/python train.py                        # 2.1 DoRA 16-bit  -> model_bf16/   (GPU)
../.venv-quant/bin/python quantize.py               # 2.2 FP8 + NVFP4 con ModelOpt -> model_fp8/, model_nvfp4/ (GPU)
cd ..

scripts/serve_vllm.sh model_nvfp4     # 3.1 vLLM su :8000 (nome servito: sast-nemotron)
scripts/test_guided_decoding.sh       # 3.2 curl: la risposta deve essere SOLO il JSON del verdetto
scripts/run_engine.sh                 # 4.x motore su 127.0.0.1:8080
curl -X POST localhost:8080/analyze -H 'content-type: application/json' \
     -d '{"repo_url":"https://github.com/we45/Vulnerable-Flask-App"}'

cd ai-pipeline                        # 5.x
python3 benchmark.py accuracy --systems semgrep llm-noslice llm-slice
python3 benchmark.py hardware --models bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4   # avvia/ferma vLLM da solo, un modello alla volta
```

## I tre modelli: BF16, FP8, NVFP4

Alla fine ci sono **tre cartelle**, tutte caricabili da vLLM con lo stesso comando e senza flag di quantizzazione (vLLM legge il formato da `config.json`):

| Cartella | Cosa è | Chi la produce |
|---|---|---|
| `model_bf16/` | il modello fine-tuned (DoRA fuso) a precisione piena: il "vanilla" di riferimento | `train.py` |
| `model_fp8/` | FP8 W8A8 (E4M3) con KV cache FP8, come il checkpoint NVIDIA 4B-FP8 | `quantize.py` |
| `model_nvfp4/` | NVFP4 (FP4 con scale per blocchi da 16) con alcuni layer in FP8, per i Tensor Core Blackwell | `quantize.py` |

`quantize.py` parte da `model_bf16/` e produce gli altri due (`--only fp8` / `--only nvfp4` per uno solo); `benchmark.py hardware` li confronta.
**Strumento: NVIDIA Model Optimizer (ModelOpt)**, quello con cui NVIDIA produce i suoi checkpoint Nemotron (il blog del 4B: «PTQ using the ModelOpt library»;
il NVFP4 del 30B-A3B: «Software used for quantization: Model Optimizer»). Il piano diceva «llm-compressor o nvidia-modelopt»: nella prima versione avevo scelto llm-compressor per
comodità, ma non sa esprimere la ricetta di NVIDIA (formati diversi per layer, `lm_head` solo-pesi), quindi è stato tolto.

**Ricette** (le legge `quantize.py` da ModelOpt, non sono copiate qui):

| | Ricetta | Cosa fa |
|---|---|---|
| NVFP4 | `models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16` | linear (`in_proj`, `out_proj`, `up_proj`, `q/k/v/o_proj`) in **NVFP4 W4A4**, **anche le attention**; `down_proj` di 9 layer MLP (1, 3, 5, 8, 10, 18, 25, 33, 41) in **FP8 W8A8**; embedding **NVFP4 W4A16**; `lm_head` **FP8 W8A16**; `conv1d`, norme e parametri SSM in BF16. Nessuna quantizzazione della KV cache |
| FP8 | `general/ptq/fp8_default-kv_fp8` + esclusioni | FP8 W8A8 e KV FP8; in BF16 i moduli di `exclude_modules` del 4B-FP8 ufficiale: `lm_head`, `conv1d`, q/k/v/o delle 4 attention e `in_proj`/`out_proj` del Mamba subito prima di ciascuna |

La ricetta NVFP4 del 4B dichiara di «rispecchiare l'allocazione di bit del GGUF Q4_K_M» (da lì vengono i 9 `down_proj` in FP8): non è uno studio di sensibilità e non ho trovato numeri di accuratezza
pubblicati per questo NVFP4. Il 30B-A3B-NVFP4 di NVIDIA, invece, tiene le attention in BF16: il benchmark dirà quanto pesa la differenza.
Non esiste su Hugging Face un `NVIDIA-Nemotron-3-Nano-4B-NVFP4` (l'API risponde come per un repo inesistente): NVIDIA pubblica la ricetta in ModelOpt, non il checkpoint.

**Rischio noto, non verificato (serve vLLM sulla tua GPU):** la ricetta quantizza anche la tabella degli embedding. Dal sorgente di vLLM (`quantization/modelopt.py`, ramo mixed-precision)
risultano gestiti i layer Linear e `lm_head`, **non** `VocabParallelEmbedding`: se `vllm serve model_nvfp4` si ferma con un errore di shape su `embed_tokens`, rilancia
`quantize.py --only nvfp4 --bf16-embeddings` (una regola in coda che lascia gli embedding in BF16: ~0,8 GB invece di ~0,2 GB). Se invece parte, il flag non serve.

Il quantizzatore di `out_proj` Mamba è calibrato solo se il modello è in `eval()`: in training i kernel fusi leggono `out_proj.weight` senza passare dal modulo (`quantize.py` lo imposta).
Opzionale, come nella model card NVIDIA del FP8: `scripts/serve_vllm.sh model_fp8 --kv-cache-dtype fp8`.

Su 16 GB di VRAM il training va lanciato con `--batch-size 1 --grad-accum 8` (stesso batch effettivo di 2×4 del piano): con batch 2 è andato in OOM al passo ~92 sulle sequenze più lunghe.
Se non bastasse ancora, `--max-seq-length 4096` (il 95° percentile dei prompt è ~3k token). Il benchmark `hardware` usa `--gpu-util 0.80`: chiudi altri programmi che usano la GPU.

## Configurazione del motore (variabili d'ambiente)

| Variabile | Default | Note |
|---|---|---|
| `SAST_BIND` | `127.0.0.1:8080` | Il servizio clona URL scelti dal client: non esporlo su reti non fidate |
| `VLLM_URL` / `VLLM_MODEL` | `http://localhost:8000` / `sast-nemotron` | `--served-model-name` di vLLM |
| `VLLM_CONCURRENCY` | `8` | richieste in volo (allineato a `--max-num-seqs 8`) |
| `SLICE_DEPTH` / `SLICE_MAX_CALLERS` / `SLICE_MAX_CHARS` | `3` / `4` / `12000` | profondità, chiamanti per nodo, budget caratteri per traccia |
| `MAX_SINKS` / `MAX_FILES` / `MAX_FILE_BYTES` | `300` / `20000` / `1000000` | tetti di scansione |
| `ALLOWED_GIT_HOSTS` | `github.com` | solo `https://<host>/<owner>/<repo>` |
| `MAX_REPO_MB` / `GIT_CLONE_TIMEOUT_SECS` | `500` / `120` | un clone oltre il limite viene interrotto e cancellato |
| `MAX_CONCURRENT_ANALYSES` / `ANALYSIS_TIMEOUT_SECS` | `1` / `900` | le richieste in eccesso ricevono 429 |
| `SYFT_BIN` | — | se presente usa `syft` per l'SBOM, altrimenti il parser dei manifest |
| `CWE_KB_PATH` | KB incorporata | KB alternativa |
| `IPFS_API` / `ETH_RPC_URL` / `ATTEST_CONTRACT` / `ETH_FROM` / `MODEL_DIR` / `MODEL_HASH_WEIGHTS` | — | notarizzazione (`"notarize": true` nella richiesta) |

API: `POST /analyze {"repo_url": "...", "notarize": false}` → `{summary, bom, errors[, attestation]}`; `GET /health`.
Errori: 400 URL non valido · 429 occupato · 502 clone fallito · 503 vLLM spento · 504 timeout.

## Dataset: fonti, sink veri e limiti

`prepare_data.py` produce chat `system / user (regola CWE + sink + traccia) / assistant (JSON VEX)` con lo **stesso template**
del motore (`shared/prompt_template.json`, `shared/golden.json` verificati sia in Rust sia in Python). Il sink di ogni esempio è
**localizzato**, dal più al meno affidabile: commento `FLAW` (Juliet, CodeQL) → chiamata sulle righe che la fix tocca (diff vulnerabile/corretta)
→ solo senza controfattuale, la prima chiamata-sink della KB → altrimenti la *funzione stessa* alla riga del bug (`meta.sink_kind == "function"`).

| Fonte | Licenza | Cosa dà | Sink |
|---|---|---|---|
| PrimeVul (train/valid/test ufficiali) | MIT | C/C++ da CVE, coppie vulnerabile/corretta | pochi sink-chiamata, molti function-level |
| SecVulEval | MIT | C/C++, 10k coppie vulnerabile↔corretta con commit | idem, localizzati col diff |
| CVEfixes + PyVul (`janhvi-ag/cvefixes_pyvul_merged`) | non dichiarata (CVEfixes: CC-BY-4.0) | Python, `code_before`/`code_after`; la CWE mancante si ricava dal sink toccato dalla fix | diff |
| Juliet C/C++ 1.3 (SARD) | CC0 | **sintetico**, sink marcato da `/* FLAW */`; solo train/val, mai nel test (varianti quasi identiche) | commento, per costruzione |
| Test di sicurezza CodeQL | MIT | Python e C/C++ scritti a mano, `# $ Alert` = sink vulnerabile, sink senza annotazione = sicuro; **un verdetto per sink** anche nella stessa funzione. Le annotazioni e i commenti GOOD/BAD vengono tolti dal prompt | commento |

Dopo la costruzione: il test contiene solo fonti reali o curate; tutto ciò che compare nel test (o è la controparte di qualcosa che vi compare) viene tolto da train/val
(per hash della funzione) e i duplicati esatti sono rimossi; l'asimmetria fra etichette è limitata (`--label-ratio 1.25`, si scartano prima i sintetici) e gli esempi
function-level non superano la metà (`--function-share 0.5`). Train: **5.985** esempi, **41% con una vera chiamata-sink** (era 8%), 56% vulnerabili.

**Scartata di proposito: CyberNative/Code_Vulnerability_Security_DPO.** Ha molti sink espliciti, ma l'audit dei suoi stessi manutentori
(`audit/2026-10-06`) dice che non ha CWE native, che `training_validity` è `not_certified` per tutte le 4.656 righe e che nelle 66 coppie riviste
la parte "chosen" ha un vantaggio di sicurezza dimostrabile solo in 5 casi. Come fonte di etichette sarebbe rumore.

**Limite che resta.** Nelle fonti reali la KB riconosce pochi sink: su 7.315 funzioni vulnerabili di SecVulEval con CWE e fix noti, solo ~640 chiamate-sink
cadono sulle righe modificate dalla fix (il resto sono API di progetto: `GetInput`, `PyErr_SetString`…). Quindi i sink-chiamata del train sono per metà sintetici/curati
(Juliet, CodeQL) e per metà reali; nel **test solo il 7%** ha una chiamata-sink (`call_sinks`), ed è il sottoinsieme che il benchmark riporta a parte insieme alla ripartizione per fonte.
`--sinkless skip` e `--function-share 0` danno un dataset solo-chiamata, più piccolo e più coerente con l'inferenza. In training la traccia ha profondità 0
(le fonti hanno funzioni isolate); la catena di chiamanti compare solo in inferenza.

## Deviazioni dal piano (e perché)

- **Modello**: `unsloth/NVIDIA-Nemotron-3-Nano-4B` (l'ID del piano, `unsloth/Nemotron-3-Nano-4B`, non esiste su Hugging Face).
- **Thinking spento ovunque**: il chat template di Nemotron lo accende di default. Training (`enable_thinking=False`) e richieste
  (`chat_template_kwargs`) usano `<|im_start|>assistant\n<think></think>`; la loss è solo sulla risposta (`train_on_responses_only`).
- **Formato di quantizzazione e flag di vLLM**: ModelOpt scrive `hf_quant_config.json` (e `quantization_config` in `config.json`); vLLM sceglie da solo `modelopt` (FP8),
  `modelopt_fp4` (NVFP4) o `modelopt_mixed` (FP8+NVFP4 per layer, il caso della ricetta del 4B), come nelle model card NVIDIA: nessun `--quantization`.
  Se serve forzarlo: `VLLM_QUANT_FLAG=modelopt_mixed scripts/serve_vllm.sh ...`.
- **Target di DoRA**: `q/k/v/o_proj`, `up/down_proj` e Mamba `in_proj`. **Non `out_proj` né `conv1d`**: PEFT (≥ 0.18) li rifiuta nei modelli Mamba perché i kernel fusi
  leggono `out_proj.weight` direttamente e l'adapter sarebbe ignorato in silenzio (con PEFT 0.21 il piano originale fallirebbe con un errore; il notebook Unsloth li elenca ma è precedente a quel controllo).
  `gate_proj` non esiste in NemotronH, quindi è tolto. Caricamento e opzioni (`trust_remote_code`, `unsloth_force_compile`, attention `eager`) come nel notebook ufficiale Unsloth per Nemotron;
  `use_dora=True` è inoltrato a PEFT da entrambi i percorsi di Unsloth (letto nel sorgente) e `train.py` si ferma se i vettori di magnitudine non ci sono.
- **Loss con testa LM fusa** (`fused_compute_loss` in `train.py`): con il vocabolario di Nemotron i logit pieni di un batch non entrano in 16 GB (OOM al passo 0). La testa LM
  è applicata solo alle posizioni con etichetta (`add_fused_lm_head` di TRL), ma la `compute_loss` compilata da Unsloth la ignora, quindi `train.py` la sostituisce.
- **DoRA e righe a norma zero** (`patch_dora_dead_rows`): PEFT 0.21.2 calcola `magnitude / weight_norm` senza epsilon e il `in_proj` Mamba ha righe di peso tutte a zero in 8 layer,
  quindi loss e gradienti diventavano NaN. La correzione limita la norma a `1e-6`; è un workaround di un bug upstream e il suo test si salta da solo se PEFT lo risolve.
- **Batch 1 × accumulo 8** invece di 2 × 4 (stesso batch effettivo): vedi sopra, solo per la VRAM.
- **Tre virtualenv** (`.venv`, `.venv-quant`, `.venv-vllm`): vLLM fissa la propria versione di torch e `nvidia-modelopt[hf]` fissa `transformers < 5.19`, mentre il training usa la più recente.
- **Niente `git2`/`alloy`**: `git` da riga di comando (senza shell, hook disattivati, timeout, tetto di dimensione) e JSON-RPC diretto
  con ABI codificata a mano: meno dipendenze e compilazioni molto più leggere.
- **FP8 con scale statiche calibrate** (512 campioni da `train.jsonl`), come il checkpoint NVIDIA 4B-FP8.
- **VEX**: in CycloneDX non esiste `affected`; è `analysis.state = exploitable`, con `vex:status` conservato nelle `properties`.
- `sink_function/file/line` nel report sono quelli **noti al motore**, non quelli scritti dal modello.

## Note sul benchmark e sulla notarizzazione

- **VRAM**: con `--gpu-memory-utilization 0.80` vLLM prealloca la stessa quota per ogni formato, quindi il picco di `nvidia-smi` è quasi uguale per BF16/FP8/NVFP4.
  `benchmark.py hardware` riporta perciò anche i **pesi in VRAM** e i **token di KV cache** letti dal log di vLLM (le misure che cambiano davvero), oltre a disco, tok/s, latenza e F1.
- **Attestazione verificabile**: il file caricato su IPFS è esattamente il JSON di cui si calcola lo SHA-256 (il report *senza* i campi `attestation*`),
  quindi `ipfs cat <cid> | sha256sum` deve coincidere con `report_sha256` registrato on-chain. L'hash di *Model Provenance* copre nome servito + `config.json`,
  tokenizer, `generation_config.json`, `hf_quant_config.json` (e i pesi con `MODEL_HASH_WEIGHTS=1`). Lo stesso report non può essere attestato due volte.

## Limiti noti del motore

Call graph euristico (risoluzione per nome, nessuna type inference); le chiamate a livello di modulo non producono sink;
linguaggi: Python, C, C++; l'SBOM da manifest è approssimato (versioni solo se esatte) — usa `syft` per quello completo.
`notarizer` richiede un nodo IPFS (Kubo) e uno sviluppo-node con account sbloccati: `scripts/deploy_contract.sh` (Foundry) pubblica il contratto.

## Struttura

```
ai-pipeline/   prepare_data.py sources.py localize.py train.py quantize.py benchmark.py common.py vllm_flags.py tests/ data/{raw,train,val,test}.jsonl
sast-engine/   Cargo.toml cwe_kb.json src/{main,lib,config,ingestor,graph_slicer,rag_cwe,prompt,vllm_client,vex,notarizer,pipeline,server}.rs tests/ examples/
shared/        prompt_template.json  vex_schema.json  golden.json     (contratto prompt/schema condiviso Rust ↔ Python)
contracts/     AttestationRegistry.sol
scripts/       check_env.sh setup_env.sh serve_vllm.sh test_guided_decoding.sh run_engine.sh deploy_contract.sh
```
