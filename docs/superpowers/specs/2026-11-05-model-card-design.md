# Model card automatica nel notebook Colab

Data: 2026-11-05 · Stato: bozza da rivedere · Percorso: architetturale

## Obiettivo

Al termine del training su Colab (`notebooks/train_colab.ipynb`), salvare in `Drive/Tunatron/` una **model card in inglese**
(`MODEL_CARD.md` + grafici PNG) con: breve descrizione, dettagli di training, dati, **prestazioni** (precision/recall/F1/FPR),
**latenza** e throughput, memoria e dimensioni, per ciascun formato disponibile (BF16, FP8, NVFP4).

Le misure vengono dal benchmark vLLM già presente (`ai-pipeline/benchmark.py hardware`), non da una misura nuova.

## Fuori ambito

- Confronto con il modello base non fine-tuned e con Semgrep (`benchmark.py accuracy`): si può aggiungere dopo, non serve alla card.
- Pubblicazione su Hugging Face. La card ha l'intestazione YAML di Hugging Face, ma il caricamento resta manuale.
- Intervalli di confidenza. La card riporta N e lo dichiara; non calcola incertezze.
- Modifiche a `train.py`: il notebook continua a lanciarlo così com'è.

## Architettura

```
train.log ─────────────┐
results/<fmt>/hardware.json (benchmark.py, uno per formato) ─┤
data/{train,val,test}.jsonl ──┤                              ├─► model_card.py ─► MODEL_CARD.md
outputs/adapter/adapter_config.json, model_*/config.json,    │                    model_card_assets/*.png
model_*/hf_quant_config.json, colab_commit.txt, colab_freeze.txt ┘
```

- **`ai-pipeline/model_card.py`** (nuovo). Funzioni pure per leggere gli input e produrre testo e tabelle; i grafici con matplotlib
  importato in modo pigro. Si lancia con il `python3` di Colab, che ha già matplotlib: i venv a versioni fissate non cambiano.
- **`ai-pipeline/benchmark.py`** (modifica additiva): vedi sotto.
- **`notebooks/train_colab.ipynb`**: nuova sezione 11 «Model card»; l'attuale 11 (rilascio runtime) diventa 12.
- Il modulo nuovo arriva su Colab solo dopo `git push`, perché il notebook clona la repo.

## Notebook: sezione 11

Impostazioni nuove (cella 0):

| Nome | Default | Significato |
|---|---|---|
| `MODEL_CARD` | `True` | genera la card |
| `RUN_BENCHMARK` | `True` | esegue il benchmark vLLM; se `False` la card ha solo i dati di training |
| `VENV_VLLM` | `/content/venv-vllm` | ambiente di vLLM, separato da training e quantizzazione |
| `BENCH_SAMPLES` | `200` | esempi del test set per formato (`0` = tutti, 3.266). Stesso default di `benchmark.py` |
| `BENCH_CONCURRENCY` | `8` | richieste concorrenti |
| `BENCH_EXTRA_ARGS` | `""` | argomenti aggiuntivi per `benchmark.py hardware` |

Celle:

1. **11.1 Ambiente vLLM.** `uv venv --python 3.12 $VENV_VLLM`, `uv pip install vllm --torch-backend=auto` (come `scripts/setup_env.sh`),
   **sempre l'ultima versione, mai fissata**; verifica con `vllm --version`. Se fallisce, la cella lo segnala e il resto prosegue senza benchmark.
2. **11.2 Benchmark.** Per ogni formato disponibile (`model_bf16` sempre; `model_fp8` e `model_nvfp4` se presenti su disco o copiabili da
   Drive) lancia `python3 benchmark.py hardware --models <fmt>=<dir> --vllm-bin $VENV_VLLM/bin/vllm --samples $BENCH_SAMPLES
   --concurrency $BENCH_CONCURRENCY --out-dir results/<fmt> $BENCH_EXTRA_ARGS`. **Una chiamata per formato**: `hardware` scrive il JSON solo a fine ciclo e un errore (es. NVFP4 non
   caricabile da vLLM, rischio già noto nel README) farebbe perdere anche gli altri formati. Un formato che fallisce viene saltato e
   segnalato. Richiede la GPU libera.
   I `results/<fmt>/` (JSON e log di vLLM) vengono copiati anche in `Drive/Tunatron/benchmark_results/`.
3. **11.3 Card.** Lancia `model_card.py` e copia `MODEL_CARD.md` e `model_card_assets/` in `Drive/Tunatron/`.

## `benchmark.py`: modifica additiva

In ogni riga di `hardware.json` si aggiungono, senza toccare i campi esistenti né `markdown_table`:
`report` (il dizionario completo di `evaluate`: precision, recall, fpr, conteggi, per fonte), `latency_p50_s`, `latency_p95_s`,
`n` (esempi), `concurrency`. `test_benchmark.py` verifica i nuovi campi e che la tabella resti identica.

## `model_card.py`

Interfaccia a riga di comando (percorsi relativi a `ai-pipeline/`):

```
python3 model_card.py --out-dir <dir> [--log outputs/train.log] [--results results] [--data data]
                      [--adapter outputs/adapter] [--models bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4]
                      [--commit-file <DRIVE_DIR>/colab_commit.txt] [--freeze-file <DRIVE_DIR>/colab_freeze.txt]
                      [--gpu "<nome>, <VRAM>"] [--train-args "<argomenti passati a train.py>"]
```

Il notebook passa i file di commit e versioni che ha già salvato in Drive (celle 3 e 6) e la GPU letta dalla cella 2.
`--train-args` serve a riportare gli iperparametri **davvero usati** (learning rate, epoche, ottimizzatore, batch, lunghezza massima,
seed): `model_card.py` li ricava con `train.parse_args(shlex.split(...))`, cioè i default di `train.py` più le sovrascritture del
notebook (`--batch-size`, `--grad-accum`, `EXTRA_TRAIN_ARGS`). Se gli argomenti non sono validi, la sezione resta senza iperparametri.

Funzioni principali (tutte testabili senza GPU):

- `parse_train_log(text)`: toglie le sequenze ANSI e le barre tqdm, legge i dizionari `{'loss': …}`, `{'eval_loss': …}` e
  `{'train_runtime': …}`; restituisce curve (asse x: `epoch`) e tempo di training. Tollera righe non riconosciute.
- `load_results(results_dir)`: unisce i `hardware.json` per formato nell'ordine BF16, FP8, NVFP4.
- `dataset_summary(data_dir)`: esempi, etichette, fonti e linguaggi contati dai `.jsonl`.
- `render_card(ctx)`: testo Markdown. Ogni sezione con input mancante è sostituita da una riga «not available: …», mai da un errore.
- `make_charts(ctx, out_dir)`: scrive i PNG; se matplotlib manca, la card esce senza grafici e lo dice.

### Contenuto della card

Intestazione YAML (`base_model`, `library_name`, `pipeline_tag`, `tags`, `license: other` con rimando alla licenza del modello base).
Poi: Summary · Intended use e limitations · Training (base, DoRA r=64/alpha=64 e moduli target, ottimizzatore, lr, epoche, batch
effettivo, lunghezza massima, seed, GPU, tempo, commit, versioni) · Data · Evaluation · Efficiency · Files e riproduzione.

### Grafici

1. Train e eval loss per epoca.
2. Precision, recall, F1, FPR per formato.
3. Latenza media (con p50/p95) e tok/s per formato.
4. Pesi in VRAM, disco e capacità della KV cache per formato.

I grafici seguono la guida del skill `dataviz` (palette, accessibilità, tema chiaro e scuro dove applicabile).

### Limiti dichiarati nella card

- La latenza è per richiesta **sotto carico** (concorrenza `BENCH_CONCURRENCY`), non a vuoto.
- Singola esecuzione, una GPU, una versione di vLLM (letta dal log di avvio e riportata).
- FP8 e NVFP4 sono valutati sugli stessi N esempi del BF16; un output non valido conta come «non vulnerabile».
- Il prompt di sistema e il campo `reasoning` sono in italiano; il codice è C, C++ e Python.

## Errori

| Situazione | Comportamento |
|---|---|
| vLLM non si installa o non parte | niente benchmark; card con i soli dati di training |
| un formato non si carica | quel formato è «not measured»; gli altri restano |
| `train.log` assente o senza metriche | sezione Training senza curve |
| matplotlib assente | card senza grafici, con avviso |
| GPU occupata (training in corso) | la cella 11.2 si ferma con un messaggio, come la 10.3 |

## Test

- `tests/test_model_card.py`: parsing del log con rumore ANSI/tqdm, unione dei risultati, card completa e con input mancanti,
  front matter, creazione dei PNG (`skipTest` se matplotlib manca), ordine dei formati.
- `tests/test_benchmark.py`: nuovi campi in `hardware.json`; `markdown_table` invariata.
- `tests/test_setup_env.py` continua a passare (legge le impostazioni del notebook).
- Controllo manuale: card di prova generata da fixture e PNG guardati a occhio.

## Cosa non è verificato

La sezione 11 non è mai stata eseguita su Colab. In particolare: il formato esatto delle righe di metriche nel log di
transformers 5.17 (non c'è un log locale; il parser si basa su ciò che `log_status` del notebook già assume) e lo schema reale di
`hf_quant_config.json`. Dopo la prima esecuzione su Colab vanno controllati.
