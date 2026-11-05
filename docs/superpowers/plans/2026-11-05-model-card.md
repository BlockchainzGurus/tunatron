# Model card automatica nel notebook Colab: piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Al termine del training su Colab, generare in `Drive/Tunatron/` una model card in inglese (`MODEL_CARD.md` + grafici) con descrizione, training, dati, prestazioni, latenza e memoria per BF16/FP8/NVFP4.

**Architecture:** `benchmark.py hardware` (già esistente) misura i formati, uno per chiamata; il nuovo `ai-pipeline/model_card.py` (stdlib + matplotlib pigro) legge log di training, risultati, dataset e config e produce card e PNG; il notebook ottiene una sezione 11 che installa vLLM in un venv separato, lancia il benchmark e la card.

**Tech Stack:** Python 3 stdlib + `unittest` (come gli altri moduli di `ai-pipeline/`), matplotlib solo per i PNG, notebook Jupyter (`nbformat` 4.5, celle con `id`).

**Spec:** `docs/superpowers/specs/2026-11-05-model-card-design.md`

## Global Constraints

- Card in **inglese**; codice, commenti, docstring e messaggi del repo in **italiano** (come il resto di `ai-pipeline/`).
- Ogni stringa che finisce in `MODEL_CARD.md` o in un PNG (titoli, assi, legende, `not available: …`, `not measured`) è **in inglese**; le stringhe dei grafici stanno in una costante `CHART_TITLES` (dizionario) del modulo.
- Mai presentare risultati vecchi come attuali: `bench` svuota `results/<fmt>` prima di ogni misura e `main` svuota `model_card_assets/` prima di disegnare.
- `train.py` non si modifica. `benchmark.py`: solo modifiche additive (nessun campo esistente cambia; `markdown_table` e `parse_vllm_log` restano identici, con i loro test).
- `model_card.py`: solo libreria standard a livello di modulo; `matplotlib` importato dentro `_pyplot()`; deve girare con il `python3` di Colab (3.12), senza toccare i venv a versioni fissate.
- **vLLM si installa sempre alla sua ultima versione**: mai `vllm==…`, mai una versione fissata in nessun file (decisione dell'utente). Nessun dubbio di compatibilità di vLLM con Blackwell da scrivere nel notebook o nei report.
- Ordine dei formati ovunque: `bf16`, `fp8`, `nvfp4`.
- Impostazioni nuove del notebook, con questi default: `MODEL_CARD = True`, `RUN_BENCHMARK = True`, `VENV_VLLM = "/content/venv-vllm"`, `BENCH_SAMPLES = 200`, `BENCH_CONCURRENCY = 8`, `BENCH_EXTRA_ARGS = ""`.
- Un input mancante o illeggibile non solleva mai un errore: la sezione corrispondente dice `not available: <motivo>`; un formato presente ma non misurato dice `not measured`.
- Test: dalla cartella `ai-pipeline/`, `python3 -m unittest tests.<modulo> -v`; suite intera `python3 -m unittest discover -s tests -t .`.
- **Nessun commit**: l'utente ha chiesto di committare solo su richiesta. Gli step «Commit» del modello sono sostituiti da un controllo con `git status`; la proposta di commit va fatta alla fine.

## Review Focus

Casi che la spec implica ma che nessun task coprirebbe da solo; ognuno ha il suo test nel task indicato.

1. Loss `nan` nel log (training divergente): il punto si scarta, nessuna eccezione (Task 2).
2. Righe di `hardware.json` prodotte da un `benchmark.py` più vecchio, senza `report`, p50/p95 e `n`: la card mostra `n/d`, non un `KeyError` (Task 3).
3. `--train-args` con un flag che `train.py` non conosce: `argparse` farebbe `SystemExit`; la card deve uscire comunque, senza iperparametri (Task 2).
4. Un solo formato misurato (solo BF16) o un formato presente ma senza riga di risultati: grafici e tabelle reggono una sola categoria; il formato mancante è `not measured` (Task 3 e 4).
5. Esempi del test/val/train senza chiavi in `meta` (`label`, `source`, `lang`): `dataset_summary` li conta sotto `"?"` invece di fallire (Task 2).

---

## File Structure

| File | Azione | Responsabilità |
|---|---|---|
| `ai-pipeline/benchmark.py` | modifica | `percentile`, `parse_vllm_version`, `hardware_row`; `cmd_hardware` usa `hardware_row` |
| `ai-pipeline/tests/test_benchmark.py` | modifica | test dei tre punti sopra |
| `ai-pipeline/model_card.py` | nuovo | lettura input, `render_card`, `make_charts`, CLI |
| `ai-pipeline/tests/test_model_card.py` | nuovo | test di `model_card.py` |
| `notebooks/train_colab.ipynb` | modifica | impostazioni, sezione 11, sezione 12 rinumerata, intro |
| `ai-pipeline/tests/test_notebook.py` | nuovo | controlli strutturali sul notebook |
| `README.md` | modifica | albero del progetto e paragrafo Colab |

### Forma del contesto della card (usata dai Task 2, 3, 4)

`collect_context` produce, `render_card` e `make_charts` consumano, questo dizionario (chiavi sempre presenti; valore vuoto = input mancante):

```python
ctx = {
  "base_model": str,            # adapter_config["base_model_name_or_path"], altrimenti train.DEFAULT_MODEL
  "hyper": dict | None,         # vars(train.parse_args(shlex.split(train_args))); None se train_args manca o non è valido
  "adapter": dict | None,       # {"r", "lora_alpha", "target_modules", "use_dora"} da adapter_config.json
  "curves": {"train": [(epoch, loss)], "eval": [(epoch, eval_loss)]},   # liste vuote se manca il log
  "runtime_s": float | None,    # train_runtime dal log
  "commit": str | None,         # contenuto di colab_commit.txt
  "gpu": str | None,
  "vllm_version": str | None,   # da `vllm --version`; serve solo se le righe di risultati non hanno la propria
  "versions": dict[str, str],   # pacchetti di KEY_PACKAGES da colab_freeze.txt
  "data": dict | None,          # dataset_summary(...)
  "formats": list[str],         # formati presenti, in ordine bf16, fp8, nvfp4
  "results": list[dict],        # righe di hardware.json, in ordine dei formati
  "quant": dict[str, dict],     # formato -> contenuto di <dir>/hf_quant_config.json (solo se esiste)
  "charts": dict[str, str],     # nome -> percorso relativo alla card; lo riempie main()
}
```

Una riga di `results` (da `hardware_row`, Task 1): `format, weights_gb, kv_tokens, vram_gb, disk_gb, tok_s, latency_s,
latency_p50_s, latency_p95_s, scan_s, f1, f1_call_sinks, invalid, n, concurrency, vllm_version, report`, dove `report` è l'output
di `benchmark.evaluate` (`n, invalid, all{precision,recall,f1,fpr,accuracy,counts}, call_sinks{…}, by_source{…}`). Le righe vecchie
possono non avere i campi nuovi.

---

### Task 1: Campi nuovi in `benchmark.py hardware`

**Files:**
- Modify: `ai-pipeline/benchmark.py` (sezione «GPU»/«report» per le funzioni; `cmd_hardware` righe ~345-388)
- Test: `ai-pipeline/tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `percentile(values: list[float], q: float) -> float` (nearest-rank; lista vuota → `0.0`)
  - `parse_vllm_version(text: str) -> str | None`
  - `hardware_row(name: str, examples: list[dict], raw: list[dict], wall: float, log_info: dict, vllm_version: str | None, vram_delta_mib: int, disk_gb: float, scan_s: float | None, concurrency: int) -> dict` con le chiavi della riga descritta sopra.

- [ ] **Step 1: Scrivere i test (falliscono)** in `test_benchmark.py`, tre nuove classi:
  - `PercentileTests.test_nearest_rank`: `percentile([10,9,8,7,6,5,4,3,2,1], 50) == 5`, `…, 95) == 10`, `percentile([7], 95) == 7`, `percentile([], 50) == 0.0`.
  - `VllmVersionTests`: `"INFO 11-04 [api_server.py:1] vLLM API server version 0.11.0"` → `"0.11.0"`; `"Initializing a V1 LLM engine (v0.11.2) with config"` → `"0.11.2"`; `"nothing"` → `None`.
  - `HardwareRowTests.test_row_keeps_the_old_fields_and_adds_the_new_ones`: quattro esempi (`meta.label` 1,1,0,0; `sink_kind` qualsiasi), `raw` con tre risultati validi (`content='{"vulnerable": true}'` o `false`, `completion_tokens` 100, `latency` 1.0, 2.0, 3.0) e uno con `content=None`; `wall=2.0`, `log_info={"weights_gib": 7.5, "kv_tokens": 1000}`, `vram_delta_mib=2048`, `disk_gb=7.9`, `scan_s=None`, `concurrency=8`, `vllm_version="0.11.0"`. Asserzioni: `row["tok_s"] == 150.0` (300 token / 2 s), `row["latency_s"] == 2.0`, `row["latency_p50_s"] == 2.0`, `row["latency_p95_s"] == 3.0`, `row["vram_gb"] == 2.0`, `row["weights_gb"] == 7.5`, `row["n"] == 4`, `row["concurrency"] == 8`, `row["vllm_version"] == "0.11.0"`, `row["report"]["n"] == 4`, `row["f1"] == row["report"]["all"]["f1"]`, `row["invalid"] == 1`; e che `benchmark.markdown_table([row])` non solleva.

- [ ] **Step 2: Verificare che falliscano** — `python3 -m unittest tests.test_benchmark -v` → `AttributeError` sulle tre funzioni nuove.

- [ ] **Step 3: Implementare** le tre funzioni in `benchmark.py`: `percentile` con il rango più vicino (`sorted(values)[ceil(q/100*n)-1]`); `parse_vllm_version` con due regex (`vLLM API server version (\S+)` e `\(v(\S+?)\) with config`), la prima che combacia; `hardware_row` sposta qui la costruzione del dizionario oggi inline in `cmd_hardware` (stessi valori per i campi esistenti), calcola latenze solo sui risultati con `content is not None` e aggiunge i campi nuovi (`report = evaluate(...)`).

- [ ] **Step 4: Usare `hardware_row` in `cmd_hardware`**: leggere il log di vLLM una volta, ricavarne `parse_vllm_log` e `parse_vllm_version`, e sostituire il blocco `rows.append({...})` con `rows.append(hardware_row(...))`.

- [ ] **Step 5: Verificare** — `python3 -m unittest tests.test_benchmark -v` → tutti verdi, compresi `test_markdown_table_with_f1_delta_against_the_first_row` e `VllmLogTests` rimasti invariati; poi `git status --short` mostra solo `benchmark.py` e `tests/test_benchmark.py` fra i file del piano.

---

### Task 2: `model_card.py`: lettura degli input

**Files:**
- Create: `ai-pipeline/model_card.py`
- Test: `ai-pipeline/tests/test_model_card.py`

**Interfaces:**
- Consumes: `train.parse_args`, `train.DEFAULT_MODEL` (import a livello di modulo: `train.py` importa i pacchetti pesanti solo dentro `main`).
- Produces (tutte in `model_card.py`):
  - `FORMATS = ("bf16", "fp8", "nvfp4")`, `KEY_PACKAGES = ("torch", "unsloth", "unsloth-zoo", "trl", "peft", "transformers", "accelerate", "bitsandbytes", "mamba-ssm", "causal-conv1d")`
  - `parse_train_log(text: str) -> dict` → `{"train": [(epoch, loss)], "eval": [(epoch, eval_loss)], "runtime_s": float | None}`
  - `parse_freeze(text: str) -> dict[str, str]` (solo `KEY_PACKAGES`; nomi normalizzati in minuscolo con `-`)
  - `load_results(results_dir: Path) -> list[dict]`
  - `dataset_summary(data_dir: Path) -> dict | None` → `{split: {"examples", "vulnerable", "not_vulnerable", "sources": {…}, "languages": {…}}}` per i soli split presenti
  - `collect_context(*, log, results, data, adapter, models, commit_file, freeze_file, gpu, train_args, vllm_version) -> dict` (la forma `ctx` sopra; `log, adapter, commit_file, freeze_file, gpu, train_args, vllm_version` sono `Path | str | None`, `results` e `data` `Path`, `models` `dict[str, Path]`)

- [ ] **Step 1: Scrivere i test (falliscono)** in `test_model_card.py`, con `tempfile.TemporaryDirectory` per i file:
  - `ParseTrainLogTests`: un testo con sequenze ANSI (`\x1b[A`), barre tqdm separate da `\r` e le righe `{'loss': '0.5321', 'grad_norm': '1.2', 'learning_rate': '0.0002', 'epoch': '0.05'}`, `{'loss': 0.31, 'epoch': 0.1}`, `{'eval_loss': '0.28', 'epoch': '0.1'}`, `{'train_runtime': '5432.1', 'train_loss': '0.3'}` → `train == [(0.05, 0.5321), (0.1, 0.31)]`, `eval == [(0.1, 0.28)]`, `runtime_s == 5432.1`. **Review Focus 1:** una riga `{'loss': 'nan', 'epoch': '0.2'}` e una senza `epoch` non compaiono; il testo `"garbage"` dà liste vuote e `runtime_s is None`.
  - `ParseFreezeTests`: `"torch==2.14.1\nUnsloth_Zoo==2026.11.2\nnumpy==2.0\n"` → `{"torch": "2.14.1", "unsloth-zoo": "2026.11.2"}`.
  - `LoadResultsTests`: due cartelle `bf16/` e `nvfp4/` con `hardware.json` (lista di una riga ciascuna), più `fp8/` assente → `[bf16, nvfp4]` in quell'ordine anche se create al contrario; cartella inesistente → `[]`.
  - `DatasetSummaryTests`: tre `.jsonl` minimi → conteggi per etichetta, fonte e linguaggio; **Review Focus 5:** una riga con `"meta": {}` conta come `"?"`; cartella senza `.jsonl` → `None`.
  - `CollectContextTests`: cartella con `adapter_config.json` (`base_model_name_or_path`, `r=64`, `lora_alpha=64`, `target_modules=["q_proj"]`, `use_dora=true`), un log, `results/bf16/hardware.json`, `model_bf16/` e `model_fp8/hf_quant_config.json`, e `train_args="--batch-size 2 --grad-accum 4 --epochs 1"` → `ctx["hyper"]["batch_size"] == 2`, `ctx["hyper"]["epochs"] == 1.0`, `ctx["adapter"]["r"] == 64`, `ctx["vllm_version"] == "0.11.0"` (passato come argomento), `ctx["formats"] == ["bf16", "fp8"]`, `"fp8" in ctx["quant"]`, `ctx["charts"] == {}`. **Review Focus 3:** `train_args="--non-esiste"` → `ctx["hyper"] is None` senza `SystemExit` (argparse scrive sull'stderr: il test lo reindirizza). Tutti gli input `None`/inesistenti → nessuna eccezione, `ctx["formats"] == []`, `ctx["results"] == []`, `ctx["data"] is None`.

- [ ] **Step 2: Verificare che falliscano** — `python3 -m unittest tests.test_model_card -v` → `ModuleNotFoundError: model_card`.

- [ ] **Step 3: Implementare** le funzioni in `model_card.py`. `parse_train_log`: stesso approccio di `log_status` del notebook (toglie ANSI con `\x1b\[[0-9;?]*[A-Za-z]`, spezza su `[\r\n]+`, cerca `\{'loss'[^}]*\}`, `\{'eval_loss'[^}]*\}`, `\{'train_runtime'[^}]*\}`), valori letti con `ast.literal_eval` e `float()` (stringhe o numeri), scartando i non finiti e le righe senza `epoch`. `collect_context`: file illeggibili o JSON non validi → il campo corrispondente resta vuoto; `train.parse_args` va chiamato dentro `try/except SystemExit`.

- [ ] **Step 4: Verificare** — `python3 -m unittest tests.test_model_card -v` → verdi.

---

### Task 3: `render_card`: il testo della card

**Files:**
- Modify: `ai-pipeline/model_card.py`
- Test: `ai-pipeline/tests/test_model_card.py`

**Interfaces:**
- Consumes: la forma `ctx` (sopra) e le righe di `hardware_row`.
- Produces: `render_card(ctx: dict) -> str` (Markdown in inglese).

**Contenuto pinnato** (la spec fissa sezioni e limiti; qui i testi che i test controllano):
- Front matter YAML in testa: `language: en`, `base_model: <ctx["base_model"]>`, `library_name: transformers`, `pipeline_tag: text-generation`, `license: other`, `tags:` con almeno `security`, `sast`, `vulnerability-detection`, `dora`.
- Titoli `##` in quest'ordine: `Summary`, `Intended use and limitations`, `Training`, `Data`, `Evaluation`, `Efficiency`, `Files and reproduction`. I titoli ci sono sempre; il contenuto manca solo con `not available: <motivo>`.
- Summary: dice che il modello riceve regola CWE, sink candidato e backward slice e risponde con un JSON di finding in stile VEX, con `reasoning` e `suggested_fix` **scritti in italiano**; elenca C, C++ e Python.
- Intended use and limitations: include una riga `license: see the base model card` e che il modello non sostituisce un analizzatore.
- Training: rank/alpha/moduli/DoRA da `adapter`; lr, epoche, ottimizzatore, batch effettivo (`batch_size * grad_accum`), `max_seq_length`, seed da `hyper`; GPU, tempo (`runtime_s` in ore/minuti), commit e `versions`.
- Evaluation: una riga per formato di `ctx["formats"]`: precision, recall, F1, FPR, `invalid`, `n`; per un formato in `formats` senza riga di risultati, la cella dice `not measured`. Sotto la tabella, il protocollo: temperature 0, JSON-schema guided decoding, un output non valido conta come «non vulnerable».
- Efficiency: tabella con latenza media/p50/p95, tok/s, pesi in VRAM, KV cache, disco, e il testo **«per-request latency under load (concurrency C)»** con C preso dalle righe; riporta la versione di vLLM (quella delle righe di risultati, altrimenti `ctx["vllm_version"]`) e la GPU. Per i formati quantizzati, algoritmo e `group_size` da `ctx["quant"]` se presenti.
- Immagini: `![…](<percorso>)` solo per i nomi presenti in `ctx["charts"]` (`loss_curves`, `quality`, `latency_throughput`, `memory`).

- [ ] **Step 1: Scrivere i test (falliscono)** in `test_model_card.py`, classe `RenderCardTests`, con una `full_ctx()` di esempio (tre formati, curve, dati, grafici):
  - `test_front_matter_and_sections_in_order`: il testo inizia con `---\n`, contiene `license: other`, `library_name: transformers`, e i sette titoli `## …` compaiono nell'ordine dato (`text.index` crescente).
  - `test_tables_have_one_row_per_format_and_the_load_caveat`: contiene `per-request latency under load (concurrency 8)` e le tre etichette `bf16`, `fp8`, `nvfp4` nella tabella di Evaluation; il valore `0.812` (F1 della riga di esempio) formattato a tre decimali.
  - `test_missing_inputs_leave_the_headings_and_say_why`: `ctx` vuoto (come da `collect_context` senza input) → tutti i titoli presenti e almeno quattro occorrenze di `not available:`.
  - `test_a_format_without_results_is_not_measured`: `formats=["bf16","nvfp4"]` e `results` con la sola riga bf16 → `not measured` presente.
  - **Review Focus 2** `test_old_result_rows_render_with_n_d`: una riga con soli i campi vecchi (`format, f1, tok_s, latency_s, vram_gb, disk_gb, …` senza `report`, `n`, `latency_p50_s`) → `n/d` presente, nessuna eccezione.
  - `test_vllm_version_falls_back_to_the_context`: righe senza `vllm_version` e `ctx["vllm_version"] = "0.12.0"` → `0.12.0` nel testo; con la versione nella riga vince quella della riga.
  - `test_the_card_is_english_only`: né la card completa né quella vuota contengono lettere accentate italiane (`[àèéìòùÀÈÉÌÒÙ]`).
  - `test_images_only_for_existing_charts`: `charts={"loss_curves": "model_card_assets/loss_curves.png"}` → quel link c'è, `quality.png` no.

- [ ] **Step 2: Verificare che falliscano** — `python3 -m unittest tests.test_model_card.RenderCardTests -v` → `AttributeError: render_card`.

- [ ] **Step 3: Implementare** `render_card` e piccoli helper privati (`_fmt(value, spec)` che restituisce `n/d` per `None`/chiave assente, `_table(headers, rows)`); una funzione per sezione, così ogni sezione può degradare da sola.

- [ ] **Step 4: Verificare** — `python3 -m unittest tests.test_model_card -v` → verdi.

---

### Task 4: Grafici e CLI

**Files:**
- Modify: `ai-pipeline/model_card.py`
- Test: `ai-pipeline/tests/test_model_card.py`

**Interfaces:**
- Consumes: la forma `ctx`; `collect_context` (Task 2); `render_card` (Task 3).
- Produces:
  - `_pyplot()` → modulo `matplotlib.pyplot` con backend `Agg`; solleva `ImportError` se manca.
  - `make_charts(ctx: dict, assets_dir: Path) -> dict[str, str]`: scrive i PNG possibili e ritorna `nome -> "<assets_dir.name>/<file>.png"`. File: `loss_curves.png`, `quality.png`, `latency_throughput.png`, `memory.png`. Un grafico viene creato solo se i suoi dati ci sono; se `_pyplot()` solleva `ImportError`, ritorna `{}`.
  - `CHART_TITLES: dict[str, str]`: titoli, etichette degli assi e legende, in inglese.
  - `main(argv: list[str] | None = None) -> int` con le opzioni della spec (`--out-dir` obbligatorio; `--log`, `--results` default `results`, `--data` default `data`, `--adapter`, `--models fmt=dir …`, `--commit-file`, `--freeze-file`, `--gpu`, `--train-args`, `--vllm-version`). Scrive `<out-dir>/MODEL_CARD.md` (UTF-8) e `<out-dir>/model_card_assets/`, che svuota **prima** di disegnare (nessun PNG di una corsa precedente sopravvive).

**Aspetto dei grafici:** *prima di scrivere il codice dei grafici carica il skill `dataviz`* e segui la sua guida. Vincoli fissi: un colore per formato, lo stesso in tutti i grafici; valori scritti sulle barre; titoli e assi con unità (s, tok/s, GiB, GB, token); sfondo bianco opaco; `dpi=150`; nessun grafico con una sola barra dà errore.

- [ ] **Step 1: Scrivere i test (falliscono)**:
  - `MakeChartsTests.test_without_matplotlib_returns_no_charts_and_does_not_fail`: con `unittest.mock.patch.object(model_card, "_pyplot", side_effect=ImportError)` → `make_charts(full_ctx(), tmp) == {}`. Gira sempre.
  - `MakeChartsTests.test_charts_are_written` (`skipTest` se `matplotlib` non è importabile): con `full_ctx()` i quattro file esistono, non vuoti, e il ritorno li mappa come `model_card_assets/<file>.png`.
  - **Review Focus 4** `test_single_format_and_no_curves`: `ctx` con la sola riga bf16 e curve vuote → vengono creati `quality`, `latency_throughput`, `memory` e **non** `loss_curves`; con `results=[]` e curve presenti → solo `loss_curves`.
  - `MainTests.test_writes_the_card_with_no_inputs`: `main(["--out-dir", tmp, "--results", tmp/"nessuno", "--data", tmp/"nessuno"])` con `_pyplot` che solleva `ImportError` → ritorna `0`, `MODEL_CARD.md` esiste, è leggibile come UTF-8 e contiene `not available:`.
  - `MainTests.test_stale_charts_are_removed`: `model_card_assets/quality.png` preesistente + `main` senza risultati (e `_pyplot` che solleva `ImportError`) → il file non esiste più e la card non lo cita.
  - `ChartTitlesTests.test_titles_are_english`: nessun valore di `CHART_TITLES` contiene lettere accentate italiane, e almeno un titolo contiene la parola `Latency`.
  - `MainTests.test_links_charts_that_exist` (`skipTest` senza matplotlib): con risultati e log di esempio in una cartella temporanea, `MODEL_CARD.md` contiene `![` per `model_card_assets/quality.png`.

- [ ] **Step 2: Verificare che falliscano** — `python3 -m unittest tests.test_model_card -v` → errori sulle funzioni mancanti.

- [ ] **Step 3: Implementare** `_pyplot`, `make_charts` (una funzione privata per grafico, `plt.close` dopo ogni salvataggio) e `main`. `main` costruisce `ctx` con `collect_context`, scrive prima i PNG (`ctx["charts"] = make_charts(...)`; se vuoto e matplotlib manca, avviso su stderr), poi `render_card`. `--models` accetta `fmt=dir`; i formati sconosciuti sono un errore di `argparse`.

- [ ] **Step 4: Ambiente di prova con matplotlib** — matplotlib non è installato nel Python locale. Crea un venv usa e getta nello scratchpad (`uv venv --python 3.12` oppure `python3.12 -m venv`; se il Python locale non ha wheel di matplotlib, usa 3.12) e installa `matplotlib`. Esegui `$VENV/bin/python -m unittest tests.test_model_card -v` da `ai-pipeline/`: i test dei PNG non devono risultare saltati. Non aggiungere matplotlib a nessun venv del progetto.

- [ ] **Step 5: Controllo visivo** — genera una card di prova con dati finti (tre formati, curve di un centinaio di punti) in una cartella dello scratchpad con `$VENV/bin/python model_card.py --out-dir …`, apri i quattro PNG con il tool `Read` e verifica: etichette leggibili, nessun testo sovrapposto, colori coerenti, unità presenti. Correggi e ripeti finché sono a posto.

- [ ] **Step 6: Verificare** — suite intera `python3 -m unittest discover -s tests -t .` (con il Python di sistema) → verde, test dei PNG saltati; stessa suite col venv → verde, nessun salto per matplotlib.

---

### Task 5: Notebook, test strutturali e README

**Files:**
- Modify: `notebooks/train_colab.ipynb` (usa il tool `NotebookEdit`: ToolSearch `select:NotebookEdit`; non riscrivere il JSON a mano)
- Create: `ai-pipeline/tests/test_notebook.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: CLI di `model_card.py` (Task 4); `benchmark.py hardware --models <fmt>=<dir> --vllm-bin … --samples … --concurrency … --out-dir …` (esistente); helper del notebook già presenti: `sh`, `out`, `running`, `dir_size`, variabili `AI`, `WORK`, `DRIVE_DIR`, `LOG`, `GPU_NAME`, `VRAM`, `BATCH_SIZE`, `GRAD_ACCUM`, `EXTRA_TRAIN_ARGS`, `QUANT_FORMATS`.
- Produces: celle con id `hmc` (markdown 11), `hmc1` (11.1), `vllmenv`, `hmc2` (11.2), `bench`, `hmc3` (11.3), `card`, inserite dopo `qcopy` e prima di `h11` (che diventa «12. Rilascia il runtime»).

- [ ] **Step 1: Scrivere `tests/test_notebook.py`** (falliscono finché il notebook non è modificato):
  - `test_ids_are_unique_and_code_cells_parse`: gli `id` sono unici; ogni cella `code` passa `ast.parse`.
  - `test_model_card_settings_and_defaults`: la cella `settings` (eseguita come in `test_setup_env.notebook_settings`) definisce `MODEL_CARD is True`, `RUN_BENCHMARK is True`, `VENV_VLLM == "/content/venv-vllm"`, `BENCH_SAMPLES == 200`, `BENCH_CONCURRENCY == 8`, `BENCH_EXTRA_ARGS == ""`.
  - `test_section_order`: gli id appaiono nell'ordine `qcopy`, `hmc`, `hmc1`, `vllmenv`, `hmc2`, `bench`, `hmc3`, `card`, `h11`, `release`; il testo di `h11` inizia con `## 12.`.
  - `test_benchmark_runs_one_format_per_call`: il sorgente di `bench` contiene `benchmark.py hardware`, `--vllm-bin`, `--out-dir results/{fmt}` e `--models {fmt}=model_{fmt}`, e un ciclo `for fmt in` (un formato per chiamata, così un formato che fallisce non fa perdere gli altri).
  - `test_vllm_is_never_pinned`: il sorgente di `vllmenv` contiene `uv pip install` con `vllm` e `--torch-backend=auto`, e nessuna delle celle del notebook contiene `vllm==` né `vllm>=`.
  - `test_bench_clears_stale_results`: il sorgente di `bench` contiene `rmtree` e `results/{fmt}` (un risultato di una corsa precedente non deve sopravvivere a una misura fallita).
  - `test_vllm_version_is_recorded`: il sorgente di `vllmenv` contiene `VLLM_VERSION`, e quello di `card` passa `--vllm-version`.
  - `test_card_cell_passes_what_the_card_needs`: il sorgente di `card` contiene `model_card.py`, `--train-args`, `--gpu`, `--commit-file`, `--freeze-file`, e il ripiego su Drive per l'adapter (`{DRIVE_DIR}/adapter`).

- [ ] **Step 2: Verificare che falliscano** — `python3 -m unittest tests.test_notebook -v` → falliscono i test su impostazioni, ordine e celle.

- [ ] **Step 2b: Strumento per il notebook** — carica lo schema di `NotebookEdit` (ToolSearch `select:NotebookEdit`) e controlla che permetta di fissare l'`id` della cella all'inserimento. Se non lo permette, modifica il JSON con un piccolo script che rispetti la serializzazione attuale del file (indentazione, `ensure_ascii`, newline finale) e dia alle celle codice nuove `metadata: {}`, `execution_count: null`, `outputs: []`. In ogni caso, a fine task `git diff --stat notebooks/train_colab.ipynb` deve mostrare solo le celle volute.

- [ ] **Step 3: Impostazioni** — nella cella `settings` aggiungere, dopo il blocco della quantizzazione, un blocco «Model card (sezione 11)» con le sei costanti e un commento breve ciascuna (significati nella spec, tabella «Notebook: sezione 11»).

- [ ] **Step 4: Celle 11.1 e 11.2** (markdown `hmc`, `hmc1`, `hmc2` con una frase ciascuna; `hmc` dice anche che, se la sezione 10 si è fermata con un errore o è stata saltata, si eseguono direttamente le celle della sezione 11, che funzionano anche con il solo BF16; codice):
  - `vllmenv`: se `not MODEL_CARD or not RUN_BENCHMARK` stampa che salta e termina; altrimenti crea il venv (`uv venv --python 3.12 {VENV_VLLM}` se manca), `uv pip install --python {VENV_VLLM}/bin/python vllm --torch-backend=auto` (sempre l'ultima versione: nessun pin, nemmeno via `PINS`), poi `{VENV_VLLM}/bin/vllm --version` con `check=False`. **Non solleva mai**: "Esegui tutto" si fermerebbe prima della card. Imposta `VLLM_OK = True/False` (anche l'installazione va in `try/except`) e, se è `False`, stampa che la card uscirà senza benchmark. Salva in `VLLM_VERSION` l'output di `{VENV_VLLM}/bin/vllm --version` (con `out()`), perché con vLLM sempre all'ultima versione è ciò che rende riproducibili le latenze.
  - `bench`: se `running()` solleva `RuntimeError` (GPU occupata, come `qrun`); se `RUN_BENCHMARK` è falso o `VLLM_OK` non è `True` (usa `globals().get("VLLM_OK")`, perché in un runtime riavviato la cella 11.1 può non essere stata eseguita), stampa il motivo e termina. Formati = `["bf16"]` più `fp8`/`nvfp4` se `Path(AI)/f"model_{f}"` **o** `Path(DRIVE_DIR)/f"model_{f}"` contiene `.safetensors`; per quelli solo su Drive, copia in locale con `shutil.copytree` e controllo `dir_size` (stesso schema di `qrun`). Ciclo `for fmt in formats:` che per prima cosa cancella `Path(AI)/"results"/fmt` con `shutil.rmtree(..., ignore_errors=True)` (niente risultati vecchi presentati come attuali), poi lancia con `sh(..., cwd=AI, check=False)` il comando `python3 benchmark.py hardware --models {fmt}=model_{fmt} --vllm-bin {VENV_VLLM}/bin/vllm --samples {BENCH_SAMPLES} --concurrency {BENCH_CONCURRENCY} --out-dir results/{fmt} {BENCH_EXTRA_ARGS}`; se il codice è diverso da 0 lo annota in una lista `bench_failed` e continua. Alla fine stampa i formati misurati e quelli falliti, e copia `Path(AI)/"results"` in `Path(DRIVE_DIR)/"benchmark_results"` (`dirs_exist_ok=True`).

- [ ] **Step 5: Cella 11.3 `card`** (markdown `hmc3` + codice): se `not MODEL_CARD` termina. Costruisce `models` con `fmt=dir` per ogni formato presente (cartella locale se c'è, altrimenti quella su Drive); usa `LOG` se esiste, altrimenti `{DRIVE_DIR}/train_colab.log`, e l'adapter in `outputs/adapter` se esiste, altrimenti `{DRIVE_DIR}/adapter` (in un runtime nuovo la sezione 9 ha già copiato lì entrambi); passa `--vllm-version` con `globals().get("VLLM_VERSION")` quando c'è; lancia `sh("python3 model_card.py --out-dir outputs/model_card --log … --results results --data data --adapter outputs/adapter --models … --commit-file {DRIVE_DIR}/colab_commit.txt --freeze-file {DRIVE_DIR}/colab_freeze.txt --gpu '{GPU_NAME}, {VRAM}' --train-args '--batch-size {BATCH_SIZE} --grad-accum {GRAD_ACCUM} {EXTRA_TRAIN_ARGS}'", cwd=AI)` (con `shlex.quote` dove serve) e poi `shutil.copytree(f"{AI}/outputs/model_card", DRIVE_DIR, dirs_exist_ok=True)`; stampa il percorso di `MODEL_CARD.md` su Drive.

- [ ] **Step 6: Rinumerare e aggiornare il testo** — titolo di `h11` → `## 12. Rilascia il runtime (facoltativo)`; nella cella `intro` aggiungere a «Cosa produce» `MODEL_CARD.md`, `model_card_assets/`, `benchmark_results/`, e ai «Limiti» una riga: la sezione 11 è provata solo con test sul PC (parser, card, struttura del notebook), non ancora eseguita su Colab; nessun riferimento a vLLM su Blackwell. Nella nota della sezione 10 non serve nulla.

- [ ] **Step 7: README** — nell'albero aggiungere `│   ├── model_card.py` dopo `quantize.py`; nel paragrafo Colab (riga ~97) aggiungere una frase: la sezione 11 esegue il benchmark vLLM e salva `MODEL_CARD.md` con i grafici in `Il mio Drive/Tunatron/`.

- [ ] **Step 8: Verificare** — `python3 -m unittest tests.test_notebook tests.test_setup_env -v` → verdi; suite intera `python3 -m unittest discover -s tests -t .` → verde; `python3 -c "import json; json.load(open('../notebooks/train_colab.ipynb'))"` (da `ai-pipeline/`) senza errori; `git status --short` elenca solo i file del piano e la cartella `docs/`.

- [ ] **Step 9: Chiusura** — riportare all'utente: file toccati, test eseguiti, cosa **non** è stato verificato (esecuzione della sezione 11 su Colab, formato reale del log di transformers 5.17, schema reale di `hf_quant_config.json`; niente dubbi su vLLM e Blackwell), il promemoria `git push` prima di rilanciare il notebook, e proporre il commit.
