# Dati di training con il grafo: piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produrre i dati di training e di valutazione con slice reali del grafo (Juliet C + Python) e le misure S1–S5, perché un solo run su Colab insegni al modello a usare i chiamanti.

**Architecture:** Il motore Rust guadagna `sast-engine slice` (stesso `build_graph`/`CodeGraph::slice` del server) che scrive JSONL. Moduli Python separati (anonimizzazione C, Juliet, esempi, tracce Python, verifica delle perdite) trasformano le slice in esempi; `prepare_data.py` li unisce ai dati esistenti; `benchmark.py` confronta «con chiamanti» e «solo sink».

**Tech Stack:** Rust (tree-sitter, petgraph, serde_json, tempfile per i test), Python 3 stdlib + `unittest` (come i test esistenti), vLLM solo per le valutazioni.

**Spec:** `docs/superpowers/specs/2026-11-06-graph-aware-training-data-design.md`

## Global Constraints

Dalla spec (valori verbatim):

- Slice: profondità **5** in training e valutazione; **4** chiamanti per nodo; **12.000** caratteri (default di `SliceOptions`). Il default del motore (3) cambia a 5 solo se S1 con profondità 5 batte quello con 3 (Task 15).
- Juliet: solo `.c`, CWE **120, 78, 190, 416**; fonte `https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip`, sha256 `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb`.
- Sink Juliet = chiamate del catalogo dopo aver riportato le macro in maiuscolo al nome vero (`SYSTEM`→`system`); etichetta dal nome originale della funzione col sink (`bad` → 1, `good*` → 0). Il commento FLAW non si usa.
- Un esempio entra solo se il suo prompt completo non compare con l'etichetta opposta. Miscela: **0,7** slice completa, **0,3** solo blocco a profondità 0 (ammessa solo se quel prompt non è in conflitto).
- Gruppo = test case (nome senza lettera finale). ≈**12%** delle varianti di flusso (≈6 su 48) è escluso da train e val e forma `unseen_variants`.
- Juliet ≤ ~20% del train. P3 sintetici ≤ **15%** degli esempi Python con slice e mai nei file di valutazione.
- Target: niente id CVE, descrizione CVE o messaggio di commit; `suggested_fix` = `fix_hint` del catalogo (vulnerabili) o "No change needed." (dopo il Task 5c); per le slice il reasoning usa un vocabolario chiuso (origine del dato, percorso con profondità, motivo del verdetto) **scritto in inglese** (decisione dell'utente, R13: "depth N"); system prompt, template, schede e hint del catalogo diventano inglesi nel Task 5c.
- Ogni file dati < 50 MB; esempi oltre `--max-seq-length` (8192) scartati e contati.
- Soglie (proposte, Task 15): S1 con chiamanti ≥ 70% (senza: 50% per costruzione); S2 scarto ≤ 10 punti; S3 ≤ 50% dei falsi positivi pygoat ancora «affected»; S4 troncati < 2% e 0 «CVE-»; S5 F1 non peggiore di 2 punti.

Del progetto:

- `cargo build/test` sempre sotto `systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0` (come `scripts/run_engine.sh`). vLLM parte solo su richiesta esplicita dell'utente, col venv `/home/gabs/.venvs/tunatron-vllm` (`VLLM_BIN=…/bin/vllm`), mai `pkill -f`.
- Test: `cd ai-pipeline && python3 -m unittest discover -s tests -t .`; Rust: `cargo test --manifest-path sast-engine/Cargo.toml`. Commenti e docstring in italiano come nel resto del codice.
- Ultime versioni, nessun pin a mano. Nessun `git push` senza conferma: il notebook Colab clona GitHub, il push lo decide l'utente.

## Review Focus

Condizioni che la spec implica ma nessun test del piano originario esercitava; ognuna ha il test nel task indicato.

1. Slice con `truncated=true` o test case i cui prompt sono tutti in conflitto: nessun esempio, conteggiato nel rapporto, nessun crash (Task 5).
2. Identificatori che contengono una parola-spia solo come sottostringa (`badge_count`, `fixture`) restano intatti; stringhe letterali con parole-spia («Benign, fixed string») vengono neutralizzate (Task 3).
3. Target con riga fuori da ogni funzione: saltato con messaggio, gli altri proseguono (Task 1).
4. Sorgenti con `\r\n` o byte non UTF-8: estratti normalizzati, così righe e offset del motore coincidono (Task 4).
5. Membro dello zip con percorso che esce dalla destinazione: rifiutato (Task 4).

---

### Task 1: `sast-engine slice`

**Files:**
- Modify: `sast-engine/src/graph_slicer.rs` (in `impl CodeGraph`, accanto a `slice`, ~riga 398)
- Modify: `sast-engine/src/rag_cwe.rs` (helper KB), `sast-engine/src/pipeline.rs:56-59` (usa l'helper), `sast-engine/src/lib.rs`, `sast-engine/src/main.rs`
- Create: `sast-engine/src/slice_cli.rs`
- Test: `sast-engine/tests/slice_cli.rs`

**Interfaces:**
- Produces (Rust):
  - `CodeGraph::add_target_sink(&mut self, file: &str, line: u32, callee: &str, cwe_id: &str) -> Option<usize>`: aggiunge un `Sink` sulla funzione più interna di `file` che contiene `line` e ne restituisce l'indice in `self.sinks`; `None` se nessuna funzione la contiene.
  - `CweKb::load_or_embedded(path: Option<&Path>) -> anyhow::Result<CweKb>`.
  - `slice_cli::SliceArgs { root: PathBuf, targets: Option<PathBuf>, depth: Option<usize>, callers: Option<usize>, max_chars: Option<usize> }`
  - `slice_cli::parse_args(args: &[String]) -> anyhow::Result<SliceArgs>`: `<ROOT>` obbligatorio, flag `--targets --depth --callers --max-chars`; flag ignoto o root mancante = errore.
  - `slice_cli::slice_to_json(slice: &Slice, cwe_id: &str) -> serde_json::Value` = `{"sink":{"function","file","line","cwe_id"},"blocks":[{"file","name","start_line","end_line","depth","lang","code"}…],"truncated":bool}`, blocchi dal chiamante più esterno al sink.
  - `slice_cli::run(args: &SliceArgs, cfg: &Config, kb: &CweKb, out: &mut dyn Write, err: &mut dyn Write) -> anyhow::Result<usize>` (righe scritte).
- Consumes: `build_graph`, `CodeGraph::slice`, `Config.{scan,slice,kb_path}`.
- CLI: `sast-engine slice <ROOT> [--targets FILE] [--depth N] [--callers N] [--max-chars N]`, JSONL su stdout, messaggi su stderr. Targets: righe `{"file","line","callee","cwe_id"}` con `file` relativo alla radice, separatori `/`.

- [ ] **Step 1: Scrivi i test in `sast-engine/tests/slice_cli.rs`** (helper `tree(&[(&str,&str)]) -> TempDir` e `slice(root, targets: Option<&str>, depth: Option<usize>) -> (Vec<Value>, String)` che chiama `slice_cli::run` con `Config::default()` e `CweKb::embedded()`; i target stanno in un file temporaneo fuori dalla radice). Fixture `a.c`: `f5` chiama `system(x)`, `f4`→`f5`, `f3`→`f4`, `f2`→`f3`, `f1`→`f2(getenv("X"))`.
  - `depth_5_returns_five_blocks_outermost_first`: profondità dei blocchi `[4,3,2,1,0]`, `blocks[0]["name"] == "f1"`, `sink["function"] == "system"`, `sink["cwe_id"] == "CWE-78"`.
  - `depth_3_cuts_the_root`: con `Some(3)` profondità `[3,2,1,0]`.
  - `target_slices_the_enclosing_function_of_a_macro_sink`: file con `void h(char *d)\n{\n    SYSTEM(d);\n}` e target `{"file":"m.c","line":3,"callee":"system","cwe_id":"CWE-78"}` → un record, `sink.function == "system"`, ultimo blocco `name == "h"`, `depth == 0`.
  - `target_outside_any_function_is_skipped_with_a_message`: due target (riga 1 globale, riga 3 valida) → 1 record, `stderr` contiene `fuori da ogni funzione`.
  - `catalog_sinks_equal_server_slices`: albero Python con `os.system(cmd)`; l'uscita senza target è uguale a `build_graph` + `CodeGraph::slice` + `slice_to_json` con `SliceOptions::default()`.
  - `parse_args_rejects_missing_root_and_unknown_flag`.

- [ ] **Step 2: Verifica che falliscano**: `systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 cargo test --manifest-path sast-engine/Cargo.toml --test slice_cli` → errore di compilazione (`slice_cli` non esiste).

- [ ] **Step 3: Implementa** le firme sopra. `run` costruisce il grafo con `ScanLimits { max_files: usize::MAX, max_sinks: usize::MAX, ..cfg.scan.clone() }` (strumento offline, nessun tetto), parte da `cfg.slice.clone()` e sovrascrive con i flag; senza `--targets` emette una riga per ogni `graph.sinks`, con `--targets` chiama `add_target_sink` per riga e salta con `riga {n}: {file}:{line} fuori da ogni funzione` su stderr. `main.rs`: se il primo argomento è `slice`, esegue il sottocomando e termina, altrimenti avvia il server come oggi; la KB si sceglie con `CweKb::load_or_embedded(cfg.kb_path.as_deref())`, e `Engine::new` usa lo stesso helper.

- [ ] **Step 4: Verifica che passino** tutti i test: stesso comando di Step 2, poi `cargo test --manifest-path sast-engine/Cargo.toml` intero (102 test preesistenti ancora verdi) sotto lo stesso guard.

- [ ] **Step 5: Ricompila il binario**: `systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 cargo build --release --manifest-path sast-engine/Cargo.toml`; prova `sast-engine/target/release/sast-engine slice sast-engine/tests/fixtures/vulnerable_app | head -c 400`.

- [ ] **Step 6: Commit**: `git add sast-engine && git commit -m "feat(engine): sottocomando slice per esportare le backward slice in JSONL"`

---

### Task 2: `slice_io.py`

**Files:**
- Create: `ai-pipeline/slice_io.py`
- Test: `ai-pipeline/tests/test_slice_io.py`

**Interfaces:**
- Produces:
  - `engine_bin() -> Path | None`: `$SAST_ENGINE_BIN` o `../sast-engine/target/release/sast-engine` rispetto al modulo, se esiste.
  - `run_slice(root: Path, targets: list[dict] | None = None, *, depth: int = 5, callers: int = 4, max_chars: int = 12000, binary: Path | None = None) -> tuple[list[dict], str]`: lancia `<binary> slice <root> --depth … --callers … --max-chars … [--targets <file temporaneo>]`, restituisce (record JSON, stderr); errore (`RuntimeError`) se esce con codice ≠ 0.
  - `write_targets(path: Path, targets: list[dict]) -> None` (JSONL).

- [ ] **Step 1: Test con un motore finto** (script `sh` scritto nella cartella temporanea che stampa su stderr i propri argomenti e su stdout due righe JSON):
  - `test_run_slice_passes_defaults_and_parses_jsonl`: gli argomenti contengono `--depth 5 --callers 4 --max-chars 12000`, i record sono 2.
  - `test_targets_are_written_as_jsonl_and_passed`: con `targets=[{...}]` compare `--targets`.
  - `test_nonzero_exit_raises_runtime_error`.
  - `test_engine_bin_reads_environment` (`SAST_ENGINE_BIN` impostata a un file esistente).
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_slice_io -v` → `ModuleNotFoundError`.
- [ ] **Step 3: Implementa** con `subprocess.run` (stdlib, nessun `shell=True`).
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Commit**: `git add ai-pipeline/slice_io.py ai-pipeline/tests/test_slice_io.py && git commit -m "feat(pipeline): wrapper Python del sottocomando slice"`

---

### Task 3: `c_anonymize.py`

**Files:**
- Create: `ai-pipeline/c_anonymize.py`
- Test: `ai-pipeline/tests/test_c_anonymize.py`

**Interfaces:**
- Produces:
  - `LEAK_RE = re.compile(r"(?i)bad|good|g2b|b2g|flaw|fix|benign|cwe\d")`: controllo finale a sottostringa.
  - `strip_comments(code: str) -> tuple[str, list[int]]`: toglie `/* */` e `//` rispettando stringhe e caratteri letterali, elimina le righe rimaste vuote per un commento a riga intera e comprime le righe vuote consecutive; il secondo valore dà, per ogni riga nuova (indice 0-based), la riga vecchia (0-based).
  - `leaky_identifier(name: str, extra_parts: frozenset[str] = frozenset()) -> bool`: spezza su `_`, cifre e confini camelCase e controlla le *parti* `bad|good|g2b|b2g|flaw|fix|benign|cwe` più `extra_parts` (le tracce Python aggiungono `safe|unsafe|vuln|vulnerable|secure|insecure`).
  - `anonymize_slice(rec: dict, seed: str, macros: dict[str, str]) -> dict`: `rec` è un record di `slice` (Task 1). Restituisce una copia con: commenti tolti; macro riscritte (`SYSTEM`→`system`); funzioni definite nei blocchi → `fn_<6 esadecimali di sha256(seed:nome)>`; altri identificatori con `leaky_identifier` → `v_<6 hex>`; stringhe letterali che soddisfano `LEAK_RE` → `"text"`; file → `src/<8 hex>.c` (uno per file originale); `end_line = start_line + righe − 1` per blocco; `sink.line` ricalcolato con la mappa di `strip_comments`; `sink.file` uguale al file del blocco a profondità 0.

- [ ] **Step 1: Scrivi i test** (fixture: i blocchi 54d e 54e di `CWE78_OS_Command_Injection__char_environment_system_54*` con `badSink`, `goodG2BSink`, la chiamata `SYSTEM(data)` e `printLine("command execution failed!")`):
  - `test_strip_comments_removes_both_kinds_and_keeps_string_content`: `'x = "/* no */"; /* c */ // d'` → `'x = "/* no */";'`.
  - `test_strip_comments_line_map_points_to_original_lines`.
  - `test_whole_line_comment_leaves_no_blank_line`.
  - `test_leaky_identifier_matches_parts_not_substrings`: True per `dataGoodBuffer`, `goodG2BSource`, `badSink`, `CWE78_x_54e_badSink`; False per `badge_count`, `fixture`, `prefix`, `data`, `globalReturnsTrue`.
  - `test_defined_functions_get_one_consistent_name_across_blocks`: il nome chiamato nel blocco a profondità 1 è uguale al `name` del blocco a profondità 0.
  - `test_macro_is_rewritten_and_sink_line_follows`: `system(` compare nel blocco e `sink.line − start_line` indica proprio quella riga del codice del blocco.
  - `test_file_names_are_hidden_and_consistent`: tutti iniziano con `src/` e finiscono con `.c`; `sink.file` = file del blocco a profondità 0.
  - `test_leaky_literal_replaced_but_neutral_literal_kept`: `"Benign, fixed string"` → `"text"`; `"command execution failed!"` resta.
  - `test_deterministic_per_seed_and_different_across_seeds`.
  - `test_no_leak_pattern_survives`: `LEAK_RE.search(json.dumps(out)) is None` sulla fixture.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_c_anonymize -v`.
- [ ] **Step 3: Implementa** le firme sopra. Tokenizza il C con una regex che separa commenti, stringhe, caratteri e identificatori, così che nulla venga riscritto dentro stringhe o commenti.
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Commit**: `git add ai-pipeline/c_anonymize.py ai-pipeline/tests/test_c_anonymize.py && git commit -m "feat(pipeline): anonimizzazione del C per le slice Juliet"`

---

### Task 4: `juliet_graph.py`

**Files:**
- Create: `ai-pipeline/juliet_graph.py`
- Test: `ai-pipeline/tests/test_juliet_graph.py`

**Interfaces:**
- Consumes: `localize.split_c_functions(source) -> [{"name","code","start_line"}]`, `localize.find_calls(code, "c") -> [(callee, line_relativa_1based)]`, `common.match_sink`, `common.resolve_cwe`.
- Produces:
  - Costanti `JULIET_URL`, `JULIET_SHA256`, `JULIET_CWES = ("CWE-120","CWE-78","CWE-190","CWE-416")`, `MACROS: dict[str, str]` (almeno `SYSTEM EXECL EXECLP EXECV EXECVP POPEN SNPRINTF`, completata nello Step 5).
  - `fetch_juliet(dest_dir: Path) -> Path`: scarica se manca, verifica lo sha256 (`ValueError` se diverso), restituisce il percorso dello zip.
  - `extract_c_sources(zip_path: Path, out_dir: Path, kb: dict, cwes: tuple[str, ...] = JULIET_CWES) -> list[Path]`: estrae solo i `.c` sotto `C/testcases/CWE<n>_*/` la cui CWE si riporta (alias inclusi, `common.resolve_cwe`) a una di `cwes`; scrive UTF-8 (`errors="replace"`) con `\n`; rifiuta (`ValueError`) membri che escono da `out_dir`; restituisce una radice per directory di CWE Juliet (es. `CWE121_Stack_Based_Buffer_Overflow`).
  - `sink_targets(root: Path, kb: dict) -> list[dict]`: per ogni funzione `bad*`/`good*` (non `main`) e ogni chiamata il cui callee, dopo `MACROS`, è un sink del catalogo per la CWE del test case: `{"file": relativo a root con "/", "line": assoluta, "callee": nome vero, "cwe_id"}`.
  - `sink_label(function_name: str) -> int | None`: `bad` → 1; `good*` → 0; altro → None.
  - `variant_kind(function_name: str) -> str | None`: `"g2b" | "b2g" | "bad" | "good"`.
  - `testcase_group(file_name: str) -> str` (senza estensione né lettera finale dopo le cifre), `flow_variant(file_name: str) -> str | None` (le due cifre, es. `"54"`), `source_token(file_name: str) -> str | None` (es. `"environment"`).

- [ ] **Step 1: Scrivi i test** (zip costruito con `zipfile` in cartella temporanea):
  - `test_extract_writes_only_selected_cwes_and_c_files`: un `.cpp` e una CWE non selezionata non compaiono.
  - `test_extract_normalizes_crlf_and_invalid_bytes`: il file estratto non contiene `\r`.
  - `test_extract_refuses_members_outside_destination`: membro `C/testcases/CWE78_X/../../../evil.c` → `ValueError`.
  - `test_fetch_juliet_rejects_wrong_sha256`: file locale con il nome atteso e contenuto sbagliato.
  - `test_sink_targets_map_macros_and_give_absolute_lines`: `static void goodG2B()` con `SYSTEM(data);` alla riga 12 → `{"file":"s01/…_01.c","line":12,"callee":"system","cwe_id":"CWE-78"}`.
  - `test_sink_label`: `..._54e_badSink`→1, `..._54e_goodG2BSink`→0, `main`→None, `helper`→None.
  - `test_variant_kind`: `goodG2BSink`→`g2b`, `goodB2G`→`b2g`, `badSink`→`bad`, `good1`→`good`.
  - `test_group_variant_and_source`: `CWE78_OS_Command_Injection__char_environment_system_54e.c` → gruppo `CWE78_OS_Command_Injection__char_environment_system_54`, variante `"54"`, sorgente `"environment"`; `…_01.c` → variante `"01"`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_juliet_graph -v`.
- [ ] **Step 3: Implementa** le firme sopra (`urllib.request` per il download, `zipfile` con controllo del percorso risolto).
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Completa `MACROS` sui dati veri** (richiede rete, 146 MB): `fetch_juliet(Path("data/raw/juliet-nist"))`, `extract_c_sources(…, Path("data/raw/juliet-nist/src"), common.load_kb())`; elenca i callee in maiuscolo che compaiono nelle funzioni `bad*/good*` e il cui nome minuscolo è un sink del catalogo per la CWE; aggiungili a `MACROS` e aggiungi un test per ciascun nuovo nome. Annota nel commit anche quante radici e quanti target risultano per CWE (la CWE-22 non è nell'elenco, vedi spec).
- [ ] **Step 6: Commit**: `git add ai-pipeline/juliet_graph.py ai-pipeline/tests/test_juliet_graph.py && git commit -m "feat(pipeline): sorgenti, sink ed etichette Juliet dall'archivio NIST"`

---

### Task 5: `graph_examples.py` (funzioni pure)

**Files:**
- Create: `ai-pipeline/graph_examples.py`
- Test: `ai-pipeline/tests/test_graph_examples.py`

**Interfaces:**
- Consumes: `common.build_messages`, `common.render_card`, `common.validate_finding`, `hashlib` (stdlib); record di slice anonimizzati (Task 3) o rinominati (Task 8).
- Produces:
  - `PHRASES: dict[tuple[str, str], str]`: frase di motivazione per `(cwe_id, variant_kind)` con `variant_kind ∈ {"bad","g2b","b2g","good"}` per le 4 CWE Juliet; `GENERIC: dict[int, str]` con una frase per etichetta (1: origine non fidata che arriva al sink senza validazione; 0: valore costante o controllato) usata per ogni CWE fuori da `PHRASES` (tracce Python); `ORIGINS: dict[str, str]` da `source_token` a frase italiana, con fallback «dati esterni». Nessuna parola di `LEAK_RE`.
  - `build_example(rec: dict, kb: dict, *, label: int, variant_kind: str, origin: str | None, meta: dict, view: str = "callers") -> dict`: `{"messages": [system, user, assistant], "meta": {…}}`. User = `common.build_messages(cwe_id, render_card(entry), sink, blocks)`; assistant = JSON dello schema con `reasoning` del vocabolario chiuso (sink, funzione del blocco a profondità massima con la sua profondità, origine, motivo) e `suggested_fix` = `fix_hint` o `"Nessuna modifica necessaria."`. Fuori da Juliet `variant_kind` è `"bad"` se `label == 1`, altrimenti `"good"`. `meta` include le chiavi usate da `benchmark.evaluate` (`label`, `kind` = `"vulnerable"`/`"patched"` come in `prepare_data`, `cwe`, `lang`, `sink_function`, `sink_line`, `sink_kind="call"`, `source`, `synthetic`, `func_hash` = hash del prompt utente) più `variant_kind`, `track`, `group`, `variant`, `view`, `pair_id=None`, `trace={"cwe_id","sink","blocks"}`.
  - `sink_only(rec: dict) -> dict`: record con il solo blocco a profondità 0.
  - `prompt_key(example: dict) -> str`.
  - `drop_conflicts(examples: list[dict]) -> tuple[list[dict], int]`: toglie gli esempi il cui `prompt_key` compare con l'etichetta opposta; restituisce (rimasti, scartati).
  - `choose_views(candidates: list[dict], seed: int, p_callers: float = 0.7) -> list[dict]`: ogni candidato è `{"full": example, "sink_only": example | None}`; sceglie la view completa con probabilità `p_callers`, la solo-sink solo se esiste (cioè se non in conflitto).
  - `link_pairs(examples: list[dict]) -> list[dict]`: assegna `meta.pair_id` agli esempi con uguale `prompt_key` della view solo-sink ed etichette opposte; restituisce solo quelli.
  - `split_of_juliet(group: str, variant: str, seed: int, val_pct: int, test_pct: int, held_variants: set[str]) -> str`: `"unseen_variants"` se `variant in held_variants`, altrimenti `"train" | "val" | "test"` per hash del gruppo.
  - `holdout_variants(variants: list[str], seed: int, pct: int = 12) -> set[str]`: `round(pct/100 · n)` varianti scelte per hash.
  - `cap_balanced(examples: list[dict], max_total: int, seed: int) -> list[dict]`: campionamento uniforme per `(cwe, variant)` con etichette bilanciate.

- [ ] **Step 1: Scrivi i test** (record di slice a 2 blocchi costruiti a mano):
  - `test_example_has_multi_block_prompt_and_trace_meta`: `user` contiene 2 intestazioni `depth` (dopo il Task 5c; prima `profondità`); `len(meta.trace.blocks) == 2`; `meta.view == "callers"`; `common.validate_finding(json.loads(assistant)) == []`.
  - `test_reasoning_covers_every_cwe_and_kind`: per `JULIET_CWES × {"bad","g2b","b2g","good"}` esiste una frase e `LEAK_RE.search(frase) is None`; per `GENERIC[0]` e `GENERIC[1]` idem.
  - `test_cwe_outside_the_table_uses_the_generic_phrase`: un esempio CWE-601 con `label=1` ha un `reasoning` non vuoto che nomina il sink.
  - `test_reasoning_names_root_function_and_depth_and_has_no_cve`: contiene il nome del blocco più esterno e `depth 4` (per una catena a 5 blocchi; il reasoning è in inglese, R13) e non contiene `CVE-`.
  - `test_vulnerable_fix_is_catalog_hint_and_safe_fix_is_none_needed`.
  - `test_sink_only_keeps_depth_zero_block`.
  - `test_drop_conflicts_removes_prompts_seen_with_both_labels` → (rimasti, 2).
  - `test_choose_views_mixes_seventy_thirty_and_respects_conflicts`: su 1000 candidati la quota solo-sink sta in [0,25; 0,35]; se `sink_only is None` non viene mai scelta.
  - `test_link_pairs_pairs_equal_sink_only_prompt_with_opposite_labels`.
  - `test_split_is_stable_per_group_and_holds_out_variants`: tutti i file di un gruppo → stesso split; ogni gruppo di una variante tenuta fuori → `"unseen_variants"`.
  - `test_holdout_variants_picks_six_of_forty_eight`.
  - `test_cap_balanced_limits_total_and_keeps_label_balance` (|vulnerabili − sicuri| ≤ 1).
  - `test_truncated_slice_is_rejected`: `build_example` su `truncated=True` solleva `ValueError`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_graph_examples -v`.
- [ ] **Step 3: Implementa** le firme sopra. Le frasi di `PHRASES`/`ORIGINS` le scrive chi implementa, in inglese (decisione dell'utente, R13), nello stile «`system` in `fn_3a1f` receives a value that in `fn_91c2` (depth 4) is a constant string and cannot be influenced by the user»; per `ORIGINS` si parte dai `source_token` che compaiono nei file estratti (Task 4 Step 5).
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Commit**: `git add ai-pipeline/graph_examples.py ai-pipeline/tests/test_graph_examples.py && git commit -m "feat(pipeline): costruzione di esempi con slice, view, conflitti e split"`

---

### Task 5c: prompt, schede e hint in inglese (decisione dell'utente)

**Perché.** Nemotron è un modello addestrato soprattutto in inglese; il system prompt attuale gli chiede di scrivere `reasoning` e `suggested_fix` in italiano. Riaddestriamo dal modello base, quindi il contratto prompt può cambiare ora, prima di generare i dati. Cambiare `shared/` = cambiare il contratto prompt/matching (lo dice `shared/golden.json`): si rigenera tutto nel Task 13.

**Files:**
- Modify: `shared/prompt_template.json` (testi inglesi, `version` 1 → 2), `shared/golden.json` (sezioni `card` e `prompts`, vedi sotto)
- Modify: `sast-engine/cwe_kb.json` (solo `summary`, `indicators`, `safe_patterns`, `fix_hint` delle 20 voci)
- Modify: `ai-pipeline/common.py` (`render_card`), `sast-engine/src/rag_cwe.rs` (`render_card`)
- Modify: `ai-pipeline/prepare_data.py` (`NO_FIX`), `ai-pipeline/benchmark.py` (~r.39: prompt di sistema «senza slice»), `ai-pipeline/model_card.py` (~r.300 e ~r.402: le frasi che dicono che prompt e output sono in italiano)
- Modify: i test che citano il vecchio testo: `sast-engine/tests/pipeline.rs` (~r.69-71, intestazioni `righe…, profondità…`), test Rust e Python che citano stringhe italiane del prompt, della scheda o di `NO_FIX` (`grep -rn` sui file elencati; includi `ai-pipeline/tests/test_graph_examples.py`, `test_prepare_data.py`, `test_benchmark.py`, `test_model_card.py`, `test_common.py` se esistono).
- Non modificare: id, alias, severità, `sinks` e `name` del catalogo; il codice dei motori; commenti e docstring (restano in italiano come nel resto del codice); `ai-pipeline/data/*.jsonl` (si rigenerano nel Task 13); `README.md` (Task 13).

**Interfaces:**
- Produces (testi esatti):
  - `system`: `You are a SAST security analyzer. You receive a CWE rule, a candidate sink and the code trace (backward slice) that reaches it. Decide whether the sink is really vulnerable to that CWE: untrusted data must reach the sink without adequate validation or sanitization. Answer ONLY with a JSON object that conforms to the requested schema. Write "reasoning" and "suggested_fix" in English, citing files and lines.`
  - `user`: righe nell'ordine di oggi, con i segnaposto invariati (`{cwe_card}`, `{sink_function}`, `{sink_file}`, `{sink_line}`, `{trace}`, `{cwe_id}`) e questi testi: `## CWE rule`, `## Candidate sink`, `- Function: …`, `- File: …`, `- Line: …`, `## Code trace (from the outermost caller to the sink)`, `## Task`, `Decide whether the sink \`{sink_function}\` is vulnerable to {cwe_id}. If it is not, answer with "vulnerable": false, "vex_status": "not_affected" and "severity": "NONE".`
  - `block_header`: `### {file} · {name} (lines {start_line}-{end_line}, depth {depth})`
  - `render_card` (Python e Rust identici): `{id} — {name}` / `Description: {summary}` / `Vulnerability indicators:` / `- …` / `Safe patterns:` / `- …`
  - `NO_FIX = "No change needed."`
  - Catalogo: traduzione fedele in inglese tecnico e conciso dei testi italiani, senza cambiarne il significato né gli esempi di codice (es. `cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))` resta identico). Tutte le 20 voci.
- Consumes: i test di parità esistenti (`prompt.rs`, `rag_cwe.rs`, Python `tests/` che leggono `shared/golden.json`).

- [ ] **Step 1: Elenca** con `grep -rn` le stringhe italiane che finiscono nel prompt, nella scheda o nel target (distinguendole dai commenti/docstring, che restano). Scrivile nel report.
- [ ] **Step 2: Scrivi i test che falliscono**: (a) `ai-pipeline/tests/test_prompt_language.py` — `shared/prompt_template.json` e, per OGNI voce di `cwe_kb.json`, la scheda renderizzata e `fix_hint` non contengono parole funzionali italiane (`\b(il|lo|la|gli|della|delle|degli|dei|nel|nella|che|non|viene|vengono|senza|dati|usare|evitare|una|sono)\b`, case-insensitive); `system` contiene `in English`; `NO_FIX == "No change needed."`; `version == 2`; (b) aggiorna i test esistenti che citano il vecchio testo. Verifica che falliscano (test sul testo italiano).
- [ ] **Step 3: Implementa** le modifiche sopra. Poi rigenera `shared/golden.json` — SOLO le sezioni `card` e `prompts` (`normalize`, `callee_matches`, `kb_match`, `alias` restano identiche byte per byte): le righe attese si ricostruiscono con `common.render_card`/`common.render_user_prompt` sulle stesse entrate di oggi; aggiorna il campo `_comment` solo se dice qualcosa di falso.
- [ ] **Step 4: Verifica**: `cd ai-pipeline && python3 -m unittest discover -s tests -t .` (l'unico fallimento ammesso è il noto `tests.test_notebook…test_section_order`), poi `systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 cargo test --manifest-path sast-engine/Cargo.toml` (tutti verdi, nessun warning nuovo), poi ricompila il binario `cargo build --release` sotto lo stesso guard (serve ai task seguenti). Controlla a mano che `git diff --stat` tocchi solo i file elencati e che nel catalogo `id`, `aliases`, `default_severity`, `sinks` e `name` siano identici (confronto con `git show HEAD:sast-engine/cwe_kb.json` via script).
- [ ] **Step 5: Commit**: `git add shared sast-engine ai-pipeline && git commit -m "feat(prompt): prompt, schede e hint del catalogo in inglese"`

---

### Task 6: Verifica delle perdite e traccia Juliet end-to-end

**Files:**
- Create: `ai-pipeline/leak_audit.py`, `ai-pipeline/build_graph_data.py`
- Test: `ai-pipeline/tests/test_leak_audit.py`, `ai-pipeline/tests/test_build_graph_data.py`

**Interfaces:**
- Produces:
  - `leak_audit.forbidden_hits(examples: list[dict], mode: str = "substring") -> list[str]`: occorrenze nei `code`, `file`, `name` di `meta.trace.blocks`. `"substring"` usa `LEAK_RE` (per il C anonimizzato di Juliet); `"parts"` controlla gli identificatori con `leaky_identifier(…, extra_parts)` (per il codice Python vero, dove `prefix` o `fixture` non sono perdite).
  - `leak_audit.top_tokens(examples: list[dict], k: int = 30) -> list[tuple[str, float, int, int]]`: (token, log-odds con smoothing +1 fra etichetta 1 e 0, n_vulnerabili, n_sicuri), per modulo decrescente, solo dai blocchi di codice.
  - `leak_audit.bow_accuracy(train: list[dict], heldout: list[dict]) -> float`: naive Bayes a sacco di token addestrato su `train`, valutato su `heldout`.
  - CLI `python3 leak_audit.py --train F --heldout F` stampa le tre cose e esce con 1 se `forbidden_hits` non è vuoto.
  - `build_graph_data.py juliet --raw-dir data/raw --out-dir data/graph --eval-dir data/eval --seed 42 --max-train N --max-eval 600 [--zip PATH] [--engine-bin PATH]`: scarica/estrae (Task 4), per ogni radice scrive i target, chiama `run_slice` (profondità 5), unisce l'etichetta e `variant_kind`/`source_token`, anonimizza, scarta le slice troncate, costruisce gli esempi, `drop_conflicts`, `choose_views`, split, tetto. Scrive `data/graph/juliet_train.jsonl`, `juliet_val.jsonl`, `data/eval/juliet_test.jsonl`, `juliet_unseen_variants.jsonl`, `juliet_pairs.jsonl` e `data/graph/juliet_report.json` (candidati, troncati, conflitti, quota non decidibile, conteggi per CWE e variante, test case senza esempi).

- [ ] **Step 1: Scrivi i test**:
  - `test_leak_audit`: `forbidden_hits` trova `goodG2B` in un blocco e nulla in uno pulito; `top_tokens` mette in testa un token presente solo nelle sicure; `bow_accuracy` = 1,0 su un caso separabile; la CLI esce con 1 se ci sono perdite.
  - `test_build_graph_data` (motore finto di Task 2 che restituisce due slice con le stesse funzioni in versione `bad` e `goodG2B`): i file di output esistono; nessun `LEAK_RE` nei blocchi; `juliet_pairs.jsonl` contiene la coppia con `pair_id` uguale ed etichette opposte; `juliet_report.json` ha `truncated_dropped` e `conflicts_dropped`; un test case i cui prompt sono tutti in conflitto non produce esempi ed è contato in `testcases_without_examples`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_leak_audit tests.test_build_graph_data -v`.
- [ ] **Step 3: Implementa** le firme sopra.
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Esecuzione vera (cancello)**: `cd ai-pipeline && python3 build_graph_data.py juliet --raw-dir data/raw --out-dir data/graph --eval-dir data/eval --max-train 1300` poi `python3 leak_audit.py --train data/graph/juliet_train.jsonl --heldout data/eval/juliet_test.jsonl`. Atteso: codice di uscita 0 (nessuna perdita). Se ci sono perdite: aggiungi la regola in `c_anonymize.py` con il suo test e ripeti. Leggi a mano i 30 token e lascia in `data/graph/juliet_report.json` accuratezza del sacco di token, quota non decidibile e esempi per CWE. Soglie di allarme mie, non della spec: se una CWE ha < 50 esempi o la quota non decidibile supera 30%, fermati e mostra i numeri all'utente prima di proseguire.
- [ ] **Step 6: Commit** (codice e test; i dati si committano nel Task 13): `git add ai-pipeline/leak_audit.py ai-pipeline/build_graph_data.py ai-pipeline/tests/test_leak_audit.py ai-pipeline/tests/test_build_graph_data.py && git commit -m "feat(pipeline): traccia Juliet end-to-end con verifica delle perdite"`

---

### Task 7: `prepare_data.py`: target senza CVE, unione, dimensioni

**Files:**
- Modify: `ai-pipeline/prepare_data.py` (`_reasoning` ~r.107, `build_example` ~r.156-239, `ALL_SOURCES`, `main` ~r.464), `ai-pipeline/sources.py` (`codeql_records`)
- Test: `ai-pipeline/tests/test_prepare_data.py`, `ai-pipeline/tests/test_sources.py`

**Interfaces:**
- Consumes: file `data/graph/*_train.jsonl` e `*_val.jsonl` (Task 6 e successivi).
- Produces:
  - `_reasoning` non usa più `cve_desc`, `cve` né `note`; per i vulnerabili, in inglese (R13): "The sink `X` at line N of F receives unvalidated data and is vulnerable to CWE-NN (name)."; per i sicuri una frase inglese analoga ("... does not receive untrusted data ..."), senza parole di `LEAK_RE`.
  - `suggested_fix` = `entry["fix_hint"]` per i vulnerabili; `_upstream_fix`, `FIX_PREFIX`, `MAX_DIFF_LINES` rimossi (`grep -n` e aggiorna i test che li citano).
  - `ALL_SOURCES` senza `juliet`; `--juliet` resta ma l'help dice «legacy: contiene etichette nel codice».
  - `--graph-dir DIR`: dopo il ciclo delle fonti accoda a `train` e `val` gli esempi di `DIR/*_train.jsonl` e `DIR/*_val.jsonl`; con `--graph-dir` il lettore CodeQL salta i file Python (`sources.codeql_records(files, kb, skip_python=True)`).
  - Rapporto finale: dimensione di ogni file scritto, avviso se > 50 MB, conteggi per `meta.source`, e il numero di esempi con `len(user)+len(assistant) > 8192·3,5` caratteri (stima dei token oltre `max_seq_length`), scartati e contati.

- [ ] **Step 1: Scrivi/aggiorna i test**:
  - `test_vulnerable_reasoning_has_no_cve_id_description_or_commit_note`: record con `cve="CVE-2020-1"`, `cve_desc="boom"`, `note="fix x"` → il `reasoning` non contiene `CVE-`, `boom`, `fix x`.
  - `test_vulnerable_suggested_fix_is_catalog_hint_not_upstream_diff`.
  - `test_graph_dir_examples_go_to_train_and_val_only_and_test_is_unchanged`: stessa composizione di `test.jsonl` con e senza `--graph-dir`.
  - `test_graph_dir_skips_python_codeql_but_keeps_c`.
  - `test_report_warns_for_files_over_50_mb` (soglia iniettabile).
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_prepare_data tests.test_sources -v`.
- [ ] **Step 3: Implementa** le modifiche sopra.
- [ ] **Step 4: Passano**: `cd ai-pipeline && python3 -m unittest discover -s tests -t .` intero (nessuna regressione sugli altri 200+ test).
- [ ] **Step 5: Commit**: `git add ai-pipeline && git commit -m "feat(data): target senza CVE, fix dal catalogo, unione degli esempi con slice"`

---

### Task 8: Traccia P1 (test CodeQL Python)

**Files:**
- Create: `ai-pipeline/codeql_graph.py`
- Modify: `ai-pipeline/build_graph_data.py` (sottocomando `codeql`)
- Test: `ai-pipeline/tests/test_codeql_graph.py`

**Interfaces:**
- Consumes: `localize.scrub_labels(code, "#")`, `localize.annotated_lines(code, "#") -> {"alert": set[int], "ambiguous": set[int]}`, `slice_io.run_slice`.
- Produces:
  - `prepare_tree(codeql_root: Path, dest: Path) -> dict[str, dict]`: copia i `.py` sotto `python/ql/test/query-tests/Security/CWE-*` (esclusi i percorsi con `py2`) con le annotazioni tolte e le righe invariate; restituisce `{percorso_relativo: {"alert", "ambiguous", "cwe"}}` con `cwe` dal nome della directory (`CWE-078-…` → `CWE-78`, via `common.resolve_cwe`).
  - `label_slices(slices: list[dict], notes: dict[str, dict]) -> list[tuple[dict, int]]`: etichetta 1 se `sink.line` è in `alert`; salta se in `ambiguous`; 0 se `sink.cwe_id` coincide con la CWE della directory; salta gli altri.
  - `rename_leaky_names(rec: dict, seed: str) -> dict`: sui record Python, i nomi di funzione e variabile definiti nella slice per cui `leaky_identifier(name, extra_parts)` è vero (nei test CodeQL ricorrono `bad`, `good`, `safe`, `unsafe`) diventano `fn_<6 hex>` / `v_<6 hex>` con la tokenizzazione `tokenize` della stdlib (stringhe e commenti intatti); i file diventano `src/<8 hex>.py`.
  - Una radice di `run_slice` per directory CWE. Gruppo per lo split = directory della query (`split_of` di `prepare_data`); scrive `data/graph/py_codeql_{train,val}.jsonl` e `data/eval/py_codeql.jsonl`, con le stesse regole di Task 5 (conflitti, view 0,7/0,3); controllo finale `forbidden_hits(…, mode="parts")` vuoto.

- [ ] **Step 1: Scrivi i test** (albero finto `python/ql/test/query-tests/Security/CWE-078-CommandInjection/test.py` con `os.system(cmd)  # $ Alert` e `os.system("ls")`, più una funzione di passaggio):
  - `test_prepare_tree_strips_annotations_and_keeps_line_numbers`.
  - `test_cwe_comes_from_directory_name`: `CWE-078-…` → `CWE-78`.
  - `test_label_slices_alert_is_1_plain_sink_is_0_ambiguous_skipped` (record di slice finti).
  - `test_py2_paths_are_ignored`.
  - `test_rename_leaky_names_hides_bad_good_safe_unsafe_and_keeps_calls_consistent` (`def bad_cmd(x)` chiamata da `good_caller` → nomi neutri, chiamata coerente, `fixture` e `prefix` intatti, `"unsafe"` dentro una stringa intatta).
  - Se `slice_io.engine_bin()` esiste: `test_end_to_end_with_real_engine` (`skipUnless`) → 1 vulnerabile e 1 sicuro.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_codeql_graph -v`.
- [ ] **Step 3: Implementa** le firme sopra e il sottocomando.
- [ ] **Step 4: Passano**: stesso comando; poi esecuzione vera su `data/raw/codeql-tests` e controllo a mano di 5 esempi (annotazione assente dal prompt, etichetta coerente).
- [ ] **Step 5: Commit**: `git add ai-pipeline && git commit -m "feat(pipeline): traccia P1 con le slice dei test CodeQL Python"`

---

### Task 9: Traccia P3 (sintetici Python)

**Files:**
- Create: `ai-pipeline/synth_python.py`
- Modify: `ai-pipeline/build_graph_data.py` (sottocomando `synth`)
- Test: `ai-pipeline/tests/test_synth_python.py`

**Interfaces:**
- Produces:
  - `generate(out_dir: Path, n: int, seed: int) -> list[dict]`: scrive `n` piccoli pacchetti Python e restituisce il manifesto `{"file","line","callee","cwe_id","label","chain_len"}` (un sink per pacchetto). Sink con argomento costante (label 0) o derivato da input (label 1) attraverso `chain_len ∈ {0,1,2,3}` funzioni intermedie; nomi e struttura variati; CWE 601, 78, 89, 22; i callee vengono dai sink Python del catalogo.
  - `cap_share(synthetic: list[dict], real: list[dict], share: float = 0.15) -> list[dict]`: tiene al più `share` del totale `real + synthetic`.
  - Gli esempi vanno solo in `data/graph/py_synth_{train,val}.jsonl`, mai in `data/eval/`.

- [ ] **Step 1: Scrivi i test**:
  - `test_sources_parse_and_manifest_line_holds_the_callee` (`ast.parse`; la riga indicata contiene il callee).
  - `test_callees_are_catalog_sinks` (`common.match_sink(kb, "python", callee)` non `None`).
  - `test_labels_are_balanced_and_chain_lengths_vary` (|1 − 0| ≤ 1; tutte le `chain_len` presenti).
  - `test_deterministic_for_same_seed`.
  - `test_no_identifier_carries_label_words` (nessuna parte tra `safe, unsafe, tainted, vuln, vulnerable, clean, const, bad, good`).
  - `test_cap_share_never_exceeds_fifteen_percent`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_synth_python -v`.
- [ ] **Step 3: Implementa** le firme sopra con template composti in Python (nessun modello, nessuna rete).
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Commit**: `git add ai-pipeline && git commit -m "feat(pipeline): traccia P3 con varianti Python sintetiche"`

---

### Task 10: Pilota P2 (CVEfixes/PyVul con checkout)

**Files:**
- Create: `ai-pipeline/pyvul_pilot.py`
- Test: `ai-pipeline/tests/test_pyvul_pilot.py`

**Interfaces:**
- Consumes: righe del parquet (`repo_url`, `commit_hash`, `function_name`, `code_before`, `code_after`, `cwe_id`, `cve_id`) via `sources.read_rows`; `localize.localize_sink`.
- Produces:
  - `select_repos(rows: list[dict], n: int, seed: int) -> list[str]`: gli `n` repo con più righe utilizzabili (spareggio per hash con seme).
  - `fetch_commit(repo_url: str, sha: str, dest: Path, timeout: int = 300) -> bool`: `git init`, `git fetch --depth 2 origin <sha>`, checkout del commit; `False` su errore o timeout.
  - `find_function(root: Path, function_name: str, code: str) -> str | None`: percorso relativo del `.py` che contiene una `def function_name` il cui corpo, normalizzato negli spazi, coincide con `code`.
  - `pilot_report(rows_total, rows_resolved, examples, seconds, bytes_on_disk) -> dict`.
  - CLI `python3 pyvul_pilot.py --parquet data/raw/cvefixes_pyvul.parquet --repos 20 --work-dir PATH --out-dir data/graph --eval-dir data/eval`: per ogni riga, vulnerabile = slice del sink toccato dalla fix sul **genitore** (`fetch_commit(sha + "^")`), corretto = stessa funzione sul commit di fix; gruppo per lo split = repo (`split_of`); scrive `py_pyvul_{train,val}.jsonl` in `--out-dir` e la parte di test in `--eval-dir/py_cvefixes.jsonl`; `forbidden_hits(…, mode="parts")` viene solo **riportato**, non blocca (nei repo veri nomi come `is_safe_url` sono legittimi); nessuna rinomina.

- [ ] **Step 1: Scrivi i test** (nessuna rete: repo git locale creato con `git init` in cartella temporanea e `fetch_commit` puntato a `file://`):
  - `test_find_function_matches_by_name_and_normalized_body` e `…_returns_none_when_body_differs`.
  - `test_select_repos_prefers_more_rows_and_is_deterministic`.
  - `test_fetch_commit_checks_out_the_sha` (repo locale con due commit).
  - `test_fetch_commit_returns_false_on_unknown_sha`.
  - `test_pilot_report_computes_yield_per_repo`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_pyvul_pilot -v`.
- [ ] **Step 3: Implementa** le firme sopra (`subprocess` con `timeout`, nessun `shell=True`; `--work-dir` fuori da `/tmp`, ad esempio `ai-pipeline/outputs/pyvul-pilot`).
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Pilota vero (cancello)**: esegui la CLI su 20 repo e mostra all'utente `pilot_report` (righe risolte su totali, esempi per repo, tempo, spazio). Non estendere oltre 20 repo senza la sua decisione.
- [ ] **Step 6: Commit**: `git add ai-pipeline/pyvul_pilot.py ai-pipeline/tests/test_pyvul_pilot.py && git commit -m "feat(pipeline): pilota P2 con checkout dei commit CVEfixes/PyVul"`

---

### Task 11: `benchmark.py`: «con chiamanti» contro «solo sink»

**Files:**
- Modify: `ai-pipeline/benchmark.py` (`SYSTEMS` ~r.45, `cmd_accuracy` r.433, `parse_args` r.486, `evaluate` r.77)
- Test: `ai-pipeline/tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `SYSTEMS` esteso con `"llm-graph"` e `"llm-sink-only"`.
  - `rerender_messages(example: dict, max_depth: int | None = None, sink_only: bool = False, kb: dict | None = None) -> list[dict]`: da `meta.trace` ricostruisce system+user con `common.build_messages`, tenendo i blocchi con `depth <= max_depth` (o solo `depth == 0` se `sink_only`); senza `meta.trace` restituisce `slice_messages(example)`.
  - `pair_report(examples: list[dict], preds: list[bool | None]) -> dict | None`: sugli esempi con `meta.pair_id`: `{"pairs": n, "accuracy": esatti/esempi, "both_correct": coppie_giuste/n}`; `None` se non ce ne sono. `evaluate` lo aggiunge come `"pairs"`.
  - `accuracy` accetta `--depth N` (default `None`).

- [ ] **Step 1: Scrivi i test**:
  - `test_sink_only_prompt_has_exactly_one_block_header` (conta `depth` == 1 nelle intestazioni e il codice a profondità 0 c'è; intestazione inglese dopo il Task 5c).
  - `test_max_depth_filters_blocks` (`max_depth=1` su blocchi 0..4 → intestazioni a profondità 0 e 1).
  - `test_example_without_trace_falls_back_to_stored_messages`.
  - `test_pair_report_accuracy_and_both_correct` (4 esempi, 2 coppie, una giusta per intero e una a metà → `accuracy == 0.75`, `both_correct == 0.5`).
  - `test_pair_report_is_none_without_pairs`.
  - `test_accuracy_parser_accepts_new_systems_and_depth`.
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_benchmark -v`.
- [ ] **Step 3: Implementa** (in `cmd_accuracy`, `llm-graph` → `rerender_messages(ex, args.depth)`, `llm-sink-only` → `rerender_messages(ex, sink_only=True)`).
- [ ] **Step 4: Passano**: `cd ai-pipeline && python3 -m unittest discover -s tests -t .` intero.
- [ ] **Step 5: Commit**: `git add ai-pipeline/benchmark.py ai-pipeline/tests/test_benchmark.py && git commit -m "feat(benchmark): sistemi con chiamanti e solo sink, metrica sulle coppie"`

---

### Task 12: Lista congelata pygoat (S3, S4)

**Files:**
- Create: `ai-pipeline/eval_frozen.py`, `ai-pipeline/data/eval/pygoat_frozen.json`
- Modify: `ai-pipeline/benchmark.py` (sottocomando `frozen`; `chat_completion` r.127 restituisce anche `"finish_reason"`, additivo)
- Test: `ai-pipeline/tests/test_eval_frozen.py`, `ai-pipeline/tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `candidates(repo: Path, binary: Path | None = None) -> list[dict]`: una voce per sink del catalogo (profondità 5): `{"file","line","callee","cwe_id","excerpt"}` dove `excerpt` sono 5 righe attorno al sink, per la revisione a mano.
  - `load_frozen(path: Path) -> dict`: valida `{"repo","commit","entries":[{"file","line","callee","cwe_id","expected":"safe"|"vulnerable","evidence"}]}`; `ValueError` se manca un campo o `expected` è diverso.
  - `score_frozen(frozen: dict, results: list[dict]) -> dict`: `results[i] = {"vulnerable": bool|None, "truncated": bool, "reasoning": str}` per `entries[i]`; restituisce `fp_still_affected` (quota di `safe` con `vulnerable is True`), `tp_recall` (quota di `vulnerable` con `True`), `truncated_rate` (su tutte), `cve_mentions` (occorrenze di `CVE-\d{4}-\d+` nei reasoning).
  - `benchmark.py frozen --frozen F --repo DIR --url U --model M --depth 5 --max-tokens 768`: slice dei sink della lista (`slice_io` con `--targets`), chiamata a vLLM con lo stesso `response_format` json_schema dello schema (`common.load_schema()`), `temperature` 0 e `enable_thinking=False` del motore, stampa e salva `score_frozen`. Troncato = JSON non valido o `finish_reason == "length"`.

- [ ] **Step 1: Scrivi i test**: `test_chat_completion_returns_finish_reason` (server finto che risponde `"finish_reason": "length"` → il dizionario restituito lo riporta); `test_load_frozen_rejects_missing_field_and_bad_expected`; `test_score_frozen_counts_fp_tp_truncation_and_cve` (6 voci con risultati noti → numeri esatti); `test_candidates_use_the_engine_and_add_an_excerpt` (motore finto di Task 2).
- [ ] **Step 2: Falliscono**: `cd ai-pipeline && python3 -m unittest tests.test_eval_frozen -v`.
- [ ] **Step 3: Implementa** le firme sopra.
- [ ] **Step 4: Passano**: stesso comando.
- [ ] **Step 5: Costruisci la lista vera**: `git -C ai-pipeline/outputs/repos/pygoat/repo rev-parse HEAD` per il `commit`; `candidates(...)` sui sink di pygoat; leggi ogni sink sul sorgente e inserisci in `pygoat_frozen.json` solo quelli che puoi giudicare con certezza (`expected` + `evidence` di una riga): i falsi positivi certi (destinazione costante di `redirect`, `subprocess` con lista costante, `CREATE TABLE`, `requests.get` con URL da settings) e i veri positivi che verifichi uno per uno. Quelli dubbi restano fuori. Mostra all'utente la lista e i conteggi prima del commit.
- [ ] **Step 6: Commit**: `git add ai-pipeline/eval_frozen.py ai-pipeline/benchmark.py ai-pipeline/tests/test_eval_frozen.py ai-pipeline/data/eval/pygoat_frozen.json && git commit -m "feat(eval): lista pygoat congelata e sottocomando frozen"`

---

### Task 13: Genera i dati finali e documenta

**Files:**
- Modify: `README.md` (sezioni *Dataset* e *Stato*), `ai-pipeline/data/{train,val,test}.jsonl`, `ai-pipeline/data/stats.json`
- Create: `ai-pipeline/data/graph/*.jsonl`, `ai-pipeline/data/eval/*.jsonl`

- [ ] **Step 1: Baseline di composizione**: `wc -l ai-pipeline/data/*.jsonl` e `python3 -c` che stampa il numero di esempi di `train.jsonl` senza Juliet (4.860 attesi) per calcolare il tetto Juliet = 0,25 × quel numero.
- [ ] **Step 2: Rigenera le tracce** con i tetti: `build_graph_data.py juliet --max-train <tetto>`, `codeql`, `synth` (con `cap_share` 15% sul totale Python con slice); i sintetici non scrivono in `data/eval/`.
- [ ] **Step 3: Rigenera i dataset** (il README documenta solo `--download --test-benign-ratio 3`; qui si usano i file grezzi già presenti e lo stesso rapporto): `cd ai-pipeline && python3 prepare_data.py --train data/raw/primevul_train.jsonl --valid data/raw/primevul_valid.jsonl --test data/raw/primevul_test.jsonl --secvuleval data/raw/secvuleval.parquet --python-pairs data/raw/cvefixes_pyvul.parquet --codeql data/raw/codeql-tests --test-benign-ratio 3 --graph-dir data/graph --out-dir data` (senza `--juliet`). Verifica, confrontando con `git show HEAD:ai-pipeline/data/stats.json`: `test.jsonl` ha ancora 3.266 esempi e la stessa composizione per fonte; il train senza `graph` ha 4.860 esempi meno eventuali tolti dal passaggio CodeQL Python; nessun file > 50 MB; `python3 -c "import json;print(sum('CVE-' in json.loads(l)['messages'][-1]['content'] for l in open('data/train.jsonl')))"` stampa 0; `python3 leak_audit.py --train data/graph/juliet_train.jsonl --heldout data/eval/juliet_test.jsonl` esce con 0.
- [ ] **Step 4: Aggiorna `README.md`** con i nuovi conteggi per fonte, il formato `meta.trace`, i file di valutazione e il comando di rigenerazione.
- [ ] **Step 5: Test completi**: `cd ai-pipeline && python3 -m unittest discover -s tests -t .` e `cargo test` (con guard) verdi.
- [ ] **Step 6: Commit** (anche i dati, che il notebook clona): `git add README.md ai-pipeline/data/*.jsonl ai-pipeline/data/stats.json ai-pipeline/data/graph ai-pipeline/data/eval && git commit -m "data: dataset con slice reali (Juliet C, Python), target senza CVE"`
- [ ] **Step 7: Chiedi all'utente di fare `git push`** e lanciare il run su Colab (un solo run, dal modello base, configurazione attuale). Non fare il push.

---

### Task 14: Baseline «prima» (richiede il via dell'utente per vLLM)

- [ ] **Step 1: Chiedi conferma** per avviare vLLM col modello attuale (`scripts/serve_vllm.sh` con `VLLM_BIN=/home/gabs/.venvs/tunatron-vllm/bin/vllm`, sotto `systemd-run … MemoryMax=14G`).
- [ ] **Step 2: Misura il modello attuale** sugli stessi file del dopo (nota dal Task 5c: il modello attuale ha visto SOLO prompt italiani; per S5 usa il vecchio `data/test.jsonl` italiano da `git show 13aceea:ai-pipeline/data/test.jsonl`, e per S1–S4 i nuovi file inglesi dichiarando che per il vecchio modello è un prompt mai visto): `benchmark.py accuracy --test data/eval/juliet_pairs.jsonl --systems llm-graph llm-sink-only --depth 5 --out results/before/juliet_pairs_d5.json`; stesso con `--depth 3`; poi `juliet_test`, `juliet_unseen_variants`, `py_codeql`, `data/test.jsonl` (S5) e `benchmark.py frozen`.
- [ ] **Step 3: Spegni vLLM** per PID (mai `pkill -f`) e committa `results/before/` (`git add results/before && git commit -m "results: baseline del modello attuale sui nuovi insiemi di valutazione"`).

### Task 15: Valutazione «dopo» e decisione sulla profondità

- [ ] **Step 1: Dopo il run Colab** e la quantizzazione NVFP4 (flusso esistente), ripeti Task 14 Step 2 sul nuovo modello in `results/after/`.
- [ ] **Step 2: Riporta S1–S5** in una tabella (valori «prima», «dopo», soglia) con i conteggi, per traccia, per famiglia di variante e per profondità 3 e 5. Se una soglia non è rispettata, dillo in chiaro con i numeri.
- [ ] **Step 3: Solo se S1 con profondità 5 batte quello con 3**: porta il default di `SLICE_DEPTH` a 5 in `sast-engine/src/config.rs`, aggiorna il test che ne verifica il default e il README; verifica `cargo test` con guard e committa (`git commit -m "feat(engine): profondità di slice predefinita 5"`). Altrimenti lascia 3 e scrivilo nel report.
