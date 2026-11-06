# Dati di training con il grafo: slice reali

Data: 2026-11-06 · Stato: bozza da rivedere · Percorso: architetturale

## Obiettivo

Detto dall'utente: «io volevo che il modello imparasse a individuare codice non sicuro anche con l'aiuto del "grafo"».
Oggi non lo impara: nessun esempio di training contiene un chiamante. Questa spec definisce i **dati di training** (e le valutazioni
che li giudicano) perché il modello usi i chiamanti della backward slice quando servono. L'addestramento resta su Colab, **un solo run**.

Assunzioni mie, da correggere in revisione:

- «Grafo» = la backward slice che il motore Rust già costruisce (profondità, 4 chiamanti per nodo, 12.000 caratteri). Nessun grafo nuovo.
- La prima corsa copre **C (Juliet) e Python**. Solidity e altri linguaggi sono una traccia successiva (vedi *Fuori ambito*).
- La valutazione gira sul PC (vLLM + NVFP4), non su Colab.

## Perché serve (misurato nello spike del 2026-11-06)

- Train 5.985 / val 402 / test 3.266: **il 100% ha un solo blocco a profondità 0**. Il motore invece manda fino a profondità 3.
  Anche `benchmark.py` «llm-slice» usa lo stesso prompt a un blocco: nulla misura l'apporto dei chiamanti.
- Su pygoat, 52 verdetti «affected»: almeno 44 sono falsi positivi certi (`redirect("login")` con destinazione costante e simili).
  Con la sola funzione intera, 7 casi su 8 tornano «vulnerable=True»: il modello non segue il flusso del dato.
- 29 risposte su 82 sono JSON troncato a 768 token; il reasoning contiene descrizioni di CVE inventate, perché `prepare_data._reasoning`
  accoda `Descrizione {CVE}: …` presa dall'esempio di origine.
- I 1.125 esempi Juliet in train (19%) si risolvono senza guardare il codice: 1.124 hanno `bad`/`good` nel nome di funzione o file e tutti
  contengono `/* FLAW */` o `/* FIX */`. Nel test non c'è Juliet, quindi le metriche non lo mostrano.
- Il target dei negativi è una frase fissa («i dati raggiungono il sink con controlli o limiti adeguati»), uguale per qualsiasi codice:
  non insegna a indicare da dove arriva il dato.

## Criteri di successo

Le soglie sono **proposte mie**, da confermare in revisione. Si misurano su file di valutazione mai visti in training.

| # | Cosa | Come | Soglia proposta |
|---|------|------|-----------------|
| S1 | Il grafo serve (test causale) | Coppie Juliet con lo stesso prompt «solo sink» ed etichette opposte; stesso modello, con e senza chiamanti | con chiamanti ≥ 70%; senza chiamanti 50% per costruzione |
| S2 | Niente scorciatoia sul template | Accuratezza su varianti di flusso mai viste in training, contro il val | scarto ≤ 10 punti |
| S3 | Falsi positivi reali | Le 44 funzioni pygoat «certe» (lista congelata e verificata sul sorgente): quante restano «affected» | ≤ 50% (oggi 100% per selezione); recall sui veri positivi verificati non peggiore |
| S4 | Risposte complete e senza CVE inventate | pygoat con `max_tokens` 768: JSON troncati; occorrenze di `CVE-\d{4}-` nel reasoning | troncati < 2% (oggi 29/82); CVE inventate 0 |
| S5 | Nessuna regressione | `test.jsonl` esistente (3.266, un blocco) | F1 non peggiore di 2 punti del modello attuale, misurato prima |

S1 è il risultato che risponde alla domanda dell'utente; gli altri dicono se l'aggiunta ha rotto qualcosa o corretto i difetti noti.

## Ambito

**Dentro**

- Sottocomando `sast-engine slice` (lo stesso codice di slice del server, su una cartella locale).
- Nuovo modulo `ai-pipeline/graph_examples.py` che trasforma le slice in esempi di training e valutazione.
- Quattro tracce: **J** Juliet C; **P1** test CodeQL Python; **P2** sottoinsieme di CVEfixes/PyVul (prima un pilota); **P3** varianti Python sintetiche.
- Riscrittura del target (reasoning e fix) per tutte le fonti; tolti dal train i vecchi esempi Juliet.
- Estensioni di `benchmark.py accuracy` per confrontare «con chiamanti» e «solo sink».

**Fuori ambito**

- **Solidity e altri linguaggi.** Richiedono: grammatica tree-sitter compatibile con la 0.27 del motore (non verificata) e una variante in `Lang`;
  `sinks` per linguaggio nel catalogo; una mappatura CWE/SWC; una fonte di dati con licenza verificata; un test set; la verifica che il modello base
  conosca il linguaggio. Nessuna delle 20 CWE attuali è tipica degli smart contract. Traccia a parte, con la sua spec.
- **Juliet C++.** Juliet C++ mette le funzioni in `namespace`, quindi lo stesso nome (`badSink`) ricorre in molti test case e la risoluzione per
  nome del motore (`MAX_AMBIGUOUS` = 3) scarterebbe o sbaglierebbe gli archi. Non verificato sull'albero: da controllare nel piano.
- Nuove CWE o nuovi sink nel catalogo di produzione (`cwe_kb.json`). I sink macro di Juliet **non** vanno nel catalogo.
- Chiamate tramite puntatore a funzione (variante Juliet 65): il motore non le risolve, quindi nemmeno qui.
- Cambi di iperparametri in `train.py`, un secondo training, quantizzazione e pubblicazione (restano il flusso esistente).

## Disegno

```
Juliet (NIST zip) ──┐
test CodeQL ────────┤   sast-engine slice            slices.jsonl      graph_examples.py        data/{train,val,test}.jsonl
repo a un commit ───┼─► (build_graph + CodeGraph ──►  (TraceBlock,  ──► anonimizza, etichetta, ─► data/eval/*.jsonl
sintetici Python ───┘    ::slice, come il server)      sink, flag)       deduplica, dividi
```

### 1. `sast-engine slice`

`main.rs` oggi avvia solo il server, senza argomenti. Si aggiunge il sottocomando, con il server invariato come comportamento predefinito:

```
sast-engine slice <ROOT> [--targets FILE] [--depth N] [--callers N] [--max-chars N]   # JSONL su stdout
```

- Per default i sink sono quelli del catalogo, come in `/analyze`. Ogni riga: `{"sink": {function, file, line, cwe_id}, "blocks": [TraceBlock…], "truncated": bool}`
  con `TraceBlock` identico a quello di `prompt.rs`.
- `--targets` (JSONL di `{file, line, callee, cwe_id}`) calcola la slice della funzione che contiene la riga indicata. Serve a Juliet, i cui sink sono macro
  (vedi sotto). Una riga che non cade in una funzione viene segnalata su stderr e saltata.
- Nessuna rete, nessun git: la radice è una cartella locale. `build_graph` e `CodeGraph::slice` non cambiano, così addestramento e inferenza vedono la stessa slice.
- Test: catena di 3 funzioni con profondità tagliata, `--targets`, riga fuori funzione, e uguaglianza fra l'uscita di `slice` e la `Slice` del server sullo stesso input.

### 2. Traccia J: Juliet C

**Fonte.** NIST SARD, test suite 112, `https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip`,
152.957.342 byte, sha256 `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb` (scaricato il 2026-11-06). Licenza: la pagina dice «not subject to copyright
protection … in the public domain» e, per i diritti esteri, CC0 1.0. Si scarica in `data/raw/juliet-nist/` (già ignorato da git) ed estrae **solo** i `.c` delle CWE usate, controllando che
nessun percorso esca dalla cartella di destinazione. Il `juliet_test.csv` locale non basta: ha catene incomplete.

**CWE.** 120, 78, 190, 416: sono le sole per cui, con il catalogo, Juliet ha sink-chiamata (misure in appendice). Le altre CWE di Juliet non sono raggiungibili dal motore:
401, 476, 369, 362 e 377 non hanno sink C nel catalogo; la 327 usa funzioni (`CryptCreateHash`…) che il catalogo non elenca; per la 22 il catalogo ha sink (`fopen`, `open`…) ma la stima dà zero, e va capito perché.

**Sink.** Ogni chiamata in una funzione `bad*`/`good*` il cui nome, con le macro in maiuscolo riportate al nome vero (`SYSTEM`→`system`, `EXECL`→`execl`, `SNPRINTF`→`snprintf`),
è un sink del catalogo per la CWE del test case. Si scrivono come `--targets`. Non si usa il commento FLAW per trovare il sink: per la CWE-78 il sink che `localize.flaw_sink` ricava dal commento è più spesso `strcat`
(2.650 volte, la preparazione del dato), non la chiamata che esegue il comando (`SYSTEM`). Nel codice dei blocchi le macro vengono riscritte col nome vero, perché il prompt dice `system`.

**Etichetta.** Dal nome originale della funzione che contiene il sink, *prima* dell'anonimizzazione: `bad` → vulnerabile, `good*` → non vulnerabile.

**Anonimizzazione** (sul JSONL delle slice, coerente dentro l'esempio):

1. Via **tutti** i commenti (`/* */`, `//`): contengono FLAW, FIX, INCIDENTAL e descrizioni.
2. Ogni funzione definita nella slice riceve un nome neutro (`fn_<4 esadecimali>`); ogni identificatore che contiene `bad|good|g2b|b2g|flaw|fix|benign|cwe\d*`
   (anche variabili, per esempio `dataGoodBuffer`) riceve un nome neutro. I percorsi dei file diventano `src/<hash>.c`.
3. Righe ricalcolate sul codice ripulito: intestazioni dei blocchi (`righe a-b`), `sink.line` e `sink_file` restano coerenti.
4. **Verifica delle perdite** (bloccante): nessun blocco o intestazione contiene più i pattern sopra; elenco dei 30 token con log-odds più alto fra le due etichette, da leggere a mano;
   accuratezza di un classificatore a sacco di token su gruppi tenuti fuori, riportata. Una stringa letterale rivelatrice trovata qui si aggiunge alle regole.

**Profondità 5.** In una catena `_54` il sink è in `_54e` e la sorgente del dato in `_54a`: sono 4 salti, e il motore oggi si ferma a 3, cioè esclude proprio la funzione che decide l'etichetta.
Training e valutazione usano profondità 5; il default del motore (3) si porta a 5 solo se S1 con profondità 5 batte quello con 3 (lo stesso modello si misura a entrambe le profondità).

**Ambiguità.** Un esempio entra solo se il suo prompt completo non compare con l'etichetta opposta. Quelli in conflitto vanno in un rapporto (la loro quota dice quanto il grafo del motore
non basta a decidere, per esempio con puntatori a funzione).

**Divisione.** Gruppo = test case (nome senza lettera finale: `_54a…_54e` insieme). In più, circa il 12% delle varianti di flusso (≈6 su 48, scelte con hash e seme) è escluso da train e val e
forma `J-test-var`. Le coppie per S1 sono i sink con lo stesso prompt «solo sink» ed etichette opposte (per esempio `54e_badSink` e `54e_goodG2BSink`, corpi identici).

**Tetti.** Juliet al più ~20% del train, campionato in modo uniforme per (CWE, variante) e con etichette bilanciate (si riusa `limit_label_skew`).

### 3. Tracce Python

- **P1, test CodeQL Python.** La directory di ogni query è una radice (evita collisioni di nomi fra test diversi). Si copia l'albero togliendo annotazioni e commenti GOOD/BAD senza spostare le righe
  (`localize.scrub_labels`, già usato), si lancia `slice` col catalogo, si assegna l'etichetta unendo `(file, riga)` con le annotazioni `$ Alert` originali.
  Gli attuali 66 esempi Python CodeQL del train (a un blocco) sono sostituiti da queste versioni; i 177 C/C++ restano come sono.
- **P2, CVEfixes/PyVul.** Il parquet ha `repo_url`, `commit_hash`, `function_name`, `code_before`, `code_after` ma **nessun percorso di file**. Per ogni riga: fetch superficiale del commit e del suo genitore, ricerca della funzione per nome e corpo,
  sink della fix come oggi (`localize.localize_sink`), etichetta «vulnerabile» sul genitore e «corretta» sul commit di fix. Si parte con un **pilota su 20 repo** e si riporta la resa (righe risolte, esempi per repo, tempo, spazio);
  si estende solo dopo che l'utente ha visto i numeri. Gruppo per divisione = repo (più severo del commit usato oggi).
- **P3, sintetici.** Piccoli pacchetti generati: sink con argomento costante o derivato da input, attraverso 0–3 funzioni intermedie, con nomi e struttura variati, per le CWE Python del catalogo (601, 78, 89, 22…). Al più 15% degli esempi Python, **mai** nei file di valutazione.
- Nelle view Django/Flask il chiamante è il framework e la slice non ha chiamanti: lì il modello impara a ragionare dentro la funzione. È voluto, e per questo c'è la miscela sotto.

### 4. Target (reasoning e fix)

- Per tutte le fonti: **niente id CVE, descrizione CVE o messaggio di commit** nel reasoning. `suggested_fix` = `fix_hint` del catalogo per i vulnerabili e «Nessuna modifica necessaria.» per gli altri; i diff di altri progetti escono dal target (vedi domande aperte).
- Per gli esempi con slice, il reasoning è costruito da un vocabolario chiuso e dice (a) da dove arriva l'argomento del sink (funzione anonimizzata e profondità), (b) il percorso, (c) il motivo del verdetto: origine non fidata, costante, oppure limite o controllo.
  Esempio non vulnerabile: «`system` in `fn_3a1f` riceve `data`; `data` è una stringa costante assegnata in `fn_91c2` (profondità 4) e non è influenzabile dall'utente.»
  L'origine si ricava dai token del nome Juliet *prima* di anonimizzare (`environment`, `console`, `connect_socket`, `fixed_string`…); la tabella dei termini sta nel piano.

### 5. Miscela con e senza chiamanti

Per ogni sink con chiamanti, con probabilità 0,7 l'esempio ha la slice completa, con 0,3 solo il blocco a profondità 0, e quest'ultima view è ammessa solo se il suo prompt non è in conflitto di etichetta con un altro.
Il modello impara a non dipendere dai chiamanti quando non ci sono. Ogni esempio salva in `meta.trace` (`cwe_id`, sink, blocchi) la forma strutturata, così le view si ricostruiscono da codice e non da regex sul prompt.

### 6. Unione, dimensioni, Colab

- `prepare_data.py` si estende (non si duplica): rigenera train/val/test da `data/raw` più le nuove tracce, senza i vecchi Juliet; la **composizione** di `test.jsonl` non cambia (cambiano solo i target, che non contengono più CVE), così i confronti restano validi.
- File di valutazione nuovi in `data/eval/`: `juliet_pairs`, `juliet_unseen_variants`, `py_codeql`, `py_cvefixes` e la lista congelata `pygoat_frozen.json`.
- Dimensioni: oggi train+val+test sono circa 49 MB (30 + 2 + 17). Le slice allungano i prompt (fino a 12.000 caratteri). Ogni file resta sotto 50 MB; se `train.jsonl` li supera, `--train` di `train.py` accetta più file.
  Gli esempi oltre `--max-seq-length` (8192) si scartano e si contano.
- Il notebook clona la repo da GitHub: serve `git push` dei dati prima del run. Un solo run, dal modello base, con la configurazione di oggi. La lunghezza media in token e il tempo per passo sono ignoti finché non si misurano.

### 7. Valutazione (sul PC)

- `benchmark.py accuracy` guadagna due sistemi: `llm-graph` (messaggi come salvati, con `--depth` per ricostruire a profondità diversa) e `llm-sink-only` (solo il blocco a profondità 0, da `meta.trace`).
- Si misura prima il modello attuale sugli stessi file («prima»), poi il nuovo («dopo»). Risultati per traccia, per famiglia di variante (lunghezza della catena) e per profondità.
- La lista pygoat va costruita e controllata sul sorgente: oggi le 44 «certe» sono state viste, i ~7 veri positivi no uno per uno.

## Appendice: Juliet misurato il 2026-11-06

Stime con i moduli stdlib del progetto (`localize.find_calls`, regex, non tree-sitter), `.c` e `.cpp` insieme: valgono per il dimensionamento, non come conteggio finale.

- 105.727 sorgenti in `testcases/`, 118 CWE. Con gli alias del catalogo, 68.805 file ricadono in 11 delle nostre CWE (120: 26.476; 190: 17.288; 22: 7.814; 78: 7.813; 401: 3.336; 416: 2.170; 476: 2.096; 369: 1.480; 377: 146; 327: 110; 362: 76).
- Funzioni con un sink del catalogo per la propria CWE (macro in maiuscolo riportate al nome vero): **120: 6.900 bad / 9.756 good; 78: 2.280 / 3.180; 190: 684 / 954; 416: 468 / 822; 25.044 in tutto, in 10.230 test case.**
- Catene multi-file (conteggio per nome file sulle 8 CWE con nome diretto): 4.671 test case su 14.323 hanno 2 o più file; 747 ne hanno 3–5.
- 48 varianti di flusso distinte: 01–18, 21, 22, 31–34, 41–45, 51–54, 61–68, 72–74, 81–84.
- Corpi uguali con etichette opposte: 2.560 delle 9.926 funzioni `goodG2B*` con un sink hanno un gemello `bad` dal corpo identico (nomi e commenti normalizzati). In `_54e`, `badSink` e `goodG2BSink` sono identici anche nel testo; cambia solo la sorgente in `_54a`.

## Rischi

1. **Juliet è sintetico e uniforme.** Il modello può imparare la struttura dei template invece del ragionamento. Mitigazioni: anonimizzazione con verifica delle perdite, varianti tenute fuori (S2), valutazione su codice vero (pygoat, CVEfixes). Finché S3 non lo conferma, si dice «usa i chiamanti», non «generalizza a repo reali».
2. **Limiti del grafo per nome:** puntatori a funzione, `namespace` C++, nomi ambigui. Una quota di Juliet non sarà decidibile e viene esclusa e dichiarata.
3. **Profondità:** se 5 costa troppo su repo reali (tronca la slice o gonfia il prompt), il default resta 3 e la traccia J perde le catene più lunghe; si decide con S1.
4. **Costo del run:** prompt più lunghi significano più token e più tempo su Colab; ignoto finché non si misura la distribuzione.
5. **Copertura:** Juliet porta 4 CWE; le altre 16 del catalogo non hanno esempi con grafo da questa fonte e non si può dire nulla su di esse.
6. **Campione piccolo per S3:** 44 casi: si riportano i conteggi, non solo le percentuali.
7. **Limite di GitHub** sui file (100 MB, avviso a 50 MB).

## Domande aperte per la revisione

1. Le soglie S1–S5 vanno bene, o vuoi altri numeri?
2. Togliere i diff upstream da `suggested_fix` (resta solo `fix_hint`) va bene? Perdi la «fix come diff», ma lo spike li ha mostrati allucinati.
3. Il pilota P2 su 20 repo: ti basta decidere sui numeri dopo averli visti?
4. Portare il default del motore da profondità 3 a 5 dipende da S1: confermi che lo decidiamo dopo la misura?
