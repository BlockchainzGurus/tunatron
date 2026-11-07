# Architettura Desktop Tunatron: Tauri v2, Motore SAST e Runner LLM Multi-Target

Specifica tecnica per la realizzazione dell'applicazione desktop multipiattaforma (macOS, Windows, Linux) con interfaccia grafica Tauri v2, motore di analisi statica Rust integrato in-process e inferenza LLM locale accelerata su 3 target hardware distinti.

---

## 1. Visione Architetturale e Dimensionamento

L'applicazione separa rigorosamente il codice applicativo (GUI + motore di analisi + runtime di inferenza) dai pesi del modello:

- **Dimensione Installer:** ~25–40 MB (distribuito tramite GitHub Releases / Web).
- **Dimensione su Disco (senza pesi):** ~35–50 MB.
- **Pesi del Modello:** scaricati al primo avvio nella directory dati utente dell'OS (`AppData` / `Application Support`).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Tauri v2 Desktop App                            │
│                                                                        │
│  ┌───────────────────────┐            ┌─────────────────────────────┐  │
│  │   Frontend UI         │ ◄── IPC ──►│   Tauri Core (Rust)         │  │
│  │   (Vite + React /     │            │   - Lifecycle manager       │  │
│  │    Svelte + Tailwind) │            │   - Model Downloader        │  │
│  └───────────────────────┘            └──────────────┬──────────────┘  │
│                                                      │                 │
│                 ┌────────────────────────────────────┴──────────┐      │
│                 │ (Chiamata Rust diretta in-process)            │      │
│                 ▼                                               ▼      │
│  ┌──────────────────────────────┐              ┌────────────────────┐  │
│  │ sast-engine (Rust nativo)    │              │ Sidecar Manager    │  │
│  │ - Tree-sitter AST parsers    │              │ - Porta casuale    │  │
│  │ - Petgraph call graph        │              │ - Healthcheck      │  │
│  │ - Backward slicing           │              │ - Termina su exit  │  │
│  │ - CycloneDX SBOM + VEX gen   │              └─────────┬──────────┘  │
│  └──────────────┬───────────────┘                        │             │
└─────────────────┼────────────────────────────────────────┼─────────────┘
                  │                                        │
                  │ HTTP POST /v1/chat/completions         │ (Spawn / Kill)
                  │ (Guided JSON decoding)                 │
                  ▼                                        ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Inference Engine Locale (Sidecar)              │
    │  - Target Apple:  MLX / Metal Server                        │
    │  - Target NVIDIA: TensorRT-LLM / vLLM (NVFP4 nativo)        │
    │  - Target Vulkan: llama-server (Vulkan + CPU AVX2)          │
    └─────────────────────────────────────────────────────────────┘
```

---

## 2. Matrice dei 3 Installer Indipendenti

Per evitare complessità di compilazione incrociata e dispatcher dinamici a runtime, la distribuzione prevede 3 pacchetti distinti:

| Parametro | 1. Apple Silicon | 2. NVIDIA Workstation | 3. Universal Fallback |
|---|---|---|---|
| **File Installer** | `Tunatron-macOS-AppleSilicon.dmg` | `Tunatron-Windows-Nvidia.exe` | `Tunatron-Windows-Universal.exe` |
| **Piattaforma OS** | macOS (`aarch64-apple-darwin`) | Windows 10/11 (`x86_64-pc-windows-msvc`) | Windows / Linux (`x86_64`) |
| **Target Hardware** | Apple Silicon (M1/M2/M3/M4) | GPU NVIDIA GeForce/RTX (Ada / Blackwell) | GPU AMD Radeon, Intel Arc, iGPU, CPU |
| **Backend Inferenza** | **MLX / Metal** | **TensorRT-LLM / vLLM** | **llama-server (Vulkan + AVX2)** |
| **Formato Modello** | Pesi MLX 4-bit / 6-bit | Checkpoint ModelOpt **NVFP4 (W4A4/W8A8)** | File GGUF `Q4_K_M` |
| **Consumo RAM/VRAM** | ~2.5 – 3.2 GB (Unified Memory) | ~3.4 – 3.8 GB VRAM | ~2.6 – 3.4 GB VRAM/RAM |
| **Dimensione Installer** | ~25 MB | ~40 MB | ~30 MB |

### Dettaglio dei target:

### Target 1: Apple Silicon (macOS)
- **Motivazione:** Sfrutta la memoria unificata (UMA) a larghezza di banda elevata e l'architettura ibrida Mamba-2/Attention tramite il supporto nativo `mlx_lm.models.nemotron_h`.
- **Generazione Pesi:** Script `scripts/export_mlx.sh -s <HF_REPO> -b 4,6`.
- **Runtime:** Micro-servizio compilato `mlx_lm.server` con binding Metal.

### Target 2: NVIDIA Workstation (Windows / Linux)
- **Motivazione:** Sfrutta i Tensor Core hardware a 4-bit delle architetture Blackwell (`sm_120`, es. RTX 5070 Ti, RTX PRO 6000) e Ada Lovelace (`sm_89`). L'uso di GGUF su questo target è escluso per non perdere le accelerazioni NVFP4 prodotte da `ai-pipeline/quantize.py`.
- **Formato Pesi:** Checkpoint `model_nvfp4` (W4A4 per lineari con scale a blocchi da 16, down-projection MLP in FP8 W8A8, lm-head ed embedding in BF16).
- **Runtime:** Binario C++ `trtllm-executor` (TensorRT-LLM for Windows) o vLLM (`serve_vllm.sh`).

### Target 3: Universal Fallback (AMD / Intel / CPU)
- **Motivazione:** Compatibilità garantita su qualsiasi PC senza dipendere da driver proprietari CUDA o ambienti complessi ROCm/OpenVINO non testabili in locale.
- **Formato Pesi:** `tunatron-3-nano-4b-Q4_K_M.gguf`.
- **Runtime:** Binario standalone `llama-server` compilato con flag `GGML_VULKAN=ON`. Funziona out-of-the-box tramite le librerie grafiche di sistema (Vulkan 1.2+).

---

## 3. Motore SAST Integrato in-process

A differenza della configurazione di sviluppo (dove `sast-engine` è un server HTTP Axum indipendente), nell'app desktop il motore Rust viene importato come crate di libreria:

```toml
# In src-tauri/Cargo.toml
[dependencies]
sast-engine = { path = "../../sast-engine" }
tauri = { version = "2", features = ["tray-icon"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["stream"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Vantaggi:
1. **Zero latency IPC per l'AST:** Il parsing Tree-sitter e la generazione del grafo Petgraph avvengono nello stesso spazio di memoria del backend Tauri.
2. **Nessun microservizio da configurare:** L'utente non deve impostare porte, variabili d'ambiente o proxy.

---

## 4. Gestione Pesi: Download al Primo Avvio

I pesi non risiedono nel pacchetto di installazione. Al primo avvio, l'applicazione rileva l'assenza del modello e avvia la procedura di bootstrap.

### 4.1 Directory di Destinazione OS-Specific
Risolte tramite l'API `app.path().app_data_dir()` di Tauri:
- **macOS:** `~/Library/Application Support/com.tunatron.app/models/`
- **Windows:** `%LOCALAPPDATA%\tunatron\models\`
- **Linux:** `~/.local/share/tunatron/models/`

### 4.2 Requisiti del Downloader Rust
1. **Streaming asincrono:** Download a blocchi da 64 KB tramite `reqwest::get(...).bytes_stream()`.
2. **Eventi di avanzamento:** Emissione verso la webview dell'evento `model-download-progress`:
   ```json
   {
     "downloaded_bytes": 125829120,
     "total_bytes": 2576980377,
     "percentage": 4.88,
     "speed_mbps": 18.4
   }
   ```
3. **Resumable Download:** Utilizzo dell'header HTTP `Range: bytes=<offset>-` per riprendere il trasferimento se interrotto.
4. **Verifica Integrità:** Calcolo SHA-256 on-the-fly durante la scrittura su disco; blocco dell'inizializzazione se l'hash non corrisponde al valore atteso su Hugging Face.

---

## 5. Comunicazione con il Modello e Guided Decoding

Tutti e tre i runner sidecar espongono un'interfaccia HTTP OpenAI-compatibile su `127.0.0.1:<PORT>` (porta casuale libera allocata all'avvio):

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "tunatron-nano-4b",
  "messages": [ ... ],
  "temperature": 0.2,
  "response_format": {
    "type": "json_object",
    "schema": <vex_schema.json>
  }
}
```

### Meccanismo di Costrizione Sintattica per Runner:
- **NVIDIA (vLLM / TensorRT-LLM):** Supporto nativo `guided_json` con compilazione regex/schema DFA in kernel CUDA.
- **Vulkan / CPU (`llama-server`):** Parametro nativo `--grammar` / BNF grammar derivata automaticamente dallo schema JSON.
- **MLX (`mlx_lm.server`):** Template Jinja vincolante con schema enforcement a livello di token sampler.

---

## 6. Configurazione Build e Release (Cargo.toml)

Per garantire la dimensione target (~30 MB) dell'eseguibile, il profilo di release in `src-tauri/Cargo.toml` deve essere configurato come segue:

```toml
[profile.release]
opt-level = "z"       # Ottimizzazione per dimensione del codice
lto = true            # Link Time Optimization tra tutti i crate dipendenti
codegen-units = 1     # Massimizza le ottimizzazioni inter-modulo
panic = "abort"       # Elimina tabelle di stack unwinding non necessarie
strip = true          # Rimozione totale dei simboli di debug dal binario
```

---

## 7. Pipeline CI/CD GitHub Actions

Un unico workflow (`.github/workflows/release-desktop.yml`) genera i 3 binari sfruttando runner GitHub dedicati:

```yaml
name: Release Desktop App

on:
  push:
    tags:
      - 'v*'

jobs:
  build-macos:
    runs-on: macos-14 # Apple Silicon M1/M2 runner
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-apple-darwin
      - name: Build Tauri App (MLX Target)
        run: cargo tauri build --target aarch64-apple-darwin
      - name: Upload DMG
        uses: softprops/action-gh-release@v2
        with:
          files: src-tauri/target/aarch64-apple-darwin/release/bundle/dmg/*.dmg

  build-windows:
    strategy:
      matrix:
        include:
          - name: Tunatron-Windows-Nvidia
            features: "backend-nvfp4"
          - name: Tunatron-Windows-Universal
            features: "backend-vulkan"
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Build Tauri App
        run: cargo tauri build --features ${{ matrix.features }}
      - name: Upload Installer
        uses: softprops/action-gh-release@v2
        with:
          files: src-tauri/target/release/bundle/nsis/*.exe
```
