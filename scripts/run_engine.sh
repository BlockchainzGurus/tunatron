#!/usr/bin/env bash
# Compila (con RAM/job limitati) e avvia sast-engine su 127.0.0.1:8080. Configurazione via variabili d'ambiente (vedi README).
set -euo pipefail
cd "$(dirname "$0")/../sast-engine"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
if command -v systemd-run >/dev/null; then
  systemd-run --user --scope -q -p MemoryMax="${BUILD_MEM_MAX:-10G}" -p MemorySwapMax=0 cargo build --release
else
  cargo build --release
fi
exec ./target/release/sast-engine
