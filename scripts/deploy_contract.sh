#!/usr/bin/env bash
# Fase 4.6 (opzionale) — pubblica AttestationRegistry su un nodo locale (Anvil o Hardhat) con account sbloccati.
# Richiede Foundry (forge). Prima, in un altro terminale:  anvil
#   scripts/deploy_contract.sh      # stampa l'indirizzo da esportare come ATTEST_CONTRACT
set -euo pipefail
cd "$(dirname "$0")/.."
RPC="${ETH_RPC_URL:-http://127.0.0.1:8545}"
FROM="${ETH_FROM:-0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266}"   # account #0 di Anvil (indirizzo pubblico, nessuna chiave nel repo)
forge create contracts/AttestationRegistry.sol:AttestationRegistry --rpc-url "$RPC" --unlocked --from "$FROM" --broadcast
echo "Esporta:  export ATTEST_CONTRACT=<Deployed to: ...>   ETH_RPC_URL=$RPC"
