//! (Opzionale) Notarizzazione: report su IPFS + attestazione on-chain (hash del report e provenance del modello).
//!
//! Nessuna dipendenza pesante: l'ABI della singola funzione `attest(string,bytes32,bytes32)` è codificata a mano
//! e la transazione viene inviata via JSON-RPC a un nodo di sviluppo con account sbloccati (Anvil/Hardhat).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sha3::Keccak256;

pub const ATTEST_SIGNATURE: &str = "attest(string,bytes32,bytes32)";

#[derive(Debug, Clone)]
pub struct NotarizerConfig {
    pub ipfs_api: String,
    pub rpc_url: String,
    /// Indirizzo dello smart contract `AttestationRegistry`.
    pub contract: Option<String>,
    /// Account mittente; se assente si usa il primo di `eth_accounts`.
    pub from: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub model_name: String,
    /// Include nell'hash di provenance anche i pesi `*.safetensors` (lento su modelli grandi).
    pub hash_weights: bool,
    pub timeout: Duration,
}

impl Default for NotarizerConfig {
    fn default() -> Self {
        NotarizerConfig {
            ipfs_api: "http://127.0.0.1:5001".into(),
            rpc_url: "http://127.0.0.1:8545".into(),
            contract: None,
            from: None,
            model_dir: None,
            model_name: "sast-nemotron".into(),
            hash_weights: false,
            timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Attestation {
    pub cid: String,
    pub report_sha256: String,
    pub model_provenance_sha256: String,
    pub tx_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub hash: [u8; 32],
    pub files: BTreeMap<String, String>,
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn keccak256(bytes: &[u8]) -> [u8; 32] {
    Keccak256::digest(bytes).into()
}

/// Primi 4 byte di keccak256 della firma (selettore ABI).
pub fn selector(signature: &str) -> [u8; 4] {
    let h = keccak256(signature.as_bytes());
    [h[0], h[1], h[2], h[3]]
}

fn word(n: usize) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&(n as u64).to_be_bytes());
    w
}

/// Calldata di `attest(string cid, bytes32 reportHash, bytes32 modelHash)`.
pub fn encode_attest_call(cid: &str, report_hash: [u8; 32], model_hash: [u8; 32]) -> Vec<u8> {
    let mut out = selector(ATTEST_SIGNATURE).to_vec();
    // head: offset del dato dinamico (3 parole), poi i due bytes32
    out.extend(word(0x60));
    out.extend(report_hash);
    out.extend(model_hash);
    // tail: lunghezza e byte della stringa, con padding a multipli di 32
    out.extend(word(cid.len()));
    out.extend(cid.as_bytes());
    out.extend(std::iter::repeat_n(0u8, (32 - cid.len() % 32) % 32));
    out
}

/// File che descrivono la configurazione del modello (inclusa la quantizzazione, che sta in `config.json`).
const PROVENANCE_FILES: [&str; 6] = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "hf_quant_config.json", "recipe.yaml"];

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path).with_context(|| format!("apertura {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Hash di configurazione del modello (`Model Provenance`): nome servito + hash dei file di config/tokenizer/quantizzazione
/// (e, se richiesto, dei pesi). Deterministico e indipendente dal percorso.
pub fn model_provenance(model_dir: Option<&Path>, model_name: &str, hash_weights: bool) -> anyhow::Result<Provenance> {
    let mut files = BTreeMap::new();
    if let Some(dir) = model_dir {
        if !dir.is_dir() {
            bail!("directory del modello non trovata: {}", dir.display());
        }
        for name in PROVENANCE_FILES {
            let path = dir.join(name);
            if path.is_file() {
                files.insert(name.to_string(), sha256_file(&path)?);
            }
        }
        if hash_weights {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".safetensors") && entry.path().is_file() {
                    files.insert(name, sha256_file(&entry.path())?);
                }
            }
        }
    }
    let canonical = serde_json::to_vec(&json!({"model_name": model_name, "files": files}))?;
    Ok(Provenance { hash: sha256(&canonical), files })
}

pub async fn ipfs_add(http: &reqwest::Client, api: &str, bytes: &[u8]) -> anyhow::Result<String> {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec()).file_name("report.json").mime_str("application/json")?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let url = format!("{}/api/v0/add?pin=true&cid-version=1", api.trim_end_matches('/'));
    let response = http.post(&url).multipart(form).send().await.with_context(|| format!("nodo IPFS non raggiungibile ({api})"))?;
    let status = response.status();
    let text = response.text().await.context("lettura risposta IPFS")?;
    if !status.is_success() {
        bail!("IPFS ha risposto {status}: {}", text.chars().take(200).collect::<String>());
    }
    // Kubo risponde con NDJSON: per un singolo file c'è una sola riga.
    let line = text.lines().rev().find(|l| !l.trim().is_empty()).ok_or_else(|| anyhow!("risposta IPFS vuota"))?;
    let value: Value = serde_json::from_str(line).context("risposta IPFS non JSON")?;
    value["Hash"].as_str().map(String::from).ok_or_else(|| anyhow!("risposta IPFS senza campo Hash"))
}

async fn rpc(http: &reqwest::Client, url: &str, method: &str, params: Value) -> anyhow::Result<Value> {
    let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let response = http.post(url).json(&body).send().await.with_context(|| format!("nodo RPC non raggiungibile ({url})"))?;
    let value: Value = response.json().await.context("risposta RPC non valida")?;
    if let Some(error) = value.get("error") {
        bail!("RPC {method}: {}", error["message"].as_str().unwrap_or("errore sconosciuto"));
    }
    Ok(value["result"].clone())
}

fn is_address(s: &str) -> bool {
    s.len() == 42 && s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
}

fn contract_address(cfg: &NotarizerConfig) -> anyhow::Result<&str> {
    let contract = cfg.contract.as_deref().ok_or_else(|| anyhow!("ATTEST_CONTRACT non configurato"))?;
    if !is_address(contract) {
        bail!("ATTEST_CONTRACT non è un indirizzo valido: {contract}");
    }
    Ok(contract)
}

pub async fn send_attestation(
    http: &reqwest::Client,
    cfg: &NotarizerConfig,
    cid: &str,
    report_hash: [u8; 32],
    model_hash: [u8; 32],
) -> anyhow::Result<String> {
    let contract = contract_address(cfg)?;
    let from = match &cfg.from {
        Some(from) => from.clone(),
        None => rpc(http, &cfg.rpc_url, "eth_accounts", json!([]))
            .await?
            .get(0)
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| anyhow!("il nodo non espone account sbloccati: imposta ETH_FROM"))?,
    };
    let data = format!("0x{}", hex::encode(encode_attest_call(cid, report_hash, model_hash)));
    let tx = rpc(http, &cfg.rpc_url, "eth_sendTransaction", json!([{"from": from, "to": contract, "data": data}]))
        .await?
        .as_str()
        .map(String::from)
        .ok_or_else(|| anyhow!("eth_sendTransaction non ha restituito un hash"))?;
    for _ in 0..40 {
        let receipt = rpc(http, &cfg.rpc_url, "eth_getTransactionReceipt", json!([tx])).await?;
        if !receipt.is_null() {
            if receipt["status"] == "0x1" {
                return Ok(tx);
            }
            bail!("transazione revertita ({tx}), status {}", receipt["status"]);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    bail!("receipt non ricevuto per la transazione {tx}")
}

/// Pipeline completa: hash del report, upload su IPFS, provenance, attestazione on-chain.
pub async fn notarize(report_json: &[u8], cfg: &NotarizerConfig) -> anyhow::Result<Attestation> {
    // Prima i controlli locali: niente upload su IPFS se l'attestazione non potrebbe comunque partire.
    contract_address(cfg)?;
    let provenance = model_provenance(cfg.model_dir.as_deref(), &cfg.model_name, cfg.hash_weights)?;
    let report_hash = sha256(report_json);
    let http = reqwest::Client::builder().timeout(cfg.timeout).build()?;
    let cid = ipfs_add(&http, &cfg.ipfs_api, report_json).await?;
    let tx_hash = send_attestation(&http, cfg, &cid, report_hash, provenance.hash).await?;
    Ok(Attestation {
        cid,
        report_sha256: hex::encode(report_hash),
        model_provenance_sha256: hex::encode(provenance.hash),
        tx_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn keccak_matches_known_vectors() {
        assert_eq!(hex::encode(keccak256(b"")), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
        assert_eq!(hex::encode(selector("transfer(address,uint256)")), "a9059cbb");
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(hex::encode(sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    fn word(data: &[u8], i: usize) -> &[u8] {
        &data[4 + 32 * i..4 + 32 * (i + 1)]
    }

    #[test]
    fn attest_calldata_follows_the_abi_layout() {
        let (rh, mh) = ([0xaa; 32], [0xbb; 32]);
        let data = encode_attest_call("bafy123", rh, mh);
        assert_eq!(&data[..4], &selector(ATTEST_SIGNATURE));
        // head: offset della stringa (3 parole = 0x60), poi i due bytes32
        assert_eq!(word(&data, 0), &[&[0u8; 31][..], &[0x60]].concat()[..]);
        assert_eq!(word(&data, 1), &rh);
        assert_eq!(word(&data, 2), &mh);
        // tail: lunghezza e dati con padding a 32 byte
        assert_eq!(word(&data, 3), &[&[0u8; 31][..], &[7]].concat()[..]);
        assert_eq!(&word(&data, 4)[..7], b"bafy123");
        assert!(word(&data, 4)[7..].iter().all(|b| *b == 0));
        assert_eq!(data.len(), 4 + 32 * 5);
    }

    #[test]
    fn long_cid_is_padded_to_whole_words() {
        let cid = "b".repeat(59); // CIDv1 base32 tipico
        let data = encode_attest_call(&cid, [0; 32], [0; 32]);
        assert_eq!(data.len(), 4 + 32 * 3 + 32 + 64);
        assert_eq!(&data[4 + 32 * 4..4 + 32 * 4 + 59], cid.as_bytes());
    }

    fn model_dir() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("config.json"), "{\"a\":1}").unwrap();
        fs::write(d.path().join("tokenizer_config.json"), "{}").unwrap();
        fs::write(d.path().join("model.safetensors"), vec![7u8; 4096]).unwrap();
        fs::write(d.path().join("README.md"), "ignorato").unwrap();
        d
    }

    #[test]
    fn provenance_covers_config_files_and_optionally_weights() {
        let d = model_dir();
        let light = model_provenance(Some(d.path()), "m", false).unwrap();
        assert_eq!(light.files.keys().map(String::as_str).collect::<Vec<_>>(), vec!["config.json", "tokenizer_config.json"]);
        let heavy = model_provenance(Some(d.path()), "m", true).unwrap();
        assert!(heavy.files.contains_key("model.safetensors"));
        assert_ne!(light.hash, heavy.hash);
    }

    #[test]
    fn provenance_is_deterministic_and_sensitive_to_content_and_name() {
        let d = model_dir();
        let a = model_provenance(Some(d.path()), "m", false).unwrap();
        assert_eq!(a, model_provenance(Some(d.path()), "m", false).unwrap());
        assert_ne!(a.hash, model_provenance(Some(d.path()), "other", false).unwrap().hash);
        fs::write(d.path().join("config.json"), "{\"a\":2}").unwrap();
        assert_ne!(a.hash, model_provenance(Some(d.path()), "m", false).unwrap().hash);
    }

    #[test]
    fn provenance_without_a_model_dir_still_binds_the_model_name() {
        let a = model_provenance(None, "m", false).unwrap();
        assert!(a.files.is_empty());
        assert_ne!(a.hash, model_provenance(None, "n", false).unwrap().hash);
    }

    #[test]
    fn provenance_reports_a_missing_directory() {
        assert!(model_provenance(Some(Path::new("/definitely/not/here")), "m", false).is_err());
    }
}
