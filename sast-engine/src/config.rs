//! Configurazione del motore da variabili d'ambiente.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::graph_slicer::{ScanLimits, SliceOptions};
use crate::ingestor::IngestConfig;
use crate::notarizer::NotarizerConfig;
use crate::vllm_client::VllmConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: String,
    pub vllm: VllmConfig,
    pub ingest: IngestConfig,
    pub scan: ScanLimits,
    pub slice: SliceOptions,
    pub notarizer: NotarizerConfig,
    /// KB CWE alternativa a quella incorporata.
    pub kb_path: Option<PathBuf>,
    /// Analisi contemporanee ammesse (le altre ricevono 429).
    pub max_concurrent_analyses: usize,
    pub analysis_timeout: Duration,
}

impl Config {
    pub fn from_map(vars: &HashMap<String, String>) -> Result<Config, String> {
        let get = |key: &str| vars.get(key).map(|v| v.trim()).filter(|v| !v.is_empty());
        let num = |key: &str, default: u64| -> Result<u64, String> {
            match get(key) {
                None => Ok(default),
                Some(v) => v.parse().map_err(|_| format!("{key}: valore non valido {v:?}")),
            }
        };
        let positive = |key: &str, default: usize| -> Result<usize, String> {
            match num(key, default as u64)? {
                0 => Err(format!("{key} deve essere maggiore di zero")),
                n => Ok(n as usize),
            }
        };
        let secs = |key: &str, default: u64| num(key, default).map(Duration::from_secs);
        let text = |key: &str, default: &str| get(key).unwrap_or(default).to_string();

        let vd = VllmConfig::default();
        let vllm = VllmConfig {
            base_url: text("VLLM_URL", &vd.base_url),
            model: text("VLLM_MODEL", &vd.model),
            concurrency: positive("VLLM_CONCURRENCY", vd.concurrency)?,
            timeout: secs("VLLM_TIMEOUT_SECS", vd.timeout.as_secs())?,
            max_tokens: num("VLLM_MAX_TOKENS", u64::from(vd.max_tokens))? as u32,
            ..vd
        };

        let id = IngestConfig::default();
        let allowed_hosts = match get("ALLOWED_GIT_HOSTS") {
            Some(list) => list.split(',').map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect(),
            None => id.allowed_hosts.clone(),
        };
        let ingest = IngestConfig {
            allowed_hosts,
            clone_timeout: secs("GIT_CLONE_TIMEOUT_SECS", id.clone_timeout.as_secs())?,
            max_repo_bytes: positive("MAX_REPO_MB", (id.max_repo_bytes / 1_000_000) as usize)? as u64 * 1_000_000,
            syft_bin: get("SYFT_BIN").map(String::from),
            // `allow_file_urls` resta spento: serve solo ai test e non è configurabile dall'ambiente.
            ..id
        };

        let sd = ScanLimits::default();
        let scan = ScanLimits {
            max_file_bytes: num("MAX_FILE_BYTES", sd.max_file_bytes)?,
            max_files: num("MAX_FILES", sd.max_files as u64)? as usize,
            max_sinks: num("MAX_SINKS", sd.max_sinks as u64)? as usize,
        };

        let ld = SliceOptions::default();
        let slice = SliceOptions {
            max_depth: num("SLICE_DEPTH", ld.max_depth as u64)? as usize,
            max_callers_per_node: positive("SLICE_MAX_CALLERS", ld.max_callers_per_node)?,
            max_chars: positive("SLICE_MAX_CHARS", ld.max_chars)?,
        };

        let nd = NotarizerConfig::default();
        let notarizer = NotarizerConfig {
            ipfs_api: text("IPFS_API", &nd.ipfs_api),
            rpc_url: text("ETH_RPC_URL", &nd.rpc_url),
            contract: get("ATTEST_CONTRACT").map(String::from),
            from: get("ETH_FROM").map(String::from),
            model_dir: get("MODEL_DIR").map(PathBuf::from),
            model_name: vllm.model.clone(),
            hash_weights: matches!(get("MODEL_HASH_WEIGHTS"), Some("1" | "true" | "yes")),
            ..nd
        };

        Ok(Config {
            bind: text("SAST_BIND", "127.0.0.1:8080"),
            vllm,
            ingest,
            scan,
            slice,
            notarizer,
            kb_path: get("CWE_KB_PATH").map(PathBuf::from),
            max_concurrent_analyses: positive("MAX_CONCURRENT_ANALYSES", 1)?,
            analysis_timeout: secs("ANALYSIS_TIMEOUT_SECS", 900)?,
        })
    }

    pub fn from_env() -> Result<Config, String> {
        Config::from_map(&std::env::vars().collect())
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::from_map(&HashMap::new()).expect("i default sono validi")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn defaults_match_the_plan() {
        let c = Config::default();
        assert_eq!(c.bind, "127.0.0.1:8080");
        assert_eq!(c.vllm.base_url, "http://localhost:8000");
        assert_eq!(c.slice.max_depth, 3);
        assert_eq!(c.ingest.allowed_hosts, vec!["github.com"]);
        assert!(!c.ingest.allow_file_urls);
        assert_eq!(c.max_concurrent_analyses, 1);
        assert!(c.notarizer.contract.is_none());
    }

    #[test]
    fn overrides_are_applied() {
        let c = Config::from_map(&vars(&[
            ("SAST_BIND", "0.0.0.0:9000"),
            ("VLLM_URL", "http://gpu:8001"),
            ("VLLM_MODEL", "m"),
            ("VLLM_CONCURRENCY", "16"),
            ("SLICE_DEPTH", "2"),
            ("MAX_SINKS", "10"),
            ("ALLOWED_GIT_HOSTS", "github.com, gitlab.com"),
            ("CWE_KB_PATH", "/tmp/kb.json"),
            ("ATTEST_CONTRACT", "0x5FbDB2315678afecb367f032d93F642f64180aa3"),
            ("MODEL_HASH_WEIGHTS", "1"),
            ("SYFT_BIN", "syft"),
        ]))
        .unwrap();
        assert_eq!(c.bind, "0.0.0.0:9000");
        assert_eq!((c.vllm.base_url.as_str(), c.vllm.model.as_str(), c.vllm.concurrency), ("http://gpu:8001", "m", 16));
        assert_eq!((c.slice.max_depth, c.scan.max_sinks), (2, 10));
        assert_eq!(c.ingest.allowed_hosts, vec!["github.com", "gitlab.com"]);
        assert_eq!(c.kb_path, Some(PathBuf::from("/tmp/kb.json")));
        assert!(c.notarizer.hash_weights);
        assert_eq!(c.ingest.syft_bin.as_deref(), Some("syft"));
        assert!(c.notarizer.contract.is_some());
    }

    #[test]
    fn invalid_numbers_name_the_offending_variable() {
        let err = Config::from_map(&vars(&[("SLICE_DEPTH", "tre")])).unwrap_err();
        assert!(err.contains("SLICE_DEPTH"), "{err}");
    }

    #[test]
    fn zero_concurrency_is_rejected() {
        assert!(Config::from_map(&vars(&[("MAX_CONCURRENT_ANALYSES", "0")])).is_err());
        assert!(Config::from_map(&vars(&[("VLLM_CONCURRENCY", "0")])).is_err());
    }

    #[test]
    fn repository_size_limit_is_configured_in_megabytes() {
        assert_eq!(Config::default().ingest.max_repo_bytes, 500_000_000);
        let c = Config::from_map(&vars(&[("MAX_REPO_MB", "10")])).unwrap();
        assert_eq!(c.ingest.max_repo_bytes, 10_000_000);
        assert!(Config::from_map(&vars(&[("MAX_REPO_MB", "0")])).is_err());
    }

    #[test]
    fn file_urls_cannot_be_enabled_from_the_environment() {
        let c = Config::from_map(&vars(&[("ALLOW_FILE_URLS", "1")])).unwrap();
        assert!(!c.ingest.allow_file_urls);
    }
}
