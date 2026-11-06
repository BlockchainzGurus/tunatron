//! Orchestrazione: clone -> SBOM -> call graph + slicing -> prompt -> vLLM -> report VEX (-> notarizzazione).

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::graph_slicer::{Slice, build_graph};
use crate::ingestor::{clone_repo, generate_sbom, validate_repo_url};
use crate::notarizer;
use crate::prompt::{render_user_prompt, system_prompt};
use crate::rag_cwe::{CweKb, render_card};
use crate::vex::{AnalysisItem, Report, Summary, build_report};
use crate::vllm_client::{Prompt, VllmClient};

#[derive(Debug)]
pub enum EngineError {
    InvalidRequest(String),
    CloneFailed(String),
    ModelUnavailable(String),
    Internal(String),
}

impl EngineError {
    pub fn http_status(&self) -> u16 {
        match self {
            EngineError::InvalidRequest(_) => 400,
            EngineError::CloneFailed(_) => 502,
            EngineError::ModelUnavailable(_) => 503,
            EngineError::Internal(_) => 500,
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidRequest(m) => write!(f, "{m}"),
            EngineError::CloneFailed(m) => write!(f, "clone non riuscito: {m}"),
            EngineError::ModelUnavailable(m) => write!(f, "modello non disponibile: {m}"),
            EngineError::Internal(m) => write!(f, "errore interno: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct Engine {
    cfg: Config,
    kb: Arc<CweKb>,
    client: VllmClient,
}

impl Engine {
    pub fn new(cfg: Config) -> anyhow::Result<Engine> {
        let kb = CweKb::load_or_embedded(cfg.kb_path.as_deref())?;
        let client = VllmClient::new(cfg.vllm.clone())?;
        Ok(Engine { cfg, kb: Arc::new(kb), client })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub async fn model_healthy(&self) -> bool {
        self.client.health().await
    }

    /// Analizza una directory già presente su disco (nessun clone).
    pub async fn analyze_path(&self, root: &Path, name: &str) -> Result<Report, EngineError> {
        let bom = generate_sbom(root, name, &self.cfg.ingest).await;

        let (root, kb, scan, opts) = (root.to_path_buf(), self.kb.clone(), self.cfg.scan.clone(), self.cfg.slice.clone());
        let (stats, slices) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let graph = build_graph(&root, &kb, &scan)?;
            let slices: Vec<(Slice, String)> = (0..graph.sinks.len()).map(|i| (graph.slice(i, &opts), graph.sinks[i].cwe_id.clone())).collect();
            Ok((graph.stats.clone(), slices))
        })
        .await
        .map_err(|e| EngineError::Internal(e.to_string()))?
        .map_err(|e| EngineError::Internal(format!("{e:#}")))?;

        let summary = Summary {
            model: self.cfg.vllm.model.clone(),
            files_scanned: stats.files_scanned,
            functions: stats.functions,
            parse_errors: stats.parse_errors,
            sinks_found: slices.len(),
            sinks_truncated: stats.sinks_truncated,
            ..Summary::default()
        };
        if slices.is_empty() {
            return Ok(build_report(bom, Vec::new(), summary));
        }
        // Fallire subito evita centinaia di richieste (e di retry) verso un server spento.
        if !self.client.health().await {
            return Err(EngineError::ModelUnavailable(format!("{} non risponde a /health", self.cfg.vllm.base_url)));
        }

        let system = system_prompt();
        let prompts: Vec<Prompt> = slices
            .iter()
            .map(|(slice, cwe)| {
                let card = self.kb.get(cwe).map(render_card).unwrap_or_default();
                Prompt { system: system.clone(), user: render_user_prompt(cwe, &card, &slice.location, &slice.blocks) }
            })
            .collect();
        let verdicts = self.client.analyze_all(prompts).await;

        let items = slices
            .into_iter()
            .zip(verdicts)
            .map(|((slice, cwe), outcome)| AnalysisItem { sink: slice.location, candidate_cwe: cwe, outcome })
            .collect();
        Ok(build_report(bom, items, summary))
    }

    /// Clona `url`, analizza, rimuove il clone e, se richiesto, notarizza il report.
    pub async fn analyze_url(&self, url: &str, notarize: bool) -> Result<Report, EngineError> {
        if notarize && self.cfg.notarizer.contract.is_none() {
            return Err(EngineError::InvalidRequest("ATTEST_CONTRACT non configurato: impossibile notarizzare".into()));
        }
        validate_repo_url(url, &self.cfg.ingest).map_err(EngineError::InvalidRequest)?;
        let repo = clone_repo(url, &self.cfg.ingest).await.map_err(|e| EngineError::CloneFailed(format!("{e:#}")))?;
        let mut report = self.analyze_path(&repo.path, &repo.name).await?;
        drop(repo);

        if notarize {
            let bytes = serde_json::to_vec(&report).map_err(|e| EngineError::Internal(e.to_string()))?;
            match notarizer::notarize(&bytes, &self.cfg.notarizer).await {
                Ok(attestation) => report.attestation = Some(attestation),
                // L'analisi è costata minuti di GPU: un'attestazione fallita non deve buttarla via.
                Err(e) => report.attestation_error = Some(format!("{e:#}")),
            }
        }
        Ok(report)
    }
}
