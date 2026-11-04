//! Client asincrono verso il server vLLM (API OpenAI-compatibile) con guided decoding via JSON Schema.

use std::time::Duration;

use futures::stream::{self, StreamExt};
use serde_json::{Value, json};

use crate::vex::Finding;

#[derive(Debug, Clone)]
pub struct VllmConfig {
    pub base_url: String,
    /// Nome passato a `vllm serve --served-model-name`.
    pub model: String,
    /// Richieste in volo contemporaneamente.
    pub concurrency: usize,
    pub timeout: Duration,
    pub max_tokens: u32,
    /// Tentativi aggiuntivi su errori di rete / 5xx / 429.
    pub retries: u32,
    pub retry_backoff: Duration,
}

impl Default for VllmConfig {
    fn default() -> Self {
        VllmConfig {
            base_url: "http://localhost:8000".into(),
            model: "sast-nemotron".into(),
            concurrency: 8,
            timeout: Duration::from_secs(120),
            max_tokens: 768,
            retries: 2,
            retry_backoff: Duration::from_millis(250),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}

#[derive(Clone)]
pub struct VllmClient {
    http: reqwest::Client,
    cfg: VllmConfig,
}

struct Failure {
    retryable: bool,
    message: String,
}

fn excerpt(text: &str) -> String {
    let t: String = text.chars().take(300).collect();
    if t.len() < text.len() { format!("{t}…") } else { t }
}

impl VllmClient {
    pub fn new(cfg: VllmConfig) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder().timeout(cfg.timeout).build()?;
        Ok(VllmClient { http, cfg })
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.cfg.base_url.trim_end_matches('/'), path)
    }

    /// Corpo della richiesta: `temperature` 0, JSON Schema forzato (guided decoding) e thinking disattivato
    /// (il chat template di Nemotron lo accende di default; il training è fatto con `enable_thinking=False`).
    pub fn request_body(&self, prompt: &Prompt) -> Value {
        let schema: Value = serde_json::from_str(include_str!("../../shared/vex_schema.json")).expect("vex_schema.json non valido");
        json!({
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": 0.0,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "vex_finding", "strict": true, "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": false},
        })
    }

    async fn attempt(&self, body: &Value) -> Result<Finding, Failure> {
        let url = self.endpoint("/v1/chat/completions");
        let response = self.http.post(&url).json(body).send().await.map_err(|e| Failure {
            retryable: true,
            message: format!("vLLM non raggiungibile ({url}): {e}"),
        })?;
        let status = response.status();
        let text = response.text().await.map_err(|e| Failure { retryable: true, message: format!("vLLM: lettura risposta fallita: {e}") })?;
        if !status.is_success() {
            return Err(Failure {
                retryable: status.is_server_error() || status.as_u16() == 429,
                message: format!("vLLM ha risposto {status}: {}", excerpt(&text)),
            });
        }
        let fatal = |message: String| Failure { retryable: false, message };
        let envelope: Value = serde_json::from_str(&text).map_err(|e| fatal(format!("vLLM: risposta non JSON: {e}")))?;
        let content = envelope["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| fatal("vLLM: risposta senza contenuto".to_string()))?;
        serde_json::from_str::<Finding>(content.trim())
            .map_err(|e| fatal(format!("output del modello non valido ({e}): {}", excerpt(content))))
    }

    /// Una richiesta, con retry; l'errore è già un messaggio leggibile per il report.
    pub async fn complete(&self, prompt: &Prompt) -> Result<Finding, String> {
        let body = self.request_body(prompt);
        let mut last = String::new();
        for attempt in 0..=self.cfg.retries {
            if attempt > 0 {
                tokio::time::sleep(self.cfg.retry_backoff * 2u32.pow(attempt - 1)).await;
            }
            match self.attempt(&body).await {
                Ok(finding) => return Ok(finding),
                Err(f) => {
                    last = f.message;
                    if !f.retryable {
                        break;
                    }
                }
            }
        }
        Err(last)
    }

    /// Tutte le richieste in parallelo (al massimo `concurrency`); l'ordine dei risultati è quello dei prompt.
    pub async fn analyze_all(&self, prompts: Vec<Prompt>) -> Vec<Result<Finding, String>> {
        let mut results: Vec<(usize, Result<Finding, String>)> = stream::iter(prompts.into_iter().enumerate())
            .map(|(i, prompt)| {
                let client = self.clone();
                async move { (i, client.complete(&prompt).await) }
            })
            .buffer_unordered(self.cfg.concurrency.max(1))
            .collect()
            .await;
        results.sort_by_key(|(i, _)| *i);
        results.into_iter().map(|(_, r)| r).collect()
    }

    pub async fn health(&self) -> bool {
        self.http
            .get(self.endpoint("/health"))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }
}
