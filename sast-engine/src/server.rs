//! Server HTTP (axum): `POST /analyze` e `GET /health`.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Semaphore;

use crate::pipeline::Engine;

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    permits: Arc<Semaphore>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalyzeRequest {
    repo_url: String,
    #[serde(default)]
    notarize: bool,
}

fn error(status: u16, message: impl Into<String>) -> Response {
    (StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(json!({"error": message.into()}))).into_response()
}

async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "vllm": st.engine.model_healthy().await}))
}

async fn analyze(State(st): State<AppState>, Json(req): Json<AnalyzeRequest>) -> Response {
    // Una sola analisi alla volta (configurabile): ogni scansione occupa GPU e RAM della macchina.
    let Ok(_permit) = st.permits.clone().try_acquire_owned() else {
        return error(429, "analisi già in corso: riprova più tardi");
    };
    let limit = st.engine.config().analysis_timeout;
    match tokio::time::timeout(limit, st.engine.analyze_url(&req.repo_url, req.notarize)).await {
        Err(_) => error(504, format!("analisi interrotta dopo {}s", limit.as_secs())),
        Ok(Err(e)) => error(e.http_status(), e.to_string()),
        Ok(Ok(report)) => Json(report).into_response(),
    }
}

pub fn router(engine: Arc<Engine>) -> Router {
    let permits = Arc::new(Semaphore::new(engine.config().max_concurrent_analyses));
    Router::new()
        .route("/health", get(health))
        .route("/analyze", post(analyze))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(AppState { engine, permits })
}
