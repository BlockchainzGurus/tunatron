//! Server vLLM finto (compatibile con `/v1/chat/completions`) per i test, senza GPU.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

pub struct Reply {
    pub status: u16,
    pub delay_ms: u64,
    pub body: String,
}

/// Risposta 200 con `content` come messaggio dell'assistente.
pub fn chat_reply(content: &str) -> Reply {
    chat_reply_with_delay(content, 0)
}

pub fn chat_reply_with_delay(content: &str, delay_ms: u64) -> Reply {
    let body = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    });
    Reply { status: 200, delay_ms, body: body.to_string() }
}

pub fn status_reply(status: u16, body: &str) -> Reply {
    Reply { status, delay_ms: 0, body: body.to_string() }
}

pub fn finding_json(vulnerable: bool, cwe: &str, reasoning: &str) -> String {
    json!({
        "vulnerable": vulnerable,
        "cwe_id": cwe,
        "severity": if vulnerable { "HIGH" } else { "NONE" },
        "sink_function": "ignored",
        "sink_file": "ignored.py",
        "sink_line": 1,
        "vex_status": if vulnerable { "affected" } else { "not_affected" },
        "reasoning": reasoning,
        "suggested_fix": "fix"
    })
    .to_string()
}

pub struct MockVllm {
    pub url: String,
    pub calls: Arc<AtomicUsize>,
    pub max_in_flight: Arc<AtomicUsize>,
    pub bodies: Arc<Mutex<Vec<Value>>>,
}

type Responder = dyn Fn(&Value, usize) -> Reply + Send + Sync;

struct Inner {
    responder: Box<Responder>,
    calls: Arc<AtomicUsize>,
    in_flight: AtomicUsize,
    max_in_flight: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<Value>>>,
}

async fn chat(State(st): State<Arc<Inner>>, Json(body): Json<Value>) -> Response {
    let index = st.calls.fetch_add(1, Ordering::SeqCst);
    let now = st.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    st.max_in_flight.fetch_max(now, Ordering::SeqCst);
    st.bodies.lock().unwrap().push(body.clone());
    let reply = (st.responder)(&body, index);
    if reply.delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(reply.delay_ms)).await;
    }
    st.in_flight.fetch_sub(1, Ordering::SeqCst);
    (StatusCode::from_u16(reply.status).unwrap(), [("content-type", "application/json")], reply.body).into_response()
}

pub async fn start(responder: impl Fn(&Value, usize) -> Reply + Send + Sync + 'static) -> MockVllm {
    let calls = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let inner = Arc::new(Inner {
        responder: Box::new(responder),
        calls: calls.clone(),
        in_flight: AtomicUsize::new(0),
        max_in_flight: max_in_flight.clone(),
        bodies: bodies.clone(),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/health", get(|| async { "ok" }))
        .with_state(inner);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    MockVllm { url, calls, max_in_flight, bodies }
}

// --------------------------------------------------------------------------- IPFS e nodo RPC finti
pub const CONTRACT: &str = "0x5FbDB2315678afecb367f032d93F642f64180aa3";
pub const ACCOUNT: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";

#[derive(Default)]
pub struct IpfsSeen {
    pub content_type: String,
    pub body: Vec<u8>,
    pub query: String,
}

pub async fn spawn(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

pub async fn mock_ipfs(cid: &'static str) -> (String, Arc<Mutex<IpfsSeen>>) {
    let seen = Arc::new(Mutex::new(IpfsSeen::default()));
    let app = Router::new()
        .route(
            "/api/v0/add",
            post(
                move |State(seen): State<Arc<Mutex<IpfsSeen>>>, axum::extract::RawQuery(q): axum::extract::RawQuery, headers: HeaderMap, body: Bytes| async move {
                    let mut s = seen.lock().unwrap();
                    s.content_type = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                    s.body = body.to_vec();
                    s.query = q.unwrap_or_default();
                    format!("{{\"Name\":\"report.json\",\"Hash\":\"{cid}\",\"Size\":\"{}\"}}\n", body.len())
                },
            ),
        )
        .with_state(seen.clone());
    (spawn(app).await, seen)
}

pub struct RpcState {
    pub calls: Mutex<Vec<(String, Value)>>,
    pub receipt_status: &'static str,
}

pub async fn mock_rpc(receipt_status: &'static str) -> (String, Arc<RpcState>) {
    let st = Arc::new(RpcState { calls: Mutex::new(vec![]), receipt_status });
    let app = Router::new()
        .route(
            "/",
            post(|State(st): State<Arc<RpcState>>, Json(req): Json<Value>| async move {
                let method = req["method"].as_str().unwrap().to_string();
                st.calls.lock().unwrap().push((method.clone(), req["params"].clone()));
                let result = match method.as_str() {
                    "eth_accounts" => json!([ACCOUNT]),
                    "eth_sendTransaction" => json!("0xabc123"),
                    "eth_getTransactionReceipt" => json!({"status": st.receipt_status}),
                    _ => return Json(json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "method not found"}})),
                };
                Json(json!({"jsonrpc": "2.0", "id": 1, "result": result}))
            }),
        )
        .with_state(st.clone());
    (spawn(app).await, st)
}


// --------------------------------------------------------------------------- repository di prova
pub fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vulnerable_app")
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Repository git locale (una commit) con il contenuto della fixture; si clona con `file://<path>`.
pub fn git_repo_from_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    copy_dir(&fixture_dir(), dir.path());
    for args in [
        vec!["init", "-q", "-b", "main"],
        vec!["add", "."],
        vec!["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ] {
        let out = std::process::Command::new("git").current_dir(dir.path()).args(&args).output().expect("git non disponibile");
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    dir
}
