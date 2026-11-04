mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{CONTRACT, Reply, chat_reply, chat_reply_with_delay, finding_json, git_repo_from_fixture, mock_ipfs, mock_rpc, spawn, start};
use sast_engine::config::Config;
use sast_engine::pipeline::Engine;
use sast_engine::server::router;
use serde_json::{Value, json};

fn config_for(vllm_url: &str) -> Config {
    let mut c = Config::default();
    c.vllm.base_url = vllm_url.into();
    c.vllm.retry_backoff = Duration::from_millis(5);
    c.ingest.allow_file_urls = true;
    c
}

fn judge(body: &Value, _: usize) -> Reply {
    let vulnerable = body["messages"][1]["content"].as_str().unwrap().contains("f\"SELECT");
    chat_reply(&finding_json(vulnerable, "CWE-89", "r"))
}

async fn serve(cfg: Config) -> String {
    spawn(router(Arc::new(Engine::new(cfg).unwrap()))).await
}

async fn post_analyze(base: &str, body: Value) -> (u16, Value) {
    let r = reqwest::Client::new().post(format!("{base}/analyze")).json(&body).send().await.unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap_or(Value::Null))
}

#[tokio::test]
async fn health_reports_model_availability() {
    let mock = start(judge).await;
    let up = serve(config_for(&mock.url)).await;
    let down = serve(config_for("http://127.0.0.1:1")).await;
    let ok: Value = reqwest::get(format!("{up}/health")).await.unwrap().json().await.unwrap();
    assert_eq!(ok, json!({"status": "ok", "vllm": true}));
    let degraded: Value = reqwest::get(format!("{down}/health")).await.unwrap().json().await.unwrap();
    assert_eq!(degraded, json!({"status": "ok", "vllm": false}));
}

#[tokio::test]
async fn analyze_returns_the_report_for_a_repository() {
    let mock = start(judge).await;
    let base = serve(config_for(&mock.url)).await;
    let src = git_repo_from_fixture();
    let (status, body) = post_analyze(&base, json!({"repo_url": format!("file://{}", src.path().display())})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["summary"]["sinks_found"], 2);
    assert_eq!(body["summary"]["affected"], 1);
    assert_eq!(body["bom"]["bomFormat"], "CycloneDX");
    assert!(body.get("attestation").is_none());
}

#[tokio::test]
async fn bad_requests_get_4xx_with_a_json_error() {
    let base = serve(config_for("http://unused")).await;
    let (status, body) = post_analyze(&base, json!({"repo_url": "https://evil.example.com/o/r"})).await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("URL"), "{body}");

    let (status, _) = post_analyze(&base, json!({"repo_url": "https://github.com/o/r", "shell": "rm -rf /"})).await;
    assert!((400..500).contains(&status), "campi sconosciuti vanno rifiutati, status={status}");

    let (status, _) = post_analyze(&base, json!({})).await;
    assert!((400..500).contains(&status));
}

#[tokio::test]
async fn model_outage_maps_to_503() {
    let base = serve(config_for("http://127.0.0.1:1")).await;
    let src = git_repo_from_fixture();
    let (status, body) = post_analyze(&base, json!({"repo_url": format!("file://{}", src.path().display())})).await;
    assert_eq!(status, 503, "{body}");
}

#[tokio::test]
async fn a_second_concurrent_analysis_is_refused_with_429() {
    let mock = start(|_, _| chat_reply_with_delay(&finding_json(true, "CWE-89", "r"), 600)).await;
    let base = serve(config_for(&mock.url)).await; // max_concurrent_analyses = 1
    let src = git_repo_from_fixture();
    let url = format!("file://{}", src.path().display());
    let (a, b) = tokio::join!(post_analyze(&base, json!({"repo_url": url})), async {
        tokio::time::sleep(Duration::from_millis(150)).await;
        post_analyze(&base, json!({"repo_url": url})).await
    });
    let mut statuses = [a.0, b.0];
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 429]);
}

#[tokio::test]
async fn analysis_longer_than_the_timeout_gets_504() {
    let mock = start(|_, _| chat_reply_with_delay(&finding_json(true, "CWE-89", "r"), 2000)).await;
    let mut cfg = config_for(&mock.url);
    cfg.analysis_timeout = Duration::from_millis(300);
    let base = serve(cfg).await;
    let src = git_repo_from_fixture();
    let (status, _) = post_analyze(&base, json!({"repo_url": format!("file://{}", src.path().display())})).await;
    assert_eq!(status, 504);
}

#[tokio::test]
async fn notarize_flag_returns_the_attestation() {
    let mock = start(judge).await;
    let (ipfs, _) = mock_ipfs("bafkreihttp").await;
    let (rpc, _) = mock_rpc("0x1").await;
    let mut cfg = config_for(&mock.url);
    cfg.notarizer.ipfs_api = ipfs;
    cfg.notarizer.rpc_url = rpc;
    cfg.notarizer.contract = Some(CONTRACT.into());
    let base = serve(cfg).await;
    let src = git_repo_from_fixture();
    let (status, body) = post_analyze(&base, json!({"repo_url": format!("file://{}", src.path().display()), "notarize": true})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["attestation"]["cid"], "bafkreihttp");
    assert_eq!(body["attestation"]["tx_hash"], "0xabc123");
}

#[tokio::test]
async fn oversized_request_bodies_are_rejected() {
    let base = serve(config_for("http://unused")).await;
    let big = "a".repeat(100_000);
    let (status, _) = post_analyze(&base, json!({"repo_url": big})).await;
    assert_eq!(status, 413);
}
