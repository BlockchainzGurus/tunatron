mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{CONTRACT, Reply, chat_reply, finding_json, fixture_dir, git_repo_from_fixture, mock_ipfs, mock_rpc, start, status_reply};
use sast_engine::config::Config;
use sast_engine::notarizer::sha256;
use sast_engine::pipeline::{Engine, EngineError};
use serde_json::Value;

fn config_for(vllm_url: &str) -> Config {
    let mut c = Config::default();
    c.vllm.base_url = vllm_url.into();
    c.vllm.retry_backoff = Duration::from_millis(5);
    c
}

/// Il "modello" finto giudica vulnerabile solo la query costruita con f-string.
fn judge(body: &Value, _: usize) -> Reply {
    let user = body["messages"][1]["content"].as_str().unwrap();
    if user.contains("f\"SELECT * FROM orders") {
        chat_reply(&finding_json(true, "CWE-89", "f-string con user_id"))
    } else {
        chat_reply(&finding_json(false, "CWE-89", "query parametrizzata"))
    }
}

fn prop(v: &Value, name: &str) -> String {
    v["properties"].as_array().unwrap().iter().find(|p| p["name"] == name).unwrap()["value"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn fixture_repo_produces_sbom_and_vex_report() {
    let mock = start(judge).await;
    let engine = Engine::new(config_for(&mock.url)).unwrap();
    let r = engine.analyze_path(&fixture_dir(), "vulnerable_app").await.unwrap();

    assert_eq!((r.summary.files_scanned, r.summary.functions, r.summary.sinks_found), (3, 5, 2));
    assert_eq!((r.summary.affected, r.summary.not_affected, r.summary.errors), (1, 1, 0));
    assert_eq!(r.summary.model, "sast-nemotron");

    let purls: Vec<&str> = r.bom["components"].as_array().unwrap().iter().map(|c| c["purl"].as_str().unwrap()).collect();
    assert_eq!(purls, vec!["pkg:pypi/flask@2.0.1", "pkg:pypi/requests"]);

    let vulns = r.bom["vulnerabilities"].as_array().unwrap();
    let bad = vulns.iter().find(|v| v["analysis"]["state"] == "exploitable").unwrap();
    assert_eq!((prop(bad, "sast:sink_file").as_str(), prop(bad, "sast:sink_line").as_str()), ("app/db.py", "5"));
    assert_eq!(prop(bad, "sast:sink_function"), "cursor.execute");
    assert_eq!(bad["cwes"], serde_json::json!([89]));
    let safe = vulns.iter().find(|v| v["analysis"]["state"] == "not_affected").unwrap();
    assert_eq!(prop(safe, "sast:sink_line"), "10");
}

#[tokio::test]
async fn prompt_for_the_vulnerable_sink_carries_the_rule_and_the_backward_slice() {
    let mock = start(judge).await;
    Engine::new(config_for(&mock.url)).unwrap().analyze_path(&fixture_dir(), "vulnerable_app").await.unwrap();

    let bodies = mock.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let body = bodies.iter().find(|b| b["messages"][1]["content"].as_str().unwrap().contains("f\"SELECT * FROM orders")).unwrap();
    let user = body["messages"][1]["content"].as_str().unwrap();
    for expected in [
        "CWE-89 — SQL Injection",
        "- Function: cursor.execute",
        "- File: app/db.py",
        "- Line: 5",
        "### app/routes.py · orders_handler (lines 3-5, depth 2)",
        "### app/service.py · list_orders (lines 3-4, depth 1)",
        "### app/db.py · fetch_orders (lines 3-6, depth 0)",
    ] {
        assert!(user.contains(expected), "manca {expected:?} nel prompt:\n{user}");
    }
    let pos = |needle: &str| user.find(needle).unwrap();
    assert!(pos("app/routes.py ·") < pos("app/service.py ·") && pos("app/service.py ·") < pos("app/db.py ·"), "ordine: dal chiamante più esterno al sink");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert!(body["messages"][0]["content"].as_str().unwrap().contains("SAST"));
}

#[tokio::test]
async fn repository_without_sinks_never_calls_the_model() {
    let mock = start(judge).await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.py"), "def f():\n    return 1\n").unwrap();
    let r = Engine::new(config_for(&mock.url)).unwrap().analyze_path(dir.path(), "empty").await.unwrap();
    assert_eq!(r.summary.sinks_found, 0);
    assert_eq!(r.bom["vulnerabilities"].as_array().unwrap().len(), 0);
    assert_eq!(mock.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unreachable_model_fails_fast_with_a_specific_error() {
    let engine = Engine::new(config_for("http://127.0.0.1:1")).unwrap();
    let err = engine.analyze_path(&fixture_dir(), "vulnerable_app").await.unwrap_err();
    assert!(matches!(err, EngineError::ModelUnavailable(_)), "{err}");
    assert_eq!(err.http_status(), 503);
}

#[tokio::test]
async fn a_failing_sink_is_reported_without_losing_the_others() {
    let mock = start(|body, _| {
        if body["messages"][1]["content"].as_str().unwrap().contains("orders") {
            chat_reply(&finding_json(true, "CWE-89", "ok"))
        } else {
            status_reply(400, "boom")
        }
    })
    .await;
    let r = Engine::new(config_for(&mock.url)).unwrap().analyze_path(&fixture_dir(), "vulnerable_app").await.unwrap();
    assert_eq!((r.summary.affected, r.summary.errors), (1, 1));
    assert_eq!(r.errors[0].sink_line, 10);
    assert!(r.errors[0].error.contains("400"));
}

#[tokio::test]
async fn analyze_url_clones_scans_and_removes_the_clone() {
    let mock = start(judge).await;
    let src = git_repo_from_fixture();
    let work = tempfile::tempdir().unwrap();
    let mut cfg = config_for(&mock.url);
    cfg.ingest.allow_file_urls = true;
    cfg.ingest.work_dir = work.path().to_path_buf();
    let engine = Engine::new(cfg).unwrap();

    let r = engine.analyze_url(&format!("file://{}", src.path().display()), false).await.unwrap();
    assert_eq!(r.summary.sinks_found, 2);
    assert!(r.attestation.is_none());
    assert_eq!(std::fs::read_dir(work.path()).unwrap().count(), 0, "il clone temporaneo va rimosso");
}

#[tokio::test]
async fn invalid_url_is_a_request_error_and_clone_failure_is_a_gateway_error() {
    let engine = Engine::new(config_for("http://unused")).unwrap();
    let e = engine.analyze_url("--upload-pack=evil", false).await.unwrap_err();
    assert!(matches!(e, EngineError::InvalidRequest(_)), "{e}");
    assert_eq!(e.http_status(), 400);

    let mut cfg = config_for("http://unused");
    cfg.ingest.allow_file_urls = true;
    let e = Engine::new(cfg).unwrap().analyze_url("file:///definitely/not/a/repo", false).await.unwrap_err();
    assert!(matches!(e, EngineError::CloneFailed(_)), "{e}");
    assert_eq!(e.http_status(), 502);
}

#[tokio::test]
async fn notarization_attaches_an_attestation_over_the_report_hash() {
    let mock = start(judge).await;
    let (ipfs, _) = mock_ipfs("bafkreireport").await;
    let (rpc, _) = mock_rpc("0x1").await;
    let src = git_repo_from_fixture();
    let mut cfg = config_for(&mock.url);
    cfg.ingest.allow_file_urls = true;
    cfg.notarizer.ipfs_api = ipfs;
    cfg.notarizer.rpc_url = rpc;
    cfg.notarizer.contract = Some(CONTRACT.into());
    let engine = Engine::new(cfg).unwrap();

    let report = engine.analyze_url(&format!("file://{}", src.path().display()), true).await.unwrap();
    let att = report.attestation.clone().expect("attestazione");
    assert_eq!((att.cid.as_str(), att.tx_hash.as_str()), ("bafkreireport", "0xabc123"));

    let mut bare = report.clone();
    bare.attestation = None;
    assert_eq!(att.report_sha256, hex::encode(sha256(&serde_json::to_vec(&bare).unwrap())), "l'hash copre il report senza l'attestazione");
}

#[tokio::test]
async fn notarization_failure_keeps_the_analysis() {
    let mock = start(judge).await;
    let src = git_repo_from_fixture();
    let mut cfg = config_for(&mock.url);
    cfg.ingest.allow_file_urls = true;
    cfg.notarizer.ipfs_api = "http://127.0.0.1:1".into();
    cfg.notarizer.contract = Some(CONTRACT.into());
    let report = Engine::new(cfg).unwrap().analyze_url(&format!("file://{}", src.path().display()), true).await.unwrap();
    assert!(report.attestation.is_none());
    assert!(report.attestation_error.as_deref().unwrap().contains("IPFS"));
    assert_eq!(report.summary.affected, 1);
}

#[tokio::test]
async fn notarize_without_contract_is_rejected_before_cloning() {
    let engine = Engine::new(config_for("http://unused")).unwrap();
    let e = engine.analyze_url("https://github.com/o/r", true).await.unwrap_err();
    assert!(matches!(e, EngineError::InvalidRequest(ref m) if m.contains("ATTEST_CONTRACT")), "{e}");
}
