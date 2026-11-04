mod common;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use common::{chat_reply, chat_reply_with_delay, finding_json, start, status_reply};
use sast_engine::vllm_client::{Prompt, VllmClient, VllmConfig};

fn cfg(url: &str) -> VllmConfig {
    VllmConfig {
        base_url: url.to_string(),
        model: "sast-nemotron".into(),
        concurrency: 4,
        timeout: Duration::from_secs(5),
        max_tokens: 512,
        retries: 2,
        retry_backoff: Duration::from_millis(5),
    }
}

fn prompt(user: &str) -> Prompt {
    Prompt { system: "SYS".into(), user: user.into() }
}

#[tokio::test]
async fn request_is_deterministic_guided_and_without_thinking() {
    let mock = start(|_, _| chat_reply(&finding_json(true, "CWE-89", "ok"))).await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    client.complete(&prompt("USER")).await.unwrap();

    let body = mock.bodies.lock().unwrap()[0].clone();
    assert_eq!(body["model"], "sast-nemotron");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["max_tokens"], 512);
    assert_eq!(body["messages"][0], serde_json::json!({"role": "system", "content": "SYS"}));
    assert_eq!(body["messages"][1], serde_json::json!({"role": "user", "content": "USER"}));
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    let rf = &body["response_format"];
    assert_eq!(rf["type"], "json_schema");
    assert_eq!(rf["json_schema"]["strict"], true);
    let schema = &rf["json_schema"]["schema"];
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema["required"].as_array().unwrap().iter().any(|v| v == "vex_status"));
}

#[tokio::test]
async fn parses_the_finding_from_the_assistant_message() {
    let mock = start(|_, _| chat_reply(&finding_json(true, "CWE-89", "motivo"))).await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    let f = client.complete(&prompt("x")).await.unwrap();
    assert!(f.vulnerable);
    assert_eq!((f.cwe_id.as_str(), f.reasoning.as_str()), ("CWE-89", "motivo"));
}

#[tokio::test]
async fn analyze_all_keeps_order_and_runs_in_parallel_up_to_the_limit() {
    let mock = start(|body, _| {
        let user = body["messages"][1]["content"].as_str().unwrap().to_string();
        chat_reply_with_delay(&finding_json(false, "CWE-89", &format!("echo {user}")), 100)
    })
    .await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    let prompts: Vec<Prompt> = (0..8).map(|i| prompt(&format!("p{i}"))).collect();

    let started = Instant::now();
    let results = client.analyze_all(prompts).await;
    let elapsed = started.elapsed();

    let echoed: Vec<String> = results.iter().map(|r| r.as_ref().unwrap().reasoning.clone()).collect();
    assert_eq!(echoed, (0..8).map(|i| format!("echo p{i}")).collect::<Vec<_>>());
    assert_eq!(mock.max_in_flight.load(Ordering::SeqCst), 4, "deve usare tutta la concorrenza ma non oltre");
    assert!(elapsed < Duration::from_millis(700), "8 richieste da 100ms con concorrenza 4 non devono essere sequenziali: {elapsed:?}");
}

#[tokio::test]
async fn one_failing_request_does_not_poison_the_others() {
    let mock = start(|body, _| {
        if body["messages"][1]["content"] == "bad" {
            status_reply(400, "schema non supportato")
        } else {
            chat_reply(&finding_json(true, "CWE-89", "ok"))
        }
    })
    .await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    let results = client.analyze_all(vec![prompt("a"), prompt("bad"), prompt("c")]).await;
    assert!(results[0].is_ok());
    assert!(results[1].as_ref().unwrap_err().contains("400"));
    assert!(results[2].is_ok());
}

#[tokio::test]
async fn retries_server_errors_then_succeeds() {
    let mock = start(|_, i| if i < 2 { status_reply(503, "overloaded") } else { chat_reply(&finding_json(true, "CWE-89", "ok")) }).await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    assert!(client.complete(&prompt("x")).await.is_ok());
    assert_eq!(mock.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn gives_up_after_the_configured_retries() {
    let mock = start(|_, _| status_reply(500, "boom")).await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    let err = client.complete(&prompt("x")).await.unwrap_err();
    assert!(err.contains("500"), "{err}");
    assert_eq!(mock.calls.load(Ordering::SeqCst), 3, "1 tentativo + 2 retry");
}

#[tokio::test]
async fn client_errors_are_not_retried() {
    let mock = start(|_, _| status_reply(422, "bad schema")).await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    assert!(client.complete(&prompt("x")).await.unwrap_err().contains("422"));
    assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn malformed_model_output_is_an_error_not_a_panic() {
    let mock = start(|_, i| match i {
        0 => chat_reply("questo non è JSON"),
        1 => chat_reply(r#"{"vulnerable": true}"#),
        _ => status_reply(200, r#"{"choices":[{"message":{"role":"assistant","content":null}}]}"#),
    })
    .await;
    let client = VllmClient::new(cfg(&mock.url)).unwrap();
    for _ in 0..3 {
        assert!(client.complete(&prompt("x")).await.is_err());
    }
}

#[tokio::test]
async fn unreachable_server_is_reported() {
    let client = VllmClient::new(cfg("http://127.0.0.1:1")).unwrap();
    let err = client.complete(&prompt("x")).await.unwrap_err();
    assert!(err.contains("vLLM"), "{err}");
}

#[tokio::test]
async fn health_reflects_server_availability() {
    let mock = start(|_, _| chat_reply("x")).await;
    assert!(VllmClient::new(cfg(&mock.url)).unwrap().health().await);
    assert!(!VllmClient::new(cfg("http://127.0.0.1:1")).unwrap().health().await);
}
