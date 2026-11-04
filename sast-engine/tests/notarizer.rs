mod common;

use common::{ACCOUNT, CONTRACT, mock_ipfs, mock_rpc};
use sast_engine::notarizer::{ATTEST_SIGNATURE, NotarizerConfig, encode_attest_call, ipfs_add, notarize, selector, send_attestation, sha256};

fn cfg(ipfs: &str, rpc: &str) -> NotarizerConfig {
    NotarizerConfig {
        ipfs_api: ipfs.into(),
        rpc_url: rpc.into(),
        contract: Some(CONTRACT.into()),
        ..NotarizerConfig::default()
    }
}

#[tokio::test]
async fn ipfs_add_uploads_multipart_and_returns_the_cid() {
    let (url, seen) = mock_ipfs("bafkreiexample").await;
    let cid = ipfs_add(&reqwest::Client::new(), &url, b"{\"report\":true}").await.unwrap();
    assert_eq!(cid, "bafkreiexample");
    let s = seen.lock().unwrap();
    assert!(s.content_type.starts_with("multipart/form-data"), "{}", s.content_type);
    assert!(String::from_utf8_lossy(&s.body).contains("{\"report\":true}"));
    assert!(s.query.contains("pin=true") && s.query.contains("cid-version=1"), "{}", s.query);
}

#[tokio::test]
async fn ipfs_add_reports_unreachable_node() {
    let err = ipfs_add(&reqwest::Client::new(), "http://127.0.0.1:1", b"x").await.unwrap_err();
    assert!(err.to_string().contains("IPFS"), "{err}");
}

#[tokio::test]
async fn attestation_sends_the_encoded_call_from_the_first_account() {
    let (url, st) = mock_rpc("0x1").await;
    let (rh, mh) = ([1u8; 32], [2u8; 32]);
    let tx = send_attestation(&reqwest::Client::new(), &cfg("http://unused", &url), "bafy", rh, mh).await.unwrap();
    assert_eq!(tx, "0xabc123");

    let calls = st.calls.lock().unwrap();
    let send = calls.iter().find(|(m, _)| m == "eth_sendTransaction").unwrap();
    let tx_obj = &send.1[0];
    assert_eq!(tx_obj["from"], ACCOUNT);
    assert_eq!(tx_obj["to"], CONTRACT);
    let expected = format!("0x{}", hex::encode(encode_attest_call("bafy", rh, mh)));
    assert_eq!(tx_obj["data"], expected.as_str());
    assert_eq!(&hex::decode(&expected[2..]).unwrap()[..4], &selector(ATTEST_SIGNATURE));
}

#[tokio::test]
async fn reverted_transaction_is_an_error() {
    let (url, _) = mock_rpc("0x0").await;
    let err = send_attestation(&reqwest::Client::new(), &cfg("http://unused", &url), "bafy", [0; 32], [0; 32]).await.unwrap_err();
    assert!(err.to_string().contains("revert"), "{err}");
}

#[tokio::test]
async fn missing_contract_address_is_reported_before_any_network_call() {
    let mut c = cfg("http://unused", "http://127.0.0.1:1");
    c.contract = None;
    let err = send_attestation(&reqwest::Client::new(), &c, "bafy", [0; 32], [0; 32]).await.unwrap_err();
    assert!(err.to_string().contains("ATTEST_CONTRACT"), "{err}");
}

#[tokio::test]
async fn malformed_contract_address_is_rejected() {
    let mut c = cfg("http://unused", "http://127.0.0.1:1");
    c.contract = Some("0x1234".into());
    assert!(send_attestation(&reqwest::Client::new(), &c, "bafy", [0; 32], [0; 32]).await.is_err());
}

#[tokio::test]
async fn notarize_ties_report_hash_cid_and_transaction_together() {
    let (ipfs, _) = mock_ipfs("bafkreitest").await;
    let (rpc, st) = mock_rpc("0x1").await;
    let report = br#"{"summary":{"affected":1}}"#;
    let a = notarize(report, &cfg(&ipfs, &rpc)).await.unwrap();

    assert_eq!(a.cid, "bafkreitest");
    assert_eq!(a.report_sha256, hex::encode(sha256(report)));
    assert_eq!(a.tx_hash, "0xabc123");
    assert_eq!(a.model_provenance_sha256.len(), 64);

    let calls = st.calls.lock().unwrap();
    let data = calls.iter().find(|(m, _)| m == "eth_sendTransaction").unwrap().1[0]["data"].as_str().unwrap().to_string();
    // il calldata contiene l'hash del report e il CID
    assert!(data.contains(&a.report_sha256));
    assert!(data.contains(&hex::encode("bafkreitest")));
}
