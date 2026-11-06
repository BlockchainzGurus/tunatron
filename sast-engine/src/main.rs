use std::io::{BufWriter, Write};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use sast_engine::config::Config;
use sast_engine::pipeline::Engine;
use sast_engine::rag_cwe::CweKb;
use sast_engine::{server, slice_cli};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "slice" => run_slice(rest),
        _ => serve(),
    }
}

/// `sast-engine slice <ROOT> ...`: JSONL su stdout, messaggi su stderr, poi termina.
fn run_slice(args: &[String]) -> anyhow::Result<()> {
    let args = slice_cli::parse_args(args)?;
    let cfg = Config::from_env().map_err(|e| anyhow!("configurazione non valida: {e}"))?;
    let kb = CweKb::load_or_embedded(cfg.kb_path.as_deref())?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    slice_cli::run(&args, &cfg, &kb, &mut out, &mut std::io::stderr())?;
    out.flush()?;
    Ok(())
}

#[tokio::main]
async fn serve() -> anyhow::Result<()> {
    let cfg = Config::from_env().map_err(|e| anyhow!("configurazione non valida: {e}"))?;
    let bind = cfg.bind.clone();
    let engine = Arc::new(Engine::new(cfg)?);

    let listener = tokio::net::TcpListener::bind(&bind).await.with_context(|| format!("bind su {bind}"))?;
    let addr = listener.local_addr()?;
    eprintln!("sast-engine in ascolto su http://{addr} (vLLM: {})", engine.config().vllm.base_url);
    if !addr.ip().is_loopback() {
        eprintln!("ATTENZIONE: il servizio clona URL scelti dal client; esponilo solo su reti fidate.");
    }
    axum::serve(listener, server::router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
