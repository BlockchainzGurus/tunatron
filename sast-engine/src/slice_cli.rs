//! Sottocomando `sast-engine slice`: esporta le backward slice del call graph come JSONL.
//!
//! Stesse slice che il server passa al modello (`build_graph` + `CodeGraph::slice`), ma offline: nessun tetto
//! su file e sink, così la pipeline di training può costruire i dati con la stessa traccia usata in inferenza.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;
use crate::graph_slicer::{CodeGraph, ScanLimits, Slice, SliceOptions, build_graph};
use crate::rag_cwe::CweKb;

const USAGE: &str = "uso: sast-engine slice <ROOT> [--targets FILE] [--depth N] [--callers N] [--max-chars N]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceArgs {
    /// Radice dei sorgenti da analizzare.
    pub root: PathBuf,
    /// JSONL di sink espliciti (`{"file","line","callee","cwe_id"}`): se presente sostituisce il catalogo della KB.
    pub targets: Option<PathBuf>,
    pub depth: Option<usize>,
    pub callers: Option<usize>,
    pub max_chars: Option<usize>,
}

/// Un sink indicato dall'utente; `file` è relativo alla radice, con separatore `/`.
#[derive(Debug, Deserialize)]
struct Target {
    file: String,
    line: u32,
    callee: String,
    cwe_id: String,
}

/// Interpreta gli argomenti che seguono `slice`: la radice è obbligatoria, i flag possono stare in qualunque posizione.
pub fn parse_args(args: &[String]) -> anyhow::Result<SliceArgs> {
    let mut root: Option<PathBuf> = None;
    let (mut targets, mut depth, mut callers, mut max_chars) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().with_context(|| format!("{flag} richiede un valore\n{USAGE}"));
        match arg.as_str() {
            "--targets" => targets = Some(PathBuf::from(value("--targets")?)),
            "--depth" => depth = Some(number("--depth", value("--depth")?, false)?),
            "--callers" => callers = Some(number("--callers", value("--callers")?, true)?),
            "--max-chars" => max_chars = Some(number("--max-chars", value("--max-chars")?, true)?),
            flag if flag.starts_with('-') && flag.len() > 1 => bail!("flag sconosciuto {flag}\n{USAGE}"),
            positional => {
                if root.replace(PathBuf::from(positional)).is_some() {
                    bail!("più di una radice indicata ({positional})\n{USAGE}");
                }
            }
        }
    }
    let root = root.with_context(|| format!("radice mancante\n{USAGE}"))?;
    Ok(SliceArgs { root, targets, depth, callers, max_chars })
}

fn number(flag: &str, raw: &str, positive: bool) -> anyhow::Result<usize> {
    let n: usize = raw.parse().map_err(|_| anyhow::anyhow!("{flag}: valore non valido {raw:?}\n{USAGE}"))?;
    if positive && n == 0 {
        bail!("{flag} deve essere maggiore di zero");
    }
    Ok(n)
}

/// Record JSONL di una slice: `sink`, `blocks` (dal chiamante più esterno al sink) e `truncated`.
pub fn slice_to_json(slice: &Slice, cwe_id: &str) -> Value {
    let blocks: Vec<Value> = slice
        .blocks
        .iter()
        .map(|b| {
            json!({
                "file": b.file,
                "name": b.name,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "depth": b.depth,
                "lang": b.lang,
                "code": b.code,
            })
        })
        .collect();
    json!({
        "sink": {
            "function": slice.location.function,
            "file": slice.location.file,
            "line": slice.location.line,
            "cwe_id": cwe_id,
        },
        "blocks": blocks,
        "truncated": slice.truncated,
    })
}

fn write_record(out: &mut dyn Write, graph: &CodeGraph, sink_index: usize, opts: &SliceOptions) -> anyhow::Result<()> {
    let record = slice_to_json(&graph.slice(sink_index, opts), &graph.sinks[sink_index].cwe_id);
    serde_json::to_writer(&mut *out, &record)?;
    out.write_all(b"\n")?;
    Ok(())
}

/// Scrive su `out` un record per sink e restituisce quante righe ha scritto; i messaggi vanno su `err`.
/// Senza `--targets` i sink sono quelli del catalogo della KB; con `--targets` solo quelli indicati
/// (uno per riga valida, nell'ordine del file).
pub fn run(args: &SliceArgs, cfg: &Config, kb: &CweKb, out: &mut dyn Write, err: &mut dyn Write) -> anyhow::Result<usize> {
    // Strumento offline: nessun tetto su file e sink.
    let limits = ScanLimits { max_files: usize::MAX, max_sinks: usize::MAX, ..cfg.scan.clone() };
    let mut opts = cfg.slice.clone();
    if let Some(depth) = args.depth {
        opts.max_depth = depth;
    }
    if let Some(callers) = args.callers {
        opts.max_callers_per_node = callers;
    }
    if let Some(max_chars) = args.max_chars {
        opts.max_chars = max_chars;
    }

    let mut graph = build_graph(&args.root, kb, &limits)?;
    let (mut written, mut skipped) = (0, 0);
    match &args.targets {
        None => {
            for index in 0..graph.sinks.len() {
                write_record(out, &graph, index, &opts)?;
                written += 1;
            }
        }
        Some(path) => {
            let text = fs::read_to_string(path).with_context(|| format!("lettura target {}", path.display()))?;
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let n = i + 1;
                let target: Target = serde_json::from_str(line).with_context(|| format!("{}: riga {n} non valida", path.display()))?;
                match graph.add_target_sink(&target.file, target.line, &target.callee, &target.cwe_id) {
                    Some(index) => {
                        write_record(out, &graph, index, &opts)?;
                        written += 1;
                    }
                    None => {
                        writeln!(err, "riga {n}: {}:{} fuori da ogni funzione", target.file, target.line)?;
                        skipped += 1;
                    }
                }
            }
        }
    }
    writeln!(
        err,
        "slice: {written} record scritti, {skipped} target saltati ({} file, {} funzioni)",
        graph.stats.files_scanned, graph.stats.functions
    )?;
    Ok(written)
}
