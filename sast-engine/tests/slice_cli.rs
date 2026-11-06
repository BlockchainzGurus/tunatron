//! Sottocomando `slice`: backward slice esportate come JSONL, con test sull'albero di sorgenti reale (nessun mock).

use std::fs;
use std::path::{Path, PathBuf};

use sast_engine::config::Config;
use sast_engine::graph_slicer::{ScanLimits, SliceOptions, build_graph};
use sast_engine::rag_cwe::CweKb;
use sast_engine::slice_cli::{SliceArgs, parse_args, run, slice_to_json};
use serde_json::Value;
use tempfile::TempDir;

/// Albero di sorgenti temporaneo: ogni coppia è (percorso relativo, contenuto).
fn tree(files: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (rel, content) in files {
        let path = dir.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
    dir
}

/// Esegue `slice_cli::run` con i default e restituisce (record JSONL, stderr).
/// I target, se presenti, stanno in un file temporaneo fuori dalla radice.
fn slice(root: &Path, targets: Option<&str>, depth: Option<usize>) -> (Vec<Value>, String) {
    let targets_dir = tempfile::tempdir().unwrap();
    let targets_path = targets.map(|t| {
        let p = targets_dir.path().join("targets.jsonl");
        fs::write(&p, t).unwrap();
        p
    });
    let args = SliceArgs { root: root.to_path_buf(), targets: targets_path, depth, callers: None, max_chars: None };
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let written = run(&args, &Config::default(), &CweKb::embedded(), &mut out, &mut err).unwrap();
    let records: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(written, records.len(), "run deve restituire le righe scritte");
    (records, String::from_utf8(err).unwrap())
}

fn depths(record: &Value) -> Vec<u64> {
    record["blocks"].as_array().unwrap().iter().map(|b| b["depth"].as_u64().unwrap()).collect()
}

/// `f1` -> `f2` -> `f3` -> `f4` -> `f5`, e solo `f5` chiama il sink `system`.
const CHAIN: &str = "void f5(char *x)\n{\n    system(x);\n}\n\nvoid f4(char *x)\n{\n    f5(x);\n}\n\nvoid f3(char *x)\n{\n    f4(x);\n}\n\nvoid f2(char *x)\n{\n    f3(x);\n}\n\nvoid f1(void)\n{\n    f2(getenv(\"X\"));\n}\n";

#[test]
fn depth_5_returns_five_blocks_outermost_first() {
    let d = tree(&[("a.c", CHAIN)]);
    let (records, _) = slice(d.path(), None, Some(5));
    assert_eq!(records.len(), 1);
    let r = &records[0];
    assert_eq!(depths(r), vec![4, 3, 2, 1, 0]);
    assert_eq!(r["blocks"][0]["name"], "f1");
    assert_eq!(r["blocks"][0]["file"], "a.c");
    assert_eq!(r["blocks"][0]["lang"], "c");
    assert_eq!(r["blocks"][4]["name"], "f5");
    assert_eq!(r["sink"]["function"], "system");
    assert_eq!(r["sink"]["file"], "a.c");
    assert_eq!(r["sink"]["line"], 3);
    assert_eq!(r["sink"]["cwe_id"], "CWE-78");
    assert_eq!(r["truncated"], false);
}

#[test]
fn depth_3_cuts_the_root() {
    let d = tree(&[("a.c", CHAIN)]);
    let (records, _) = slice(d.path(), None, Some(3));
    assert_eq!(records.len(), 1);
    assert_eq!(depths(&records[0]), vec![3, 2, 1, 0]);
    assert_eq!(records[0]["blocks"][0]["name"], "f2");
}

#[test]
fn target_slices_the_enclosing_function_of_a_macro_sink() {
    let d = tree(&[("m.c", "void h(char *d)\n{\n    SYSTEM(d);\n}\n")]);
    let target = r#"{"file":"m.c","line":3,"callee":"system","cwe_id":"CWE-78"}"#;
    let (records, stderr) = slice(d.path(), Some(target), None);
    assert_eq!(records.len(), 1, "stderr: {stderr}");
    let r = &records[0];
    assert_eq!(r["sink"]["function"], "system");
    assert_eq!(r["sink"]["file"], "m.c");
    assert_eq!(r["sink"]["line"], 3);
    assert_eq!(r["sink"]["cwe_id"], "CWE-78");
    let last = r["blocks"].as_array().unwrap().last().unwrap();
    assert_eq!(last["name"], "h");
    assert_eq!(last["depth"], 0);
    assert!(last["code"].as_str().unwrap().contains("SYSTEM(d)"));
}

#[test]
fn targets_mode_emits_only_the_targets_not_the_catalog_sinks() {
    // `g` contiene un sink del catalogo (`system`), `h` una macro che il catalogo non conosce.
    let d = tree(&[("m.c", "void g(char *d)\n{\n    system(d);\n}\n\nvoid h(char *d)\n{\n    RUN_SHELL(d);\n}\n")]);

    let (catalog, _) = slice(d.path(), None, None);
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0]["sink"]["function"], "system");

    let target = r#"{"file":"m.c","line":8,"callee":"RUN_SHELL","cwe_id":"CWE-78"}"#;
    let (records, _) = slice(d.path(), Some(target), None);
    assert_eq!(records.len(), 1, "il sink del catalogo non va emesso con --targets");
    assert_eq!(records[0]["sink"]["function"], "RUN_SHELL");
    assert_eq!(records[0]["sink"]["line"], 8);
    assert_eq!(records[0]["blocks"].as_array().unwrap().last().unwrap()["name"], "h");
}

#[test]
fn target_outside_any_function_is_skipped_with_a_message() {
    let d = tree(&[("m.c", "int g = 1;\nvoid h(char *d) {\n    SYSTEM(d);\n}\n")]);
    let targets = concat!(
        r#"{"file":"m.c","line":1,"callee":"system","cwe_id":"CWE-78"}"#,
        "\n",
        r#"{"file":"m.c","line":3,"callee":"system","cwe_id":"CWE-78"}"#,
        "\n"
    );
    let (records, stderr) = slice(d.path(), Some(targets), None);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["sink"]["line"], 3);
    assert!(stderr.contains("fuori da ogni funzione"), "{stderr}");
    assert!(stderr.contains("riga 1: m.c:1"), "{stderr}");
}

#[test]
fn catalog_sinks_equal_server_slices() {
    let d = tree(&[("app.py", "import os\n\ndef run(cmd):\n    os.system(cmd)\n\ndef main():\n    run(input())\n")]);
    let (records, _) = slice(d.path(), None, None);

    let limits = ScanLimits { max_files: usize::MAX, max_sinks: usize::MAX, ..Config::default().scan };
    let graph = build_graph(d.path(), &CweKb::embedded(), &limits).unwrap();
    let expected: Vec<Value> =
        (0..graph.sinks.len()).map(|i| slice_to_json(&graph.slice(i, &SliceOptions::default()), &graph.sinks[i].cwe_id)).collect();

    assert!(!expected.is_empty());
    assert_eq!(records, expected);
    assert_eq!(records[0]["sink"]["function"], "os.system");
    assert_eq!(records[0]["blocks"][0]["lang"], "python");
}

#[test]
fn parse_args_rejects_missing_root_and_unknown_flag() {
    let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert!(parse_args(&args(&[])).is_err(), "radice mancante");
    assert!(parse_args(&args(&["--depth", "2"])).is_err(), "solo flag, nessuna radice");
    assert!(parse_args(&args(&["repo", "--bogus"])).is_err(), "flag ignoto");
    assert!(parse_args(&args(&["repo", "--depth"])).is_err(), "flag senza valore");
    assert!(parse_args(&args(&["repo", "--depth", "tre"])).is_err(), "valore non numerico");
    assert!(parse_args(&args(&["repo", "--callers", "0"])).is_err(), "callers deve essere positivo");
    assert!(parse_args(&args(&["repo", "altra"])).is_err(), "due radici");
}

#[test]
fn parse_args_reads_root_and_all_flags() {
    let args: Vec<String> =
        ["--depth", "5", "repo", "--targets", "t.jsonl", "--callers", "2", "--max-chars", "900"].iter().map(|s| s.to_string()).collect();
    let parsed = parse_args(&args).unwrap();
    assert_eq!(parsed.root, PathBuf::from("repo"));
    assert_eq!(parsed.targets, Some(PathBuf::from("t.jsonl")));
    assert_eq!((parsed.depth, parsed.callers, parsed.max_chars), (Some(5), Some(2), Some(900)));

    let bare = parse_args(&["repo".to_string()]).unwrap();
    assert_eq!((bare.targets, bare.depth, bare.callers, bare.max_chars), (None, None, None, None));
}
