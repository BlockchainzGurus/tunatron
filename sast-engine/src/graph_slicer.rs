//! Parsing AST con tree-sitter, Call Graph con petgraph, individuazione dei Sink e Backward Slice.
//!
//! Il call graph è euristico (risoluzione per nome semplice, senza type inference): privilegia
//! la funzione nello stesso file e scarta i nomi troppo ambigui. Le chiamate a livello di modulo
//! (fuori da qualunque funzione) non vengono attribuite a nessun nodo e quindi non producono sink.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use tree_sitter::{Node, Parser};
use walkdir::WalkDir;

use crate::prompt::{SinkLocation, TraceBlock};
use crate::rag_cwe::{CweKb, normalize_callee};

/// Directory mai scansionate (dipendenze, artefatti di build, metadati VCS).
pub const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "venv", ".venv", "__pycache__", "dist", "build", "vendor", ".tox", "site-packages"];
/// Un nome definito in più di tanti punti (e non nel file del chiamante) non genera archi.
const MAX_AMBIGUOUS: usize = 3;
/// Budget minimo sotto il quale non vale la pena aggiungere un altro chiamante alla traccia.
const MIN_BLOCK_BUDGET: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Python,
    C,
    Cpp,
}

impl Lang {
    pub fn from_path(path: &Path) -> Option<Lang> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "py" => Some(Lang::Python),
            "c" | "h" => Some(Lang::C),
            "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Some(Lang::Cpp),
            _ => None,
        }
    }

    /// Identificatore usato nella KB e nei fence markdown del prompt.
    pub fn as_str(&self) -> &'static str {
        match self {
            Lang::Python => "python",
            Lang::C => "c",
            Lang::Cpp => "cpp",
        }
    }

    fn grammar(&self) -> tree_sitter::Language {
        match self {
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSite {
    /// Testo del callee così com'è nel sorgente (es. `cursor.execute`, `std::memcpy`).
    pub callee: String,
    /// Riga 1-based.
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub qualname: String,
    /// Percorso relativo alla radice della repository.
    pub file: PathBuf,
    pub lang: Lang,
    pub start_line: u32,
    pub end_line: u32,
    pub start_byte: usize,
    pub end_byte: usize,
    pub calls: Vec<CallSite>,
}

#[derive(Debug, Clone)]
pub struct Sink {
    pub function: NodeIndex,
    pub callee: String,
    pub line: u32,
    pub cwe_id: String,
}

#[derive(Debug, Clone, Default)]
pub struct ScanStats {
    pub files_scanned: usize,
    pub files_skipped: usize,
    pub parse_errors: usize,
    pub functions: usize,
    pub sinks_truncated: bool,
}

#[derive(Debug, Clone)]
pub struct ScanLimits {
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub max_sinks: usize,
}

impl Default for ScanLimits {
    fn default() -> Self {
        ScanLimits { max_file_bytes: 1_000_000, max_files: 20_000, max_sinks: 300 }
    }
}

#[derive(Debug, Clone)]
pub struct SliceOptions {
    pub max_depth: usize,
    pub max_callers_per_node: usize,
    pub max_chars: usize,
}

impl Default for SliceOptions {
    fn default() -> Self {
        SliceOptions { max_depth: 3, max_callers_per_node: 4, max_chars: 12_000 }
    }
}

#[derive(Debug, Clone)]
pub struct Slice {
    pub sink_index: usize,
    pub location: SinkLocation,
    /// Dal chiamante più esterno al sink (ordine di flusso dei dati).
    pub blocks: Vec<TraceBlock>,
    pub truncated: bool,
}

pub struct CodeGraph {
    pub root: PathBuf,
    /// Archi caller -> callee; il peso è la riga (nel caller) della prima chiamata.
    pub graph: DiGraph<FunctionInfo, u32>,
    pub sinks: Vec<Sink>,
    pub stats: ScanStats,
}

// --------------------------------------------------------------------------- parsing

fn text<'a>(node: Node, source: &'a [u8]) -> &'a str {
    node.utf8_text(source).unwrap_or("")
}

fn is_path(t: &str) -> bool {
    !t.is_empty()
        && t.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | ':' | '-' | '>'))
        && t.chars().any(|c| c.is_alphanumeric() || c == '_')
}

/// Testo del callee se è un semplice percorso (`a.b.c`, `a::b`, `p->f`); altrimenti, per catene come
/// `db.cursor().execute`, il solo nome finale dell'attributo/campo.
fn callee_of(func: Node, source: &[u8]) -> Option<String> {
    let full = text(func, source).trim();
    if is_path(full) {
        return Some(full.to_string());
    }
    let last = match func.kind() {
        "attribute" => func.child_by_field_name("attribute"),
        "field_expression" => func.child_by_field_name("field"),
        _ => None,
    }?;
    let last = text(last, source).trim();
    is_path(last).then(|| last.to_string())
}

fn python_qualname(node: Node, name: &str, source: &[u8]) -> String {
    let mut parts = vec![name.to_string()];
    let mut parent = node.parent();
    while let Some(p) = parent {
        if p.kind() == "class_definition"
            && let Some(n) = p.child_by_field_name("name") {
                parts.push(text(n, source).to_string());
            }
        parent = p.parent();
    }
    parts.reverse();
    parts.join(".")
}

fn function_name(lang: Lang, node: Node, source: &[u8]) -> Option<(String, String)> {
    match lang {
        Lang::Python => {
            let name = text(node.child_by_field_name("name")?, source).to_string();
            let qual = python_qualname(node, &name, source);
            Some((name, qual))
        }
        Lang::C | Lang::Cpp => {
            let mut d = node.child_by_field_name("declarator")?;
            while matches!(
                d.kind(),
                "function_declarator" | "pointer_declarator" | "parenthesized_declarator" | "reference_declarator" | "array_declarator" | "attributed_declarator"
            ) {
                d = d.child_by_field_name("declarator").or_else(|| d.named_child(0))?;
            }
            let qual = text(d, source).trim().to_string();
            let name = qual.rsplit("::").next().unwrap_or(&qual).to_string();
            (!name.is_empty()).then_some((name, qual))
        }
    }
}

/// Porta l'inizio all'inizio della riga se prima del nodo c'è solo indentazione (il codice resta leggibile).
fn extend_to_line_start(source: &[u8], start: usize) -> usize {
    let line_start = source[..start].iter().rposition(|b| *b == b'\n').map_or(0, |p| p + 1);
    if source[line_start..start].iter().all(|b| *b == b' ' || *b == b'\t') { line_start } else { start }
}

fn parse_with(parser: &mut Parser, lang: Lang, file: &Path, source: &[u8]) -> (Vec<FunctionInfo>, bool) {
    parser.set_language(&lang.grammar()).expect("versione della grammatica incompatibile con tree-sitter");
    let Some(tree) = parser.parse(source, None) else { return (Vec::new(), true) };
    let root = tree.root_node();
    let mut funcs: Vec<FunctionInfo> = Vec::new();
    let mut stack: Vec<(Node, Option<usize>)> = vec![(root, None)];
    while let Some((node, mut current)) = stack.pop() {
        let kind = node.kind();
        if kind == "function_definition" {
            if let Some((name, qualname)) = function_name(lang, node, source) {
                funcs.push(FunctionInfo {
                    name,
                    qualname,
                    file: file.to_path_buf(),
                    lang,
                    start_line: node.start_position().row as u32 + 1,
                    end_line: node.end_position().row as u32 + 1,
                    start_byte: extend_to_line_start(source, node.start_byte()),
                    end_byte: node.end_byte(),
                    calls: Vec::new(),
                });
                current = Some(funcs.len() - 1);
            }
        } else if let Some(fi) = current {
            let is_call = if lang == Lang::Python { kind == "call" } else { kind == "call_expression" };
            if is_call
                && let Some(callee) = node.child_by_field_name("function").and_then(|f| callee_of(f, source)) {
                    funcs[fi].calls.push(CallSite { callee, line: node.start_position().row as u32 + 1 });
                }
        }
        for i in (0..node.child_count()).rev() {
            if let Some(child) = node.child(i) {
                stack.push((child, current));
            }
        }
    }
    (funcs, root.has_error())
}

/// Estrae le funzioni (con le loro chiamate) da un singolo sorgente. `file` è già relativo alla radice.
/// Il secondo valore indica se l'albero contiene errori di sintassi.
pub fn parse_functions(lang: Lang, file: &Path, source: &[u8]) -> (Vec<FunctionInfo>, bool) {
    parse_with(&mut Parser::new(), lang, file, source)
}

// --------------------------------------------------------------------------- grafo

fn simple_name(callee: &str) -> String {
    let n = normalize_callee(callee);
    n.rsplit('.').next().unwrap_or(&n).to_string()
}

fn skipped_dir(entry: &walkdir::DirEntry) -> bool {
    entry.depth() > 0 && entry.file_type().is_dir() && SKIP_DIRS.contains(&entry.file_name().to_string_lossy().as_ref())
}

/// Scansiona la repository e costruisce call graph + sink.
pub fn build_graph(root: &Path, kb: &CweKb, limits: &ScanLimits) -> anyhow::Result<CodeGraph> {
    let root = root.canonicalize().with_context(|| format!("radice repository non valida: {}", root.display()))?;
    let mut graph: DiGraph<FunctionInfo, u32> = DiGraph::new();
    let mut stats = ScanStats::default();
    let mut parser = Parser::new();

    let walker = WalkDir::new(&root).follow_links(false).sort_by_file_name().into_iter().filter_entry(|e| !skipped_dir(e));
    for entry in walker {
        let Ok(entry) = entry else {
            stats.files_skipped += 1;
            continue;
        };
        let Some(lang) = Lang::from_path(entry.path()) else { continue };
        let ft = entry.file_type();
        // I symlink non vengono mai seguiti: potrebbero puntare fuori dalla repository clonata.
        if ft.is_symlink() || !ft.is_file() {
            stats.files_skipped += usize::from(ft.is_symlink());
            continue;
        }
        let oversized = entry.metadata().map(|m| m.len() > limits.max_file_bytes).unwrap_or(true);
        if oversized || stats.files_scanned >= limits.max_files {
            stats.files_skipped += 1;
            continue;
        }
        let Ok(source) = fs::read(entry.path()) else {
            stats.files_skipped += 1;
            continue;
        };
        let rel = entry.path().strip_prefix(&root).unwrap_or(entry.path()).to_path_buf();
        let (funcs, had_error) = parse_with(&mut parser, lang, &rel, &source);
        stats.files_scanned += 1;
        stats.parse_errors += usize::from(had_error);
        for f in funcs {
            graph.add_node(f);
        }
    }
    stats.functions = graph.node_count();

    let mut by_name: HashMap<String, Vec<NodeIndex>> = HashMap::new();
    for i in graph.node_indices() {
        by_name.entry(graph[i].name.clone()).or_default().push(i);
    }

    let mut edges: Vec<(NodeIndex, NodeIndex, u32)> = Vec::new();
    for caller in graph.node_indices() {
        for call in &graph[caller].calls {
            let Some(candidates) = by_name.get(&simple_name(&call.callee)) else { continue };
            let same_file: Vec<NodeIndex> = candidates.iter().copied().filter(|c| graph[*c].file == graph[caller].file).collect();
            let targets = if !same_file.is_empty() {
                same_file
            } else if candidates.len() <= MAX_AMBIGUOUS {
                candidates.clone()
            } else {
                continue;
            };
            edges.extend(targets.into_iter().map(|t| (caller, t, call.line)));
        }
    }
    for (a, b, line) in edges {
        if graph.find_edge(a, b).is_none() {
            graph.add_edge(a, b, line);
        }
    }

    let mut sinks = Vec::new();
    'scan: for idx in graph.node_indices() {
        let f = &graph[idx];
        for call in &f.calls {
            let Some(entry) = kb.match_sink(f.lang.as_str(), &call.callee) else { continue };
            // Una funzione del repo con lo stesso nome "nudo" (es. `execute(q)`) è una chiamata interna, non il sink di libreria.
            let norm = normalize_callee(&call.callee);
            if !norm.contains('.') && by_name.contains_key(&norm) {
                continue;
            }
            if sinks.len() >= limits.max_sinks {
                stats.sinks_truncated = true;
                break 'scan;
            }
            sinks.push(Sink { function: idx, callee: call.callee.clone(), line: call.line, cwe_id: entry.id.clone() });
        }
    }

    Ok(CodeGraph { root, graph, sinks, stats })
}

// --------------------------------------------------------------------------- slicing

/// Finestra di righe centrata su `focus_line` che sta nel budget. Restituisce (codice, prima riga, ultima riga, troncato).
fn window(code: &str, start_line: u32, focus_line: u32, budget: usize) -> (String, u32, u32, bool) {
    let lines: Vec<&str> = code.lines().collect();
    if lines.is_empty() {
        return (String::new(), start_line, start_line, false);
    }
    let joined_len = lines.iter().map(|l| l.len() + 1).sum::<usize>() - 1;
    let last = start_line + lines.len() as u32 - 1;
    if joined_len <= budget {
        return (lines.join("\n"), start_line, last, false);
    }
    let focus = (focus_line.saturating_sub(start_line) as usize).min(lines.len() - 1);
    let (mut lo, mut hi) = (focus, focus);
    let mut used = lines[focus].len();
    if used > budget {
        let cut: String = lines[focus].chars().take(budget).collect();
        let line = start_line + focus as u32;
        return (cut, line, line, true);
    }
    loop {
        let mut progressed = false;
        if lo > 0 && used + lines[lo - 1].len() < budget {
            lo -= 1;
            used += lines[lo].len() + 1;
            progressed = true;
        }
        if hi + 1 < lines.len() && used + lines[hi + 1].len() < budget {
            hi += 1;
            used += lines[hi].len() + 1;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    (lines[lo..=hi].join("\n"), start_line + lo as u32, start_line + hi as u32, true)
}

impl CodeGraph {
    fn read_function(&self, f: &FunctionInfo) -> String {
        let path = self.root.join(&f.file);
        // Il file potrebbe essere stato sostituito da un symlink dopo la scansione.
        if fs::symlink_metadata(&path).map(|m| m.file_type().is_symlink()).unwrap_or(true) {
            return String::new();
        }
        let Ok(bytes) = fs::read(&path) else { return String::new() };
        bytes.get(f.start_byte..f.end_byte).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default()
    }

    /// Backward slice: visita all'indietro (`Direction::Incoming`) fino a `max_depth` dal sink.
    pub fn slice(&self, sink_index: usize, opts: &SliceOptions) -> Slice {
        let sink = &self.sinks[sink_index];
        let mut seen: HashSet<NodeIndex> = HashSet::from([sink.function]);
        // (nodo, profondità, riga su cui centrare la finestra se il codice va troncato)
        let mut found: Vec<(NodeIndex, u32, u32)> = vec![(sink.function, 0, sink.line)];
        let mut frontier = vec![sink.function];
        for depth in 1..=opts.max_depth {
            let mut next = Vec::new();
            for &n in &frontier {
                let mut callers: Vec<(NodeIndex, u32)> = self
                    .graph
                    .edges_directed(n, Direction::Incoming)
                    .map(|e| (e.source(), *e.weight()))
                    .filter(|(s, _)| !seen.contains(s))
                    .collect();
                callers.sort_by(|a, b| {
                    let (fa, fb) = (&self.graph[a.0], &self.graph[b.0]);
                    (&fa.file, fa.start_line).cmp(&(&fb.file, fb.start_line))
                });
                for (s, line) in callers.into_iter().take(opts.max_callers_per_node) {
                    if seen.insert(s) {
                        next.push(s);
                        found.push((s, depth as u32, line));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        let mut truncated = false;
        let mut remaining = opts.max_chars;
        let mut blocks: Vec<TraceBlock> = Vec::new();
        for (i, (node, depth, focus)) in found.iter().enumerate() {
            let f = &self.graph[*node];
            let budget = if i == 0 {
                if found.len() == 1 { opts.max_chars } else { opts.max_chars * 2 / 3 }
            } else if remaining < MIN_BLOCK_BUDGET {
                truncated = true;
                break;
            } else {
                remaining / 2
            };
            let (code, start_line, end_line, cut) = window(&self.read_function(f), f.start_line, *focus, budget);
            truncated |= cut;
            remaining = remaining.saturating_sub(code.len());
            blocks.push(TraceBlock {
                file: f.file.to_string_lossy().replace('\\', "/"),
                name: f.qualname.clone(),
                start_line,
                end_line,
                depth: *depth,
                lang: f.lang.as_str().to_string(),
                code,
            });
        }
        blocks.sort_by_key(|b| (Reverse(b.depth), b.file.clone(), b.start_line));

        let sink_fn = &self.graph[sink.function];
        Slice {
            sink_index,
            location: SinkLocation { function: sink.callee.clone(), file: sink_fn.file.to_string_lossy().replace('\\', "/"), line: sink.line },
            blocks,
            truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const PY_SRC: &str = r#"
import sqlite3

class Repo:
    def fetch(self, uid):
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM users WHERE id={uid}")
        return cur.fetchall()

def handler(request):
    uid = request.args["id"]
    return Repo().fetch(uid)

def outer():
    def inner():
        os.system("ls")
    inner()
"#;

    fn funcs(lang: Lang, src: &str) -> Vec<FunctionInfo> {
        parse_functions(lang, Path::new("t"), src.as_bytes()).0
    }

    fn by_name<'a>(fs: &'a [FunctionInfo], name: &str) -> &'a FunctionInfo {
        fs.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("funzione {name} non trovata"))
    }

    fn calls(f: &FunctionInfo) -> Vec<(String, u32)> {
        f.calls.iter().map(|c| (c.callee.clone(), c.line)).collect()
    }

    #[test]
    fn lang_is_detected_from_extension() {
        assert_eq!(Lang::from_path(Path::new("a/b.py")), Some(Lang::Python));
        assert_eq!(Lang::from_path(Path::new("a.c")), Some(Lang::C));
        assert_eq!(Lang::from_path(Path::new("a.h")), Some(Lang::C));
        assert_eq!(Lang::from_path(Path::new("a.cpp")), Some(Lang::Cpp));
        assert_eq!(Lang::from_path(Path::new("a.hpp")), Some(Lang::Cpp));
        assert_eq!(Lang::from_path(Path::new("a.rs")), None);
        assert_eq!(Lang::from_path(Path::new("Makefile")), None);
        assert_eq!(Lang::Cpp.as_str(), "cpp");
    }

    #[test]
    fn python_functions_calls_and_lines() {
        let f = funcs(Lang::Python, PY_SRC);
        let fetch = by_name(&f, "fetch");
        assert_eq!(fetch.qualname, "Repo.fetch");
        assert_eq!((fetch.start_line, fetch.end_line), (5, 8));
        assert!(calls(fetch).contains(&("cur.execute".to_string(), 7)));
        assert!(calls(fetch).contains(&("self.conn.cursor".to_string(), 6)));

        let handler = by_name(&f, "handler");
        assert_eq!((handler.start_line, handler.end_line), (10, 12));
        // `Repo().fetch(uid)`: il testo non è un percorso, si ricade sul nome dell'attributo.
        assert!(calls(handler).contains(&("fetch".to_string(), 12)), "{:?}", calls(handler));
    }

    #[test]
    fn python_calls_belong_to_the_innermost_function() {
        let f = funcs(Lang::Python, PY_SRC);
        assert_eq!(calls(by_name(&f, "outer")), vec![("inner".to_string(), 17)]);
        assert_eq!(calls(by_name(&f, "inner")), vec![("os.system".to_string(), 16)]);
    }

    #[test]
    fn python_chained_call_keeps_the_final_method_name() {
        let f = funcs(Lang::Python, "def q(db, s):\n    return db.cursor().execute(s)\n");
        assert!(calls(by_name(&f, "q")).contains(&("execute".to_string(), 2)));
    }

    #[test]
    fn c_functions_and_calls() {
        let src = "#include <string.h>\nstatic int copy_name(char *dst, const char *src) {\n    strcpy(dst, src);\n    return helper(dst);\n}\nint helper(char *p) { return p[0]; }\n";
        let f = funcs(Lang::C, src);
        let copy = by_name(&f, "copy_name");
        assert_eq!((copy.start_line, copy.end_line), (2, 5));
        assert_eq!(calls(copy), vec![("strcpy".to_string(), 3), ("helper".to_string(), 4)]);
        assert_eq!(by_name(&f, "helper").start_line, 6);
    }

    #[test]
    fn c_pointer_returning_function_name_is_found() {
        let f = funcs(Lang::C, "char *dup_str(const char *s) {\n    return strdup(s);\n}\n");
        assert_eq!(by_name(&f, "dup_str").start_line, 1);
    }

    #[test]
    fn cpp_qualified_names_and_member_calls() {
        let src = "void Foo::bar(int n) {\n    std::memcpy(buf, src, n);\n    obj->run();\n}\n";
        let f = funcs(Lang::Cpp, src);
        let bar = by_name(&f, "bar");
        assert_eq!(bar.qualname, "Foo::bar");
        assert_eq!(calls(bar), vec![("std::memcpy".to_string(), 2), ("obj->run".to_string(), 3)]);
    }

    #[test]
    fn syntax_errors_do_not_panic_and_are_reported() {
        let (_, had_error) = parse_functions(Lang::Python, Path::new("t"), b"def broken(:\n  pass\n\ndef ok():\n  os.system(x)\n");
        assert!(had_error);
    }

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn chain_repo() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "app/routes.py", "def get_orders(request):\n    uid = request.args[\"id\"]\n    return service(uid)\n");
        write(d.path(), "app/service.py", "def service(uid):\n    return fetch(uid)\n");
        write(d.path(), "app/db.py", "def fetch(uid):\n    cursor.execute(\"SELECT * FROM t WHERE id=\" + uid)\n");
        d
    }

    fn node(g: &CodeGraph, name: &str) -> NodeIndex {
        g.graph.node_indices().find(|i| g.graph[*i].name == name).unwrap_or_else(|| panic!("nodo {name}"))
    }

    #[test]
    fn graph_has_call_edges_and_the_sink() {
        let d = chain_repo();
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        assert_eq!(g.stats.files_scanned, 3);
        assert_eq!(g.stats.functions, 3);
        assert!(g.graph.find_edge(node(&g, "get_orders"), node(&g, "service")).is_some());
        assert!(g.graph.find_edge(node(&g, "service"), node(&g, "fetch")).is_some());
        assert_eq!(g.sinks.len(), 1);
        let s = &g.sinks[0];
        assert_eq!((s.callee.as_str(), s.line, s.cwe_id.as_str()), ("cursor.execute", 2, "CWE-89"));
        assert_eq!(g.graph[s.function].file, PathBuf::from("app/db.py"));
    }

    #[test]
    fn slice_walks_callers_outermost_first() {
        let d = chain_repo();
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let s = g.slice(0, &SliceOptions::default());
        let order: Vec<(&str, u32)> = s.blocks.iter().map(|b| (b.name.as_str(), b.depth)).collect();
        assert_eq!(order, vec![("get_orders", 2), ("service", 1), ("fetch", 0)]);
        assert_eq!(s.location, SinkLocation { function: "cursor.execute".into(), file: "app/db.py".into(), line: 2 });
        assert!(s.blocks[2].code.contains("cursor.execute"));
        assert!(!s.truncated);
    }

    #[test]
    fn slice_respects_max_depth() {
        let d = chain_repo();
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let s = g.slice(0, &SliceOptions { max_depth: 1, ..SliceOptions::default() });
        let names: Vec<&str> = s.blocks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["service", "fetch"]);
    }

    #[test]
    fn slice_terminates_on_call_cycles() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "a.py", "def a():\n    b()\n\ndef b():\n    a()\n    os.system(x)\n");
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let s = g.slice(0, &SliceOptions::default());
        let names: Vec<&str> = s.blocks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn slice_caps_callers_per_node() {
        let d = tempfile::tempdir().unwrap();
        let mut src = String::from("def target():\n    os.system(x)\n");
        for i in 0..6 {
            src.push_str(&format!("\ndef caller{i}():\n    target()\n"));
        }
        write(d.path(), "m.py", &src);
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let s = g.slice(0, &SliceOptions { max_callers_per_node: 2, ..SliceOptions::default() });
        assert_eq!(s.blocks.len(), 3); // 2 caller + il sink
    }

    #[test]
    fn same_file_definition_wins_over_other_files() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "a.py", "def run():\n    pass\n\ndef main():\n    run()\n");
        write(d.path(), "b.py", "def run():\n    os.system(x)\n");
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let main = node(&g, "main");
        let targets: Vec<PathBuf> = g.graph.neighbors(main).map(|n| g.graph[n].file.clone()).collect();
        assert_eq!(targets, vec![PathBuf::from("a.py")]);
    }

    #[test]
    fn too_ambiguous_names_create_no_edge() {
        let d = tempfile::tempdir().unwrap();
        for i in 0..5 {
            write(d.path(), &format!("m{i}.py"), "def run():\n    pass\n");
        }
        write(d.path(), "main.py", "def main():\n    run()\n");
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        assert_eq!(g.graph.neighbors(node(&g, "main")).count(), 0);
    }

    #[test]
    fn locally_defined_function_shadows_a_bare_sink_name_but_not_a_method_call() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "m.py", "def execute(q):\n    return q\n\ndef a(q):\n    execute(q)\n\ndef b(cur, q):\n    cur.execute(q)\n");
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        assert_eq!(g.sinks.len(), 1);
        assert_eq!(g.graph[g.sinks[0].function].name, "b");
    }

    #[test]
    fn c_repo_gets_buffer_overflow_sink() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "src/copy.c", "void f(char *d, char *s) {\n    strcpy(d, s);\n}\n");
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        assert_eq!(g.sinks.len(), 1);
        assert_eq!(g.sinks[0].cwe_id, "CWE-120");
    }

    #[test]
    fn ignored_dirs_oversized_files_and_symlinks_are_not_scanned() {
        let d = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        write(d.path(), "ok.py", "def ok():\n    os.system(x)\n");
        write(d.path(), "node_modules/dep/x.py", "def bad():\n    os.system(x)\n");
        write(d.path(), ".git/hooks/h.py", "def bad2():\n    os.system(x)\n");
        write(d.path(), "big.py", &format!("def big():\n    os.system(x)\n# {}\n", "x".repeat(5000)));
        write(outside.path(), "secret.py", "def leaked():\n    os.system(x)\n");
        std::os::unix::fs::symlink(outside.path().join("secret.py"), d.path().join("link.py")).unwrap();
        let limits = ScanLimits { max_file_bytes: 1000, ..ScanLimits::default() };
        let g = build_graph(d.path(), &CweKb::embedded(), &limits).unwrap();
        let names: Vec<&str> = g.graph.node_weights().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["ok"]);
        assert_eq!(g.stats.files_scanned, 1);
        assert!(g.stats.files_skipped >= 2, "{:?}", g.stats);
    }

    #[test]
    fn max_sinks_is_enforced_and_reported() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "m.py", "def f(c):\n    c.execute(1)\n    c.execute(2)\n    c.execute(3)\n");
        let limits = ScanLimits { max_sinks: 2, ..ScanLimits::default() };
        let g = build_graph(d.path(), &CweKb::embedded(), &limits).unwrap();
        assert_eq!(g.sinks.len(), 2);
        assert!(g.stats.sinks_truncated);
    }

    #[test]
    fn huge_functions_are_windowed_around_the_sink_line() {
        let d = tempfile::tempdir().unwrap();
        let mut src = String::from("def big(c):\n");
        for i in 0..300 {
            src.push_str(&format!("    value_{i} = {i}\n"));
        }
        src.push_str("    c.execute(value_1)\n");
        for i in 0..300 {
            src.push_str(&format!("    tail_{i} = {i}\n"));
        }
        write(d.path(), "big.py", &src);
        let g = build_graph(d.path(), &CweKb::embedded(), &ScanLimits::default()).unwrap();
        let s = g.slice(0, &SliceOptions { max_chars: 1500, ..SliceOptions::default() });
        let b = &s.blocks[0];
        assert!(s.truncated);
        assert!(b.code.contains("c.execute(value_1)"));
        assert!(b.code.len() <= 1500, "len={}", b.code.len());
        assert!(b.start_line > 1 && b.start_line <= 302 && b.end_line >= 302, "{}-{}", b.start_line, b.end_line);
    }
}
