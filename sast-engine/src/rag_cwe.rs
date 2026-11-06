//! Arricchimento di contesto: associa un sink alla scheda tecnica della CWE (knowledge base JSON in memoria).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

const EMBEDDED_KB: &str = include_str!("../cwe_kb.json");

#[derive(Debug, Clone, Deserialize)]
pub struct CweEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub default_severity: String,
    pub summary: String,
    pub indicators: Vec<String>,
    pub safe_patterns: Vec<String>,
    pub fix_hint: String,
    #[serde(default)]
    pub sinks: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CweKb {
    pub version: u32,
    pub entries: Vec<CweEntry>,
}

impl CweKb {
    /// KB incorporata nel binario (sast-engine/cwe_kb.json).
    pub fn embedded() -> Self {
        Self::from_json(EMBEDDED_KB).expect("cwe_kb.json incorporato non valido")
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        serde_json::from_str(json).context("KB CWE non valida")
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path).with_context(|| format!("lettura KB CWE {}", path.display()))?;
        Self::from_json(&json)
    }

    /// KB dal file indicato (`CWE_KB_PATH`) oppure, senza percorso, quella incorporata.
    pub fn load_or_embedded(path: Option<&Path>) -> anyhow::Result<Self> {
        match path {
            Some(path) => Self::load(path),
            None => Ok(Self::embedded()),
        }
    }

    pub fn get(&self, id: &str) -> Option<&CweEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Prima entry (nell'ordine del file) che dichiara un sink combaciante per il linguaggio.
    pub fn match_sink(&self, lang: &str, callee: &str) -> Option<&CweEntry> {
        let key = if lang == "cpp" { "c" } else { lang };
        self.entries.iter().find(|e| {
            e.sinks
                .get(key)
                .is_some_and(|patterns| patterns.iter().any(|p| callee_matches(p, callee)))
        })
    }

    /// Riporta una label di dataset (es. `CWE-787`) all'id primario della KB.
    pub fn resolve_alias(&self, label: &str) -> Option<&str> {
        let label = label.trim().to_uppercase();
        self.entries
            .iter()
            .find(|e| e.id == label || e.aliases.contains(&label))
            .map(|e| e.id.as_str())
    }
}

/// Toglie gli spazi e porta `::` e `->` al separatore `.`.
pub fn normalize_callee(callee: &str) -> String {
    let compact: String = callee.chars().filter(|c| !c.is_whitespace()).collect();
    compact.replace("::", ".").replace("->", ".")
}

/// `execute` combacia con `cursor.execute`; `os.system` solo con `os.system` (o `x.os.system`).
pub fn callee_matches(pattern: &str, callee: &str) -> bool {
    let c = normalize_callee(callee);
    let p = normalize_callee(pattern);
    c == p || c.ends_with(&format!(".{p}"))
}

/// Scheda testuale della CWE inserita nel prompt (formato identico a `common.render_card` in Python).
pub fn render_card(entry: &CweEntry) -> String {
    let mut lines = vec![
        format!("{} — {}", entry.id, entry.name),
        format!("Descrizione: {}", entry.summary),
        "Indicatori di vulnerabilità:".to_string(),
    ];
    lines.extend(entry.indicators.iter().map(|i| format!("- {i}")));
    lines.push("Pattern sicuri:".to_string());
    lines.extend(entry.safe_patterns.iter().map(|i| format!("- {i}")));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden() -> Value {
        serde_json::from_str(include_str!("../../shared/golden.json")).unwrap()
    }

    #[test]
    fn normalize_matches_golden() {
        for case in golden()["normalize"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            assert_eq!(normalize_callee(input), case["expected"].as_str().unwrap(), "input={input:?}");
        }
    }

    #[test]
    fn callee_matches_golden() {
        for case in golden()["callee_matches"].as_array().unwrap() {
            let (p, c) = (case["pattern"].as_str().unwrap(), case["callee"].as_str().unwrap());
            assert_eq!(callee_matches(p, c), case["expected"].as_bool().unwrap(), "{p} vs {c}");
        }
    }

    #[test]
    fn match_sink_golden_with_embedded_kb() {
        let kb = CweKb::embedded();
        for case in golden()["kb_match"].as_array().unwrap() {
            let (lang, callee) = (case["lang"].as_str().unwrap(), case["callee"].as_str().unwrap());
            let got = kb.match_sink(lang, callee).map(|e| e.id.as_str());
            assert_eq!(got, case["cwe"].as_str(), "{lang}:{callee}");
        }
    }

    #[test]
    fn resolve_alias_golden() {
        let kb = CweKb::embedded();
        for case in golden()["alias"].as_array().unwrap() {
            let label = case["label"].as_str().unwrap();
            assert_eq!(kb.resolve_alias(label), case["cwe"].as_str(), "{label}");
        }
    }

    #[test]
    fn card_matches_golden() {
        let g = golden();
        let entry: CweEntry = serde_json::from_value(g["card"]["entry"].clone()).unwrap();
        let expected: Vec<&str> = g["card"]["expected_lines"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(render_card(&entry), expected.join("\n"));
    }

    #[test]
    fn get_finds_entry_by_id() {
        let kb = CweKb::embedded();
        assert_eq!(kb.get("CWE-89").unwrap().name, "SQL Injection");
        assert!(kb.get("CWE-0").is_none());
    }

    #[test]
    fn from_json_rejects_garbage() {
        assert!(CweKb::from_json("{not json").is_err());
    }

    #[test]
    fn load_reads_a_file_and_reports_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("kb.json");
        std::fs::write(&p, EMBEDDED_KB).unwrap();
        assert!(!CweKb::load(&p).unwrap().entries.is_empty());
        assert!(CweKb::load(&dir.path().join("nope.json")).is_err());
    }

    #[test]
    fn load_or_embedded_uses_the_file_when_given_and_the_embedded_kb_otherwise() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("kb.json");
        std::fs::write(&p, r#"{"version": 7, "entries": []}"#).unwrap();
        assert_eq!(CweKb::load_or_embedded(Some(&p)).unwrap().version, 7);
        assert!(!CweKb::load_or_embedded(None).unwrap().entries.is_empty());
        assert!(CweKb::load_or_embedded(Some(&dir.path().join("nope.json"))).is_err());
    }
}
