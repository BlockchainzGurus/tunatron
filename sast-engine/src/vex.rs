//! Verdetti del modello e report finale: SBOM CycloneDX arricchito con le `vulnerabilities` (VEX).
//!
//! I campi di localizzazione (`sink_function/file/line`) sono noti con certezza al motore e
//! sovrascrivono quanto scritto dal modello; il modello decide solo vulnerabilità, CWE, severità e testo.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::notarizer::Attestation;
use crate::prompt::SinkLocation;

pub const SEVERITIES: [&str; 5] = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

/// Output del modello (stesso schema di `shared/vex_schema.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Finding {
    pub vulnerable: bool,
    pub cwe_id: String,
    pub severity: String,
    pub sink_function: String,
    pub sink_file: String,
    pub sink_line: i64,
    pub vex_status: String,
    pub reasoning: String,
    pub suggested_fix: String,
}

fn cwe_number(id: &str) -> Option<u32> {
    id.strip_prefix("CWE-")?.parse().ok()
}

impl Finding {
    /// Coerenza interna del verdetto; i risultati incoerenti vengono scartati, non "aggiustati".
    pub fn check(&self) -> Result<(), String> {
        if !SEVERITIES.contains(&self.severity.as_str()) {
            return Err(format!("severity non valida: {}", self.severity));
        }
        if cwe_number(&self.cwe_id).is_none() {
            return Err(format!("cwe_id non valido: {}", self.cwe_id));
        }
        let expected = if self.vulnerable { "affected" } else { "not_affected" };
        if self.vex_status != expected {
            return Err(format!("vex_status incoerente con vulnerable={}: atteso {expected}, trovato {}", self.vulnerable, self.vex_status));
        }
        if self.vulnerable == (self.severity == "NONE") {
            return Err(format!("severity {} incoerente con vulnerable={}", self.severity, self.vulnerable));
        }
        Ok(())
    }
}

/// Esito dell'analisi di un singolo sink.
#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub sink: SinkLocation,
    /// CWE proposta dal motore in base al sink.
    pub candidate_cwe: String,
    pub outcome: Result<Finding, String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct Summary {
    pub model: String,
    pub files_scanned: usize,
    pub functions: usize,
    pub parse_errors: usize,
    pub sinks_found: usize,
    pub sinks_truncated: bool,
    pub affected: usize,
    pub not_affected: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorRecord {
    pub sink_function: String,
    pub sink_file: String,
    pub sink_line: u32,
    pub cwe_id: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub summary: Summary,
    /// SBOM CycloneDX 1.5 con `vulnerabilities` (VEX).
    pub bom: Value,
    pub errors: Vec<ErrorRecord>,
    /// Presente solo se è stata richiesta e riuscita la notarizzazione (hash calcolato sul report senza questi due campi).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation: Option<Attestation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_error: Option<String>,
}

fn vulnerability(item: &AnalysisItem, f: &Finding, root_ref: &str) -> Value {
    let digest = Sha256::digest(format!("{}|{}|{}|{}", f.cwe_id, item.sink.file, item.sink.line, item.sink.function));
    let id = format!("SAST-{}-{}", f.cwe_id, &hex::encode(digest)[..8]);
    let state = if f.vulnerable { "exploitable" } else { "not_affected" };
    json!({
        "bom-ref": id,
        "id": id,
        "source": {"name": "sast-engine"},
        "cwes": [cwe_number(&f.cwe_id)],
        "description": f.reasoning,
        "recommendation": f.suggested_fix,
        "ratings": [{"source": {"name": "sast-engine"}, "severity": f.severity.to_lowercase(), "method": "other"}],
        "analysis": {"state": state, "detail": f.reasoning},
        "affects": [{"ref": root_ref}],
        "properties": [
            {"name": "vex:status", "value": f.vex_status},
            {"name": "sast:sink_function", "value": item.sink.function},
            {"name": "sast:sink_file", "value": item.sink.file},
            {"name": "sast:sink_line", "value": item.sink.line.to_string()},
        ],
    })
}

/// Aggiunge le `vulnerabilities` al BOM e calcola il riepilogo. `summary` arriva con i contatori di scansione già valorizzati.
pub fn build_report(mut bom: Value, items: Vec<AnalysisItem>, mut summary: Summary) -> Report {
    let root_ref = bom["metadata"]["component"]["bom-ref"].as_str().unwrap_or("root").to_string();
    let mut vulns: Vec<Value> = Vec::new();
    let mut errors = Vec::new();
    summary.affected = 0;
    summary.not_affected = 0;
    for item in &items {
        match item.outcome.clone().and_then(|f| f.check().map(|()| f)) {
            Ok(f) => {
                if f.vulnerable {
                    summary.affected += 1;
                } else {
                    summary.not_affected += 1;
                }
                vulns.push(vulnerability(item, &f, &root_ref));
            }
            Err(error) => errors.push(ErrorRecord {
                sink_function: item.sink.function.clone(),
                sink_file: item.sink.file.clone(),
                sink_line: item.sink.line,
                cwe_id: item.candidate_cwe.clone(),
                error,
            }),
        }
    }
    summary.errors = errors.len();
    match bom.get_mut("vulnerabilities").and_then(Value::as_array_mut) {
        Some(existing) => existing.extend(vulns),
        None => bom["vulnerabilities"] = Value::Array(vulns),
    }
    Report { summary, bom, errors, attestation: None, attestation_error: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(vulnerable: bool) -> Finding {
        Finding {
            vulnerable,
            cwe_id: "CWE-89".into(),
            severity: if vulnerable { "HIGH" } else { "NONE" }.into(),
            sink_function: "model_says_this".into(),
            sink_file: "model/says/this.py".into(),
            sink_line: 999,
            vex_status: if vulnerable { "affected" } else { "not_affected" }.into(),
            reasoning: "motivazione".into(),
            suggested_fix: "correzione".into(),
        }
    }

    fn loc(line: u32) -> SinkLocation {
        SinkLocation { function: "cursor.execute".into(), file: "src/db.py".into(), line }
    }

    fn bom() -> Value {
        json!({"bomFormat": "CycloneDX", "specVersion": "1.5", "metadata": {"component": {"bom-ref": "root", "name": "demo"}}, "components": []})
    }

    #[test]
    fn consistent_findings_pass_the_check() {
        assert!(finding(true).check().is_ok());
        assert!(finding(false).check().is_ok());
    }

    #[test]
    fn inconsistent_findings_are_rejected() {
        let mut bad_status = finding(true);
        bad_status.vex_status = "not_affected".into();
        assert!(bad_status.check().unwrap_err().contains("vex_status"));

        let mut bad_sev = finding(true);
        bad_sev.severity = "SEVERE".into();
        assert!(bad_sev.check().unwrap_err().contains("severity"));

        let mut none_but_vuln = finding(true);
        none_but_vuln.severity = "NONE".into();
        assert!(none_but_vuln.check().is_err());

        let mut bad_cwe = finding(true);
        bad_cwe.cwe_id = "89".into();
        assert!(bad_cwe.check().unwrap_err().contains("cwe_id"));
    }

    #[test]
    fn report_maps_findings_to_cyclonedx_vex() {
        let items = vec![
            AnalysisItem { sink: loc(45), candidate_cwe: "CWE-89".into(), outcome: Ok(finding(true)) },
            AnalysisItem { sink: loc(60), candidate_cwe: "CWE-89".into(), outcome: Ok(finding(false)) },
            AnalysisItem { sink: loc(70), candidate_cwe: "CWE-89".into(), outcome: Err("JSON non valido".into()) },
        ];
        let r = build_report(bom(), items, Summary { model: "m".into(), sinks_found: 3, ..Summary::default() });

        assert_eq!((r.summary.affected, r.summary.not_affected, r.summary.errors), (1, 1, 1));
        let vulns = r.bom["vulnerabilities"].as_array().unwrap();
        assert_eq!(vulns.len(), 2);

        let v = &vulns[0];
        assert_eq!(v["analysis"]["state"], "exploitable");
        assert_eq!(v["cwes"], json!([89]));
        assert_eq!(v["ratings"][0]["severity"], "high");
        assert_eq!(v["affects"][0]["ref"], "root");
        assert_eq!(v["description"], "motivazione");
        assert_eq!(v["recommendation"], "correzione");
        let props: Vec<(&str, &str)> = v["properties"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p["name"].as_str().unwrap(), p["value"].as_str().unwrap()))
            .collect();
        assert!(props.contains(&("vex:status", "affected")));
        // La posizione è quella nota al motore, non quella inventata dal modello.
        assert!(props.contains(&("sast:sink_line", "45")));
        assert!(props.contains(&("sast:sink_file", "src/db.py")));
        assert!(props.contains(&("sast:sink_function", "cursor.execute")));

        assert_eq!(vulns[1]["analysis"]["state"], "not_affected");
        assert_eq!(vulns[1]["ratings"][0]["severity"], "none");

        assert_eq!(r.errors, vec![ErrorRecord {
            sink_function: "cursor.execute".into(),
            sink_file: "src/db.py".into(),
            sink_line: 70,
            cwe_id: "CWE-89".into(),
            error: "JSON non valido".into(),
        }]);
    }

    #[test]
    fn vulnerability_ids_are_stable_and_unique_per_sink() {
        let mk = |line| AnalysisItem { sink: loc(line), candidate_cwe: "CWE-89".into(), outcome: Ok(finding(true)) };
        let a = build_report(bom(), vec![mk(1), mk(2)], Summary::default());
        let b = build_report(bom(), vec![mk(1), mk(2)], Summary::default());
        let id = |r: &Report, i: usize| r.bom["vulnerabilities"][i]["id"].as_str().unwrap().to_string();
        assert_eq!(id(&a, 0), id(&b, 0));
        assert_ne!(id(&a, 0), id(&a, 1));
        assert!(id(&a, 0).starts_with("SAST-CWE-89-"));
    }

    #[test]
    fn inconsistent_model_output_is_counted_as_an_error_not_a_finding() {
        let mut bad = finding(true);
        bad.vex_status = "not_affected".into();
        let items = vec![AnalysisItem { sink: loc(5), candidate_cwe: "CWE-89".into(), outcome: Ok(bad) }];
        let r = build_report(bom(), items, Summary::default());
        assert_eq!(r.summary.errors, 1);
        assert_eq!(r.bom["vulnerabilities"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn well_formed_model_cwe_wins_over_the_engine_candidate() {
        // CWE del modello diversa dal candidato ma ben formata: vince il modello.
        let mut other = finding(true);
        other.cwe_id = "CWE-943".into();
        let items = vec![AnalysisItem { sink: loc(5), candidate_cwe: "CWE-89".into(), outcome: Ok(other) }];
        let r = build_report(bom(), items, Summary::default());
        assert_eq!(r.bom["vulnerabilities"][0]["cwes"], json!([943]));
    }
}
