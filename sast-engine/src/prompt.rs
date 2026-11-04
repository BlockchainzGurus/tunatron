//! Costruzione del prompt `Regola CWE + Traccia Codice`.
//!
//! Il template vive in `shared/prompt_template.json` ed è lo stesso usato dalla pipeline di training
//! (`ai-pipeline/common.py`): i golden in `shared/golden.json` garantiscono output identico.

use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Template {
    system: String,
    user: String,
    block_header: String,
}

fn template() -> &'static Template {
    static T: OnceLock<Template> = OnceLock::new();
    T.get_or_init(|| {
        serde_json::from_str(include_str!("../../shared/prompt_template.json")).expect("prompt_template.json non valido")
    })
}

/// Posizione del sink candidato.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkLocation {
    pub function: String,
    pub file: String,
    pub line: u32,
}

/// Una funzione della traccia (backward slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBlock {
    pub file: String,
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub depth: u32,
    pub lang: String,
    pub code: String,
}

/// Sostituzione `{nome}` a passata singola: il testo inserito non viene mai riscansionato,
/// quindi un `{cwe_id}` dentro il codice analizzato resta letterale.
fn fill(template: &str, values: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(template.len() + 256);
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let name_len = after.bytes().take_while(|b| b.is_ascii_lowercase() || *b == b'_').count();
        if name_len > 0 && after[name_len..].starts_with('}')
            && let Some((_, value)) = values.iter().find(|(key, _)| *key == &after[..name_len]) {
                out.push_str(value);
                rest = &after[name_len + 1..];
                continue;
            }
        out.push('{');
        rest = after;
    }
    out.push_str(rest);
    out
}

pub fn system_prompt() -> String {
    template().system.clone()
}

pub fn render_block(block: &TraceBlock) -> String {
    let header = fill(
        &template().block_header,
        &[
            ("file", block.file.clone()),
            ("name", block.name.clone()),
            ("start_line", block.start_line.to_string()),
            ("end_line", block.end_line.to_string()),
            ("depth", block.depth.to_string()),
        ],
    );
    let code = block.code.replace("\r\n", "\n");
    let numbered = code
        .trim_end_matches('\n')
        .split('\n')
        .enumerate()
        .map(|(i, line)| format!("{:>5} | {}", u64::from(block.start_line) + i as u64, line))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{header}\n```{}\n{numbered}\n```", block.lang)
}

pub fn render_user_prompt(cwe_id: &str, cwe_card: &str, sink: &SinkLocation, blocks: &[TraceBlock]) -> String {
    let trace = blocks.iter().map(render_block).collect::<Vec<_>>().join("\n\n");
    fill(
        &template().user,
        &[
            ("cwe_card", cwe_card.to_string()),
            ("cwe_id", cwe_id.to_string()),
            ("sink_function", sink.function.clone()),
            ("sink_file", sink.file.clone()),
            ("sink_line", sink.line.to_string()),
            ("trace", trace),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden() -> Value {
        serde_json::from_str(include_str!("../../shared/golden.json")).unwrap()
    }

    fn lines(v: &Value) -> String {
        v.as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect::<Vec<_>>().join("\n")
    }

    fn block_from(v: &Value) -> TraceBlock {
        TraceBlock {
            file: v["file"].as_str().unwrap().into(),
            name: v["name"].as_str().unwrap().into(),
            start_line: v["start_line"].as_u64().unwrap() as u32,
            end_line: v["end_line"].as_u64().unwrap() as u32,
            depth: v["depth"].as_u64().unwrap() as u32,
            lang: v["lang"].as_str().unwrap().into(),
            code: match v.get("code_lines") {
                Some(l) => lines(l),
                None => v["code_crlf"].as_str().unwrap().into(),
            },
        }
    }

    #[test]
    fn user_prompt_matches_every_golden_case() {
        for case in golden()["prompts"].as_array().unwrap() {
            let sink = SinkLocation {
                function: case["sink"]["function"].as_str().unwrap().into(),
                file: case["sink"]["file"].as_str().unwrap().into(),
                line: case["sink"]["line"].as_u64().unwrap() as u32,
            };
            let blocks: Vec<TraceBlock> = case["blocks"].as_array().unwrap().iter().map(block_from).collect();
            let got = render_user_prompt(
                case["cwe_id"].as_str().unwrap(),
                &lines(&case["cwe_card_lines"]),
                &sink,
                &blocks,
            );
            assert_eq!(got, lines(&case["expected_user_lines"]), "caso: {}", case["name"]);
        }
    }

    #[test]
    fn system_prompt_comes_from_the_shared_template() {
        let shared: Value = serde_json::from_str(include_str!("../../shared/prompt_template.json")).unwrap();
        assert_eq!(system_prompt(), shared["system"].as_str().unwrap());
    }

    #[test]
    fn empty_code_still_renders_a_numbered_line() {
        let b = TraceBlock {
            file: "a.c".into(),
            name: "f".into(),
            start_line: 7,
            end_line: 7,
            depth: 0,
            lang: "c".into(),
            code: String::new(),
        };
        assert_eq!(render_block(&b), "### a.c · f (righe 7-7, profondità 0)\n```c\n    7 | \n```");
    }
}
