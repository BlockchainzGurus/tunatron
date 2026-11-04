//! Acquisizione della repository (git clone sicuro) e SBOM CycloneDX di base.
//!
//! Sicurezza: l'URL arriva da una richiesta HTTP, quindi viene validato in modo rigido (https, host in
//! allowlist, `owner/repo`), `git` è invocato senza shell con `--` prima degli argomenti non fidati,
//! hook e protocolli pericolosi sono disattivati e il clone ha un timeout.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::process::Command;
use uuid::Uuid;
use walkdir::WalkDir;

use crate::graph_slicer::SKIP_DIRS;

const MAX_MANIFEST_BYTES: u64 = 2_000_000;
const MAX_MANIFEST_DEPTH: usize = 6;

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub allowed_hosts: Vec<String>,
    /// Solo per i test: consente URL `file://`.
    pub allow_file_urls: bool,
    pub clone_timeout: Duration,
    /// Dimensione massima del clone su disco: oltre, il clone viene interrotto e rimosso.
    pub max_repo_bytes: u64,
    /// Directory in cui creare `repo_<uuid>`.
    pub work_dir: PathBuf,
    /// Binario `syft` opzionale; se assente o in errore si usa il parser dei manifest.
    pub syft_bin: Option<String>,
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig {
            allowed_hosts: vec!["github.com".to_string()],
            allow_file_urls: false,
            clone_timeout: Duration::from_secs(120),
            max_repo_bytes: 500_000_000,
            work_dir: std::env::temp_dir(),
            syft_bin: None,
        }
    }
}

/// Repository clonata; la directory viene rimossa al drop.
#[derive(Debug)]
pub struct ClonedRepo {
    pub path: PathBuf,
    pub name: String,
}

impl Drop for ClonedRepo {
    fn drop(&mut self) {
        // Si cancellano solo le directory create da `clone_repo` (prefisso `repo_`).
        let ours = self.path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("repo_"));
        if ours {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn invalid(reason: &str) -> String {
    format!("URL non valido: {reason}")
}

fn valid_segment(seg: &str) -> bool {
    !seg.is_empty() && seg != "." && seg != ".." && seg.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Valida l'URL e restituisce il nome della repository (ultimo segmento senza `.git`).
pub fn validate_repo_url(url: &str, cfg: &IngestConfig) -> Result<String, String> {
    if url.is_empty() || url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(invalid("vuoto o con spazi/caratteri di controllo"));
    }
    if let Some(path) = url.strip_prefix("file://") {
        if !cfg.allow_file_urls {
            return Err(invalid("lo schema file:// non è ammesso"));
        }
        let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or("local");
        return Ok(name.to_string());
    }
    let rest = url.strip_prefix("https://").ok_or_else(|| invalid("è ammesso solo https"))?;
    let (host, path) = rest.split_once('/').ok_or_else(|| invalid("manca il percorso owner/repo"))?;
    if host.contains('@') || host.contains(':') {
        return Err(invalid("credenziali o porta non ammesse"));
    }
    if !cfg.allowed_hosts.iter().any(|h| h.eq_ignore_ascii_case(host)) {
        return Err(invalid("host non in allowlist"));
    }
    let segments: Vec<&str> = path.trim_end_matches('/').split('/').collect();
    if segments.len() != 2 || !segments.iter().all(|s| valid_segment(s)) {
        return Err(invalid("atteso https://<host>/<owner>/<repo>"));
    }
    let name = segments[1].strip_suffix(".git").unwrap_or(segments[1]);
    if name.is_empty() {
        return Err(invalid("nome repository vuoto"));
    }
    Ok(name.to_string())
}

pub async fn clone_repo(url: &str, cfg: &IngestConfig) -> anyhow::Result<ClonedRepo> {
    let name = validate_repo_url(url, cfg).map_err(|e| anyhow!(e))?;
    let dest = cfg.work_dir.join(format!("repo_{}", Uuid::new_v4()));
    let file_proto = if cfg.allow_file_urls { "protocol.file.allow=always" } else { "protocol.file.allow=never" };
    let child = Command::new("git")
        .args(["-c", "core.hooksPath=/dev/null", "-c", "core.symlinks=false", "-c", "protocol.ext.allow=never", "-c", file_proto])
        .args(["clone", "--depth", "1", "--no-tags", "--single-branch", "--quiet", "--"])
        .arg(url)
        .arg(&dest)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("impossibile avviare git (installato?)")?;

    let cleanup = || {
        let _ = std::fs::remove_dir_all(&dest);
    };
    let limit_mb = cfg.max_repo_bytes / 1_000_000;
    // Sorveglia la crescita del clone: un repository enorme (o ostile) non deve riempire il disco.
    let watch = async {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let d = dest.clone();
            if tokio::task::spawn_blocking(move || dir_size(&d)).await.unwrap_or(0) > cfg.max_repo_bytes {
                break;
            }
        }
    };
    let outcome = tokio::select! {
        r = tokio::time::timeout(cfg.clone_timeout, child.wait_with_output()) => Some(r),
        () = watch => None,
    };
    match outcome {
        None => {
            // La futura di `git` è stata scartata (kill_on_drop): breve attesa e seconda pulizia per l'ultima scrittura.
            cleanup();
            tokio::time::sleep(Duration::from_millis(150)).await;
            cleanup();
            bail!("repository oltre il limite di {limit_mb} MB: clone interrotto")
        }
        Some(Err(_)) => {
            cleanup();
            bail!("timeout di {}s durante il git clone", cfg.clone_timeout.as_secs())
        }
        Some(Ok(Err(e))) => {
            cleanup();
            Err(e).context("errore durante il git clone")
        }
        Some(Ok(Ok(out))) if !out.status.success() => {
            cleanup();
            bail!("git clone fallito: {}", String::from_utf8_lossy(&out.stderr).trim())
        }
        Some(Ok(Ok(_))) => {
            let d = dest.clone();
            if tokio::task::spawn_blocking(move || dir_size(&d)).await.unwrap_or(0) > cfg.max_repo_bytes {
                cleanup();
                bail!("repository oltre il limite di {limit_mb} MB");
            }
            Ok(ClonedRepo { path: dest, name })
        }
    }
}

/// Somma delle dimensioni dei file sotto `path` (i symlink non vengono seguiti).
fn dir_size(path: &Path) -> u64 {
    WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

// --------------------------------------------------------------------------- manifest -> componenti

/// Componente di dipendenza individuato in un manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub ecosystem: &'static str,
    pub name: String,
    pub version: Option<String>,
}

impl Dependency {
    pub fn purl(&self) -> String {
        let name = if self.ecosystem == "npm" && self.name.starts_with('@') { self.name.replacen('@', "%40", 1) } else { self.name.clone() };
        match &self.version {
            Some(v) => format!("pkg:{}/{}@{}", self.ecosystem, name, v),
            None => format!("pkg:{}/{}", self.ecosystem, name),
        }
    }
}

/// Versione "esatta" (es. `4.18.2`): un intervallo (`^1`, `>=2`, `latest`) non identifica una versione.
fn exact_version(v: &str) -> Option<String> {
    let v = v.trim();
    let ok = v.chars().next().is_some_and(|c| c.is_ascii_digit()) && v.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'));
    ok.then(|| v.to_string())
}

fn requirement_line(line: &str) -> Option<Dependency> {
    let line = line.split(';').next().unwrap_or("").trim();
    let name_end = line.find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))).unwrap_or(line.len());
    let name = &line[..name_end];
    if name.is_empty() {
        return None;
    }
    let mut rest = line[name_end..].trim_start();
    if rest.starts_with('[') {
        rest = rest.split_once(']').map_or("", |(_, r)| r).trim_start();
    }
    let version = rest.strip_prefix("==").and_then(|v| exact_version(v.split(',').next().unwrap_or("")));
    Some(Dependency { ecosystem: "pypi", name: name.to_lowercase(), version })
}

pub fn parse_requirements(content: &str) -> Vec<Dependency> {
    content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty() && !l.starts_with('-'))
        .filter_map(requirement_line)
        .collect()
}

pub fn parse_package_json(content: &str) -> Vec<Dependency> {
    let Ok(v) = serde_json::from_str::<Value>(content) else { return Vec::new() };
    let mut deps = Vec::new();
    for key in ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"] {
        if let Some(obj) = v.get(key).and_then(Value::as_object) {
            for (name, ver) in obj {
                deps.push(Dependency { ecosystem: "npm", name: name.clone(), version: ver.as_str().and_then(exact_version) });
            }
        }
    }
    deps.sort_by(|a, b| a.name.cmp(&b.name));
    deps
}

pub fn parse_cargo_lock(content: &str) -> Vec<Dependency> {
    let Ok(table) = content.parse::<toml::Table>() else { return Vec::new() };
    table
        .get("package")
        .and_then(|p| p.as_array())
        .into_iter()
        .flatten()
        .filter_map(|p| {
            Some(Dependency {
                ecosystem: "cargo",
                name: p.get("name")?.as_str()?.to_string(),
                version: p.get("version").and_then(|v| v.as_str()).map(String::from),
            })
        })
        .collect()
}

pub fn parse_go_mod(content: &str) -> Vec<Dependency> {
    fn module_line(line: &str) -> Option<Dependency> {
        let mut parts = line.split_whitespace();
        Some(Dependency { ecosystem: "golang", name: parts.next()?.to_string(), version: Some(parts.next()?.to_string()) })
    }
    let mut deps = Vec::new();
    let mut in_block = false;
    for raw in content.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if in_block {
            if line == ")" {
                in_block = false;
            } else if let Some(d) = module_line(line) {
                deps.push(d);
            }
        } else if line == "require (" {
            in_block = true;
        } else if let Some(rest) = line.strip_prefix("require ") {
            deps.extend(module_line(rest));
        }
    }
    deps
}

pub fn parse_pyproject(content: &str) -> Vec<Dependency> {
    let Ok(table) = content.parse::<toml::Table>() else { return Vec::new() };
    let mut deps = Vec::new();
    if let Some(list) = table.get("project").and_then(|p| p.get("dependencies")).and_then(|d| d.as_array()) {
        deps.extend(list.iter().filter_map(|d| d.as_str()).filter_map(requirement_line));
    }
    if let Some(poetry) = table.get("tool").and_then(|t| t.get("poetry")).and_then(|p| p.get("dependencies")).and_then(|d| d.as_table()) {
        for (name, spec) in poetry.iter().filter(|(n, _)| n.as_str() != "python") {
            let version = match spec {
                toml::Value::String(s) => exact_version(s),
                toml::Value::Table(t) => t.get("version").and_then(|v| v.as_str()).and_then(exact_version),
                _ => None,
            };
            deps.push(Dependency { ecosystem: "pypi", name: name.to_lowercase(), version });
        }
    }
    deps
}

fn collect_dependencies(root: &Path) -> BTreeMap<String, Dependency> {
    let mut found: BTreeMap<String, Dependency> = BTreeMap::new();
    let walker = WalkDir::new(root)
        .follow_links(false)
        .max_depth(MAX_MANIFEST_DEPTH)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| !(e.depth() > 0 && e.file_type().is_dir() && SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref())));
    for entry in walker.flatten() {
        if !entry.file_type().is_file() || entry.metadata().map_or(true, |m| m.len() > MAX_MANIFEST_BYTES) {
            continue;
        }
        let fname = entry.file_name().to_string_lossy().to_string();
        let parse: fn(&str) -> Vec<Dependency> = match fname.as_str() {
            "pyproject.toml" => parse_pyproject,
            "package.json" => parse_package_json,
            "Cargo.lock" => parse_cargo_lock,
            "go.mod" => parse_go_mod,
            n if n.starts_with("requirements") && n.ends_with(".txt") => parse_requirements,
            _ => continue,
        };
        let Ok(content) = std::fs::read_to_string(entry.path()) else { continue };
        for dep in parse(&content) {
            found.entry(dep.purl()).or_insert(dep);
        }
    }
    found
}

fn sbom_from_manifests(root: &Path) -> Value {
    let components: Vec<Value> = collect_dependencies(root)
        .into_iter()
        .map(|(purl, dep)| {
            let mut c = json!({"type": "library", "bom-ref": purl, "name": dep.name, "purl": purl});
            if let Some(v) = dep.version {
                c["version"] = Value::String(v);
            }
            c
        })
        .collect();
    json!({
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": format!("urn:uuid:{}", Uuid::new_v4()),
        "version": 1,
        "components": components,
    })
}

async fn run_syft(bin: &str, root: &Path, timeout: Duration) -> Option<Value> {
    let child = Command::new(bin)
        .arg(format!("dir:{}", root.display()))
        .args(["-o", "cyclonedx-json", "-q"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .output();
    let out = tokio::time::timeout(timeout, child).await.ok()?.ok()?;
    if !out.status.success() {
        return None;
    }
    let bom: Value = serde_json::from_slice(&out.stdout).ok()?;
    (bom["bomFormat"] == "CycloneDX").then_some(bom)
}

/// Garantisce i campi che il resto della pipeline si aspetta, qualunque sia l'origine del BOM.
fn normalize_bom(bom: &mut Value, repo_name: &str) {
    let timestamp = OffsetDateTime::now_utc().format(&Rfc3339).unwrap_or_default();
    if bom.get("serialNumber").is_none() {
        bom["serialNumber"] = Value::String(format!("urn:uuid:{}", Uuid::new_v4()));
    }
    if !bom["metadata"].is_object() {
        bom["metadata"] = json!({});
    }
    bom["metadata"]["timestamp"] = Value::String(timestamp);
    bom["metadata"]["tools"] = json!({"components": [{"type": "application", "name": "sast-engine", "version": env!("CARGO_PKG_VERSION")}]});
    bom["metadata"]["component"] = json!({"type": "application", "bom-ref": "root", "name": repo_name});
    if !bom["components"].is_array() {
        bom["components"] = json!([]);
    }
    if !bom["vulnerabilities"].is_array() {
        bom["vulnerabilities"] = json!([]);
    }
}

/// SBOM CycloneDX 1.5 (con `syft` se configurato e funzionante, altrimenti dai manifest).
pub async fn generate_sbom(repo_root: &Path, repo_name: &str, cfg: &IngestConfig) -> Value {
    let mut bom = match &cfg.syft_bin {
        Some(bin) => run_syft(bin, repo_root, cfg.clone_timeout).await,
        None => None,
    };
    if bom.is_none() {
        let root = repo_root.to_path_buf();
        bom = tokio::task::spawn_blocking(move || sbom_from_manifests(&root)).await.ok();
    }
    let mut bom = bom.unwrap_or_else(|| json!({"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1}));
    normalize_bom(&mut bom, repo_name);
    bom
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    fn cfg() -> IngestConfig {
        IngestConfig::default()
    }

    #[test]
    fn accepts_plain_github_urls_and_returns_the_repo_name() {
        assert_eq!(validate_repo_url("https://github.com/psf/requests", &cfg()).unwrap(), "requests");
        assert_eq!(validate_repo_url("https://github.com/psf/requests.git", &cfg()).unwrap(), "requests");
        assert_eq!(validate_repo_url("https://GitHub.com/psf/requests/", &cfg()).unwrap(), "requests");
    }

    #[test]
    fn rejects_dangerous_or_unexpected_urls() {
        let bad = [
            "http://github.com/o/r",
            "git@github.com:o/r.git",
            "ssh://git@github.com/o/r",
            "https://github.com.evil.com/o/r",
            "https://evil.com/o/r",
            "https://user:pw@github.com/o/r",
            "https://github.com:8443/o/r",
            "https://github.com/o",
            "https://github.com/o/r/extra",
            "https://github.com/o/../x",
            "https://github.com/o/r with space",
            "https://github.com/o/r\nx",
            "--upload-pack=touch /tmp/pwn",
            "ext::sh -c 'touch /tmp/pwn'",
            "file:///etc",
            "",
        ];
        for url in bad {
            assert!(validate_repo_url(url, &cfg()).is_err(), "doveva essere rifiutato: {url:?}");
        }
    }

    #[test]
    fn file_urls_need_the_explicit_test_flag() {
        let on = IngestConfig { allow_file_urls: true, ..cfg() };
        assert!(validate_repo_url("file:///tmp/some_repo", &on).is_ok());
        assert!(validate_repo_url("file:///tmp/some_repo", &cfg()).is_err());
    }

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git").current_dir(dir).args(args).output().expect("git non disponibile");
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }

    #[tokio::test]
    async fn clones_a_local_repository_and_cleans_up_on_drop() {
        let src = tempfile::tempdir().unwrap();
        git(src.path(), &["init", "-q", "-b", "main"]);
        fs::write(src.path().join("app.py"), "def f():\n    pass\n").unwrap();
        git(src.path(), &["add", "."]);
        git(src.path(), &["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"]);

        let work = tempfile::tempdir().unwrap();
        let c = IngestConfig { allow_file_urls: true, work_dir: work.path().to_path_buf(), ..cfg() };
        let url = format!("file://{}", src.path().display());
        let repo = clone_repo(&url, &c).await.unwrap();
        let path = repo.path.clone();
        assert!(path.join("app.py").is_file());
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("repo_"));
        drop(repo);
        assert!(!path.exists(), "la directory clonata deve sparire al drop");
    }

    #[tokio::test]
    async fn repository_larger_than_the_limit_is_rejected_and_removed() {
        let src = tempfile::tempdir().unwrap();
        git(src.path(), &["init", "-q", "-b", "main"]);
        // dati poco comprimibili (LCG) così la dimensione su disco è davvero ~2 MB
        let mut x: u32 = 12345;
        let blob: Vec<u8> = (0..2_000_000).map(|_| { x = x.wrapping_mul(1664525).wrapping_add(1013904223); (x >> 24) as u8 }).collect();
        fs::write(src.path().join("big.bin"), blob).unwrap();
        git(src.path(), &["add", "."]);
        git(src.path(), &["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "big"]);

        let work = tempfile::tempdir().unwrap();
        let c = IngestConfig { allow_file_urls: true, work_dir: work.path().to_path_buf(), max_repo_bytes: 1_000_000, ..cfg() };
        let err = clone_repo(&format!("file://{}", src.path().display()), &c).await.unwrap_err();
        assert!(err.to_string().contains("limite"), "{err}");
        assert_eq!(fs::read_dir(work.path()).unwrap().count(), 0, "il clone troppo grande va rimosso");
    }

    #[tokio::test]
    async fn clone_failure_is_an_error_and_leaves_nothing_behind() {
        let work = tempfile::tempdir().unwrap();
        let c = IngestConfig { allow_file_urls: true, work_dir: work.path().to_path_buf(), ..cfg() };
        let r = clone_repo("file:///definitely/not/a/repo", &c).await;
        assert!(r.is_err());
        assert_eq!(fs::read_dir(work.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn invalid_url_never_reaches_git() {
        let r = clone_repo("--upload-pack=evil", &cfg()).await;
        assert!(r.unwrap_err().to_string().contains("URL"));
    }

    #[test]
    fn purl_formats_per_ecosystem() {
        let d = |e, n: &str, v: Option<&str>| Dependency { ecosystem: e, name: n.into(), version: v.map(Into::into) };
        assert_eq!(d("pypi", "flask", Some("2.0.1")).purl(), "pkg:pypi/flask@2.0.1");
        assert_eq!(d("pypi", "requests", None).purl(), "pkg:pypi/requests");
        assert_eq!(d("golang", "github.com/a/b", Some("v1.2.3")).purl(), "pkg:golang/github.com/a/b@v1.2.3");
        assert_eq!(d("npm", "@scope/pkg", Some("1.0.0")).purl(), "pkg:npm/%40scope/pkg@1.0.0");
    }

    #[test]
    fn requirements_pinned_ranges_extras_comments_and_includes() {
        let deps = parse_requirements("# comment\nflask==2.0.1\nRequests>=2.28\nnumpy[extra]==1.2 ; python_version > '3'\n-r other.txt\n--index-url https://x\n\n  django~=4.2  # inline\n");
        let got: Vec<(String, Option<String>)> = deps.iter().map(|d| (d.name.clone(), d.version.clone())).collect();
        assert_eq!(
            got,
            vec![
                ("flask".to_string(), Some("2.0.1".to_string())),
                ("requests".to_string(), None),
                ("numpy".to_string(), Some("1.2".to_string())),
                ("django".to_string(), None),
            ]
        );
        assert!(deps.iter().all(|d| d.ecosystem == "pypi"));
    }

    #[test]
    fn package_json_collects_dependencies_and_dev_dependencies() {
        let deps = parse_package_json(r#"{"dependencies":{"express":"4.18.2","lodash":"^4.17.0","left-pad":"latest"},"devDependencies":{"jest":"~29.0.0"}}"#);
        let got: Vec<(String, Option<String>)> = deps.iter().map(|d| (d.name.clone(), d.version.clone())).collect();
        assert_eq!(
            got,
            vec![
                ("express".to_string(), Some("4.18.2".to_string())),
                ("jest".to_string(), None),
                ("left-pad".to_string(), None),
                ("lodash".to_string(), None),
            ]
        );
        assert!(parse_package_json("{broken").is_empty());
    }

    #[test]
    fn cargo_lock_packages_have_exact_versions() {
        let deps = parse_cargo_lock("version = 4\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.200\"\n\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n");
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0].purl(), "pkg:cargo/serde@1.0.200");
    }

    #[test]
    fn go_mod_parses_single_and_block_requires() {
        let deps = parse_go_mod("module x\n\ngo 1.22\n\nrequire github.com/a/b v1.2.3\n\nrequire (\n\tgithub.com/c/d v0.4.0 // indirect\n\tgolang.org/x/net v0.20.0\n)\n");
        let got: Vec<String> = deps.iter().map(|d| d.purl()).collect();
        assert_eq!(got, vec!["pkg:golang/github.com/a/b@v1.2.3", "pkg:golang/github.com/c/d@v0.4.0", "pkg:golang/golang.org/x/net@v0.20.0"]);
    }

    #[test]
    fn pyproject_reads_pep621_and_poetry_dependencies() {
        let deps = parse_pyproject("[project]\nname=\"x\"\ndependencies=[\"flask==2.0.1\",\"requests>=2\"]\n\n[tool.poetry.dependencies]\npython=\"^3.11\"\nfastapi=\"0.110.0\"\nuvicorn={version=\"^0.29\", extras=[\"standard\"]}\n");
        let names: Vec<&str> = deps.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["flask", "requests", "fastapi", "uvicorn"]);
        assert_eq!(deps[2].version.as_deref(), Some("0.110.0"));
    }

    #[tokio::test]
    async fn sbom_from_manifests_is_cyclonedx_with_root_component() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("requirements.txt"), "flask==2.0.1\n").unwrap();
        fs::create_dir_all(d.path().join("web")).unwrap();
        fs::write(d.path().join("web/package.json"), r#"{"dependencies":{"express":"4.18.2"}}"#).unwrap();
        fs::create_dir_all(d.path().join("node_modules/x")).unwrap();
        fs::write(d.path().join("node_modules/x/package.json"), r#"{"dependencies":{"ignored":"1.0.0"}}"#).unwrap();

        let bom = generate_sbom(d.path(), "demo", &cfg()).await;
        assert_eq!(bom["bomFormat"], "CycloneDX");
        assert_eq!(bom["specVersion"], "1.5");
        assert!(bom["serialNumber"].as_str().unwrap().starts_with("urn:uuid:"));
        assert_eq!(bom["metadata"]["component"]["bom-ref"], "root");
        assert_eq!(bom["metadata"]["component"]["name"], "demo");
        let refs: Vec<&str> = bom["components"].as_array().unwrap().iter().map(|c| c["purl"].as_str().unwrap()).collect();
        assert_eq!(refs, vec!["pkg:npm/express@4.18.2", "pkg:pypi/flask@2.0.1"]);
        assert_eq!(bom["vulnerabilities"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn missing_syft_falls_back_to_manifests() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("requirements.txt"), "flask==2.0.1\n").unwrap();
        let c = IngestConfig { syft_bin: Some("/nonexistent/syft".into()), ..cfg() };
        let bom = generate_sbom(d.path(), "demo", &c).await;
        assert_eq!(bom["components"].as_array().unwrap().len(), 1);
    }

    /// Un file appena scritto può dare ETXTBSY ("Text file busy", errno 26) se un altro thread del processo di test
    /// sta facendo fork nello stesso istante (race nota): si attende che diventi eseguibile prima di usarlo.
    fn wait_until_executable(path: &Path) {
        for _ in 0..200 {
            match Command::new(path).output() {
                Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(Duration::from_millis(10)),
                _ => return,
            }
        }
    }

    #[tokio::test]
    async fn syft_output_is_used_and_normalized() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let tools = tempfile::tempdir().unwrap();
        let script = tools.path().join("syft");
        fs::write(&script, "#!/bin/sh\necho '{\"bomFormat\":\"CycloneDX\",\"specVersion\":\"1.5\",\"components\":[{\"type\":\"library\",\"name\":\"from-syft\"}]}'\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        wait_until_executable(&script);
        let c = IngestConfig { syft_bin: Some(script.to_string_lossy().into()), ..cfg() };
        let bom = generate_sbom(d.path(), "demo", &c).await;
        assert_eq!(bom["components"][0]["name"], "from-syft");
        assert_eq!(bom["metadata"]["component"]["bom-ref"], "root");
        assert_eq!(bom["vulnerabilities"], serde_json::json!([]));
    }
}
