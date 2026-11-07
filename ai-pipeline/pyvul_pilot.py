"""Pilota P2: coppie prima/dopo la fix di CVEfixes/PyVul con i chiamanti veri, dal checkout dei commit.

Il parquet ha `repo_url`, `commit_hash`, `function_name`, `code_before`, `code_after` ma nessun percorso di file. Per ogni
commit di un repo: un solo `git fetch --depth 2` del commit di fix (porta con sé il genitore), checkout del genitore, ricerca
della funzione per nome e corpo, slice del sink toccato dalla fix (`localize.localize_sink`); poi checkout del commit di fix
e slice della stessa funzione. Vulnerabile = genitore, corretto = commit di fix (R32: la divisione è per COMMIT con la chiave
di `prepare_data`, così i commit del test restano solo in `py_cvefixes.jsonl` e la funzione non si addestra mai due volte).

I repo scaricati sono dati non fidati: i file si leggono come testo (nessuna importazione né esecuzione), i link simbolici e
`.git` si saltano, git gira senza hook, senza prompt e senza la configurazione dell'utente (le sue `insteadOf` possono
dirottare l'URL); solo `https://github.com/…` e un SHA di 40 esadecimali, perché un valore del dataset non diventi un'opzione.

Uso (pilota su 20 repo; sotto un tetto di memoria, il motore costruisce il grafo di tutto il checkout):
  systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \\
    python3 pyvul_pilot.py --parquet data/raw/cvefixes_pyvul.parquet --repos 20 \\
      --work-dir outputs/pyvul-pilot --out-dir data/graph --eval-dir data/eval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import codeql_graph
import common
import graph_examples
import leak_audit
import localize
import prepare_data
import slice_io
import sources

SOURCE = "pyvul-graph"
DEPTH = 5
SHA_RE = re.compile(r"[0-9a-f]{40}")
REV_RE = re.compile(r"[0-9a-f]{40}\^?")
URL_RE = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+")
CWE_RE = re.compile(r"CWE-\d+")
DEF_RE = re.compile(r"\s*(?:async\s+)?def\s")
MAX_FILE_BYTES = 1_000_000
ABANDON_AFTER = 3  # fetch falliti di fila dopo i quali un repo si lascia (privato, cancellato)


# --------------------------------------------------------------------------- scelta delle righe e dei repo
def usable_rows(rows: list[dict], kb: dict) -> list[dict]:
    """Righe Python con prima e dopo diversi, un commit intero e un sink del catalogo toccato dalla fix."""
    out = []
    for r in rows:
        before, after = r.get("code_before"), r.get("code_after")
        if (r.get("programming_language") or "python").lower() != "python" or not before or not after or before == after:
            continue
        if not (r.get("repo_url") and r.get("function_name") and SHA_RE.fullmatch(str(r.get("commit_hash") or ""))):
            continue
        if localize.localize_sink(kb, before, "python", after) is not None:
            out.append(r)
    return out


def select_repos(rows: list[dict], n: int, seed: int) -> list[str]:
    """Gli `n` repo con più righe; a parità, per sha256(seme:repo)."""
    counts = Counter(r["repo_url"] for r in rows)
    ranked = sorted(counts, key=lambda repo: (-counts[repo], hashlib.sha256(f"{seed}:{repo}".encode()).hexdigest(), repo))
    return ranked[:n]


# --------------------------------------------------------------------------- git
def _git(args: list[str], cwd: Path, timeout: int, allow_local: bool) -> subprocess.CompletedProcess | None:
    """`git` senza hook, prompt né configurazione dell'utente, con i soli protocolli ammessi; None se scade o non parte."""
    protocols = ["-c", "protocol.allow=never", "-c", "protocol.https.allow=always"]
    if allow_local:
        protocols += ["-c", "protocol.file.allow=always"]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    cmd = ["git", *protocols, "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false", *args]
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except (subprocess.TimeoutExpired, OSError):
        return None


def checkout(dest: Path, rev: str, timeout: int = 120) -> bool:
    """`git checkout -f` di `rev` (uno SHA intero, anche con `^` per il genitore) nella cartella `dest`; False se fallisce."""
    if not REV_RE.fullmatch(rev):
        return False
    done = _git(["checkout", "-q", "-f", rev], Path(dest), timeout, allow_local=True)
    return done is not None and done.returncode == 0


def fetch_commit(repo_url: str, sha: str, dest: Path, timeout: int = 300, allow_local: bool = False) -> bool:
    """Porta in `dest` il commit `sha` e il suo genitore (`--depth 2`) e fa il checkout di `sha`; False su errore o timeout.

    `dest` si crea se serve e si riusa fra commit dello stesso repo (gli oggetti già scaricati non si riscaricano). Solo
    `https://github.com/…` (o `file://` con `allow_local`, per i test) e uno SHA di 40 esadecimali.
    """
    if not SHA_RE.fullmatch(sha) or not (URL_RE.fullmatch(repo_url) or (allow_local and repo_url.startswith("file://"))):
        return False
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    steps = []
    if not (dest / ".git").exists():
        steps.append(["init", "-q"])
    steps += [["config", "remote.origin.url", repo_url],
              ["fetch", "-q", "--depth", "2", "--no-tags", "--no-recurse-submodules", "origin", sha],
              ["checkout", "-q", "-f", sha]]
    for step in steps:
        done = _git(step, dest, timeout, allow_local)
        if done is None or done.returncode != 0:
            return False
    return True


# --------------------------------------------------------------------------- funzione nel checkout
def _from_def(code: str) -> tuple[int, str]:
    """(indice della riga `def`, codice da lì in poi): i decoratori (anche su più righe) non contano per il confronto."""
    lines = code.strip("\n").split("\n")
    at = next((i for i, line in enumerate(lines) if DEF_RE.match(line)), 0)
    return at, "\n".join(lines[at:])


def _squash(code: str) -> str:
    return re.sub(r"\s+", "", _from_def(code)[1])


def _python_files(root: Path):
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(files):
            path = Path(folder) / name
            if name.endswith(".py") and not path.is_symlink():
                try:
                    if path.stat().st_size <= MAX_FILE_BYTES:
                        yield path
                except OSError:
                    continue


def locate_function(root: Path, function_name: str, code: str) -> tuple[str, int] | None:
    """`(percorso relativo, riga della `def`)` del `.py` con una `def function_name` il cui corpo (spazi e decoratori a parte)
    coincide con `code`; il primo in ordine alfabetico se ce n'è più d'uno."""
    root = Path(root)
    wanted = _squash(code)
    needle = f"def {function_name}"
    for path in _python_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if needle not in text:
            continue
        for fn in localize.split_python_functions(text):
            if fn["name"] == function_name and _squash(fn["code"]) == wanted:
                return path.relative_to(root).as_posix(), fn["start_line"] + _from_def(fn["code"])[0]
    return None


def find_function(root: Path, function_name: str, code: str) -> str | None:
    """Percorso relativo del `.py` che contiene la funzione (vedi `locate_function`), altrimenti None."""
    found = locate_function(root, function_name, code)
    return found[0] if found else None


def _target(work: Path, row: dict, code: str, other: str, kb: dict, preferred: str | None) -> dict | None:
    """Sink toccato dalla fix in `code`, come bersaglio per il motore: file, riga assoluta, callee, CWE del catalogo."""
    found = locate_function(work, row["function_name"], code)
    sink = (localize.localize_sink(kb, code, "python", other, preferred) if preferred else None) or \
        localize.localize_sink(kb, code, "python", other)
    if found is None or sink is None:
        return None
    rel, def_line = found
    return {"file": rel, "line": def_line + sink["line"] - 1 - _from_def(code)[0], "callee": sink["callee"],
            "cwe_id": sink["entry"]["id"]}


# --------------------------------------------------------------------------- pilota
def _slug(repo_url: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_url.split("//", 1)[-1]).strip("_")[:120]


def _dir_size(path: Path) -> int:
    total = 0
    for folder, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                continue
    return total


def _records(targets: list[dict], work: Path, engine_bin: Path | None) -> dict[tuple, dict]:
    """Record del motore per i bersagli, per `(file, riga, callee)`; le slice troncate non ci sono."""
    if not targets:
        return {}
    records, _stderr = slice_io.run_slice(work, targets, depth=DEPTH, binary=engine_bin)
    return {(r["sink"]["file"], r["sink"]["line"], r["sink"]["function"]): r for r in records if not r.get("truncated")}


def _commit_pairs(work: Path, rows: list[dict], kb: dict, engine_bin: Path | None, stats: Counter) -> list[tuple[dict, dict, dict]]:
    """`(riga, record prima, record dopo)` per le righe del commit; il checkout `work` è sul commit di fix."""
    sha = rows[0]["commit_hash"]
    if not checkout(work, sha + "^"):
        stats["no_parent"] += len(rows)
        return []
    before: list[tuple[dict, dict, str | None]] = []
    for r in rows:
        label = r.get("cwe_id")
        preferred = common.resolve_cwe(kb, label) if label and CWE_RE.fullmatch(label) else None
        target = _target(work, r, r["code_before"], r["code_after"], kb, preferred)
        if target is None:
            stats["function_not_found"] += 1
        else:
            before.append((r, target, preferred))
    got_before = _records([t for _, t, _ in before], work, engine_bin)
    stats["records_missing_before"] += sum((t["file"], t["line"], t["callee"]) not in got_before for _, t, _ in before)
    if not checkout(work, sha):
        return []
    after: list[tuple[dict, dict, dict]] = []
    for r, t, _ in before:
        rec_before = got_before.get((t["file"], t["line"], t["callee"]))
        if rec_before is None:
            continue
        target = _target(work, r, r["code_after"], r["code_before"], kb, t["cwe_id"])
        if target is None:
            stats["fixed_side_not_found"] += 1
        elif target["cwe_id"] != t["cwe_id"]:
            stats["fixed_side_other_cwe"] += 1
        else:
            after.append((r, rec_before, target))
    got_after = _records([t for _, _, t in after], work, engine_bin)
    pairs = []
    for r, rec_before, t in after:
        rec_after = got_after.get((t["file"], t["line"], t["callee"]))
        if rec_after is None:
            stats["records_missing_after"] += 1
        else:
            pairs.append((r, rec_before, rec_after))
    return pairs


def _pair_examples(row: dict, rec_before: dict, rec_after: dict, kb: dict) -> list[dict]:
    meta = {"source": SOURCE, "synthetic": False, "track": graph_examples.PY_PYVUL_TRACK, "group": row["commit_hash"],
            "repo": row["repo_url"], "commit": row["commit_hash"], "function": row["function_name"]}
    return [graph_examples.build_example(rec_before, kb, label=1, variant_kind="bad", origin=None, meta=dict(meta)),
            graph_examples.build_example(rec_after, kb, label=0, variant_kind="good", origin=None, meta=dict(meta))]


def run_pilot(rows: list[dict], kb: dict, *, n_repos: int, work_dir: Path, seed: int = 42, val_pct: int = 5, test_pct: int = 10,
              max_rows_per_repo: int | None = None, engine_bin: Path | None = None, allow_local: bool = False,
              timeout: int = 300) -> tuple[dict[str, list[dict]], Counter]:
    """Esempi a coppie per i `n_repos` repo con più righe: `({"train", "val", "test": [esempi completi]}, conteggi)`.

    Per ogni commit (le righe dello stesso commit condividono i checkout e una sola chiamata al motore per stato) si scarica il
    commit di fix, si prende il genitore (vulnerabile) e il commit (corretto) e si tengono solo le coppie in cui il sink
    della fix si trova da entrambe le parti con la stessa CWE. La divisione è `prepare_data.split_of(commit, seed, val_pct,
    test_pct)`. Un repo con `ABANDON_AFTER` fetch falliti di fila si lascia. Le parole-spia nel codice vero si contano
    (`forbidden_hits_reported`) ma non scartano nulla: `is_safe_url` è un nome legittimo.
    """
    started = time.monotonic()
    work_dir = Path(work_dir)
    usable = usable_rows(rows, kb)
    repos = select_repos(usable, n_repos, seed)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_repo[r["repo_url"]].append(r)
    stats: Counter = Counter()
    stats["rows_usable"] = len(usable)
    stats["repos"] = len(repos)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    allow = codeql_graph.catalog_leaky_names(kb) | codeql_graph.API_LEAKY_NAMES
    for repo in repos:
        rows_r = sorted(by_repo[repo], key=lambda r: (r["commit_hash"], r["function_name"]))
        stats["rows_total"] += len(rows_r)
        if max_rows_per_repo is not None:
            rows_r = rows_r[:max_rows_per_repo]
        stats["rows_considered"] += len(rows_r)
        by_commit: dict[str, list[dict]] = defaultdict(list)
        for r in rows_r:
            by_commit[r["commit_hash"]].append(r)
        work = work_dir / _slug(repo)
        failures = 0
        for sha, rows_c in by_commit.items():
            if not fetch_commit(repo, sha, work, timeout=timeout, allow_local=allow_local):
                stats["fetch_failed"] += 1
                failures += 1
                if failures >= ABANDON_AFTER:
                    stats["repos_abandoned"] += 1
                    break
                continue
            failures = 0
            stats["commits_fetched"] += 1
            split = None
            for row, rec_before, rec_after in _commit_pairs(work, rows_c, kb, engine_bin, stats):
                try:
                    pair = _pair_examples(row, rec_before, rec_after, kb)
                except ValueError:
                    stats["build_errors"] += 1
                    continue
                split = split or prepare_data.split_of(sha, seed, val_pct, test_pct)
                stats["forbidden_hits_reported"] += len(leak_audit.forbidden_hits(
                    pair, mode="parts", extra_parts=codeql_graph.PY_EXTRA_PARTS, allow=allow))
                splits[split].extend(pair)
                stats["pairs"] += 1
                stats["rows_resolved"] += 1
    stats["seconds"] = round(time.monotonic() - started, 1)
    stats["bytes_on_disk"] = _dir_size(work_dir) if work_dir.exists() else 0
    return splits, stats


def finalize_splits(splits: dict[str, list[dict]], kb: dict, seed: int = 42, p_callers: float = 0.7) -> tuple[dict[str, list[dict]], dict]:
    """Doppioni e conflitti fuori, poi la miscela di view (completa/solo sink) su train e val; il test resta a slice completa."""
    out: dict[str, list[dict]] = {}
    removed = {"duplicates": 0, "conflicts": 0}
    for split, examples in splits.items():
        seen: set[tuple[str, int]] = set()
        unique = []
        for ex in examples:
            key = (graph_examples.prompt_key(ex), ex["meta"]["label"])
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        removed["duplicates"] += len(examples) - len(unique)
        kept, conflicts = graph_examples.drop_conflicts(unique)
        removed["conflicts"] += conflicts
        if split == "test":
            out[split] = kept
        else:
            out[split] = graph_examples.choose_views(graph_examples.make_candidates(kept, kb), seed, p_callers)
    return out, removed


def pilot_report(rows_total: int, rows_resolved: int, examples: list[dict], seconds: float, bytes_on_disk: int) -> dict:
    """Resa del pilota: righe risolte su totali, esempi per repo, tempo e spazio."""
    per_repo = Counter(ex["meta"]["repo"] for ex in examples)
    return {
        "rows_total": rows_total,
        "rows_resolved": rows_resolved,
        "resolve_rate": round(rows_resolved / rows_total, 4) if rows_total else 0.0,
        "examples": len(examples),
        "examples_per_repo": dict(sorted(per_repo.items())),
        "mean_examples_per_repo": round(len(examples) / len(per_repo), 2) if per_repo else 0.0,
        "seconds": seconds,
        "seconds_per_row": round(seconds / rows_total, 2) if rows_total else 0.0,
        "bytes_on_disk": bytes_on_disk,
        "mb_on_disk": round(bytes_on_disk / (1024 * 1024), 2),
    }


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True, help="CVEfixes/PyVul (qualunque file che sources.read_rows legge)")
    ap.add_argument("--repos", type=int, default=20, help="repo del pilota (non estendere senza decisione dell'utente)")
    ap.add_argument("--work-dir", type=Path, default=Path("outputs/pyvul-pilot"), help="checkout dei repo (fuori da /tmp, ignorata da git)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/graph"))
    ap.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    ap.add_argument("--seed", type=int, default=42, help="lo stesso di prepare_data.py")
    ap.add_argument("--val-pct", type=int, default=5, help="come --val-pct di prepare_data.py")
    ap.add_argument("--test-pct", type=int, default=10, help="come --holdout-pct di prepare_data.py")
    ap.add_argument("--max-rows-per-repo", type=int, default=None, help="tetto di righe per repo, per limitare i tempi di rete")
    ap.add_argument("--p-callers", type=float, default=0.7)
    ap.add_argument("--engine-bin", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=300, help="secondi per ogni operazione git")
    ap.add_argument("--allow-local-urls", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    kb = common.load_kb()
    rows = list(sources.read_rows(args.parquet, None))
    splits, stats = run_pilot(rows, kb, n_repos=args.repos, work_dir=args.work_dir, seed=args.seed, val_pct=args.val_pct,
                              test_pct=args.test_pct, max_rows_per_repo=args.max_rows_per_repo, engine_bin=args.engine_bin,
                              allow_local=args.allow_local_urls, timeout=args.timeout)
    final, removed = finalize_splits(splits, kb, args.seed, args.p_callers)
    _write_jsonl(args.out_dir / "py_pyvul_train.jsonl", final["train"])
    _write_jsonl(args.out_dir / "py_pyvul_val.jsonl", final["val"])
    _write_jsonl(args.eval_dir / "py_cvefixes.jsonl", final["test"])
    everything = [ex for exs in splits.values() for ex in exs]
    report = pilot_report(stats["rows_considered"], stats["rows_resolved"], everything, stats["seconds"], stats["bytes_on_disk"])
    report.update({"counts": dict(sorted(stats.items())), "removed": removed, "rows_in_dataset": len(rows),
                   "written": {"train": len(final["train"]), "val": len(final["val"]), "test": len(final["test"])},
                   "repos": args.repos, "max_rows_per_repo": args.max_rows_per_repo, "split_key": "commit (prepare_data.split_of, R32)"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "py_pyvul_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows_total", "rows_resolved", "resolve_rate", "examples", "mean_examples_per_repo",
                                              "seconds", "seconds_per_row", "mb_on_disk", "written")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
