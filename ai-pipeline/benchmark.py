#!/usr/bin/env python3
"""Benchmark (Fase 5): accuratezza SAST e prestazioni hardware BF16 vs FP8 vs NVFP4.

    # 5.1  Semgrep vs Nemotron senza slicing vs Nemotron con slicing + fine-tuning (vLLM già in esecuzione su :8000)
    python benchmark.py accuracy --test data/test.jsonl --systems semgrep llm-noslice llm-slice

    # 5.1b Il modello con i chiamanti (fino a --depth) contro lo stesso modello con il solo blocco del sink, sulle coppie di Juliet
    python benchmark.py accuracy --test data/eval/juliet_pairs.jsonl --systems llm-graph llm-sink-only --depth 5

    # 5.2  VRAM, tok/s, latenza, F1 per formato (avvia/ferma vLLM da solo, un modello alla volta)
    python benchmark.py hardware --models bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4 --test data/test.jsonl

Non lanciarlo mentre usi la GPU per altro: `hardware` occupa la GPU (gpu-memory-utilization 0.80).
Solo libreria standard; richiede `semgrep` (accuracy) e `nvidia-smi` + `vllm` (hardware).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
import graph_examples
import vllm_flags

SYSTEMS = ("semgrep", "llm-noslice", "llm-slice", "llm-graph", "llm-sink-only")
# Scarto di lunghezza (sul prompt utente) entro cui i due gemelli di una coppia si considerano di lunghezza pari (R19).
LENGTH_TOLERANCE = 0.15
NOSLICE_SYSTEM = (
    "You are a SAST security analyzer. You receive the code of a function. Decide whether it contains a security "
    "vulnerability. Answer ONLY with a JSON object that conforms to the requested schema. "
    "Write \"reasoning\" and \"suggested_fix\" in English, citing lines."
)
NOSLICE_USER = (
    "## Code\n```{lang}\n{code}\n```\n\n## Task\nIdentify any vulnerabilities in this function. "
    "If you find none, answer with \"vulnerable\": false, \"vex_status\": \"not_affected\" and \"severity\": \"NONE\"."
)
_NUMBERED_LINE = re.compile(r"^ *\d+ \| (.*)$")
_FENCE = re.compile(r"```[A-Za-z+]*\n(.*?)\n```", re.S)
_CASE = re.compile(r"case_(\d+)\.")
_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_VLLM_VERSION = (re.compile(r"vLLM API server version (\S+)"), re.compile(r"\(v(\S+?)\) with config"))


# --------------------------------------------------------------------------- metriche
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    def ratio(a, b):
        return a / b if b else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "fpr": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def confusion(labels: list[int], preds: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and not p)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and not p)
    return tp, fp, tn, fn


def evaluate(examples: list[dict], preds: list[bool | None]) -> dict:
    """Metriche complessive e sul sottoinsieme con sink-chiamata. Un output non valido conta come "non vulnerabile"."""
    def report(subset: list[int]) -> dict:
        return metrics_from_counts(*confusion([examples[i]["meta"]["label"] for i in subset], [bool(preds[i]) for i in subset]))

    everything = list(range(len(examples)))
    calls = [i for i in everything if examples[i]["meta"].get("sink_kind") == "call"]
    sources = sorted({ex["meta"].get("source", "?") for ex in examples})
    by_source = {src: report([i for i in everything if examples[i]["meta"].get("source", "?") == src]) for src in sources}
    out = {"n": len(examples), "invalid": sum(p is None for p in preds), "all": report(everything), "call_sinks": report(calls), "by_source": by_source}
    pairs = pair_report(examples, preds)
    if pairs is not None:
        out["pairs"] = pairs
    return out


def pair_report(examples: list[dict], preds: list[bool | None]) -> dict | None:
    """S1 sulle coppie (stesso prompt «solo sink», etichette opposte, `meta.pair_id`): None se non ce ne sono.

    Si contano solo le coppie complete (un vulnerabile e un sicuro con lo stesso `pair_id`); un output non valido è «non
    vulnerabile». `accuracy` = esempi giusti / esempi accoppiati, `both_correct` = coppie con entrambi giusti / coppie.
    Per non leggere S1 come prova di ragionamento sul flusso (R19) si riportano anche le coppie a lunghezza controllata
    (il prompt utente dei due gemelli differisce al più del `LENGTH_TOLERANCE`) e la base «il più lungo è il vulnerabile»:
    la quota delle coppie di lunghezza diversa in cui lo è, cioè quanto otterrebbe chi guarda solo la lunghezza.
    """
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        pid = (ex.get("meta") or {}).get("pair_id")
        if pid:
            groups.setdefault(pid, []).append(i)
    pairs = [idx for idx in groups.values() if len(idx) == 2 and {examples[i]["meta"]["label"] for i in idx} == {0, 1}]
    if not pairs:
        return None

    def right(i: int) -> bool:
        return bool(preds[i]) == bool(examples[i]["meta"]["label"])

    def summary(subset: list[list[int]]) -> dict:
        n = len(subset)
        return {"pairs": n, "accuracy": round(sum(right(i) for p in subset for i in p) / (2 * n), 4) if n else 0.0,
                "both_correct": round(sum(all(right(i) for i in p) for p in subset) / n, 4) if n else 0.0}

    size = {i: len(user_message(examples[i])) for p in pairs for i in p}
    gap = lambda p: abs(size[p[0]] - size[p[1]]) / max(size[p[0]], size[p[1]], 1)  # noqa: E731
    controlled = [p for p in pairs if gap(p) <= LENGTH_TOLERANCE]
    different = [p for p in pairs if size[p[0]] != size[p[1]]]
    longer_is_vulnerable = sum(size[next(i for i in p if examples[i]["meta"]["label"] == 1)] > size[next(i for i in p if examples[i]["meta"]["label"] == 0)]
                               for p in different)
    return {**summary(pairs),
            "length_controlled": {**summary(controlled), "tolerance": LENGTH_TOLERANCE,
                                  "identical_length": sum(size[p[0]] == size[p[1]] for p in controlled)},
            "longer_is_vulnerable": {"pairs_with_different_length": len(different),
                                     "share": round(longer_is_vulnerable / len(different), 4) if different else 0.0}}


# --------------------------------------------------------------------------- prompt e verdetti
def user_message(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def extract_code(user_text: str) -> str:
    """Ricostruisce il codice dal primo blocco numerato del prompt (inverso di common.render_block)."""
    m = _FENCE.search(user_text)
    if not m:
        raise ValueError("nessun blocco di codice nel prompt")
    lines = []
    for line in m.group(1).split("\n"):
        n = _NUMBERED_LINE.match(line)
        lines.append(n.group(1) if n else line)
    return "\n".join(lines)


def noslice_messages(example: dict) -> list[dict]:
    """Prompt senza regola CWE, senza sink e senza traccia: solo la funzione.

    Con `meta.trace` (esempi con slice) la funzione è il blocco a profondità 0: il primo blocco del prompt sarebbe il chiamante
    più esterno, non il sink.
    """
    meta = example.get("meta", {})
    sink_block = next((b for b in (meta.get("trace") or {}).get("blocks", []) if b["depth"] == 0), None)
    code = sink_block["code"] if sink_block else extract_code(user_message(example))
    return [{"role": "system", "content": NOSLICE_SYSTEM}, {"role": "user", "content": NOSLICE_USER.format(lang=meta.get("lang", "c"), code=code)}]


def slice_messages(example: dict) -> list[dict]:
    return [m for m in example["messages"] if m["role"] in ("system", "user")]


@functools.lru_cache(maxsize=1)
def _catalog() -> dict:
    return common.load_kb()


def rerender_messages(example: dict, max_depth: int | None = None, sink_only: bool = False, kb: dict | None = None) -> list[dict]:
    """System e user rifatti da `meta.trace`: i blocchi fino a `max_depth`, o il solo blocco a profondità 0 con `sink_only`.

    Delega a `graph_examples.messages_from_trace` (lo stesso codice che costruisce gli esempi, R6c), così il prompt «solo
    sink» di un esempio completo è per costruzione quello dell'esempio costruito solo-sink. Senza `meta.trace` (esempi delle
    fonti senza slice) restituisce i messaggi salvati.
    """
    trace = (example.get("meta") or {}).get("trace")
    if not trace:
        return slice_messages(example)
    return graph_examples.messages_from_trace(trace, kb or _catalog(), max_depth=max_depth, sink_only=sink_only)


def verdict_from_content(content) -> bool | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    flag = data.get("vulnerable") if isinstance(data, dict) else None
    return flag if isinstance(flag, bool) else None


# --------------------------------------------------------------------------- client vLLM (stdlib)
def chat_completion(url: str, model: str, messages: list[dict], max_tokens: int = 768, timeout: float = 300) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "vex_finding", "strict": True, "schema": common.load_schema()}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return {
        "content": data["choices"][0]["message"]["content"],
        "completion_tokens": int((data.get("usage") or {}).get("completion_tokens", 0)),
        "latency": time.perf_counter() - started,
    }


def run_llm(url: str, model: str, message_sets: list[list[dict]], concurrency: int, max_tokens: int = 768) -> list[dict]:
    def one(messages):
        started = time.perf_counter()
        try:
            return chat_completion(url, model, messages, max_tokens)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            return {"content": None, "error": str(exc), "completion_tokens": 0, "latency": time.perf_counter() - started}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, message_sets))


# --------------------------------------------------------------------------- baseline Semgrep
def write_semgrep_cases(examples: list[dict], directory: Path) -> dict[str, int]:
    index = {}
    for i, ex in enumerate(examples):
        lang = ex.get("meta", {}).get("lang", "c")
        path = Path(directory) / f"case_{i:06d}.{ {'cpp': 'cpp', 'python': 'py'}.get(lang, 'c') }"
        path.write_text(extract_code(user_message(ex)) + "\n", encoding="utf-8")
        index[str(path)] = i
    return index


def parse_semgrep(report: dict, index: dict[str, int]) -> set[int]:
    flagged = set()
    for result in report.get("results", []):
        path = result.get("path", "")
        if path in index:
            flagged.add(index[path])
        elif (m := _CASE.search(Path(path).name)) and int(m.group(1)) in index.values():
            flagged.add(int(m.group(1)))
    return flagged


def run_semgrep(examples: list[dict], config: str, timeout: int = 3600) -> list[bool]:
    with tempfile.TemporaryDirectory() as d:
        index = write_semgrep_cases(examples, Path(d))
        cmd = ["semgrep", "--config", config, "--json", "--quiet", "--metrics=off", "--timeout", "10", d]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError("semgrep non installato (uv pip install semgrep)") from exc
        flagged = parse_semgrep(json.loads(out.stdout or "{}"), index)
    return [i in flagged for i in range(len(examples))]


# --------------------------------------------------------------------------- GPU
def parse_nvidia_smi_memory(text: str) -> int:
    for line in text.splitlines():
        if m := re.match(r"\s*(\d+)", line):
            return int(m.group(1))
    raise ValueError(f"output di nvidia-smi non riconosciuto: {text!r}")


def parse_vllm_log(text: str) -> dict:
    """Pesi in VRAM e capacità della KV cache dal log di avvio di vLLM.

    Con --gpu-memory-utilization fisso vLLM prealloca la stessa quota di VRAM per ogni formato: il picco di nvidia-smi
    non distingue BF16/FP8/NVFP4, mentre pesi e token di KV cache sì.
    """
    weights, kv = _WEIGHTS.search(text), _KV_TOKENS.search(text)
    return {"weights_gib": float(weights.group(1)) if weights else None, "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None}


def parse_vllm_version(text: str) -> str | None:
    """Versione di vLLM dal log di avvio (riga dell'API server, o quella dell'engine); None se non c'è."""
    for rx in _VLLM_VERSION:
        if m := rx.search(text):
            return m.group(1)
    return None


def vram_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return parse_nvidia_smi_memory(out.stdout)


class PeakSampler:
    """Campiona la memoria GPU in un thread e ne tiene il massimo (relativo alla lettura iniziale)."""

    def __init__(self, reader=vram_used_mib, interval: float = 0.5):
        self.reader, self.interval = reader, interval
        self.baseline_mib = 0
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak_mib = max(self.peak_mib, self.reader())

    def __enter__(self):
        self.baseline_mib = self.peak_mib = self.reader()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self.reader())

    @property
    def delta_mib(self) -> int:
        return self.peak_mib - self.baseline_mib


# --------------------------------------------------------------------------- report
def markdown_table(rows: list[dict]) -> str:
    head = ["Formato", "Pesi in VRAM (GB)", "KV cache (token)", "VRAM picco (GB)", "Disco (GB)", "Throughput (tok/s)",
            "Latenza media (s)", "Scansione repo (s)", "F1", "ΔF1 vs BF16"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    base = rows[0]["f1"] if rows else 0.0
    nd = lambda v, fmt: "n/d" if v is None else format(v, fmt)  # noqa: E731
    for r in rows:
        lines.append(
            f"| {r['format']} | {nd(r.get('weights_gb'), '.1f')} | {nd(r.get('kv_tokens'), 'd')} | {r['vram_gb']:.1f} | {r['disk_gb']:.1f} | "
            f"{r['tok_s']:.1f} | {r['latency_s']:.2f} | {nd(r.get('scan_s'), '.1f')} | {r['f1']:.3f} | {r['f1'] - base:+.3f} |"
        )
    return "\n".join(lines)


def accuracy_table(results: dict[str, dict]) -> str:
    lines = ["| Sistema | Insieme | N | Precision | Recall | F1 | FPR | Non validi |", "|---|---|---|---|---|---|---|---|"]
    for name, rep in results.items():
        rows = [(rep["all"], "tutti"), (rep["call_sinks"], "solo sink-chiamata")] + [(m, f"fonte: {src}") for src, m in rep.get("by_source", {}).items()]
        for m, label in rows:
            n = sum(m["counts"].values())
            lines.append(f"| {name} | {label} | {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fpr']:.3f} | {rep['invalid']} |")
    return "\n".join(lines)


def pairs_table(results: dict[str, dict]) -> str:
    """Una riga per sistema con le coppie S1 (stringa vuota se nessun sistema ne ha)."""
    rows = [(name, rep["pairs"]) for name, rep in results.items() if "pairs" in rep]
    if not rows:
        return ""
    head = ["Sistema", "Coppie", "Accuratezza", "Entrambe giuste", "Coppie a lunghezza controllata", "Entrambe giuste (lunghezza controllata)",
            "Base «il più lungo è vulnerabile»"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for name, p in rows:
        c, b = p["length_controlled"], p["longer_is_vulnerable"]
        lines.append(f"| {name} | {p['pairs']} | {p['accuracy']:.3f} | {p['both_correct']:.3f} | {c['pairs']} | {c['both_correct']:.3f} | {b['share']:.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- riga del benchmark hardware
def percentile(values: list[float], q: float) -> float:
    """Percentile q (0-100) col metodo del rango più vicino; 0.0 se la lista è vuota."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def hardware_row(name: str, examples: list[dict], raw: list[dict], wall: float, log_info: dict, vllm_version: str | None,
                 vram_delta_mib: int, disk_gb: float, scan_s: float | None, concurrency: int) -> dict:
    """Riga di hardware.json per un formato. I campi `report`, `latency_p50_s/p95_s`, `n`, `concurrency` e `vllm_version`
    sono aggiunti per la model card; gli altri sono quelli della tabella Markdown."""
    ok = [r for r in raw if r["content"] is not None]
    latencies = [r["latency"] for r in ok]
    report = evaluate(examples, [verdict_from_content(r["content"]) for r in raw])
    return {
        "format": name,
        "weights_gb": log_info["weights_gib"],
        "kv_tokens": log_info["kv_tokens"],
        "vram_gb": vram_delta_mib / 1024,
        "disk_gb": disk_gb,
        "tok_s": sum(r["completion_tokens"] for r in ok) / wall if wall else 0.0,
        "latency_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "scan_s": scan_s,
        "f1": report["all"]["f1"],
        "f1_call_sinks": report["call_sinks"]["f1"],
        "invalid": report["invalid"],
        "n": len(examples),
        "concurrency": concurrency,
        "vllm_version": vllm_version,
        "report": report,
    }


# --------------------------------------------------------------------------- gestione vLLM
def vllm_command(vllm_bin: str, model_dir: Path, port: int = 8000, gpu_util: float = 0.80, max_len: int = 8192) -> list[str]:
    return [vllm_bin, *vllm_flags.serve_args(model_dir, port=port, gpu_util=gpu_util, max_len=max_len)]


def vllm_env(vllm_bin: str, base: dict | None = None) -> dict:
    """Ambiente di vLLM: la cartella del suo eseguibile va in testa al PATH.

    `venv/bin/vllm` si lancia senza attivare il venv, quindi `venv/bin` non è nel PATH. vLLM però compila al volo i kernel di
    FlashInfer (sampler) chiamando `ninja` per nome: senza questo, `FileNotFoundError: 'ninja'` e il motore muore all'avvio.
    Il percorso diventa assoluto perché FlashInfer lancia ninja da un'altra cartella di lavoro.
    """
    env = dict(os.environ if base is None else base)
    if os.sep in vllm_bin:  # un nome nudo ("vllm") lo risolve già il PATH
        bin_dir = os.path.dirname(os.path.abspath(vllm_bin))
        env["PATH"] = os.pathsep.join(p for p in (bin_dir, env.get("PATH", "")) if p)  # niente voci vuote: significano "cartella corrente"
    return env


def start_vllm(vllm_bin: str, model_dir: Path, port: int, gpu_util: float, log) -> subprocess.Popen:
    return subprocess.Popen(vllm_command(vllm_bin, model_dir, port, gpu_util), stdout=log, stderr=subprocess.STDOUT,
                            env=vllm_env(vllm_bin), start_new_session=True)


# Riga di log del motore di vLLM: "(EngineCore pid=47909) ERROR 11-05 12:11:38 [core.py:1584] <testo>"
_ENGINE_ERROR_PREFIX = re.compile(r"^\(EngineCore pid=\d+\) ERROR \d\d-\d\d \d\d:\d\d:\d\d \[[\w.]+:\d+\] ")
_CARET_ONLY = re.compile(r"^[\s^~]*$")


def engine_error(text: str, lines: int = 8) -> str:
    """Le ultime righe del traceback con cui il motore di vLLM spiega perché non parte ("" se il log non ne ha uno).

    L'errore vero sta nel blocco che segue "EngineCore failed to start."; in fondo al log c'è poi il traceback dell'API server, lungo,
    che dice solo "See root cause above": la sola coda del file non mostra la causa. Prefisso e righe di soli cursori sono tolti.
    """
    rows = text.splitlines()
    start = next((i for i in range(len(rows) - 1, -1, -1) if "EngineCore failed to start" in rows[i]), None)
    if start is None:
        return ""
    block = []
    for row in rows[start + 1:]:
        m = _ENGINE_ERROR_PREFIX.match(row)
        if not m:
            break
        body = row[m.end():]
        if not _CARET_ONLY.match(body):
            block.append(body)
    return "\n".join(block[-lines:])


def log_excerpt(path: Path, lines: int = 40) -> str:
    """Cosa mostrare di un log di vLLM quando il processo muore: l'errore del motore, se c'è, altrimenti la coda del file."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    engine = engine_error(text)
    if engine:
        return f"--- errore del motore vLLM ---\n{engine}"
    tail = "\n".join(text.splitlines()[-lines:])
    return f"--- ultime righe del log ---\n{tail}" if tail else ""


def wait_ready(url: str, proc: subprocess.Popen, timeout: float, log_path: Path | None = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            msg = f"vLLM terminato con codice {proc.returncode} prima di essere pronto (log: {log_path})" if log_path else \
                  f"vLLM terminato con codice {proc.returncode} prima di essere pronto"
            excerpt = log_excerpt(log_path) if log_path else ""
            raise RuntimeError(f"{msg}\n{excerpt}" if excerpt else msg)
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM non pronto dopo {timeout:.0f}s")


def stop_vllm(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9


def scan_repos(engine_url: str, repos: list[str], timeout: float = 3600) -> list[float]:
    times = []
    for repo in repos:
        req = urllib.request.Request(f"{engine_url.rstrip('/')}/analyze", json.dumps({"repo_url": repo}).encode(), {"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        times.append(time.perf_counter() - started)
    return times


# --------------------------------------------------------------------------- comandi
def load_examples(path: Path, limit: int | None) -> list[dict]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break
    return examples


def cmd_accuracy(args: argparse.Namespace) -> int:
    examples = load_examples(args.test, args.limit)
    results = {}
    for system in args.systems:
        if system == "semgrep":
            preds: list[bool | None] = run_semgrep(examples, args.semgrep_config)
        else:
            build = {"llm-noslice": noslice_messages, "llm-slice": slice_messages,
                     "llm-graph": lambda ex: rerender_messages(ex, args.depth),
                     "llm-sink-only": lambda ex: rerender_messages(ex, sink_only=True)}[system]
            raw = run_llm(args.url, args.model, [build(ex) for ex in examples], args.concurrency)
            preds = [verdict_from_content(r["content"]) for r in raw]
        results[system] = evaluate(examples, preds)
        print(f"{system}: F1={results[system]['all']['f1']:.3f}", file=sys.stderr)
    table = accuracy_table(results)
    pairs = pairs_table(results)
    if pairs:
        table += "\n\n" + pairs
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    examples = load_examples(args.test, args.samples)
    repos = [l.strip() for l in args.repos.read_text().splitlines() if l.strip()] if args.repos else []
    url = f"http://127.0.0.1:{args.port}"
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        name, _, path = spec.partition("=")
        model_dir = Path(path)
        print(f"== {name}: {model_dir}", file=sys.stderr)
        with PeakSampler() as sampler, (args.out_dir / f"vllm_{name}.log").open("w") as log:
            proc = start_vllm(args.vllm_bin, model_dir, args.port, args.gpu_util, log)
            try:
                wait_ready(url, proc, args.startup_timeout, log_path=args.out_dir / f"vllm_{name}.log")
                started = time.perf_counter()
                raw = run_llm(url, "sast-nemotron", [slice_messages(ex) for ex in examples], args.concurrency)
                wall = time.perf_counter() - started
                scan = statistics.mean(scan_repos(args.engine_url, repos)) if repos else None
            finally:
                stop_vllm(proc)
        log_text = (args.out_dir / f"vllm_{name}.log").read_text(errors="replace")
        rows.append(hardware_row(name, examples, raw, wall, parse_vllm_log(log_text), parse_vllm_version(log_text),
                                 sampler.delta_mib, dir_size_gb(model_dir), scan, args.concurrency))
        deadline = time.time() + 120  # attende che la VRAM venga rilasciata prima del modello successivo
        while time.time() < deadline and vram_used_mib() > sampler.baseline_mib + 500:
            time.sleep(2)
    table = markdown_table(rows)
    print(table)
    (args.out_dir / "hardware.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out_dir / "hardware.md").write_text(table + "\n", encoding="utf-8")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    acc = sub.add_parser("accuracy", help="5.1 Precision/Recall/F1/FPR dei tre sistemi")
    acc.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    acc.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    acc.add_argument("--url", default="http://127.0.0.1:8000")
    acc.add_argument("--model", default="sast-nemotron")
    acc.add_argument("--concurrency", type=int, default=4)
    acc.add_argument("--depth", type=int, default=None, help="llm-graph: profondità massima dei blocchi mostrati (default: tutti)")
    acc.add_argument("--limit", type=int, default=0, help="usa solo i primi N esempi (0 = tutti)")
    acc.add_argument("--semgrep-config", default="p/c", help="ruleset Semgrep (richiede rete per scaricarlo)")
    acc.add_argument("--out", type=Path, default=Path("results/accuracy.json"))
    acc.set_defaults(func=cmd_accuracy)

    hw = sub.add_parser("hardware", help="5.2 VRAM, tok/s, latenza, F1 per formato")
    hw.add_argument("--models", nargs="+", required=True, metavar="NOME=DIR", help="es. bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4")
    hw.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    hw.add_argument("--samples", type=int, default=200)
    hw.add_argument("--concurrency", type=int, default=8)
    hw.add_argument("--port", type=int, default=8000)
    hw.add_argument("--gpu-util", type=float, default=0.80)
    hw.add_argument("--vllm-bin", default="vllm")
    hw.add_argument("--startup-timeout", type=float, default=900)
    hw.add_argument("--repos", type=Path, default=None, help="file con un URL GitHub per riga: misura il tempo medio di scansione via sast-engine")
    hw.add_argument("--engine-url", default="http://127.0.0.1:8080")
    hw.add_argument("--out-dir", type=Path, default=Path("results"))
    hw.set_defaults(func=cmd_hardware)
    args = ap.parse_args(argv)
    if getattr(args, "limit", None) == 0:
        args.limit = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
