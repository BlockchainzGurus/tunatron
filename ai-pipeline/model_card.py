#!/usr/bin/env python3
"""Model card (in inglese) dell'addestramento: MODEL_CARD.md + grafici PNG.

    python model_card.py --out-dir outputs/model_card --log outputs/train.log --results results --data data \\
        --adapter outputs/adapter --models bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4

Legge il log di training, i risultati di `benchmark.py hardware` (uno per formato, in results/<formato>/hardware.json), il dataset,
la configurazione dell'adapter e dei checkpoint quantizzati. Ogni input mancante o illeggibile lascia la sezione corrispondente con
"not available: ..." invece di dare errore. Solo libreria standard, tranne matplotlib per i grafici (importato solo quando serve):
gira con il python3 di Colab, senza toccare i venv a versioni fissate.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import re
import shlex
import shutil
import sys
from pathlib import Path

import train  # solo parse_args e DEFAULT_MODEL: i pacchetti pesanti train.py li importa dentro main()

FORMATS = ("bf16", "fp8", "nvfp4")
KEY_PACKAGES = ("torch", "unsloth", "unsloth-zoo", "trl", "peft", "transformers", "accelerate", "bitsandbytes", "mamba-ssm", "causal-conv1d")
HYPER_KEYS = ("rank", "alpha", "batch_size", "grad_accum", "optim", "max_seq_length", "epochs", "max_steps", "lr", "warmup_ratio", "seed")
SPLITS = ("train", "val", "test")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # le barre tqdm annidate lasciano sequenze ANSI davanti ai dizionari delle metriche
_TRAIN_LOSS = re.compile(r"\{'loss'[^}]*\}")
_EVAL_LOSS = re.compile(r"\{'eval_loss'[^}]*\}")
_RUNTIME = re.compile(r"\{'train_runtime'[^}]*\}")
_FOLDER_VERSION = re.compile(r"-(\d[^/]*)$")


# --------------------------------------------------------------------------- file
def _read_text(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace") if path else ""
    except OSError:
        return ""


def _read_json(path):
    try:
        return json.loads(_read_text(path))
    except ValueError:
        return None


def _number(value) -> float | None:
    """Numero finito da un valore del log (stringa o numero); None per nan, inf o testo."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _order(name: str) -> tuple[int, str]:
    return (FORMATS.index(name) if name in FORMATS else len(FORMATS), name)


# --------------------------------------------------------------------------- log di training
def _entries(text: str, rx: re.Pattern):
    """I dizionari di metriche che `rx` trova nel testo (quelli che non si leggono si saltano)."""
    for m in rx.finditer(text):
        try:
            entry = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(entry, dict):
            yield entry


def _points(text: str, rx: re.Pattern, key: str) -> list[tuple[float, float]]:
    pairs = ((_number(e.get("epoch")), _number(e.get(key))) for e in _entries(text, rx))
    return [(epoch, value) for epoch, value in pairs if epoch is not None and value is not None]


def parse_train_log(text: str) -> dict:
    """Curve (epoca, loss) di training e di valutazione e tempo di training dal log di train.py."""
    text = _ANSI.sub("", text)
    runtimes = [r for r in (_number(e.get("train_runtime")) for e in _entries(text, _RUNTIME)) if r is not None]
    return {"train": _points(text, _TRAIN_LOSS, "loss"), "eval": _points(text, _EVAL_LOSS, "eval_loss"),
            "runtime_s": runtimes[-1] if runtimes else None}


def parse_freeze(text: str) -> dict[str, str]:
    """Versioni dei pacchetti di KEY_PACKAGES da un `uv pip freeze`. I kernel installati da una cartella locale compaiono come
    `nome @ file:///...`: la versione si legge dal nome della cartella (`mamba_ssm-2.3.2.post1`), altrimenti "built from source"."""
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if "==" in line:
            name, _, version = line.partition("==")
        elif " @ " in line:
            name, _, url = line.partition(" @ ")
            folder = _FOLDER_VERSION.search(url.strip())
            version = folder.group(1) if folder else "built from source"
        else:
            continue
        key = name.strip().lower().replace("_", "-")
        if key in KEY_PACKAGES:
            found[key] = version.strip()
    return found


# --------------------------------------------------------------------------- risultati e dati
def load_results(results_dir: Path) -> list[dict]:
    """Righe di results/<formato>/hardware.json, unite e ordinate bf16, fp8, nvfp4 (poi gli altri per nome)."""
    rows: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*/hardware.json")):
        data = _read_json(path)
        for row in data if isinstance(data, list) else []:
            if isinstance(row, dict) and isinstance(row.get("format"), str):
                rows[row["format"]] = row
    return sorted(rows.values(), key=lambda r: _order(r["format"]))


def dataset_summary(data_dir: Path) -> dict | None:
    """Esempi, etichette, fonti e linguaggi di train/val/test; None se non c'è nessuno dei tre file."""
    summary = {}
    for split in SPLITS:
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.is_file():
            continue
        info = {"examples": 0, "vulnerable": 0, "not_vulnerable": 0, "sources": {}, "languages": {}}
        for line in _read_text(path).splitlines():
            if not line.strip():
                continue
            try:
                meta = json.loads(line).get("meta") or {}
            except (ValueError, AttributeError):
                continue
            info["examples"] += 1
            info["vulnerable"] += meta.get("label") == 1
            info["not_vulnerable"] += meta.get("label") == 0
            for key, field in (("sources", "source"), ("languages", "lang")):
                name = str(meta.get(field, "?"))
                info[key][name] = info[key].get(name, 0) + 1
        summary[split] = info
    return summary or None


# --------------------------------------------------------------------------- contesto della card
def _hyper(train_args: str | None) -> dict | None:
    """Iperparametri davvero usati: default di train.py più le sovrascritture del notebook."""
    if train_args is None:
        return None
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # argparse scrive l'uso sullo stderr prima di uscire
            args = vars(train.parse_args(shlex.split(train_args)))
    except (SystemExit, ValueError):
        return None
    return {key: args[key] for key in HYPER_KEYS}


def collect_context(*, log, results: Path, data: Path, adapter, models: dict[str, Path], commit_file, freeze_file, gpu,
                    train_args, vllm_version) -> dict:
    """Mette insieme tutto ciò che la card racconta (forma documentata nel piano); un input mancante lascia il campo vuoto."""
    config = _read_json(Path(adapter) / "adapter_config.json") if adapter else None
    config = config if isinstance(config, dict) else None
    parsed = parse_train_log(_read_text(log))
    formats = [f for f in sorted(models, key=_order) if f in FORMATS and Path(models[f]).is_dir()]
    quant = {}
    for fmt in formats:
        found = _read_json(Path(models[fmt]) / "hf_quant_config.json")
        if isinstance(found, dict):
            quant[fmt] = found
    return {
        "base_model": (config or {}).get("base_model_name_or_path") or train.DEFAULT_MODEL,
        "hyper": _hyper(train_args),
        "adapter": {k: config.get(k) for k in ("r", "lora_alpha", "target_modules", "use_dora")} if config else None,
        "curves": {"train": parsed["train"], "eval": parsed["eval"]},
        "runtime_s": parsed["runtime_s"],
        "commit": _read_text(commit_file).strip() or None,
        "gpu": gpu or None,
        "vllm_version": vllm_version or None,
        "versions": parse_freeze(_read_text(freeze_file)),
        "data": dataset_summary(data),
        "formats": formats,
        "results": load_results(results),
        "quant": quant,
        "charts": {},
    }


# --------------------------------------------------------------------------- testo della card (tutto in inglese)
FORMAT_NOTES = {"bf16": "merged BF16 model (full precision)", "fp8": "FP8 W8A8 quantization with FP8 KV cache (NVIDIA ModelOpt)",
                "nvfp4": "NVFP4 quantization for Blackwell Tensor Cores (NVIDIA ModelOpt)"}


def _fmt(value, spec: str = "") -> str:
    """Valore formattato, oppure n/d se manca, è testo o non è finito (anche per le righe di benchmark di versioni precedenti)."""
    if value is None:
        return "n/d"
    if not spec:
        return str(value)
    number = _number(value)
    return "n/d" if number is None else format(int(number) if spec.endswith("d") else number, spec)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def _duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours} h {rest // 60} min" if hours else f"{rest // 60} min {rest % 60} s"


def _image(ctx: dict, name: str, alt: str) -> list[str]:
    return ["", f"![{alt}]({ctx['charts'][name]})"] if name in ctx["charts"] else []


def _front_matter(ctx: dict) -> list[str]:
    return ["---", "language:", "- it", "- en", f"base_model: {ctx['base_model']}", "library_name: transformers", "pipeline_tag: text-generation",
            "license: other", "tags:", "- security", "- sast", "- vulnerability-detection", "- dora", "---"]


def _summary(ctx: dict) -> list[str]:
    return [f"This model is a fine-tuned version of `{ctx['base_model']}` for static application security testing (SAST) triage. Given a CWE "
            "rule, a candidate sink and the backward code slice that reaches it, it decides whether the sink is really vulnerable and answers "
            "with a JSON finding that follows a fixed schema (`vulnerable`, `vex_status`, `severity`, `reasoning`, `suggested_fix`).",
            "",
            "The system prompt is in Italian, and `reasoning` and `suggested_fix` are written in Italian. Supported code languages: C, C++ and Python."]


def _intended_use() -> list[str]:
    return ["- Intended as a triage aid for human reviewers, not as a replacement for a SAST engine or a security review.",
            "- It expects the prompt format of this project (CWE rule, candidate sink, numbered backward slice); other prompt styles are not evaluated.",
            "- Trained and evaluated on C, C++ and Python functions from public vulnerability datasets; results may not transfer to other languages "
            "or to proprietary code.",
            "- License: see the base model card."]


def _training(ctx: dict) -> list[str]:
    lines = [f"- Base model: `{ctx['base_model']}`"]
    adapter, hyper = ctx["adapter"], ctx["hyper"]
    if adapter:
        modules = adapter.get("target_modules")
        modules = ", ".join(modules) if isinstance(modules, list) else _fmt(modules)
        lines.append(f"- Method: {'DoRA' if adapter.get('use_dora') else 'LoRA'} (rank {_fmt(adapter.get('r'))}, alpha {_fmt(adapter.get('lora_alpha'))}) "
                     f"on {modules}")
    else:
        lines.append("- Adapter: not available: adapter_config.json was not found.")
    if hyper:
        length = f"{hyper['epochs']:g} epochs" if hyper["max_steps"] <= 0 else f"{hyper['max_steps']} steps"
        lines.append(f"- Optimizer {hyper['optim']}; learning rate {hyper['lr']:g} (cosine, warmup ratio {hyper['warmup_ratio']:g}); {length}; "
                     f"effective batch size {hyper['batch_size'] * hyper['grad_accum']} ({hyper['batch_size']} x {hyper['grad_accum']} gradient accumulation); "
                     f"max sequence length {hyper['max_seq_length']}; seed {hyper['seed']}")
    else:
        lines.append("- Hyperparameters: not available: no training arguments were provided.")
    if ctx["gpu"]:
        lines.append(f"- Hardware: {ctx['gpu']}")
    if ctx["runtime_s"] is not None:
        lines.append(f"- Training time: {_duration(ctx['runtime_s'])}")
    if ctx["commit"]:
        lines.append("- Training code: " + " | ".join(l.strip() for l in ctx["commit"].splitlines() if l.strip()))
    if ctx["versions"]:
        lines.append("- Package versions: " + ", ".join(f"{name} {ctx['versions'][name]}" for name in KEY_PACKAGES if name in ctx["versions"]))
    train_curve, eval_curve = ctx["curves"]["train"], ctx["curves"]["eval"]
    if not train_curve and not eval_curve:
        lines.append("- Training curves: not available: no loss metrics were found in the training log.")
    else:
        last = [f"{label} {points[-1][1]:.4f} (epoch {points[-1][0]:g})" for label, points in (("training loss", train_curve), ("validation loss", eval_curve)) if points]
        lines.append("- Last logged: " + "; ".join(last))
    return lines + _image(ctx, "loss_curves", "Training and validation loss")


def _data(ctx: dict) -> list[str]:
    data = ctx["data"]
    if not data:
        return ["not available: no dataset files (train/val/test .jsonl) were found."]
    rows = [[split, f"{data[split]['examples']:,}", f"{data[split]['vulnerable']:,}", f"{data[split]['not_vulnerable']:,}"] for split in SPLITS if split in data]
    first = next(split for split in SPLITS if split in data)
    lines = _table(["Split", "Examples", "Vulnerable", "Not vulnerable"], rows) + [""]
    for key, label in (("sources", "Sources"), ("languages", "Languages")):
        lines.append(f"- {label} ({first}): " + ", ".join(f"{name} {count:,}" for name, count in sorted(data[first][key].items(), key=lambda kv: -kv[1])))
    return lines + ["- The prompts and the target rationale are in Italian."]


def _formats(ctx: dict) -> tuple[list[str], dict[str, dict]]:
    rows = {r["format"]: r for r in ctx["results"]}
    return sorted(set(ctx["formats"]) | set(rows), key=_order), rows


def _evaluation(ctx: dict) -> list[str]:
    if not ctx["results"]:
        return ["not available: no benchmark results were found (the benchmark was skipped or failed for every format)."]
    names, rows = _formats(ctx)
    table = []
    for name in names:
        row = rows.get(name)
        if row is None:
            table.append([name, "not measured"] + ["n/d"] * 5)
            continue
        m = (row.get("report") or {}).get("all") or {}
        table.append([name, _fmt(m.get("precision"), ".3f"), _fmt(m.get("recall"), ".3f"), _fmt(m.get("f1", row.get("f1")), ".3f"),
                      _fmt(m.get("fpr"), ".3f"), _fmt(row.get("invalid"), "d"), _fmt(row.get("n"), "d")])
    n = next((r.get("n") for r in ctx["results"] if r.get("n") is not None), None)
    return (_table(["Format", "Precision", "Recall", "F1", "FPR", "Invalid outputs", "Examples"], table) + [
        "",
        f"Evaluated on `data/test.jsonl` ({_fmt(n, 'd')} examples per format) with temperature 0, JSON-schema guided decoding and thinking disabled. "
        "The positive class is \"vulnerable\"; an invalid output counts as \"not vulnerable\"."] + _image(ctx, "quality", "Precision, recall, F1 and FPR by format"))


def _efficiency(ctx: dict) -> list[str]:
    if not ctx["results"]:
        return ["not available: no benchmark results were found (the benchmark was skipped or failed for every format)."]
    names, rows = _formats(ctx)
    table = []
    for name in names:
        row = rows.get(name)
        if row is None:
            table.append([name, "not measured"] + ["n/d"] * 7)
            continue
        table.append([name, _fmt(row.get("latency_s"), ".2f"), _fmt(row.get("latency_p50_s"), ".2f"), _fmt(row.get("latency_p95_s"), ".2f"),
                      _fmt(row.get("tok_s"), ".1f"), _fmt(row.get("weights_gb"), ".1f"), _fmt(row.get("kv_tokens"), ",d"),
                      _fmt(row.get("vram_gb"), ".1f"), _fmt(row.get("disk_gb"), ".1f")])
    concurrency = next((r["concurrency"] for r in ctx["results"] if r.get("concurrency") is not None), None)
    version = next((r["vllm_version"] for r in ctx["results"] if r.get("vllm_version")), None) or ctx["vllm_version"]
    lines = _table(["Format", "Mean latency (s)", "p50 (s)", "p95 (s)", "Throughput (tok/s)", "Weights in VRAM (GiB)", "KV cache (tokens)",
                    "VRAM peak (GB)", "Disk (GB)"], table)
    lines += ["", f"Latency is per-request latency under load (concurrency {_fmt(concurrency)}): requests run concurrently, so it is higher than on an idle server.",
              f"Single run on {_fmt(ctx['gpu'])} with vLLM {_fmt(version)}; every format is served with the same settings and evaluated on the same examples."]
    quant = []
    for name in names:
        q = (ctx["quant"].get(name) or {}).get("quantization") or {}
        if q:
            extra = [f"KV cache {q['kv_cache_quant_algo']}"] if q.get("kv_cache_quant_algo") else []
            extra += [f"group size {q['group_size']}"] if q.get("group_size") else []
            quant.append(f"- `{name}`: " + ", ".join([str(q.get("quant_algo", "n/d"))] + extra))
    lines += [""] + quant if quant else []  # riga vuota prima dell'elenco, solo se c'è
    return lines + _image(ctx, "latency_throughput", "Latency and throughput by format") + _image(ctx, "memory", "Memory footprint by format")


def _files(ctx: dict) -> list[str]:
    lines = ["- `MODEL_CARD.md` and `model_card_assets/`: this card and its charts."]
    lines += [f"- `model_{fmt}/`: {FORMAT_NOTES[fmt]}." for fmt in ctx["formats"]]
    lines += ["- `adapter/`: the unmerged DoRA adapter.", "- `benchmark_results/`: raw benchmark output and vLLM logs.",
              "- `train_colab.log`, `colab_freeze.txt`, `colab_commit.txt`: training log, installed package versions and trained commit.",
              "",
              "To reproduce: run `notebooks/train_colab.ipynb` on a GPU runtime. The training section runs `ai-pipeline/train.py`, section 10 runs "
              "`quantize.py`, and section 11 runs `benchmark.py hardware` and builds this card."]
    return lines


def render_card(ctx: dict) -> str:
    """MODEL_CARD.md: front matter YAML e sette sezioni; un input mancante dà "not available: ..." senza errori."""
    sections = (("Summary", _summary(ctx)), ("Intended use and limitations", _intended_use()), ("Training", _training(ctx)), ("Data", _data(ctx)),
                ("Evaluation", _evaluation(ctx)), ("Efficiency", _efficiency(ctx)), ("Files and reproduction", _files(ctx)))
    parts = _front_matter(ctx) + ["", "# Tunatron SAST triage model", ""]
    for title, body in sections:
        parts += [f"## {title}", "", *body, ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- grafici (matplotlib, importato solo qui)
DPI = 150
WIDTH_IN = 9
SLOT_PX = 26      # una barra da 24 px più 2 px di spazio (guida dataviz: barre <= 24 px, gap di 2 px)
GROUP_GAP = 0.6   # spazio fra gruppi di barre, in barre
TOP_PX, BOTTOM_PX = 84, 112
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Slot 1-3 della palette categorica di riferimento di dataviz, validati a coppie (validate_palette.js --pairs all): un colore per formato
# in tutti i grafici. Il contrasto di aqua sulla superficie è sotto 3:1: lo compensano i valori scritti in punta alle barre e le tabelle.
FORMAT_COLORS = {"bf16": "#2a78d6", "fp8": "#eb6834", "nvfp4": "#1baf7a"}
LOSS_COLORS = ("#2a78d6", "#eb6834")
ASSETS_DIR = "model_card_assets"

CHART_TITLES = {
    "loss_curves": "Training and validation loss",
    "loss_x": "Epoch",
    "loss_y": "Loss on the labelled tokens (the JSON verdict)",
    "train": "Training loss",
    "eval": "Validation loss",
    "quality": "Detection quality on the test set",
    "quality_note": "Precision, recall and F1: higher is better. False positive rate: lower is better.",
    "precision": "Precision", "recall": "Recall", "f1": "F1", "fpr": "False positive rate",
    "latency_throughput": "Latency and throughput",
    "latency": "Request latency under load (s), lower is better",
    "throughput": "Generation throughput (tokens/s), higher is better",
    "mean": "Mean", "p50": "p50", "p95": "p95",
    "memory": "Memory and disk footprint",
    "weights": "Weights in VRAM (GiB)",
    "disk": "Disk size (GB)",
    "kv": "KV cache capacity (tokens)",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _color(fmt: str) -> str:
    return FORMAT_COLORS.get(fmt, MUTED)


def _quality(row: dict) -> dict:
    m = (row.get("report") or {}).get("all") or {}
    return {"precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1", row.get("f1")), "fpr": m.get("fpr")}


def _start_figure(plt, slots: float, title: str, subtitle: str | None = None):
    """Figura con l'altezza che fa stare ogni barra in SLOT_PX pixel; ritorna (figura, altezza in pixel)."""
    height_px = TOP_PX + slots * SLOT_PX + BOTTOM_PX
    fig = plt.figure(figsize=(WIDTH_IN, height_px / DPI), dpi=DPI, facecolor=SURFACE)
    fig.text(0.02, 1 - 24 / height_px, title, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.02, 1 - 46 / height_px, subtitle, ha="left", va="center", fontsize=8.5, color=INK_2)
    return fig, height_px


def _axes(fig, height_px: float, panels: int, left_px: int):
    width_px = WIDTH_IN * DPI
    grid = {"left": left_px / width_px, "right": 1 - 30 / width_px, "bottom": BOTTOM_PX / height_px, "top": 1 - TOP_PX / height_px, "wspace": 0.4}
    return fig.subplots(1, panels, gridspec_kw=grid, squeeze=False)[0]


def _tick(x: float, _pos=None) -> str:
    """Tick pieno sotto 10.000, compatto (50K) da lì in su: una sola forma per asse (guida dataviz)."""
    return f"{x / 1000:g}K" if abs(x) >= 10_000 else f"{x:,.10g}"


def _style(ax, title: str | None = None) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="y", length=0, colors=INK_2, labelsize=8.5)
    ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=INK_2)


def _hbars(ax, groups: list[str], series: list[tuple[str, str, list]], fmt: str, span: float | None = None, xmax: float | None = None,
           title: str | None = None) -> None:
    """Barre orizzontali raggruppate: una per (gruppo, serie), valore scritto in punta. `series` = (nome, colore, valori per gruppo);
    un valore None non disegna la barra. Con un solo gruppo le barre prendono il nome della serie sull'asse."""
    n = len(series)
    centers, last = [], 0.0
    for g in range(len(groups)):
        base = g * (n + GROUP_GAP)
        centers.append(base + (n - 1) / 2)
        last = base + n - 1
        for k, (_, color, values) in enumerate(series):
            if values[g] is not None:
                ax.barh(base + k, values[g], height=24 / SLOT_PX, color=color, linewidth=0)
                ax.text(values[g], base + k, " " + format(values[g], fmt), va="center", ha="left", fontsize=8, color=INK_2)
    top = max((v for _, _, values in series for v in values if v is not None), default=1) or 1
    ax.set_xlim(0, xmax * 1.15 if xmax else top * 1.22)  # spazio per l'etichetta in punta
    ax.set_ylim(-0.5 + (span if span is not None else last + 1), -0.5)  # prima barra in alto; `span` allinea i pannelli più bassi
    if len(groups) == 1:
        ax.set_yticks(range(n))
        ax.set_yticklabels([name for name, _, _ in series])
    else:
        ax.set_yticks(centers)
        ax.set_yticklabels(groups)
    _style(ax, title)


def _slots(groups: int, series: int) -> float:
    return groups * series + (groups - 1) * GROUP_GAP


def _legend(fig, height_px: float, names: list[str]) -> None:
    if len(names) < 2:  # una sola serie: il titolo basta
        return
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(facecolor=_color(n), label=n) for n in names], loc="lower center", ncol=len(names), frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, 12 / height_px))


def _chart_loss(plt, ctx: dict):
    train_curve, eval_curve = ctx["curves"]["train"], ctx["curves"]["eval"]
    if not train_curve and not eval_curve:
        return None
    height_px = 4.2 * DPI
    width_px = WIDTH_IN * DPI
    fig = plt.figure(figsize=(WIDTH_IN, 4.2), dpi=DPI, facecolor=SURFACE)
    fig.text(0.02, 1 - 24 / height_px, CHART_TITLES["loss_curves"], ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    ax = fig.add_axes([90 / width_px, 80 / height_px, 1 - 150 / width_px, 1 - 160 / height_px])
    for points, color, label, marker in ((train_curve, LOSS_COLORS[0], CHART_TITLES["train"], None), (eval_curve, LOSS_COLORS[1], CHART_TITLES["eval"], "o")):
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1.6, label=label, solid_capstyle="round", solid_joinstyle="round", marker=marker, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.margins(x=0.06)
    ax.set_xlabel(CHART_TITLES["loss_x"], fontsize=8.5, color=INK_2)
    ax.set_ylabel(CHART_TITLES["loss_y"], fontsize=8.5, color=INK_2)
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(BASELINE)
    if train_curve and eval_curve:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    return fig


def _chart_quality(plt, ctx: dict):
    rows = [(r["format"], _quality(r)) for r in ctx["results"]]
    metrics = [key for key in ("precision", "recall", "f1", "fpr") if any(q[key] is not None for _, q in rows)]
    if not metrics:
        return None
    series = [(name, _color(name), [q[key] for key in metrics]) for name, q in rows]
    fig, height_px = _start_figure(plt, _slots(len(metrics), len(series)), CHART_TITLES["quality"], CHART_TITLES["quality_note"])
    (ax,) = _axes(fig, height_px, 1, left_px=185)
    _hbars(ax, [CHART_TITLES[key] for key in metrics], series, ".3f", xmax=1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _legend(fig, height_px, [name for name, _ in rows])
    return fig


def _chart_latency(plt, ctx: dict):
    rows = ctx["results"]
    latency = [(CHART_TITLES[label], key) for label, key in (("mean", "latency_s"), ("p50", "latency_p50_s"), ("p95", "latency_p95_s"))
               if any(r.get(key) is not None for r in rows)]
    has_throughput = any(r.get("tok_s") is not None for r in rows)
    if not latency and not has_throughput:
        return None
    slots = max(_slots(len(latency), len(rows)) if latency else 0, len(rows) if has_throughput else 0)
    fig, height_px = _start_figure(plt, slots, CHART_TITLES["latency_throughput"])
    axes = iter(_axes(fig, height_px, bool(latency) + has_throughput, left_px=70))
    if latency:
        series = [(r["format"], _color(r["format"]), [r.get(key) for _, key in latency]) for r in rows]
        _hbars(next(axes), [name for name, _ in latency], series, ".2f", span=slots, title=CHART_TITLES["latency"])
    if has_throughput:
        series = [(r["format"], _color(r["format"]), [r.get("tok_s")]) for r in rows]
        _hbars(next(axes), [""], series, ".1f", span=slots, title=CHART_TITLES["throughput"])
    _legend(fig, height_px, [r["format"] for r in rows])
    return fig


def _chart_memory(plt, ctx: dict):
    rows = ctx["results"]
    panels = [(CHART_TITLES[label], key, fmt) for label, key, fmt in (("weights", "weights_gb", ".1f"), ("disk", "disk_gb", ".1f"), ("kv", "kv_tokens", ",.0f"))
              if any(r.get(key) is not None for r in rows)]
    if not panels:
        return None
    fig, height_px = _start_figure(plt, len(rows), CHART_TITLES["memory"])
    for ax, (title, key, fmt) in zip(_axes(fig, height_px, len(panels), left_px=70), panels):
        _hbars(ax, [""], [(r["format"], _color(r["format"]), [r.get(key)]) for r in rows], fmt, title=title)
    _legend(fig, height_px, [r["format"] for r in rows])
    return fig


def make_charts(ctx: dict, assets_dir: Path) -> dict[str, str]:
    """Scrive i PNG possibili in `assets_dir`; ritorna nome -> percorso relativo alla card. Senza matplotlib ritorna {}."""
    try:
        plt = _pyplot()
    except ImportError:
        return {}
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    charts = {}
    for name, draw in (("loss_curves", _chart_loss), ("quality", _chart_quality), ("latency_throughput", _chart_latency), ("memory", _chart_memory)):
        try:  # un grafico che non si disegna non deve far perdere la card né gli altri grafici
            fig = draw(plt, ctx)
            if fig is None:
                continue
            fig.savefig(assets_dir / f"{name}.png", dpi=DPI, facecolor=SURFACE)
            plt.close(fig)
        except Exception as exc:
            plt.close("all")
            print(f"ATTENZIONE: grafico {name} non disegnato ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        charts[name] = f"{assets_dir.name}/{name}.png"
    return charts


# --------------------------------------------------------------------------- riga di comando
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="cartella di MODEL_CARD.md e di model_card_assets/")
    ap.add_argument("--log", type=Path, default=None, help="log di train.py (curve di loss e tempo di training)")
    ap.add_argument("--results", type=Path, default=Path("results"), help="cartella con <formato>/hardware.json di benchmark.py")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--adapter", type=Path, default=None, help="cartella dell'adapter (adapter_config.json)")
    ap.add_argument("--models", nargs="*", default=[], metavar="FORMATO=DIR", help="es. bf16=model_bf16 fp8=model_fp8 nvfp4=model_nvfp4")
    ap.add_argument("--commit-file", type=Path, default=None)
    ap.add_argument("--freeze-file", type=Path, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--train-args", default=None,
                    help="argomenti con cui è stato lanciato train.py; in forma --train-args=\"--epochs 2\" (un valore che inizia con -- ma non ha spazi sarebbe preso per un'opzione)")
    ap.add_argument("--vllm-version", default=None, help="usata se i risultati non riportano la propria versione")
    args = ap.parse_args(argv)
    args.model_dirs = {}
    for spec in args.models:
        fmt, sep, path = spec.partition("=")
        if not sep or fmt not in FORMATS:
            ap.error(f"--models vuole FORMATO=DIR con FORMATO fra {', '.join(FORMATS)}, non {spec!r}")
        args.model_dirs[fmt] = Path(path)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = collect_context(log=args.log, results=args.results, data=args.data, adapter=args.adapter, models=args.model_dirs,
                          commit_file=args.commit_file, freeze_file=args.freeze_file, gpu=args.gpu, train_args=args.train_args,
                          vllm_version=args.vllm_version)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets = args.out_dir / ASSETS_DIR
    shutil.rmtree(assets, ignore_errors=True)  # niente grafici di una corsa precedente
    ctx["charts"] = make_charts(ctx, assets)
    if not ctx["charts"] and (ctx["results"] or ctx["curves"]["train"] or ctx["curves"]["eval"]):
        print("ATTENZIONE: nessun grafico disegnato (matplotlib non installato?): la card esce senza immagini", file=sys.stderr)
    card = args.out_dir / "MODEL_CARD.md"
    card.write_text(render_card(ctx), encoding="utf-8")
    print(f"model card scritta in {card}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
