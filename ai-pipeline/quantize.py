#!/usr/bin/env python3
"""Quantizzazione post-training del modello BF16 fuso in FP8 e NVFP4 (Blackwell) con NVIDIA Model Optimizer (ModelOpt).

    python quantize.py                     # entrambi i formati
    python quantize.py --only nvfp4        # un solo formato

Serve `nvidia-modelopt[hf] >= 0.48.0` (scripts/setup_env.sh lo installa in .venv-quant). Usa 512 esempi di calibrazione presi
da data/train.jsonl (conversazioni complete, thinking spento). I due formati vengono prodotti uno alla volta e la GPU viene
liberata fra l'uno e l'altro.

Risultato: i TRE modelli da confrontare, ciascuno in una cartella che vLLM carica da sola (nessun flag di quantizzazione):
    model_bf16/    modello fine-tuned fuso, precisione piena (lo produce train.py)
    model_fp8/     FP8 W8A8 + KV cache FP8
    model_nvfp4/   NVFP4 (FP4 con scale per blocchi da 16) con FP8 su alcuni layer, per i Tensor Core Blackwell

Le ricette sono quelle di NVIDIA:
  NVFP4  la ricetta che ModelOpt pubblica per questo modello, caricata così com'è:
         modelopt_recipes/models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16.yaml
         (linear in NVFP4 W4A4; `down_proj` di 9 layer MLP in FP8; embedding NVFP4 W4A16; lm_head FP8 W8A16; conv1d e norme BF16)
  FP8    la ricetta generica FP8 + KV FP8 di ModelOpt (come il checkpoint NVIDIA-Nemotron-3-Nano-4B-FP8), con in BF16 i moduli
         elencati in `exclude_modules` del checkpoint ufficiale: lm_head, conv1d, attention e il Mamba che precede ogni attention.
         (ModelOpt non ha una ricetta FP8 per il 4B: le esclusioni sono quelle pubblicate, verificate in tests/test_quantize.py.)
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import re
import sys
from pathlib import Path

import common

MODELOPT_MIN_VERSION = (0, 48, 0)  # prima versione su PyPI che include la ricetta del 4B
NVFP4_RECIPE = "models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16"
FP8_RECIPE = "general/ptq/fp8_default-kv_fp8"
# Moduli Linear per tipo di layer del pattern `hybrid_override_pattern` di NemotronH: M = Mamba-2, * = attention, - = MLP.
LAYER_LINEARS = {"M": ["in_proj", "out_proj"], "*": ["q_proj", "k_proj", "v_proj", "o_proj"], "-": ["up_proj", "down_proj"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("model_bf16"))
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--out-fp8", type=Path, default=Path("model_fp8"))
    ap.add_argument("--out-nvfp4", type=Path, default=Path("model_nvfp4"))
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=4096, help="token per campione di calibrazione (se la VRAM non basta: 2048)")
    ap.add_argument("--only", choices=["fp8", "nvfp4"], default=None)
    ap.add_argument("--bf16-embeddings", action="store_true",
                    help="NVFP4: tiene la tabella degli embedding in BF16 invece che in NVFP4 W4A16 (ricetta NVIDIA). Da usare solo se vLLM "
                         "non carica il checkpoint: il suo loader ModelOpt non gestisce gli embedding quantizzati (vedi README)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.formats = [args.only] if args.only else ["fp8", "nvfp4"]
    return args


# --------------------------------------------------------------------------- helper puri (testati)
def exclusion_modules(pattern: str, prefix: str = "backbone", preceding_mamba: bool = True) -> list[str]:
    """Nomi completi dei moduli che restano in BF16 nel FP8, nello stesso formato di `exclude_modules` di NVIDIA.

    Riproduce esattamente la lista del checkpoint ufficiale NVIDIA-Nemotron-3-Nano-4B-FP8 e, con `preceding_mamba=False`,
    quella del NVIDIA-Nemotron-Nano-9B-v2-FP8: lm_head, conv1d di ogni Mamba, q/k/v/o di ogni attention e (solo 4B) in/out_proj
    del Mamba che precede ogni attention.
    """
    out = {"lm_head"}

    def layer(i: int, leaves: list[str]) -> None:
        out.update(f"{prefix}.layers.{i}.mixer.{leaf}" for leaf in leaves)

    for i, kind in enumerate(pattern):
        if kind == "M":
            layer(i, ["conv1d"])
        elif kind == "*":
            layer(i, LAYER_LINEARS["*"])
            if preceding_mamba and i > 0 and pattern[i - 1] == "M":
                layer(i - 1, LAYER_LINEARS["M"])
    return sorted(out)


def disable_entries(modules: list[str]) -> list[dict]:
    """Regole ModelOpt che spengono i quantizzatori (pesi e attivazioni) dei moduli dati.

    Il prefisso del checkpoint si toglie: `backbone.layers.N` nei file NVIDIA, `model.layers.N` nel NemotronH nativo di
    transformers 5. Il punto dopo il numero impedisce a `layers.1.` di prendere anche `layers.11.`.
    """
    rules = []
    for name in modules:
        tail = "layers." + name.split(".layers.", 1)[1] if ".layers." in name else name
        rules.append({"quantizer_name": f"*{tail}*", "enable": False})
    return rules


def fp8_quant_cfg(base: dict, pattern: str) -> dict:
    """Ricetta FP8 generica di ModelOpt + le esclusioni NVIDIA, in coda (l'ultima regola che corrisponde vince)."""
    cfg = copy.deepcopy(base)
    cfg["quant_cfg"] = list(cfg["quant_cfg"]) + disable_entries(exclusion_modules(pattern))
    return cfg


def nvfp4_quant_cfg(base: dict, bf16_embeddings: bool = False) -> dict:
    """Ricetta NVFP4 di NVIDIA per il 4B, invariata; con `bf16_embeddings` una regola in coda spegne la quantizzazione degli embedding."""
    cfg = copy.deepcopy(base)
    if bf16_embeddings:
        cfg["quant_cfg"] = list(cfg["quant_cfg"]) + [{"quantizer_name": "*embeddings*weight_quantizer", "enable": False}]
    return cfg


def check_modelopt_version(version: str) -> None:
    nums = [int(m) for m in re.findall(r"\d+", version.split("rc")[0].split("dev")[0])[:3]]
    nums += [0] * (3 - len(nums))
    if tuple(nums) < MODELOPT_MIN_VERSION:
        need = ".".join(map(str, MODELOPT_MIN_VERSION))
        raise RuntimeError(f"nvidia-modelopt {version} è troppo vecchio: la ricetta NVIDIA per il 4B c'è dalla {need} (uv pip install -U 'nvidia-modelopt[hf]>={need}')")


def read_hybrid_pattern(model_dir: Path) -> str:
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("hybrid_override_pattern"):
        raise RuntimeError(f"{model_dir}/config.json senza hybrid_override_pattern: non è un NemotronH?")
    return cfg["hybrid_override_pattern"]


def dtype_kwarg(transformers_version: str, dtype) -> dict:
    """transformers >= 4.56 usa `dtype`, le versioni precedenti `torch_dtype`."""
    major, minor = (int(p) for p in transformers_version.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def check_loading_info(info: dict, show: int = 8) -> None:
    """Ferma la quantizzazione se transformers ha dovuto inventare dei pesi.

    Se i nomi dei pesi e quelli della classe del modello non coincidono (es. `model.*` contro `backbone.*`) il caricamento riesce lo
    stesso: i tensori mancanti restano inizializzati a caso, transformers avvisa soltanto, e calibrazione ed export "riuscirebbero"
    su un modello rotto. Le chiavi in più (`unexpected_keys`) sono pesi ignorati: stesso sintomo, ma qui basta il loro conteggio.
    """
    missing = sorted(info.get("missing_keys") or ())
    if missing:
        head = ", ".join(missing[:show]) + (", ..." if len(missing) > show else "")
        raise RuntimeError(f"{len(missing)} pesi del modello non sono nel checkpoint e resterebbero casuali: {head}")


def check_layer_count(num_hidden_layers: int, pattern: str) -> None:
    """Il config caricato deve descrivere tanti layer quanti ne dichiara `hybrid_override_pattern` del config.json."""
    if num_hidden_layers != len(pattern):
        raise RuntimeError(f"il modello ha {num_hidden_layers} layer ma hybrid_override_pattern ne descrive {len(pattern)}: config.json non coerente")


def fix_legacy_tied_weights_keys(modules, tie_word_embeddings: bool) -> list[str]:
    """Porta `_tied_weights_keys` del modeling remoto NVIDIA al formato di transformers 5. Restituisce le classi corrette.

    Il modeling remoto del 4B dichiara `_tied_weights_keys = ["lm_head.weight"]` (una lista, convenzione di transformers 4);
    transformers 5 si aspetta un dict `{peso legato: sorgente}` e in `save_pretrained`, cioè nell'export di ModelOpt, chiama
    `.keys()` sulla lista (AttributeError). Il 4B ha `tie_word_embeddings=False`: nessun peso è legato e il valore giusto è `{}`
    (come nel NemotronH nativo). Se invece fosse legato, la lista non dice a quale sorgente: meglio fermarsi che inventarla.
    """
    fixed = []
    for m in modules:
        keys = getattr(m, "_tied_weights_keys", None)
        if keys is None or hasattr(keys, "keys"):
            continue
        if keys and tie_word_embeddings:
            raise RuntimeError(f"{type(m).__name__}._tied_weights_keys è una lista ({list(keys)}) ma gli embedding sono legati: serve il dict {{peso: sorgente}} di transformers 5")
        m._tied_weights_keys = {}
        fixed.append(type(m).__name__)
    return fixed


def select_calibration(path: Path, n: int, seed: int) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh if line.strip()]
    picked = random.Random(seed).sample(range(len(examples)), min(n, len(examples)))
    return [examples[i] for i in picked]


def calibration_texts(examples: list[dict], tokenizer) -> list[str]:
    """Conversazioni complete: in generazione le attivazioni includono sia il prompt sia il JSON prodotto."""
    return [common.chat_text(tokenizer, ex["messages"]) for ex in examples]


def verify_checkpoint(path: Path) -> dict:
    """Legge `hf_quant_config.json` scritto da ModelOpt: formato, KV cache, moduli esclusi, layer per formato (serve a vLLM)."""
    file = Path(path) / "hf_quant_config.json"
    if not file.exists():
        raise RuntimeError(f"{path}: manca hf_quant_config.json (l'export ModelOpt non è andato a buon fine)")
    q = json.loads(file.read_text(encoding="utf-8")).get("quantization") or {}
    if not q.get("quant_algo"):
        raise RuntimeError(f"{file}: nessun quant_algo (checkpoint non quantizzato?)")
    by_algo: dict[str, int] = {}
    for layer in (q.get("quantized_layers") or {}).values():
        algo = str(layer.get("quant_algo", "?"))
        by_algo[algo] = by_algo.get(algo, 0) + 1
    return {"quant_algo": q["quant_algo"], "kv_cache_quant_algo": q.get("kv_cache_quant_algo"),
            "excluded": len(q.get("exclude_modules") or []), "layers_by_algo": by_algo}


# --------------------------------------------------------------------------- quantizzazione (richiede GPU)
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import importlib.metadata

    check_modelopt_version(importlib.metadata.version("nvidia-modelopt"))

    import torch
    import transformers
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_recipe
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Il merge di Unsloth salva il config nativo di transformers 5 accanto al codice remoto NVIDIA, che non lo sa leggere: va riscritto
    # PRIMA di qualunque from_pretrained (anche AutoTokenizer legge il config).
    if common.normalize_nemotron_config(args.model):
        print(f"{args.model}/config.json riscritto nella forma NVIDIA (hybrid_override_pattern al posto di layers_block_type)", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    texts = calibration_texts(select_calibration(args.data, args.num_samples, args.seed), tokenizer)
    calib = [tokenizer(t, add_special_tokens=False, truncation=True, max_length=args.max_seq_length)["input_ids"] for t in texts]
    pattern = read_hybrid_pattern(args.model)

    for fmt in args.formats:
        out = args.out_fp8 if fmt == "fp8" else args.out_nvfp4
        base = load_recipe(FP8_RECIPE if fmt == "fp8" else NVFP4_RECIPE).quantize.model_dump()
        cfg = fp8_quant_cfg(base, pattern) if fmt == "fp8" else nvfp4_quant_cfg(base, args.bf16_embeddings)
        print(f"== {fmt}: {len(cfg['quant_cfg'])} regole ModelOpt -> {out}", file=sys.stderr)

        model, info = AutoModelForCausalLM.from_pretrained(
            str(args.model), device_map="cuda", trust_remote_code=True, output_loading_info=True,
            **dtype_kwarg(transformers.__version__, torch.bfloat16),
        )
        check_loading_info(info)  # pesi mancanti = inizializzati a caso: la calibrazione "riuscirebbe" su un modello rotto
        check_layer_count(model.config.num_hidden_layers, pattern)
        fixed = fix_legacy_tied_weights_keys(model.modules(), model.config.tie_word_embeddings)  # senza, l'export fallisce in save_pretrained
        if fixed:
            print(f"_tied_weights_keys portato al formato di transformers 5 su: {', '.join(fixed)}", file=sys.stderr)
        if info.get("unexpected_keys"):
            print(f"ATTENZIONE: {len(info['unexpected_keys'])} pesi del checkpoint ignorati dal modello", file=sys.stderr)
        model.eval()  # in training i kernel Mamba fusi leggono out_proj.weight direttamente e il suo quantizzatore resterebbe non calibrato

        def forward_loop(m):
            with torch.no_grad():
                for ids in calib:
                    m(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)

        mtq.quantize(model, cfg, forward_loop=forward_loop)
        export_hf_checkpoint(model, export_dir=str(out))
        tokenizer.save_pretrained(str(out))
        common.copy_remote_code(args.model, out)
        print(f"{fmt}: {verify_checkpoint(out)}", file=sys.stderr)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
