#!/usr/bin/env python3
"""Fine-tuning DoRA 16-bit di Nemotron-3-Nano-4B con Unsloth + merge in BF16.

    python train.py --dry-run          # controlla dati e parametri, senza GPU né import pesanti
    python train.py                    # addestra (GPU) e salva il modello fuso in ./model_bf16

Parametri di default = piano (r=64, alpha=64, DoRA, adamw_8bit, batch 2 x accumulo 4, seq 8192); caricamento e opzioni del
modello come nel notebook ufficiale Unsloth per Nemotron-3-Nano (trust_remote_code, unsloth_force_compile, attention "eager").
Se la VRAM (16 GB) non basta: --batch-size 1 --grad-accum 8 e/o --max-seq-length 4096.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
from pathlib import Path

import common

DEFAULT_MODEL = "unsloth/NVIDIA-Nemotron-3-Nano-4B"
# Linear di NemotronH su cui si applica DoRA:  attention q/k/v/o_proj · MLP up/down_proj (relu², non esiste gate_proj) · Mamba in_proj.
# Esclusi di proposito:
#   out_proj, conv1d  PEFT (>= 0.18) li rifiuta nei modelli Mamba: i kernel fusi leggono out_proj.weight / conv1d.weight
#                     direttamente, quindi l'adapter verrebbe ignorato in silenzio. Anche il notebook Unsloth li elenca, ma è
#                     precedente a quel controllo.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj"]
CHARS_PER_TOKEN = 3.5  # stima prudente per codice, usata solo dal dry-run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("model_bf16"), help="modello fuso BF16")
    ap.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"), help="adapter DoRA non fuso (backup)")
    ap.add_argument("--run-dir", type=Path, default=Path("outputs/trainer"), help="checkpoint del Trainer")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--attn-implementation", default="eager", help="come nel notebook Unsloth; se manca VRAM prova sdpa")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="se > 0 sostituisce --epochs (utile per una prova breve)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--dry-run", action="store_true", help="solo controllo di dati e parametri")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- helper puri (testati)
def unmatched_targets(module_names: list[str], targets: list[str]) -> list[str]:
    """Target che non corrispondono all'ultimo segmento di nessun modulo."""
    leaves = {name.rsplit(".", 1)[-1] for name in module_names}
    return [t for t in targets if t not in leaves]


def lora_targets(module_names: list[str]) -> list[str]:
    """I target di TARGET_MODULES che il modello ha davvero (stesso ordine)."""
    leaves = {name.rsplit(".", 1)[-1] for name in module_names}
    found = [t for t in TARGET_MODULES if t in leaves]
    if not found:
        raise RuntimeError("il modello non ha nessuno dei moduli target di NemotronH: è il modello giusto?")
    return found


def dora_is_active(model) -> bool:
    """DoRA è davvero attivo solo se la config lo dichiara E esistono i vettori di magnitudine."""
    configs = getattr(model, "peft_config", None)
    if not configs or not any(getattr(c, "use_dora", False) for c in configs.values()):
        return False
    named = getattr(model, "named_parameters", None)
    return named is not None and any("lora_magnitude_vector" in name for name, _ in named())


def fused_compute_loss(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Loss del causal LM calcolata solo sulle posizioni con etichetta, con la testa del modello fusa di TRL.

    Il `compute_loss` che Unsloth compila per SFTTrainer chiama il modello senza `fused_lm_head=True`, quindi materializza i logit
    di tutta la sequenza: con 131.072 parole di vocabolario sono gigabyte (OOM sui 16 GB già al primo passo con 2 x ~2.700 token).
    Con `train_on_responses_only` le etichette sono solo il JSON del verdetto (~200 token): la testa si applica a quelli.
    Stessa normalizzazione di TRL: somma della NLL / token con etichetta (o `num_items_in_batch` con l'accumulo dei gradienti).
    """
    import torch

    outputs = model(**{**inputs, "use_cache": False}, fused_lm_head=True)
    count = outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
    loss = (-outputs.log_probs).sum() / torch.as_tensor(count).clamp(min=1)
    return (loss, outputs) if return_outputs else loss


def patch_dora_dead_rows(eps: float = 1e-6) -> None:
    """Evita il NaN di DoRA sulle righe di pesi interamente a zero.

    In 8 layer Mamba di NemotronH `in_proj` ha righe morte (es. 25 su 17.504 nel layer 0). DoRA calcola `magnitudine / norma_riga`
    (peft/tuners/lora/dora.py, nessun epsilon): 0/0 = NaN, che si propaga a tutti i layer (loss e gradienti NaN al primo passo).
    La norma è portata ad almeno `eps`, nel solo punto da cui passano inizializzazione, forward e merge: per una riga morta
    la magnitudine parte da `eps` e la riga resta praticamente nulla; le altre righe non cambiano. Idempotente.
    """
    from peft.tuners.lora import dora

    cls = dora.DoraLinearLayer
    original = cls.get_weight_norm
    if getattr(original, "_dead_rows_patched", False):
        return

    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name=None):
        return original(self, weight, lora_weight, scaling, adapter_name).clamp_min(eps)

    get_weight_norm._dead_rows_patched = True
    cls.get_weight_norm = get_weight_norm


def first_supported(factory, options: list[dict]) -> dict:
    """Primo set di kwargs i cui nomi esistono nella firma di `factory` (TRL/transformers rinominano gli argomenti)."""
    params = inspect.signature(factory).parameters
    for option in options:
        if all(key in params for key in option):
            return option
    raise ValueError(f"nessuna variante di argomenti supportata da {factory}: {options}")


def fits(text: str, tokenizer, max_len: int) -> bool:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_len


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def dry_run(args: argparse.Namespace) -> dict:
    """Riepilogo di dati e parametri senza importare torch/unsloth."""
    summary = {"model": args.model, "effective_batch_size": args.batch_size * args.grad_accum, "max_seq_length": args.max_seq_length}
    for name, path in (("train", args.train), ("val", args.val)):
        count, labels, longest = 0, {}, 0
        for ex in _iter_jsonl(path):
            count += 1
            label = str(ex.get("meta", {}).get("label", "?"))
            labels[label] = labels.get(label, 0) + 1
            longest = max(longest, sum(len(m["content"]) for m in ex["messages"]))
        summary[f"{name}_examples"] = count
        summary[f"{name}_labels"] = labels
        summary[f"{name}_longest_chars"] = longest
        summary[f"{name}_longest_tokens_estimate"] = int(longest / CHARS_PER_TOKEN)
    return summary


# --------------------------------------------------------------------------- addestramento (richiede GPU)
def tokenized_dataset(path: Path, tokenizer, max_len: int):
    from datasets import Dataset

    ids, kept, dropped = [], 0, 0
    for ex in _iter_jsonl(path):
        text = common.chat_text(tokenizer, ex["messages"])
        if not fits(text, tokenizer, max_len):
            dropped += 1
            continue
        enc = tokenizer(text, add_special_tokens=False)
        ids.append({"input_ids": enc["input_ids"], "attention_mask": [1] * len(enc["input_ids"])})
        kept += 1
    print(f"{path}: {kept} esempi, {dropped} scartati perché > {max_len} token", file=sys.stderr)
    return Dataset.from_list(ids)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args), indent=2))
        return 0

    # unsloth va importato per primo (patcha transformers/trl).
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation=args.attn_implementation,
    )
    targets = lora_targets([n for n, _ in model.named_modules()])
    patch_dora_dead_rows()  # senza, DoRA dà NaN sulle righe morte di in_proj (vedi la funzione)
    print(f"DoRA r={args.rank} alpha={args.alpha} su: {targets}", file=sys.stderr)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=targets,
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        use_dora=True,  # inoltrato da Unsloth a peft.LoraConfig (verificato nel sorgente); controllato subito sotto
    )
    if not dora_is_active(model):
        raise RuntimeError("DoRA non risulta attivo (use_dora ignorato da Unsloth/PEFT?): interrompo per non addestrare un LoRA semplice.")

    train_ds = tokenized_dataset(args.train, tokenizer, args.max_seq_length)
    val_ds = tokenized_dataset(args.val, tokenizer, args.max_seq_length)

    cfg_kwargs = dict(
        output_dir=str(args.run_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim=args.optim,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        seed=args.seed,
        report_to="none",
        packing=False,
        prediction_loss_only=True,  # in valutazione serve solo la loss: niente log-prob per token da accumulare
    )
    cfg_kwargs.update(first_supported(SFTConfig, [{"eval_strategy": "steps"}, {"evaluation_strategy": "steps"}]))
    cfg_kwargs.update(first_supported(SFTConfig, [{"max_length": args.max_seq_length}, {"max_seq_length": args.max_seq_length}]))
    try:
        cfg_kwargs.update(first_supported(SFTConfig, [{"dataset_kwargs": {"skip_prepare_dataset": True}}]))
    except ValueError:
        pass  # versioni vecchie riconoscono da sole i dataset già tokenizzati
    trainer_kwargs = first_supported(SFTTrainer, [{"processing_class": tokenizer}, {"tokenizer": tokenizer}])
    trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=SFTConfig(**cfg_kwargs), **trainer_kwargs)

    # Loss solo sulla risposta dell'assistente (il JSON): marker verificati sul template reale.
    trainer = train_on_responses_only(trainer, instruction_part=common.INSTRUCTION_PART, response_part=common.RESPONSE_PART)
    trainer.compute_loss = types.MethodType(fused_compute_loss, trainer)  # vedi fused_compute_loss: senza, OOM al primo passo
    print("Loss: testa fusa, solo sui token con etichetta (il JSON del verdetto).", file=sys.stderr)
    sample = trainer.train_dataset[0]["labels"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print("Parte addestrata (primo esempio):\n" + tokenizer.decode([pad if t == -100 else t for t in sample]), file=sys.stderr)

    trainer.train()

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_dir))
    model.save_pretrained_merged(str(args.out_dir), tokenizer, save_method="merged_16bit")
    copy_remote_code_from_hub(args.model, args.out_dir)
    print(f"Modello BF16 fuso salvato in {args.out_dir}", file=sys.stderr)
    return 0


def copy_remote_code_from_hub(model_id: str, out_dir: Path) -> None:
    """vLLM con --trust-remote-code cerca configuration/modeling nella cartella del modello: li copio dal repo base."""
    try:
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_id, allow_patterns=["*.py"]))
        copied = common.copy_remote_code(src, out_dir)
        if copied:
            print(f"copiati i file del codice remoto: {copied}", file=sys.stderr)
    except Exception as exc:  # non deve far perdere un addestramento già concluso
        print(f"ATTENZIONE: codice remoto non copiato ({exc}); copialo a mano da {model_id}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
