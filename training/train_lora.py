"""QLoRA fine-tuning of a small instruct LLM to list PII spans as JSON.

    python training/train_lora.py configs/train/r16.yaml [--dry-run] [--max-steps N]

TRL SFTTrainer on prompt/completion pairs (loss on the completion only), PEFT LoRA
on the layers in `lora.target_modules`, 4-bit NF4 base weights with bf16 compute
when `quantization: nf4` (needs CUDA + bitsandbytes). Writes the adapter plus
run_info.json (GPU, peak VRAM, wall time, tokens/s, trainable params, adapter MB)
to `output_dir`.

--dry-run prints one tokenized example with its loss mask and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import yaml

from pii_gateway.detectors.llm import format_completion, prompt_text
from pii_gateway.spans import read_examples

ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("output_dir", f"outputs/{Path(path).stem}")
    return cfg


def make_dataset(path: str, tokenizer, limit: int | None = None):
    from datasets import Dataset

    rows = [
        {
            "prompt": prompt_text(tokenizer, ex.text),
            "completion": format_completion(ex.text, ex.spans),
        }
        for ex in read_examples(path, limit)
    ]
    return Dataset.from_list(rows)


def target_modules(spec):
    if spec == "attention":
        return ATTENTION_ONLY
    return spec  # "all-linear" or an explicit list


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(args.output_dir or cfg["output_dir"])

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = make_dataset(cfg["train_file"], tok, cfg.get("train_limit"))
    eval_ds = (
        make_dataset(cfg["eval_file"], tok, cfg.get("eval_limit", 200))
        if cfg.get("eval_file")
        else None
    )

    lens = [
        len(tok(r["prompt"] + r["completion"])["input_ids"])
        for r in train_ds.select(range(min(500, len(train_ds))))
    ]
    print(
        f"train examples: {len(train_ds)}; tokens/example (first 500): "
        f"mean {sum(lens) / len(lens):.0f}, max {max(lens)}"
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        max_length=cfg.get("max_seq_length", 1024),
        completion_only_loss=cfg.get("completion_only_loss", True),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg.get("batch_size", 8),
        per_device_eval_batch_size=cfg.get("batch_size", 8),
        gradient_accumulation_steps=cfg.get("grad_accum", 2),
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # < 1 means a fraction of total steps
        logging_steps=10,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", 100),
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=torch.cuda.is_available(),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=out_dir.name,
        seed=cfg.get("seed", 13),
    )

    if args.dry_run:
        trainer = SFTTrainer(
            model=cfg["base_model"] if not torch.cuda.is_available() else _load_model(cfg),
            args=sft_cfg,
            train_dataset=train_ds.select(range(2)),
            eval_dataset=eval_ds.select(range(2)) if eval_ds is not None else None,
            processing_class=tok,
            peft_config=_lora(cfg, LoraConfig),
        )
        row = trainer.train_dataset[0]
        ids = row["input_ids"]
        if "labels" in row:  # TRL >= 1.x: prompt positions are -100
            mask = [int(lab != -100) for lab in row["labels"]]
        else:
            mask = row.get("completion_mask") or row.get("assistant_masks") or [1] * len(ids)
        print("=== full example ===")
        print(tok.decode(ids))
        print("=== tokens with loss ===")
        print(repr(tok.decode([i for i, m in zip(ids, mask, strict=True) if m])))
        print(f"eos_token={tok.eos_token!r} pad_token={tok.pad_token!r}")
        return

    model = (
        _load_model(cfg)
        if torch.cuda.is_available()
        else AutoModelForCausalLM.from_pretrained(cfg["base_model"])
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=_lora(cfg, LoraConfig),
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    adapter_mb = sum(f.stat().st_size for f in out_dir.glob("adapter_model*")) / 2**20
    n_tokens = sum(len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in train_ds)
    epochs = cfg.get("epochs", 1) if args.max_steps < 0 else None
    info = {
        "config": cfg,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "wall_clock_min": round(wall / 60, 2),
        "train_tokens": n_tokens * (epochs or 1),
        "tokens_per_sec": round(n_tokens * (epochs or 1) / wall) if epochs else None,
        "trainable_params": trainable,
        "total_params": total,
        "adapter_mb": round(adapter_mb, 1),
        "train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "gpu_price_per_hour": cfg.get("gpu_price_per_hour"),
    }
    if cfg.get("gpu_price_per_hour"):
        info["cost_usd"] = round(cfg["gpu_price_per_hour"] * wall / 3600, 2)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(
        json.dumps({k: v for k, v in info.items() if k not in ("log_history", "config")}, indent=2)
    )


def _lora(cfg: dict, LoraConfig):
    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc.get("alpha", 2 * lc["r"]),
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=target_modules(lc.get("target_modules", "all-linear")),
        task_type="CAUSAL_LM",
    )


def _load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw = {"dtype": torch.bfloat16, "attn_implementation": cfg.get("attn_implementation", "sdpa")}
    if cfg.get("quantization") == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw)
    if cfg.get("quantization") == "nf4":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


if __name__ == "__main__":
    main()
