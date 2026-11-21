"""QLoRA supervised fine-tuning with TRL, tracked in MLflow. Run inside WSL.

Usage:
    python -m training.sft_train --config training/configs/qlora_3b.yaml
    python -m training.sft_train --config training/configs/qlora_3b.yaml --set max_steps=30 base_model=Qwen/Qwen2.5-Coder-1.5B-Instruct
Resumes automatically from the latest checkpoint in output_dir.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    # View with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{REPO / 'mlflow.db'}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sqlforge-sft")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    data_dir = REPO / cfg["data_dir"]
    ds = load_dataset(
        "json",
        data_files={"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "val.jsonl")},
    )
    select_ex = cfg.get("select_metric", "loss") == "ex"
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if cfg["load_in_4bit"]
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    out_dir = REPO / cfg["output_dir"]
    eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    total_steps = (
        cfg["max_steps"] if cfg["max_steps"] > 0
        else math.ceil(len(ds["train"]) / eff_batch) * cfg["num_train_epochs"]
    )
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))
    print(f"train={len(ds['train'])} eff_batch={eff_batch} total_steps={total_steps} warmup={warmup_steps}")
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=True,
        max_length=cfg["max_length"],
        completion_only_loss=True,      # loss on the SQL only, not the schema/question
        packing=False,
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        # run 1 lesson: val loss picked the wrong checkpoint; pick by execution accuracy instead
        metric_for_best_model="eval_ex" if select_ex else "eval_loss",
        greater_is_better=select_ex,
        report_to=["mlflow"],
        seed=cfg["seed"],
        dataloader_num_workers=2,
    )

    callbacks = []
    if select_ex:
        from training.ex_callback import ExecutionAccuracyCallback, load_ex_examples

        ex_rows = load_ex_examples(data_dir / "val.jsonl", cfg.get("ex_eval_samples", 150))
        if not ex_rows:
            raise SystemExit("select_metric=ex needs db_path in val.jsonl: rerun training.build_sft_data")
        callbacks.append(ExecutionAccuracyCallback(ex_rows, tok, batch_size=cfg.get("ex_batch_size", 2)))
        print(f"checkpoint selection: execution accuracy on {len(ex_rows)} val examples")

    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=ds["train"], eval_dataset=ds["validation"],
        processing_class=tok, peft_config=peft_cfg, callbacks=callbacks,
    )
    trainer.model.print_trainable_parameters()

    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    final = out_dir / "final_adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"saved adapter -> {final}")


if __name__ == "__main__":
    main()
