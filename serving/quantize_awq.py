"""AWQ 4-bit weight quantization (W4A16, group size 128) with llm-compressor, for vLLM.

Calibration size matters on an 8 GB GPU: 256 x 2048 tokens filled VRAM to 7.9 GB and made the
grid search ~13 h; 64 x 1024 (the usual AWQ budget, ~65k tokens) keeps it small and fast.
Calibration uses our own SFT prompts (schema + question + SQL), not generic web text: AWQ picks
per-channel scales from the activations it sees, so domain calibration protects the channels
that matter for SQL. Output is compressed-tensors format, which vLLM loads directly.

Run inside WSL in the quant venv (scripts/wsl_setup_quant.sh):
    python -m serving.quantize_awq --model models/sqlforge-3b-v2-step1014 --out models/sqlforge-3b-awq
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="data/processed/sft_spider_bird/train.jsonl")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(REPO / args.calib, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    # balanced Spider/BIRD calibration set, full conversations (prompt + gold SQL)
    texts = [tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False) for r in rows[: args.n_calib]]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=args.max_len, add_special_tokens=False),
        batched=True, remove_columns=["text"],
    )

    # loaded on CPU: llm-compressor onloads one decoder layer at a time to the GPU,
    # so calibration fits on an 8 GB card even though bf16 weights are ~6 GB
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    recipe = [AWQModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head"])]
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_len,
            num_calibration_samples=len(texts), sequential_targets=["Qwen2DecoderLayer"])
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    print(f"AWQ W4A16 model -> {args.out}")


if __name__ == "__main__":
    main()
