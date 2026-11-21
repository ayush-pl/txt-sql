"""Merge a LoRA adapter into its bf16 base model and save a standalone model
(used for evaluation, vLLM serving and quantization).

Usage:
    python -m training.merge_and_export --adapter outputs/checkpoints/qlora-3b-r16/final_adapter \
        --out models/sqlforge-3b
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu is safest on 8 GB VRAM")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = json.load(open(f"{args.adapter}/adapter_config.json"))["base_model_name_or_path"]
    print(f"base: {base_id}")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map={"": args.device})
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
