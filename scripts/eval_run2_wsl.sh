#!/usr/bin/env bash
# Run 2: merge candidate adapters and generate predictions inside WSL (stable on this laptop;
# Windows-side generation at high VRAM triggered hardware-error reboots). Resumable: rerun to continue.
# Scoring (CPU only) happens on Windows afterwards with eval.run_eval.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=~/venvs/sqlforge/bin/python
CK=outputs/checkpoints/qlora-3b-r16-v2
for pair in "v2-step400:$CK/final_adapter" "v2-step1014:$CK/checkpoint-1014"; do
  tag=${pair%%:*}; ad=${pair#*:}
  [ -f models/sqlforge-3b-$tag/config.json ] || $PY -m training.merge_and_export --adapter $ad --out models/sqlforge-3b-$tag
done
for tag in v2-step400 v2-step1014; do
  $PY -m eval.generate --data data/processed/spider_dev.jsonl --model models/sqlforge-3b-$tag --batch-size 8 \
      --max-gpu-gb 7 --max-batch-tokens 6144 --out outputs/preds/sqlforge3b_${tag}_spider_dev.jsonl
done
for tag in v2-step400 v2-step1014; do
  $PY -m eval.generate --data data/processed/bird_dev.jsonl --model models/sqlforge-3b-$tag --limit 500 --batch-size 4 \
      --max-gpu-gb 7 --max-batch-tokens 4096 --out outputs/preds/sqlforge3b_${tag}_bird_dev500.jsonl
done
echo ALL_GENERATED
