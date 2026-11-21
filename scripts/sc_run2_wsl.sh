#!/usr/bin/env bash
# Self-correction (schema hints, 2 rounds) on the final run-2 model, generated inside WSL. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=~/venvs/sqlforge/bin/python; M=models/sqlforge-3b-v2-step1014; N=sqlforge3b_v2-step1014
$PY -m eval.self_correct --data data/processed/spider_dev.jsonl --pred outputs/preds/${N}_spider_dev.jsonl --model $M \
    --rounds 2 --batch-size 4 --out outputs/preds/${N}_sc2h_spider_dev.jsonl
$PY -m eval.self_correct --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_bird_dev500.jsonl --model $M \
    --rounds 2 --batch-size 2 --timeout 60 --out outputs/preds/${N}_sc2h_bird_dev500.jsonl
echo ALL_DONE
