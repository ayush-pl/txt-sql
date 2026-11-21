#!/usr/bin/env bash
# Self-correction post-pass + scoring for the step-956 model on Spider dev and BIRD-500.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
M=models/sqlforge-3b-step956; N=sqlforge3b_step956
$PY -m eval.self_correct --data data/processed/spider_dev.jsonl --pred outputs/preds/${N}_spider_dev.jsonl --model $M --rounds 2 --batch-size 4 --out outputs/preds/${N}_sc2_spider_dev.jsonl
$PY -m eval.run_eval --data data/processed/spider_dev.jsonl --pred outputs/preds/${N}_sc2_spider_dev.jsonl --out outputs/eval/${N}_sc2_spider_dev.json
$PY -m eval.self_correct --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_bird_dev500.jsonl --model $M --rounds 2 --batch-size 2 --timeout 60 --out outputs/preds/${N}_sc2_bird_dev500.jsonl
$PY -m eval.run_eval --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_sc2_bird_dev500.jsonl --out outputs/eval/${N}_sc2_bird_dev500.json --timeout 60
echo ALL_DONE
