#!/usr/bin/env bash
# Self-correction post-pass + scoring on Spider dev and BIRD-500.
# Usage: bash scripts/eval_selfcorrect.sh <model_tag> <run_tag> [extra self_correct args...]
#   bash scripts/eval_selfcorrect.sh step956 sc2h            # schema hints + retry sampling (default)
#   bash scripts/eval_selfcorrect.sh step956 sc2 --no-hints --retry-temp 0
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
M=models/sqlforge-3b-$1; N=sqlforge3b_$1; T=$2; shift 2
$PY -m eval.self_correct --data data/processed/spider_dev.jsonl --pred outputs/preds/${N}_spider_dev.jsonl --model $M --rounds 2 --batch-size 4 --out outputs/preds/${N}_${T}_spider_dev.jsonl "$@"
$PY -m eval.run_eval --data data/processed/spider_dev.jsonl --pred outputs/preds/${N}_${T}_spider_dev.jsonl --out outputs/eval/${N}_${T}_spider_dev.json
$PY -m eval.self_correct --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_bird_dev500.jsonl --model $M --rounds 2 --batch-size 2 --timeout 60 --out outputs/preds/${N}_${T}_bird_dev500.jsonl "$@"
$PY -m eval.run_eval --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_${T}_bird_dev500.jsonl --out outputs/eval/${N}_${T}_bird_dev500.json --timeout 60
echo ALL_DONE
