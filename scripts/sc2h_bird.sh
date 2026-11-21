#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
N=sqlforge3b_step956
$PY -m eval.self_correct --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_bird_dev500.jsonl --model models/sqlforge-3b-step956 --rounds 2 --batch-size 2 --timeout 60 --out outputs/preds/${N}_sc2h_bird_dev500.jsonl
$PY -m eval.run_eval --data data/processed/bird_dev.jsonl --pred outputs/preds/${N}_sc2h_bird_dev500.jsonl --out outputs/eval/${N}_sc2h_bird_dev500.json --timeout 60
echo ALL_DONE
