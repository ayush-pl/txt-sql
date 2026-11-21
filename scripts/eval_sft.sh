#!/usr/bin/env bash
# Evaluate merged SFT models on Spider dev (full) and BIRD dev (500 subset). Windows .venv.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
for tag in "$@"; do
  m=models/sqlforge-3b-$tag; n=sqlforge3b_$tag
  $PY -m eval.generate --data data/processed/spider_dev.jsonl --model $m --batch-size 4 --out outputs/preds/${n}_spider_dev.jsonl
  $PY -m eval.run_eval --data data/processed/spider_dev.jsonl --pred outputs/preds/${n}_spider_dev.jsonl --out outputs/eval/${n}_spider_dev.json
  $PY -m eval.generate --data data/processed/bird_dev.jsonl --model $m --limit 500 --batch-size 2 --out outputs/preds/${n}_bird_dev500.jsonl
  $PY -m eval.run_eval --data data/processed/bird_dev.jsonl --pred outputs/preds/${n}_bird_dev500.jsonl --out outputs/eval/${n}_bird_dev500.json --timeout 60
done
echo ALL_DONE
