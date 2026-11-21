#!/usr/bin/env bash
# Accuracy through an OpenAI-compatible server (vLLM :8000 / Ollama :11434), client run inside WSL.
# (On Windows, Smart App Control blocks the openai package's compiled dependency.)
# Usage: bash scripts/eval_api_wsl.sh <name> <base_url> <served_model> [concurrency]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; URL=$2; MODEL=$3; C=${4:-32}
PY=~/venvs/serve/bin/python; export PYTHONUNBUFFERED=1
$PY -m eval.generate --backend openai --base-url $URL --model $MODEL --data data/processed/spider_dev.jsonl \
    --batch-size $C --out outputs/preds/${NAME}_spider_dev.jsonl
$PY -m eval.generate --backend openai --base-url $URL --model $MODEL --data data/processed/bird_dev.jsonl \
    --limit 500 --batch-size $C --out outputs/preds/${NAME}_bird_dev500.jsonl
echo GENERATED
