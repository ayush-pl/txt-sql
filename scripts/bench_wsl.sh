#!/usr/bin/env bash
# Latency/throughput benchmark against a running server, Spider + BIRD prompts (client inside WSL).
# Usage: bash scripts/bench_wsl.sh <name> <base_url> <served_model>
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/venvs/serve/bin/python; export PYTHONUNBUFFERED=1
$PY -m serving.benchmark --base-url $2 --model $3 --data data/processed/spider_dev.jsonl --n 200 --concurrency 1 8 32 --name ${1}_spider
$PY -m serving.benchmark --base-url $2 --model $3 --data data/processed/bird_dev.jsonl --n 200 --concurrency 1 8 32 --name ${1}_bird
