#!/usr/bin/env bash
# Accuracy of GGUF models served by Ollama on Windows (stdlib HTTP client; Ollama on 127.0.0.1:11434).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
for m in "$@"; do
  n=ollama_${m//-/_}
  $PY -m eval.generate --backend http --base-url http://127.0.0.1:11434/v1 --model $m --data data/processed/spider_dev.jsonl --batch-size 8 --out outputs/preds/${n}_spider_dev.jsonl
  $PY -m eval.run_eval --data data/processed/spider_dev.jsonl --pred outputs/preds/${n}_spider_dev.jsonl --out outputs/eval/${n}_spider_dev.json
  $PY -m eval.generate --backend http --base-url http://127.0.0.1:11434/v1 --model $m --data data/processed/bird_dev.jsonl --limit 500 --batch-size 8 --out outputs/preds/${n}_bird_dev500.jsonl
  $PY -m eval.run_eval --data data/processed/bird_dev.jsonl --pred outputs/preds/${n}_bird_dev500.jsonl --out outputs/eval/${n}_bird_dev500.json --timeout 60
done
echo ALL_DONE
