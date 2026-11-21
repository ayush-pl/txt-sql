#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
~/venvs/quant/bin/python -m serving.quantize_awq --model models/sqlforge-3b-v2-step1014 --out models/sqlforge-3b-awq
du -sh models/sqlforge-3b-awq
