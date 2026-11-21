#!/usr/bin/env bash
# Launch SFT inside WSL. Extra args are passed through to training.sft_train.
# Usage: bash scripts/train.sh --config training/configs/qlora_3b.yaml [--set key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # weights come from the local cache
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$HOME/venvs/sqlforge/bin/python" -m training.sft_train "$@"
