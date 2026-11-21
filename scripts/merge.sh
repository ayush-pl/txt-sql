#!/usr/bin/env bash
# Usage: bash scripts/merge.sh <adapter_dir> <out_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
exec ~/venvs/sqlforge/bin/python -m training.merge_and_export --adapter "$1" --out "$2"
