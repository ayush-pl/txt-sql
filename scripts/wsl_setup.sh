#!/usr/bin/env bash
# One-time setup of the WSL2 training/serving environment.
# Usage (inside WSL):  bash scripts/wsl_setup.sh
set -euo pipefail

VENV="${VENV:-$HOME/venvs/sqlforge}"
UV="$HOME/.local/bin/uv"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$VENV" ] || "$UV" venv --python 3.12 "$VENV"
export VIRTUAL_ENV="$VENV"

# RTX 50-series (Blackwell, sm_120) needs CUDA 12.8+ wheels
"$UV" pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$VENV/bin/python" -e "$REPO[train,dev]"

"$VENV/bin/python" "$REPO/scripts/check_env.py"
