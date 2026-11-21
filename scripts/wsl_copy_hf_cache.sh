#!/usr/bin/env bash
# Copy already-downloaded HF models from the Windows cache into the WSL cache
# (WSL downloads can be flaky; Windows already has the weights).
# Usage (inside WSL): bash scripts/wsl_copy_hf_cache.sh Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct
set -euo pipefail
WIN_CACHE="${WIN_CACHE:-/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/.cache/huggingface/hub}"
DST="$HOME/.cache/huggingface/hub"
mkdir -p "$DST"
for repo in "$@"; do
  name="models--${repo//\//--}"
  [ -d "$WIN_CACHE/$name" ] || { echo "missing in Windows cache: $repo"; exit 1; }
  rm -rf "${DST:?}/$name"
  cp -rL "$WIN_CACHE/$name" "$DST/"
  du -sh "$DST/$name"
done
