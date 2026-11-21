#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible API on :8000 (run inside WSL).
# Usage: bash scripts/serve_vllm.sh <model_dir> [served_name] [extra vllm args...]
#   bash scripts/serve_vllm.sh models/sqlforge-3b-v2-step1014 sqlforge
#   bash scripts/serve_vllm.sh models/sqlforge-3b-awq sqlforge-awq
# Sized for an 8 GB laptop GPU (Windows keeps ~1.1 GB for the desktop): bf16 3B weights (5.8 GB) leave no
# room for KV cache, so serve FP8 (--quantization fp8) or AWQ. 8k context: some BIRD prompts exceed 4k.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; NAME=${2:-sqlforge}; shift $(( $# >= 2 ? 2 : 1 ))
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
exec ~/venvs/serve/bin/vllm serve "$MODEL" --served-model-name "$NAME" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.82 \
  --max-num-seqs 32 --enable-prefix-caching "$@"
