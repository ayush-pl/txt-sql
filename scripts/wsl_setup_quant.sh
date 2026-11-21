#!/usr/bin/env bash
# One-time: separate venv for llm-compressor (pins its own torch/transformers versions).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2
[ -d ~/venvs/quant ] || uv venv --python 3.12 ~/venvs/quant
uv pip install --python ~/venvs/quant/bin/python llmcompressor datasets
uv pip install --python ~/venvs/quant/bin/python -e .
~/venvs/quant/bin/python -c "import llmcompressor, torch, transformers; print('llmcompressor', llmcompressor.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.is_available())"
