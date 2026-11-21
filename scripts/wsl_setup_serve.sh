#!/usr/bin/env bash
# One-time: separate venv for vLLM serving (vLLM pins its own torch; keeps the training env intact).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2
[ -d ~/venvs/serve ] || uv venv --python 3.12 ~/venvs/serve
uv pip install --python ~/venvs/serve/bin/python vllm
uv pip install --python ~/venvs/serve/bin/python -e ".[dev]" openai aiohttp
~/venvs/serve/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch.cuda.get_arch_list()[-3:])"
