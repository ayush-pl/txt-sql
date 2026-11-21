#!/usr/bin/env bash
# One-time: llama.cpp converter + CPU build of llama-quantize, outside the repo (~/llama.cpp).
# cmake comes from PyPI into a small venv, so no sudo/apt is needed.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=600
[ -d ~/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
git -C ~/llama.cpp log -1 --format="llama.cpp commit %h %cd"
[ -d ~/venvs/gguf ] || uv venv --python 3.12 ~/venvs/gguf
uv pip install --python ~/venvs/gguf/bin/python cmake numpy sentencepiece safetensors "transformers>=4.45" \
    torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
uv pip install --python ~/venvs/gguf/bin/python -e ~/llama.cpp/gguf-py
cd ~/llama.cpp
~/venvs/gguf/bin/cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
~/venvs/gguf/bin/cmake --build build --target llama-quantize -j 8
ls -la build/bin/llama-quantize
