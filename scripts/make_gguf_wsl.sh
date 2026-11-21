#!/usr/bin/env bash
# HF safetensors -> GGUF F16 -> Q8_0 / Q4_K_M (llama.cpp tools, CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=models/sqlforge-3b-v2-step1014; OUT=models/gguf; mkdir -p $OUT
[ -f $OUT/sqlforge-3b-f16.gguf ] || ~/venvs/gguf/bin/python ~/llama.cpp/convert_hf_to_gguf.py $SRC \
    --outtype f16 --outfile $OUT/sqlforge-3b-f16.gguf
for q in Q8_0 Q4_K_M; do
  [ -f $OUT/sqlforge-3b-$q.gguf ] || ~/llama.cpp/build/bin/llama-quantize $OUT/sqlforge-3b-f16.gguf $OUT/sqlforge-3b-$q.gguf $q 8
done
ls -la $OUT
