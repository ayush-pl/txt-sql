"""SQLForge live demo on Modal: the Gradio page from space/demo.py, scaled to zero when idle.

The container starts on the first visit (about 1-2 min; the weights are cached in a Modal Volume
after the first run) and shuts down after 5 idle minutes, so you only pay while someone is using it.
One container at most, which caps the cost.

Hardware (SQLFORGE_HW at deploy time):
  - "L4":  vLLM with FP8 weights, the production setup; well under 1 s per answer.
  - "T4":  vLLM in fp16 (T4 has no FP8/bf16); cheaper GPU.
  - "cpu" (default): the 4-bit GGUF with llama.cpp on 4 cores; about 4-15 s per answer.
Modal asks for a payment method on the account before it runs GPU functions.

    pip install modal && modal setup                    # one-time: log in (opens the browser)
    modal deploy deploy/modal_app.py                    # prints the public https://...modal.run URL
    $env:SQLFORGE_HW="L4"; modal deploy deploy/modal_app.py   # GPU version
    modal app logs sqlforge-demo                        # follow the logs
    modal app stop sqlforge-demo                        # take it down
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
MODEL_REPO = "ay2r/sqlforge-3b"  # public Hub repo with the merged model and gguf/
HW = os.environ.get("SQLFORGE_HW", "cpu")  # the live demo runs on CPU (no card on the Modal account)
LLAMA_WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/"

COMMON = ["fastapi>=0.115", "sqlglot>=25", "rank-bm25>=0.2", "pandas>=2.2", "tqdm>=4.66", "huggingface_hub>=0.25"]
if HW == "cpu":
    packages = [*COMMON, "gradio==5.50.0", "llama-cpp-python==0.3.36"]
    hw_env = {"SQLFORGE_BACKEND": "llama_cpp", "N_THREADS": "4"}
    fn_hw = {"cpu": 4.0, "memory": 6144}
else:
    # Gradio 5 caps Pillow < 12 and vLLM 0.34 needs Pillow >= 13, so the GPU image uses Gradio 6
    packages = [*COMMON, "gradio>=6", "vllm==0.34.0"]
    quant = {"L4": {"VLLM_QUANT": "fp8"}, "T4": {"VLLM_QUANT": "", "VLLM_DTYPE": "half"}}[HW]
    hw_env = {"SQLFORGE_BACKEND": "vllm", "GPU_NAME": HW, "VLLM_CACHE_ROOT": "/cache/vllm", **quant}
    fn_hw = {"gpu": HW}


def _not_python(p: Path) -> bool:
    return p.suffix != ".py"


image = (
    modal.Image.debian_slim(python_version="3.12")
    # prebuilt CPU wheels for llama-cpp-python (find-links: only that page is fetched, not a whole index)
    .uv_pip_install(*packages, find_links=LLAMA_WHEELS if HW == "cpu" else None)
    .env({
        "MODEL_REPO": MODEL_REPO,
        "HF_HOME": "/cache/hf",  # on the Volume: download the weights once
        "DB_DIR": "/srv/dbs",
        "GITHUB_URL": "https://github.com/ayush-pl/txt-sql",
        **hw_env,
    })
    # only the pipeline modules the demo imports (no data/raw, no training code)
    .add_local_dir(REPO / "app", "/srv/app", ignore=_not_python)
    .add_local_dir(REPO / "data", "/srv/data", ignore=_not_python)
    .add_local_dir(REPO / "eval", "/srv/eval", ignore=_not_python)
    .add_local_dir(REPO / "eval" / "gate" / "dbs", "/srv/dbs")
    .add_local_file(REPO / "space" / "demo.py", "/srv/demo.py")
    .add_local_file(REPO / "space" / "suggestions.json", "/srv/suggestions.json")
)

cache = modal.Volume.from_name("sqlforge-cache", create_if_missing=True)
app = modal.App("sqlforge-demo")


@app.function(
    image=image,
    volumes={"/cache": cache},
    min_containers=0,      # scale to zero when idle
    max_containers=1,      # one container at most: caps the cost, and Gradio sessions stay on one container
    scaledown_window=300,  # keep warm for 5 min after the last request
    timeout=600,
    **fn_hw,
)
@modal.concurrent(max_inputs=32)  # Gradio queues the generations itself
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/srv")
    import demo  # loads the model
    import gradio as gr
    from fastapi import FastAPI

    demo.demo.queue(max_size=20)
    return gr.mount_gradio_app(FastAPI(), demo.demo, path="/")
