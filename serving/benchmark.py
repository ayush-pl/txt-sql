"""Latency / throughput benchmark for any OpenAI-compatible server (vLLM, Ollama, llama.cpp).

Real workload: the exact text-to-SQL prompts used for evaluation (schema DDL + sample rows +
question), streamed so time-to-first-token (TTFT) is measured, at several concurrency levels.

Per concurrency level it reports:
  - TTFT p50/p95            prefill latency (dominated by the long schema prompt)
  - latency p50/p95         full request time
  - decode tok/s per request (output tokens / (latency - TTFT))
  - throughput              requests/s and output tokens/s across all clients
  - peak GPU memory         sampled with nvidia-smi during the run

Usage:
    python -m serving.benchmark --base-url http://localhost:8000/v1 --model sqlforge \
        --data data/processed/spider_dev.jsonl --n 200 --concurrency 1 8 32 --name vllm-bf16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

from data.prepare import load_jsonl
from eval.generate import make_messages

REPO = Path(__file__).resolve().parent.parent


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[k]


class GpuSampler(threading.Thread):
    """Peak GPU memory (MiB) while the benchmark runs."""

    def __init__(self, every: float = 0.5):
        super().__init__(daemon=True)
        self.every, self.peak, self._stop = every, 0, threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.split()
                self.peak = max(self.peak, int(out[0]))
            except Exception:
                pass
            self._stop.wait(self.every)

    def stop(self) -> int:
        self._stop.set()
        return self.peak


async def one_request(client, model: str, messages: list[dict], max_tokens: int) -> dict:
    if isinstance(client, HTTPClient):
        return await asyncio.to_thread(http_request, client.base_url, model, messages, max_tokens)
    t0 = time.perf_counter()
    ttft, n_chunks, usage, text = None, 0, None, []
    stream = await client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_chunks += 1
            text.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - t0
    out_tokens = usage.completion_tokens if usage else n_chunks
    prompt_tokens = usage.prompt_tokens if usage else None
    return {"ttft": ttft if ttft is not None else total, "latency": total, "out_tokens": out_tokens,
            "prompt_tokens": prompt_tokens, "text": "".join(text)}


async def run_level(client, model: str, prompts: list[list[dict]], concurrency: int, max_tokens: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def worker(msgs):
        async with sem:
            results.append(await one_request(client, model, msgs, max_tokens))

    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    await asyncio.gather(*(worker(m) for m in prompts))
    wall = time.perf_counter() - t0
    peak = sampler.stop()

    ttft = [r["ttft"] for r in results]
    lat = [r["latency"] for r in results]
    decode = [r["out_tokens"] / (r["latency"] - r["ttft"]) for r in results if r["latency"] - r["ttft"] > 1e-3 and r["out_tokens"] > 1]
    out_tok = sum(r["out_tokens"] for r in results)
    ptoks = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    return {
        "concurrency": concurrency, "n": len(results), "wall_s": round(wall, 2),
        "ttft_p50_ms": round(1000 * pct(ttft, 0.5)), "ttft_p95_ms": round(1000 * pct(ttft, 0.95)),
        "latency_p50_ms": round(1000 * pct(lat, 0.5)), "latency_p95_ms": round(1000 * pct(lat, 0.95)),
        "decode_tok_s_per_req": round(statistics.median(decode), 1) if decode else None,
        "throughput_req_s": round(len(results) / wall, 2),
        "throughput_out_tok_s": round(out_tok / wall, 1),
        "mean_prompt_tokens": round(statistics.mean(ptoks)) if ptoks else None,
        "mean_out_tokens": round(out_tok / len(results), 1),
        "peak_gpu_mib": peak or None,
    }


def http_request(base_url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """Streaming request with the standard library only (Server-Sent Events), same measurements
    as one_request(); used where the openai package is unavailable (Windows Smart App Control)."""
    import http.client
    from urllib.parse import urlparse

    u = urlparse(base_url.rstrip("/") + "/chat/completions")
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}})
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    conn.request("POST", u.path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft, n_chunks, usage, text = None, 0, None, []
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        ch = chunk.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_chunks += 1
            text.append(ch[0]["delta"]["content"])
    conn.close()
    total = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else total, "latency": total,
            "out_tokens": usage["completion_tokens"] if usage else n_chunks,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None, "text": "".join(text)}


class HTTPClient:
    """Adapter so run_level() can drive the stdlib client from asyncio via threads."""

    def __init__(self, base_url: str):
        self.base_url = base_url


async def amain(args) -> None:
    if args.client == "http":
        client = HTTPClient(args.base_url)
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600, max_retries=0)

    data = load_jsonl(args.data)
    random.Random(42).shuffle(data)
    data = data[: args.n]
    prompts = [make_messages(r, [], args.sample_rows) for r in data]

    # warm-up: first requests pay CUDA-graph / cache allocation costs
    await asyncio.gather(*(one_request(client, args.model, p, args.max_tokens) for p in prompts[:4]))

    levels = []
    for c in args.concurrency:
        res = await run_level(client, args.model, prompts, c, args.max_tokens)
        levels.append(res)
        print(json.dumps(res))
    out = {"name": args.name, "base_url": args.base_url, "model": args.model, "data": args.data, "levels": levels}
    path = REPO / "outputs" / "bench" / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    from rich.console import Console
    from rich.table import Table

    t = Table(title=f"{args.name}  ({len(prompts)} prompts, {Path(args.data).stem})")
    cols = ["concurrency", "ttft_p50_ms", "ttft_p95_ms", "latency_p50_ms", "latency_p95_ms",
            "decode_tok_s_per_req", "throughput_req_s", "throughput_out_tok_s", "peak_gpu_mib"]
    for c in cols:
        t.add_column(c.replace("_", " "))
    for lv in levels:
        t.add_row(*(str(lv[c]) for c in cols))
    Console().print(t)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/processed/spider_dev.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--name", required=True)
    ap.add_argument("--client", choices=["openai", "http"], default="openai",
                    help="http = stdlib streaming client (no openai package needed)")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
