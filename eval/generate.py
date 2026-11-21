"""Generate SQL predictions for a prepared JSONL split.

Backends:
  hf      local transformers model (base or merged fine-tune), batched greedy decoding
  openai  any OpenAI-compatible server: vLLM, Ollama (http://localhost:11434/v1), API models

Output is resumable: already-predicted ids in --out are skipped.

Examples:
    python -m eval.generate --data data/processed/spider_dev.jsonl --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --out outputs/preds/qwen1.5b_zeroshot_spider_dev.jsonl --limit 200
    python -m eval.generate --backend openai --base-url http://localhost:8000/v1 --model sqlforge ...
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from data.prepare import load_jsonl
from data.prompt import build_messages, build_user_prompt, extract_sql
from data.schema import serialize_schema

REPO = Path(__file__).resolve().parent.parent


def build_fewshot(train_path: Path, k: int, exclude_dbs: set[str], seed: int = 0) -> list[dict]:
    """k demos from distinct training DBs (never a dev DB), schemas without sample rows."""
    if k <= 0:
        return []
    pool = [r for r in load_jsonl(train_path) if r["db_id"] not in exclude_dbs]
    random.Random(seed).shuffle(pool)
    demos, seen = [], set()
    for r in pool:
        if r["db_id"] in seen or r["difficulty"] not in ("medium", "hard"):
            continue
        seen.add(r["db_id"])
        schema = serialize_schema(str(REPO / r["db_path"]), sample_rows=0)
        demos += [
            {"role": "user", "content": build_user_prompt(schema, r["question"])},
            {"role": "assistant", "content": r["gold_sql"]},
        ]
        if len(seen) == k:
            break
    return demos


def make_messages(rec: dict, demos: list[dict], sample_rows: int, link_k: int = 0) -> list[dict]:
    db = str(REPO / rec["db_path"])
    if link_k > 0:
        from data.schema_linker import linked_schema

        schema = linked_schema(db, rec["question"], rec.get("evidence"), k=link_k, sample_rows=sample_rows)
    else:
        schema = serialize_schema(db, sample_rows=sample_rows)
    msgs = build_messages(schema, rec["question"], rec.get("evidence"))
    return [msgs[0], *demos, msgs[1]]


def token_batches(order: list[int], lens: list[int], max_batch: int, max_tokens: int, new_tokens: int) -> list[list[int]]:
    """Group length-sorted indices so each batch stays under a token budget:
    short prompts go many at a time, long ones alone. Bounds peak GPU memory."""
    batches, cur = [], []
    for i in order:
        width = max([lens[j] for j in cur] + [lens[i]]) + new_tokens
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * width > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


class HFBackend:
    def __init__(self, model_id: str, max_new_tokens: int, max_prompt_tokens: int, max_gpu_gb: float = 7.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        # Hard cap: on Windows the NVIDIA driver otherwise spills past VRAM into system RAM
        # ("sysmem fallback"); on this laptop that coincided with fatal hardware-error reboots.
        # With the cap, overflow becomes a clean CUDA OOM instead.
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if max_gpu_gb and max_gpu_gb < total:
            torch.cuda.set_per_process_memory_fraction(max_gpu_gb / total, 0)
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens

    def prompt_len(self, messages: list[dict]) -> int:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return len(self.tok(text)["input_ids"])

    def generate(self, batch: list[list[dict]], temperature: float = 0.0) -> list[str]:
        texts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch]
        # No truncation: cutting the prompt would drop the question at the end.
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        if enc["input_ids"].shape[1] > self.max_prompt_tokens:
            print(f"warning: prompt of {enc['input_ids'].shape[1]} tokens exceeds --max-prompt-tokens")
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                **({"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {"do_sample": False}),
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        del enc, out, gen
        self.torch.cuda.empty_cache()  # keep fragmentation from creeping toward the cap
        return texts


class OpenAIBackend:
    def __init__(self, model: str, base_url: str | None, max_new_tokens: int):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600) if base_url else OpenAI()
        self.model, self.max_new_tokens = model, max_new_tokens

    def generate(self, batch: list[list[dict]], temperature: float = 0.0) -> list[str]:
        """The whole batch is sent concurrently; the server (vLLM) batches it on the GPU."""
        from concurrent.futures import ThreadPoolExecutor

        def one(msgs):
            r = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=temperature, max_tokens=self.max_new_tokens
            )
            return r.choices[0].message.content or ""

        with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
            return list(pool.map(one, batch))


class HTTPBackend:
    """OpenAI-compatible chat API with the standard library only (no openai package): works on
    Windows where Smart App Control blocks openai's compiled dependency, e.g. against Ollama."""

    def __init__(self, model: str, base_url: str, max_new_tokens: int):
        self.model, self.url, self.max_new_tokens = model, base_url.rstrip("/") + "/chat/completions", max_new_tokens

    def _one(self, msgs: list[dict], temperature: float) -> str:
        import urllib.request

        body = json.dumps({"model": self.model, "messages": msgs, "temperature": temperature,
                           "max_tokens": self.max_new_tokens, "stream": False}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""

    def generate(self, batch: list[list[dict]], temperature: float = 0.0) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
            return list(pool.map(lambda m: self._one(m, temperature), batch))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["hf", "openai", "http"], default="hf")
    ap.add_argument("--base-url")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N examples (after a fixed shuffle)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=6144)
    ap.add_argument("--train", default="data/processed/spider_train.jsonl")
    ap.add_argument("--max-gpu-gb", type=float, default=7.0, help="hard GPU memory cap (hf backend)")
    ap.add_argument("--max-batch-tokens", type=int, default=4096,
                    help="token budget per batch (hf backend): long prompts are generated alone")
    ap.add_argument("--link-k", type=int, default=0, help="schema linking: keep top-k tables (0 = full schema)")
    args = ap.parse_args()

    data = load_jsonl(args.data)
    random.Random(42).shuffle(data)  # fixed subset for --limit, spread over DBs/difficulties
    if args.limit:
        data = data[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [r for r in data if r["id"] not in done]
    print(f"{len(data)} examples, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        return

    demos = build_fewshot(REPO / args.train, args.shots, {r["db_id"] for r in data})
    msgs = [make_messages(r, demos, args.sample_rows, args.link_k) for r in todo]

    if args.backend == "hf":
        be = HFBackend(args.model, args.max_new_tokens, args.max_prompt_tokens, args.max_gpu_gb)
        # length-sorted, token-budgeted batches: little padding waste, bounded memory
        lens = [be.prompt_len(m) for m in msgs]
        order = sorted(range(len(todo)), key=lambda i: lens[i])
        batches = token_batches(order, lens, args.batch_size, args.max_batch_tokens, args.max_new_tokens)
    else:
        be = (HTTPBackend if args.backend == "http" else OpenAIBackend)(args.model, args.base_url, args.max_new_tokens)
        order = list(range(len(todo)))
        batches = [order[s : s + args.batch_size] for s in range(0, len(order), args.batch_size)]

    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as f, tqdm(total=len(order)) as bar:
        for idx in batches:
            t = time.perf_counter()
            raws = be.generate([msgs[i] for i in idx])
            dt = (time.perf_counter() - t) / len(idx)
            for i, raw in zip(idx, raws, strict=True):
                rec = {"id": todo[i]["id"], "pred_sql": extract_sql(raw), "raw": raw, "sec_per_ex": round(dt, 3)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            bar.update(len(idx))
    print(f"done in {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
