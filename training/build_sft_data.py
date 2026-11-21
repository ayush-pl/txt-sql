"""Build the SFT dataset in TRL's conversational prompt/completion format.

  {"prompt": [system, user], "completion": [assistant], "db_id", "db_path", "dataset", "n_tokens"}

With prompt/completion data, TRL computes loss on the completion (the SQL) only.

- The prompt format is identical to evaluation (data.prompt + data.schema).
- DBs listed in --exclude-dbs are dropped (formula_1 overlaps with BIRD dev).
- A validation split is held out *by database*, so val loss measures cross-DB generalization.
- Examples longer than --max-length tokens are dropped, never truncated.

Usage:
    python -m training.build_sft_data --tokenizer Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from data.prepare import load_jsonl
from data.prompt import build_messages
from data.schema import serialize_schema

REPO = Path(__file__).resolve().parent.parent


def clean_sql(sql: str) -> str:
    """Spider gold SQL has odd spacing ('max(x) ,  y'). Normalize whitespace only."""
    sql = " ".join(sql.strip().rstrip(";").split())
    return re.sub(r"\s+,", ",", sql)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["data/processed/spider_train.jsonl"])
    ap.add_argument("--out-dir", default="data/processed/sft")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--exclude-dbs", nargs="*", default=["formula_1"])
    ap.add_argument("--val-db-frac", type=float, default=0.05)
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-val", type=int, default=300, help="cap val size to keep eval fast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-gold", action="store_true",
                    help="execute every gold query and drop the ones that error or time out (BIRD train is noisy)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    recs = [r for p in args.train for r in load_jsonl(REPO / p) if r["db_id"] not in set(args.exclude_dbs)]
    if args.verify_gold:
        from concurrent.futures import ThreadPoolExecutor

        from eval.metrics import execute_sql

        with ThreadPoolExecutor(8) as pool:
            ok = list(pool.map(lambda r: execute_sql(str(REPO / r["db_path"]), r["gold_sql"], 15).ok, recs))
        bad = len(recs) - sum(ok)
        recs = [r for r, good in zip(recs, ok, strict=True) if good]
        print(f"verify-gold: dropped {bad} examples whose gold SQL fails to execute")

    dbs = sorted({r["db_id"] for r in recs})
    random.Random(args.seed).shuffle(dbs)
    val_dbs = set(dbs[: max(1, int(len(dbs) * args.val_db_frac))])

    splits: dict[str, list[dict]] = {"train": [], "val": []}
    dropped = 0
    for r in recs:
        completion = [{"role": "assistant", "content": clean_sql(r["gold_sql"])}]
        # Wide schemas: fall back to fewer sample rows instead of dropping the example.
        for rows in sorted({args.sample_rows, 1, 0}, reverse=True):
            schema = serialize_schema(str(REPO / r["db_path"]), sample_rows=rows)
            prompt = build_messages(schema, r["question"], r.get("evidence"))
            n = len(tok(tok.apply_chat_template(prompt + completion, tokenize=False))["input_ids"])
            if n <= args.max_length:
                break
        if n > args.max_length:
            dropped += 1
            continue
        ex = {"prompt": prompt, "completion": completion, "db_id": r["db_id"], "db_path": r["db_path"],
              "dataset": r["dataset"], "n_tokens": n}
        splits["val" if r["db_id"] in val_dbs else "train"].append(ex)

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        random.Random(args.seed).shuffle(rows)
        if name == "val":
            del rows[args.max_val :]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    lens = sorted(ex["n_tokens"] for ex in splits["train"])
    print(
        f"train={len(splits['train'])} val={len(splits['val'])} ({len(val_dbs)} DBs) "
        f"dropped_too_long={dropped} excluded_dbs={args.exclude_dbs}\n"
        f"train tokens: median={lens[len(lens) // 2]} p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}"
    )


if __name__ == "__main__":
    main()
