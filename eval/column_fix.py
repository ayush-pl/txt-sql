"""Offline pass: apply the deterministic column-name repair (eval.self_correct.fix_column_names)
to every prediction that fails with "no such column", then write a new predictions file.
No model calls, so it measures the fix in isolation.

    python -m eval.column_fix --data data/processed/spider_dev.jsonl \
        --pred outputs/preds/sqlforge3b_v2-step1014_spider_dev.jsonl --out outputs/preds/..._colfix_spider_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.prepare import load_jsonl
from eval.metrics import execute_sql
from eval.self_correct import fix_column_names

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=30)
    args = ap.parse_args()
    recs = {r["id"]: r for r in load_jsonl(args.data)}
    n_err = n_fixed = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p in load_jsonl(args.pred):
            db = str(REPO / recs[p["id"]]["db_path"])
            sql = p["pred_sql"]
            res = execute_sql(db, sql, args.timeout)
            if not res.ok and "no such column" in (res.error or ""):
                n_err += 1
                for _ in range(4):
                    fixed = fix_column_names(db, sql, res.error)
                    if fixed is None:
                        break
                    sql = fixed
                    res = execute_sql(db, sql, args.timeout)
                    if res.ok or "no such column" not in (res.error or ""):
                        break
                n_fixed += res.ok
            f.write(json.dumps({**p, "pred_sql": sql}, ensure_ascii=False) + "\n")
    print(f"{n_err} predictions with an unknown column, {n_fixed} now execute -> {args.out}")


if __name__ == "__main__":
    main()
