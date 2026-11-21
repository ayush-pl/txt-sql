"""Score a predictions file against gold data.

Predictions JSONL: {"id": ..., "pred_sql": ...}  (other fields ignored)
Special --pred gold scores gold vs gold (sanity check, should give EX = 100%).

Usage:
    python -m eval.run_eval --data data/processed/spider_dev.jsonl --pred outputs/preds.jsonl
    python -m eval.run_eval --data data/processed/spider_dev.jsonl --pred gold
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.table import Table

from data.prepare import load_jsonl
from eval.metrics import (
    classify_error,
    exact_match,
    execute_sql,
    has_top_level_order_by,
    results_match,
)

REPO = Path(__file__).resolve().parent.parent
DIFF_ORDER = ["easy", "medium", "hard", "extra", "simple", "moderate", "challenging", "unknown"]


def score_one(rec: dict, pred_sql: str, timeout_s: float) -> dict:
    db = str(REPO / rec["db_path"])
    mode = "bird" if rec["dataset"] == "bird" else "spider"
    gold = execute_sql(db, rec["gold_sql"], timeout_s)
    pred = execute_sql(db, pred_sql, timeout_s)
    if not gold.ok:
        return {**_base(rec, pred_sql), "gold_error": gold.error, "ex": None, "valid": pred.ok}
    ex = pred.ok and results_match(
        pred.rows, gold.rows, mode=mode, ordered=has_top_level_order_by(rec["gold_sql"])
    )
    return {
        **_base(rec, pred_sql),
        "ex": bool(ex),
        "em": exact_match(pred_sql, rec["gold_sql"]),
        "valid": pred.ok,
        "error_type": None if ex else classify_error(pred),
        "pred_error": pred.error,
        "pred_seconds": round(pred.seconds, 4),
    }


def _base(rec: dict, pred_sql: str) -> dict:
    return {
        "id": rec["id"],
        "db_id": rec["db_id"],
        "difficulty": rec["difficulty"],
        "question": rec["question"],
        "gold_sql": rec["gold_sql"],
        "pred_sql": pred_sql,
    }


def evaluate(data_path: str, pred_path: str, timeout_s: float = 30.0, workers: int = 8) -> dict:
    data = load_jsonl(data_path)
    if pred_path == "gold":
        preds = {r["id"]: r["gold_sql"] for r in data}
    else:
        preds = {p["id"]: p.get("pred_sql", "") for p in load_jsonl(pred_path)}
        data = [r for r in data if r["id"] in preds]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: score_one(r, preds.get(r["id"], ""), timeout_s), data))

    scored = [r for r in rows if r["ex"] is not None]
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_diff[r["difficulty"]].append(r)

    def pct(xs: list[bool]) -> float:
        return round(100 * sum(xs) / len(xs), 2) if xs else 0.0

    summary = {
        "n": len(scored),
        "skipped_gold_errors": len(rows) - len(scored),
        "ex": pct([r["ex"] for r in scored]),
        "em": pct([r["em"] for r in scored]),
        "valid_sql": pct([r["valid"] for r in scored]),
        "by_difficulty": {
            d: {"n": len(v), "ex": pct([r["ex"] for r in v])}
            for d, v in sorted(by_diff.items(), key=lambda kv: DIFF_ORDER.index(kv[0]) if kv[0] in DIFF_ORDER else 99)
        },
        "error_types": dict(Counter(r["error_type"] for r in scored if r["error_type"]).most_common()),
    }
    return {"summary": summary, "rows": rows}


def print_summary(name: str, s: dict) -> None:
    con = Console()
    t = Table(title=f"{name}  (n={s['n']})")
    for c in ("metric", "value"):
        t.add_column(c)
    t.add_row("Execution accuracy (EX)", f"{s['ex']:.2f}")
    t.add_row("Exact match (normalized)", f"{s['em']:.2f}")
    t.add_row("Valid SQL", f"{s['valid_sql']:.2f}")
    for d, v in s["by_difficulty"].items():
        t.add_row(f"  EX {d} (n={v['n']})", f"{v['ex']:.2f}")
    for e, n in s["error_types"].items():
        t.add_row(f"  err {e}", str(n))
    con.print(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True, help="predictions JSONL, or 'gold'")
    ap.add_argument("--out", help="write per-example results + summary JSON here")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    res = evaluate(args.data, args.pred, args.timeout, args.workers)
    print_summary(Path(args.pred).stem if args.pred != "gold" else "gold-vs-gold", res["summary"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
