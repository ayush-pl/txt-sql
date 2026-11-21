"""Check that evaluation data doesn't leak into training data.

Checks:
  1. Database overlap: Spider's dev DBs should be disjoint from train DBs (cross-database setting).
  2. Question overlap: exact or normalized question text appearing in both train and eval.
  3. Same-database question overlap (real leakage). Cross-DB generic pairs such as
     "How many X are there?" are reported but not treated as leakage.

Exits non-zero if leakage is found, so it can run in CI.

Usage:
    python -m data.contamination_check --train data/processed/spider_train.jsonl \
        --eval data/processed/spider_dev.jsonl data/processed/bird_dev.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from data.prepare import load_jsonl
from eval.metrics import normalize_sql


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    args = ap.parse_args()

    train = [r for p in args.train for r in load_jsonl(p)]
    train_dbs = {r["db_id"] for r in train}
    train_qs = {norm_q(r["question"]) for r in train}
    train_pairs = {(norm_q(r["question"]), normalize_sql(r["gold_sql"])) for r in train}
    train_triples = {(r["db_id"], norm_q(r["question"])) for r in train}

    leaked = False
    for p in args.eval:
        ev = load_jsonl(p)
        db_overlap = sorted({r["db_id"] for r in ev} & train_dbs)
        q_overlap = [r for r in ev if norm_q(r["question"]) in train_qs]
        pair_overlap = [r for r in ev if (norm_q(r["question"]), normalize_sql(r["gold_sql"])) in train_pairs]
        same_db = [r for r in ev if (r["db_id"], norm_q(r["question"])) in train_triples]
        print(f"\n== {Path(p).name} (n={len(ev)}) vs train (n={len(train)})")
        print(f"  shared databases     : {len(db_overlap)} {db_overlap[:10]}")
        print(f"  shared questions     : {len(q_overlap)}")
        print(f"  shared (question,SQL): {len(pair_overlap)}  (generic cross-DB templates; info only)")
        print(f"  same-DB question hits: {len(same_db)}")
        for r in q_overlap[:5]:
            print(f"    e.g. [{r['db_id']}] {r['question']}")
        if db_overlap or same_db:
            leaked = True

    print("\nRESULT:", "LEAKAGE FOUND" if leaked else "clean")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
