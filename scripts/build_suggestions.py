"""Build space/suggestions.json: the questions the live demo suggests, per database.

Runs every question of the eval gate set (150 real Spider dev questions over the 18 bundled
databases) through the demo's own pipeline and model (space/demo.py: guardrails, column fix,
self-correction), and keeps only those whose result matches the gold query's. So every suggested
question is one the demo is known to answer correctly.

    MODEL_PATH=models/gguf/sqlforge-3b-Q4_K_M.gguf N_THREADS=12 python scripts/build_suggestions.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.metrics import execute_sql, has_top_level_order_by, results_match  # noqa: E402

GATE = REPO / "eval" / "gate" / "gate_set.jsonl"
OUT = REPO / "space" / "suggestions.json"
ORDER = {"easy": 0, "medium": 1, "hard": 2, "extra": 3}


def main() -> None:
    spec = importlib.util.spec_from_file_location("demo", REPO / "space" / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)  # loads the model; does not launch the UI

    rows = [json.loads(line) for line in GATE.open(encoding="utf-8")]
    keep: dict[str, list[dict]] = defaultdict(list)
    n_ok = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        db_path = demo.CATALOGUE[r["db_id"]]
        ans = demo.service.answer(r["db_id"], r["question"])
        gold = execute_sql(db_path, r["gold_sql"])
        ok = bool(ans.ok and gold.ok and results_match([tuple(x) for x in ans.rows], gold.rows,
                                                        ordered=has_top_level_order_by(r["gold_sql"])))
        n_ok += ok
        if ok:
            keep[r["db_id"]].append({"question": r["question"], "difficulty": r["difficulty"],
                                     "repaired": ans.repaired})
        print(f"[{i}/{len(rows)}] {'OK ' if ok else 'BAD'} {time.time() - t0:5.0f}s {r['db_id']}: {r['question']}",
              flush=True)

    out = {db: sorted(qs, key=lambda q: ORDER.get(q["difficulty"], 9)) for db, qs in sorted(keep.items())}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"correct {n_ok}/{len(rows)} ({100 * n_ok / len(rows):.1f}%); wrote {OUT}")


if __name__ == "__main__":
    main()
