"""Eval gate: a fixed, self-contained accuracy check that CI can run without a GPU.

    build    one-time: sample a fixed gate set (150 Spider dev questions, stratified by
             difficulty, from 19 small DBs) and copy those SQLite files into eval/gate/dbs/
    predict  run the model (any OpenAI-compatible server, e.g. vLLM) on the gate set and write
             eval/gate/predictions.jsonl. Run locally / on a GPU runner whenever the model,
             prompt or decoding changes, and commit the file with the change.
    check    CI: re-score the committed predictions by executing them against the committed DBs
             and fail (exit 1) if accuracy is below eval/gate/thresholds.yaml, or if the
             predictions don't cover the gate set / were made for a different prompt.

    python -m eval.gate build
    python -m eval.gate predict --base-url http://127.0.0.1:8000/v1 --model sqlforge
    python -m eval.gate check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "eval" / "gate"
GATE_SET = GATE / "gate_set.jsonl"
PREDS = GATE / "predictions.jsonl"
META = GATE / "predictions_meta.json"
THRESHOLDS = GATE / "thresholds.yaml"
EXCLUDE_DBS = {"wta_1"}  # 100 MB: too big to commit
QUOTA = {"easy": 36, "medium": 64, "hard": 26, "extra": 24}  # ~ dev distribution, n=150


def prompt_fingerprint() -> str:
    """Hash of the prompt format: predictions made with another prompt are stale."""
    from data.prompt import SYSTEM_PROMPT, build_user_prompt

    probe = SYSTEM_PROMPT + build_user_prompt("<schema>", "<question>", "<hint>")
    return hashlib.sha256(probe.encode()).hexdigest()[:12]


def build(seed: int = 0) -> None:
    from data.prepare import load_jsonl

    recs = [r for r in load_jsonl(REPO / "data/processed/spider_dev.jsonl") if r["db_id"] not in EXCLUDE_DBS]
    by_diff = defaultdict(list)
    for r in recs:
        by_diff[r["difficulty"]].append(r)
    rng = random.Random(seed)
    chosen = []
    for d, k in QUOTA.items():
        rng.shuffle(by_diff[d])
        chosen += by_diff[d][:k]
    chosen.sort(key=lambda r: r["id"])
    (GATE / "dbs").mkdir(parents=True, exist_ok=True)
    with GATE_SET.open("w", encoding="utf-8") as f:
        for r in chosen:
            dst = GATE / "dbs" / r["db_id"] / f"{r['db_id']}.sqlite"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / r["db_path"], dst)
            f.write(json.dumps({**r, "db_path": dst.relative_to(REPO).as_posix()}, ensure_ascii=False) + "\n")
    size = sum(p.stat().st_size for p in (GATE / "dbs").rglob("*.sqlite"))
    print(f"gate set: {len(chosen)} questions, {len({r['db_id'] for r in chosen})} DBs ({size / 1e6:.1f} MB) -> {GATE_SET}")


def predict(base_url: str, model: str, concurrency: int, column_fix: bool) -> None:
    from data.prepare import load_jsonl
    from eval.generate import HTTPBackend, make_messages
    from eval.metrics import execute_sql
    from eval.self_correct import fix_column_names

    data = load_jsonl(GATE_SET)
    be = HTTPBackend(model, base_url, max_new_tokens=256)
    msgs = [make_messages(r, [], 3) for r in data]
    from data.prompt import extract_sql

    out = []
    for s in range(0, len(data), concurrency):
        for r, raw in zip(data[s : s + concurrency], be.generate(msgs[s : s + concurrency]), strict=True):
            sql = extract_sql(raw)
            if column_fix:  # same deterministic repair as the serving pipeline
                db = str(REPO / r["db_path"])
                res = execute_sql(db, sql)
                for _ in range(4):
                    if res.ok or "no such column" not in (res.error or ""):
                        break
                    fixed = fix_column_names(db, sql, res.error)
                    if fixed is None:
                        break
                    sql, res = fixed, execute_sql(db, fixed)
            out.append({"id": r["id"], "pred_sql": sql})
    with PREDS.open("w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    META.write_text(json.dumps({"model": model, "base_url": base_url, "column_fix": column_fix,
                                "prompt_fingerprint": prompt_fingerprint(), "n": len(out),
                                "created": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1),
                    encoding="utf-8")
    print(f"{len(out)} predictions -> {PREDS}")


def check(preds_path: Path = PREDS, meta_path: Path = META) -> int:
    from data.prepare import load_jsonl
    from eval.run_eval import evaluate

    th = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))
    gate_ids = {r["id"] for r in load_jsonl(GATE_SET)}
    pred_ids = {p["id"] for p in load_jsonl(preds_path)}
    failures = []
    if pred_ids != gate_ids:
        failures.append(f"predictions cover {len(pred_ids & gate_ids)}/{len(gate_ids)} gate questions")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("prompt_fingerprint") != prompt_fingerprint():
        failures.append("predictions were made with a different prompt format: rerun `python -m eval.gate predict`")

    s = evaluate(str(GATE_SET), str(preds_path), timeout_s=30, workers=4)["summary"]
    print(f"gate: n={s['n']}  EX={s['ex']:.1f}  valid_sql={s['valid_sql']:.1f}  model={meta.get('model')}  "
          f"by difficulty: " + ", ".join(f"{d} {v['ex']:.0f}" for d, v in s["by_difficulty"].items()))
    if s["ex"] < th["min_ex"]:
        failures.append(f"EX {s['ex']:.1f} < min_ex {th['min_ex']}")
    if s["valid_sql"] < th["min_valid_sql"]:
        failures.append(f"valid SQL {s['valid_sql']:.1f} < min_valid_sql {th['min_valid_sql']}")
    for d, floor in (th.get("min_ex_by_difficulty") or {}).items():
        got = s["by_difficulty"].get(d, {}).get("ex", 0.0)
        if got < floor:
            failures.append(f"{d} EX {got:.1f} < {floor}")
    if failures:
        print("GATE FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("GATE PASSED")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("predict")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="sqlforge")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--no-column-fix", dest="column_fix", action="store_false")
    c = sub.add_parser("check")
    c.add_argument("--pred", default=str(PREDS))
    c.add_argument("--meta", default=str(META))
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "predict":
        predict(args.base_url, args.model, args.concurrency, args.column_fix)
    else:
        sys.exit(check(Path(args.pred), Path(args.meta)))


if __name__ == "__main__":
    main()
