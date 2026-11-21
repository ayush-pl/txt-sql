"""Execution-guided self-correction: run each predicted query, and when SQLite raises an
error, show the model its own SQL plus the error message and ask for a fix (up to N rounds).

Works as a post-pass over an existing predictions file, so only failing queries are regenerated:

    python -m eval.self_correct --data data/processed/bird_dev.jsonl \
        --pred outputs/preds/sqlforge3b_step956_bird_dev500.jsonl \
        --model models/sqlforge-3b-step956 --rounds 2 --batch-size 2 \
        --out outputs/preds/sqlforge3b_step956_sc2_bird_dev500.jsonl

The output keeps every prediction (corrected or not) and records `first_sql` and `attempts`.
`repair_messages` is shared with the serving API.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp
from tqdm import tqdm

from data.prepare import load_jsonl
from data.prompt import extract_sql
from data.schema import _ident
from eval.generate import HFBackend, OpenAIBackend, make_messages, token_batches
from eval.metrics import execute_sql

REPO = Path(__file__).resolve().parent.parent

REPAIR_TEMPLATE = (
    "That query failed when executed on the database.\n"
    "Error: {error}\n{hint}\n"
    "Write a corrected SQLite query. Use only tables and columns that exist in the schema above, "
    "and wrap names containing spaces or special characters in double quotes. "
    "Return only the SQL."
)


def _table_columns(db_path: str) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [c[1] for c in conn.execute(f"PRAGMA table_info({_ident(t)})")] for t in tables}
    finally:
        conn.close()


def _fmt_col(c: str) -> str:
    return c if c.replace("_", "").isalnum() else f'"{c}"'


def error_hint(db_path: str, bad_sql: str, error: str) -> str:
    """Schema facts that make the error actionable: the real columns of every table the query
    touches (with aliases), or, when the SQL does not parse, the names that need quoting."""
    cols = _table_columns(db_path)
    lower = {t.lower(): t for t in cols}
    try:
        tree = sqlglot.parse_one(bad_sql, read="sqlite")
    except Exception:
        tree = None
    lines = []
    if tree is not None:
        seen = set()
        for tb in tree.find_all(exp.Table):
            real = lower.get(tb.name.lower())
            if real is None or (real, tb.alias) in seen:
                continue
            seen.add((real, tb.alias))
            alias = f" (alias {tb.alias})" if tb.alias else ""
            lines.append(f"- {real}{alias}: {', '.join(_fmt_col(c) for c in cols[real])}")
        if lines:
            return "Columns of the tables your query uses:\n" + "\n".join(lines) + "\n"
    special = [f'{t}."{c}"' for t, cs in cols.items() for c in cs if not c.replace("_", "").isalnum()]
    if special:
        return "Column names that must be double-quoted: " + ", ".join(special[:30]) + "\n"
    return ""


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def fix_column_names(db_path: str, sql: str, error: str) -> str | None:
    """Deterministic repair for the commonest failure, a near-miss column name
    (`pet_type` for `PetType`, `student_id` for `StuID` does not qualify). If the unknown column
    matches exactly one real column of the query's tables after normalization (case,
    underscores, spaces), rewrite it in the AST. Returns the fixed SQL, or None."""
    import re

    m = re.search(r"no such column: (?:[\w\"`\[\]]+\.)?[\"`\[]?([^\"`\]]+)[\"`\]]?", error or "")
    if not m:
        return None
    bad = m.group(1).strip()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    cols = _table_columns(db_path)
    lower = {t.lower(): t for t in cols}
    used = {lower[t.name.lower()] for t in tree.find_all(exp.Table) if t.name.lower() in lower}
    candidates = {c for t in used for c in cols[t] if _norm(c) == _norm(bad)}
    if len(candidates) != 1:
        return None
    real = candidates.pop()
    changed = False
    for col in tree.find_all(exp.Column):
        if col.name == bad:
            col.set("this", exp.to_identifier(real, quoted=not real.replace("_", "").isalnum()))
            changed = True
    return tree.sql(dialect="sqlite") if changed else None


def repair_messages(messages: list[dict], bad_sql: str, error: str, hint: str = "") -> list[dict]:
    """Conversation that asks the model to fix `bad_sql`, given the database error."""
    return [
        *messages,
        {"role": "assistant", "content": bad_sql},
        {"role": "user", "content": REPAIR_TEMPLATE.format(error=error[:300], hint=hint)},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True, help="existing predictions JSONL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--base-url")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-gpu-gb", type=float, default=7.0, help="hard GPU memory cap (hf backend)")
    ap.add_argument("--max-batch-tokens", type=int, default=4096, help="token budget per batch (hf backend)")
    ap.add_argument("--no-hints", dest="hints", action="store_false", help="bare error message only")
    ap.add_argument("--retry-temp", type=float, default=0.7,
                    help="sampling temperature from round 2 on, so a repeated mistake is not reproduced verbatim")
    args = ap.parse_args()

    recs = {r["id"]: r for r in load_jsonl(args.data)}
    preds = load_jsonl(args.pred)
    state = {
        p["id"]: {"id": p["id"], "pred_sql": p["pred_sql"], "first_sql": p["pred_sql"], "attempts": 1}
        for p in preds
    }

    def failing(ids=None) -> dict[str, str]:
        errs = {}
        for pid in tqdm(ids or list(state), desc="executing", leave=False):
            res = execute_sql(str(REPO / recs[pid]["db_path"]), state[pid]["pred_sql"], args.timeout)
            if not res.ok:
                errs[pid] = res.error
        return errs

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Every repaired query is appended here as soon as its batch finishes, so a crash or reboot
    # loses at most one batch: rerunning the same command replays it and continues.
    progress_path = out.with_name(out.name + ".progress")
    progress: dict[int, list[dict]] = {}
    if progress_path.exists():
        for line in progress_path.open(encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                progress.setdefault(e["round"], []).append(e)
        print(f"resuming: {sum(map(len, progress.values()))} repairs already done")

    if args.backend == "hf":
        be = HFBackend(args.model, args.max_new_tokens, max_prompt_tokens=8192, max_gpu_gb=args.max_gpu_gb)
    else:
        be = OpenAIBackend(args.model, args.base_url, args.max_new_tokens)

    errs = failing()
    print(f"{len(errs)} of {len(state)} predictions raise an error")
    with progress_path.open("a", encoding="utf-8") as plog:
        for rnd in range(1, args.rounds + 1):
            if not errs:
                break
            ids = list(errs)
            # replay this round's finished repairs (errs above was computed from the state before them)
            done = set()
            for e in progress.get(rnd, []):
                if e["id"] in errs:
                    state[e["id"]]["pred_sql"] = e["pred_sql"]
                    state[e["id"]]["attempts"] += 1
                    done.add(e["id"])
            todo = [i for i in ids if i not in done]
            convs = {
                i: repair_messages(
                    make_messages(recs[i], [], args.sample_rows), state[i]["pred_sql"], errs[i],
                    error_hint(str(REPO / recs[i]["db_path"]), state[i]["pred_sql"], errs[i]) if args.hints else "",
                )
                for i in todo
            }
            if args.backend == "hf":
                lens = {i: be.prompt_len(convs[i]) for i in todo}
                todo.sort(key=lambda i: lens[i])
                pos = token_batches(list(range(len(todo))), [lens[i] for i in todo], args.batch_size,
                                    args.max_batch_tokens, args.max_new_tokens)
                batches = [[todo[p] for p in b] for b in pos]
            else:
                batches = [todo[s : s + args.batch_size] for s in range(0, len(todo), args.batch_size)]
            temp = 0.0 if rnd == 1 else args.retry_temp
            for chunk in tqdm(batches, desc=f"round {rnd}"):
                for i, raw in zip(chunk, be.generate([convs[i] for i in chunk], temperature=temp), strict=True):
                    state[i]["pred_sql"] = extract_sql(raw)
                    state[i]["attempts"] += 1
                    plog.write(json.dumps({"round": rnd, "id": i, "pred_sql": state[i]["pred_sql"]}, ensure_ascii=False) + "\n")
                plog.flush()
            before = len(errs)
            errs = failing(ids)
            print(f"round {rnd}: {before - len(errs)} of {before} now execute")

    with out.open("w", encoding="utf-8") as f:
        for s in state.values():
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    progress_path.unlink(missing_ok=True)
    changed = sum(1 for s in state.values() if s["attempts"] > 1)
    print(f"{changed} predictions revised -> {out}")


if __name__ == "__main__":
    main()
