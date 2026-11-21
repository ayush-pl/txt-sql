"""Text-to-SQL metrics: execution accuracy (EX), normalized exact match (EM), valid-SQL rate,
plus a Spider-style hardness estimate.

EX semantics:
  - "bird":   set(pred_rows) == set(gold_rows), the official BIRD rule.
  - "spider": multiset comparison, order-sensitive when gold has a top-level ORDER BY.
              Column order is also tolerated, as the test-suite evaluator does.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

import sqlglot
from sqlglot import exp


@dataclass
class ExecResult:
    rows: list[tuple] | None
    error: str | None
    seconds: float

    @property
    def ok(self) -> bool:
        return self.error is None


def execute_sql(db_path: str, sql: str, timeout_s: float = 30.0, max_rows: int = 100_000) -> ExecResult:
    """Run SQL read-only with a wall-clock timeout. Never raises."""
    start = time.perf_counter()
    if not sql or not sql.strip():
        return ExecResult(None, "empty sql", 0.0)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as e:
        return ExecResult(None, f"connect: {e}", 0.0)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = start + timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return ExecResult([tuple(r) for r in rows], None, time.perf_counter() - start)
    except sqlite3.OperationalError as e:
        msg = "timeout" if "interrupted" in str(e) else str(e)
        return ExecResult(None, msg, time.perf_counter() - start)
    except Exception as e:
        return ExecResult(None, f"{type(e).__name__}: {e}", time.perf_counter() - start)
    finally:
        conn.close()


def _norm_val(v: object) -> object:
    if isinstance(v, float):
        return round(v, 6)
    return v


def _norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_val(v) for v in r) for r in rows]


def _col_invariant(rows: list[tuple]) -> list[tuple]:
    return [tuple(sorted(r, key=lambda v: (str(type(v)), str(v)))) for r in rows]


def has_top_level_order_by(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return "order by" in sql.lower()
    return tree.args.get("order") is not None


def results_match(pred: list[tuple], gold: list[tuple], mode: str = "spider", ordered: bool = False) -> bool:
    pred, gold = _norm_rows(pred), _norm_rows(gold)
    if mode == "bird":
        return set(pred) == set(gold)
    if ordered:
        if pred == gold:
            return True
        return _col_invariant(pred) == _col_invariant(gold)
    if Counter(pred) == Counter(gold):
        return True
    return Counter(_col_invariant(pred)) == Counter(_col_invariant(gold))


def normalize_sql(sql: str) -> str | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    return tree.sql(dialect="sqlite", normalize=True, pretty=False).lower()


def exact_match(pred: str, gold: str) -> bool:
    a, b = normalize_sql(pred), normalize_sql(gold)
    return a is not None and a == b


def _own(node: exp.Expression, sel: exp.Expression, kinds) -> list[exp.Expression]:
    """Nodes of `kinds` belonging to `sel` itself, not to a nested subquery."""
    return [n for n in sel.find_all(kinds) if n.find_ancestor(exp.Select) is sel or n is sel]


def spider_hardness(sql: str) -> str:
    """Spider's hardness buckets (easy/medium/hard/extra), following the official
    evaluation.py rules (component1 / component2 / others) on a sqlglot AST."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return "unknown"

    # top-level SELECT = leftmost branch of any set operation
    top = tree
    set_ops = 0
    while isinstance(top, (exp.Union, exp.Intersect, exp.Except)):
        set_ops += 1
        top = top.left
    if not isinstance(top, exp.Select):
        return "unknown"

    where, having = top.args.get("where"), top.args.get("having")
    order = top.args.get("order") or tree.args.get("order")
    limit = top.args.get("limit") or tree.args.get("limit")
    joins = top.args.get("joins") or []
    n_from_tables = 1 + len(joins)
    conds = [c for c in (where, having) if c is not None]
    for j in joins:
        if j.args.get("on") is not None:
            conds.append(j.args["on"])

    # component1: where, group by, order by, limit, joins, ORs, LIKEs
    comp1 = sum([where is not None, top.args.get("group") is not None, order is not None, limit is not None])
    comp1 += n_from_tables - 1
    for c in conds:
        comp1 += len(_own(c, top, exp.Or)) + len(_own(c, top, exp.Like))

    # component2: nested queries (direct subqueries in conditions/from + set-op branches)
    nested = set_ops
    for sub in top.find_all(exp.Select):
        if sub is not top and sub.find_ancestor(exp.Select) is top:
            nested += 1

    # others: more than one agg / select column / where condition / group-by column
    n_aggs = len(_own(top, top, exp.AggFunc))
    n_select = len(top.expressions)
    n_where = (len(_own(where, top, exp.And)) + len(_own(where, top, exp.Or)) + 1) if where is not None else 0
    group = top.args.get("group")
    n_group = len(group.expressions) if group is not None else 0
    others = sum([n_aggs > 1, n_select > 1, n_where > 1, n_group > 1])

    if comp1 <= 1 and others == 0 and nested == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and nested == 0) or (comp1 <= 2 and others < 2 and nested == 0):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and nested == 0)
        or (2 < comp1 <= 3 and others <= 2 and nested == 0)
        or (comp1 <= 1 and others == 0 and nested <= 1)
    ):
        return "hard"
    return "extra"


def classify_error(pred: ExecResult) -> str:
    """Coarse bucket for failed executions, used in error analysis."""
    if pred.ok:
        return "wrong_result"
    e = (pred.error or "").lower()
    if "no such column" in e:
        return "unknown_column"
    if "no such table" in e:
        return "unknown_table"
    if "syntax" in e or "incomplete" in e:
        return "syntax_error"
    if "ambiguous" in e:
        return "ambiguous_column"
    if "timeout" in e:
        return "timeout"
    if "empty" in e:
        return "empty_output"
    return "other_exec_error"
