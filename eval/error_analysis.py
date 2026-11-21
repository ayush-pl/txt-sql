"""Fine-grained error analysis for scored runs (outputs of eval.run_eval --out).

Each failed example gets one category. Execution errors keep their run_eval type
(unknown_column, syntax_error, ...). Queries that ran but returned the wrong rows are
diagnosed by comparing pred vs gold ASTs, first match wins:

  wrong_tables       different set of tables referenced (missing / extra join)
  wrong_aggregation  different aggregate functions (COUNT vs SUM, missing MAX, ...)
  wrong_group_by     GROUP BY present in one only, or on different columns
  wrong_filter_value same filter columns but different literals ('Male' vs 'M', 1 vs '1')
  wrong_filter       different WHERE/HAVING columns or operators
  wrong_order_limit  ORDER BY / LIMIT differs (top-1 vs ties, wrong direction)
  wrong_distinct     DISTINCT in one only
  wrong_select       different projected columns (extra or missing output columns)
  wrong_structure    nesting or set operations differ
  other_semantic     same skeleton, still wrong (join condition, NULL handling, ...)

Usage:
    python -m eval.error_analysis outputs/eval/sqlforge3b_step956_spider_dev.json
    python -m eval.error_analysis outputs/eval/sqlforge3b_step956_bird_dev500.json \
        --compare outputs/eval/qwen3b_zs_bird_dev500.json --md outputs/reports/bird_errors.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

CATEGORY_ORDER = [
    "unknown_column", "unknown_table", "ambiguous_column", "syntax_error", "timeout",
    "other_exec_error", "wrong_tables", "wrong_aggregation", "wrong_group_by",
    "wrong_filter_value", "wrong_filter", "wrong_order_limit", "wrong_distinct",
    "wrong_select", "wrong_structure", "other_semantic", "unparseable",
]


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _tables(t: exp.Expression) -> set[str]:
    return {x.name.lower() for x in t.find_all(exp.Table)}


def _aggs(t: exp.Expression) -> Counter:
    return Counter(type(x).__name__ for x in t.find_all(exp.AggFunc))


def _cols(nodes) -> set[str]:
    return {c.name.lower() for n in nodes if n is not None for c in n.find_all(exp.Column)}


def _group(t: exp.Expression) -> set[str] | None:
    g = t.find(exp.Group)
    return None if g is None else _cols([g])


def _filters(t: exp.Expression) -> tuple[set[str], Counter, set[str]]:
    conds = [n for n in (*t.find_all(exp.Where), *t.find_all(exp.Having))]
    ops = Counter(
        type(x).__name__ for c in conds for x in c.find_all(exp.Predicate, exp.Binary)
        if not isinstance(x, (exp.And, exp.Or))
    )
    lits = {str(x.this).lower() for c in conds for x in c.find_all(exp.Literal)}
    return _cols(conds), ops, lits


def _order_limit(t: exp.Expression) -> tuple:
    o = t.args.get("order")
    lim = t.args.get("limit")
    order = None if o is None else tuple(
        (",".join(sorted(_cols([e]))), bool(e.args.get("desc"))) for e in o.expressions
    )
    return order, None if lim is None else lim.sql()


def _select_cols(t: exp.Expression) -> list[str]:
    sel = t if isinstance(t, exp.Select) else t.find(exp.Select)
    if sel is None:
        return []
    return sorted(e.sql(normalize=True).lower() for e in sel.expressions)


def _shape(t: exp.Expression) -> tuple[int, int]:
    return (len(list(t.find_all(exp.Subquery))), len(list(t.find_all(exp.Union, exp.Intersect, exp.Except))))


def categorize(row: dict) -> str | None:
    """Category for one scored row, or None if it was correct / unscorable."""
    if row.get("ex") is not False:
        return None
    if row.get("error_type") and row["error_type"] != "wrong_result":
        return row["error_type"]
    p, g = _parse(row["pred_sql"]), _parse(row["gold_sql"])
    if p is None or g is None:
        return "unparseable"
    if _tables(p) != _tables(g):
        return "wrong_tables"
    if _aggs(p) != _aggs(g):
        return "wrong_aggregation"
    if _group(p) != _group(g):
        return "wrong_group_by"
    pc, po, pl = _filters(p)
    gc, go, gl = _filters(g)
    if pc == gc and po == go and pl != gl:
        return "wrong_filter_value"
    if pc != gc or po != go:
        return "wrong_filter"
    if _order_limit(p) != _order_limit(g):
        return "wrong_order_limit"
    if bool(p.find(exp.Distinct)) != bool(g.find(exp.Distinct)):
        return "wrong_distinct"
    if _shape(p) != _shape(g):
        return "wrong_structure"
    if _select_cols(p) != _select_cols(g):
        return "wrong_select"
    return "other_semantic"


def load_rows(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def analyze(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("ex") is not None]
    cats = {r["id"]: categorize(r) for r in scored}
    counts = Counter(c for c in cats.values() if c)
    return {"n": len(scored), "wrong": sum(counts.values()), "counts": counts, "cats": cats}


def _sorted(counts: Counter) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], CATEGORY_ORDER.index(kv[0]) if kv[0] in CATEGORY_ORDER else 99))


def render(name: str, rows: list[dict], a: dict, compare: tuple[str, list[dict], dict] | None, examples: int) -> str:
    out = [f"# Error analysis: {name}", "", f"{a['wrong']} wrong of {a['n']} scored ({100 * a['wrong'] / a['n']:.1f}%).", ""]
    if compare:
        cname, _, b = compare
        out += [f"| category | {name} | {cname} | Δ |", "|---|---|---|---|"]
        keys = [k for k, _ in _sorted(a["counts"] + b["counts"])]
        for k in keys:
            x, y = a["counts"].get(k, 0), b["counts"].get(k, 0)
            out.append(f"| {k} | {x} | {y} | {x - y:+d} |")
        ids = set(a["cats"]) & set(b["cats"])
        fixed = sum(1 for i in ids if a["cats"][i] is None and b["cats"][i] is not None)
        broke = sum(1 for i in ids if a["cats"][i] is not None and b["cats"][i] is None)
        out += ["", f"Versus {cname}: **{fixed} fixed**, **{broke} newly broken** (net {fixed - broke:+d}).", ""]
    else:
        out += ["| category | count | share of errors |", "|---|---|---|"]
        for k, v in _sorted(a["counts"]):
            out.append(f"| {k} | {v} | {100 * v / a['wrong']:.1f}% |")
        out.append("")

    if examples:
        by_id = {r["id"]: r for r in rows}
        out.append("## Examples")
        for k, _ in _sorted(a["counts"]):
            ids = [i for i, c in a["cats"].items() if c == k][:examples]
            out += ["", f"### {k}"]
            for i in ids:
                r = by_id[i]
                out += [
                    f"- **{r['db_id']}**: {r['question']}",
                    f"  - gold: `{r['gold_sql']}`",
                    f"  - pred: `{r['pred_sql']}`",
                ]
                if r.get("pred_error"):
                    out.append(f"  - error: `{r['pred_error']}`")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", help="run_eval --out JSON")
    ap.add_argument("--compare", help="another scored JSON on the same data (e.g. the base model)")
    ap.add_argument("--md", help="write the markdown report here")
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    rows = load_rows(args.scored)
    a = analyze(rows)
    cmp = None
    if args.compare:
        crows = load_rows(args.compare)
        cmp = (Path(args.compare).stem, crows, analyze(crows))
    md = render(Path(args.scored).stem, rows, a, cmp, args.examples)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"saved {args.md}")
    print(md.split("## Examples")[0])


if __name__ == "__main__":
    main()
