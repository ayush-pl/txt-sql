"""Schema linking: keep only the tables a question needs, so wide schemas fit the context window
and the model has fewer wrong tables and columns to choose from.

Method (no GPU, no training):
  1. Each table becomes a BM25 "document": its name, column names (camelCase / snake_case split)
     and a sample of distinct text values (so "Los Angeles" points at the table holding counties).
  2. Score tables against question + evidence hint, keep the top-k with a positive score.
  3. Add bridge tables: if two kept tables are linked only through a third table by foreign
     keys, keep the third too, so the join path stays expressible.

Evaluate recall offline (do we keep every table the gold SQL uses?):
    python -m data.schema_linker --data data/processed/bird_dev.jsonl --k 3 4 5
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import sqlglot
from rank_bm25 import BM25Okapi
from sqlglot import exp

from data.schema import _ident, serialize_schema

REPO = Path(__file__).resolve().parent.parent
_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "what",
    "which", "who", "whom", "how", "many", "much", "list", "show", "give", "find", "all", "each",
    "with", "by", "from", "that", "this", "their", "its", "has", "have", "do", "does", "did", "be",
    "please", "name", "names", "number", "count", "than", "more", "less", "most", "least", "as",
    "at", "it", "there", "any", "me", "tell", "return", "among", "whose", "between", "refers",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel Case
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t)
    return out


@lru_cache(maxsize=256)
def _db_index(db_path: str, values_per_col: int = 30):
    """(tables, BM25 documents, FK adjacency, junction tables -> FK targets) for one database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        docs, adj, junction = [], {t.lower(): set() for t in tables}, {}
        for t in tables:
            cols = cur.execute(f"PRAGMA table_info({_ident(t)})").fetchall()
            # table name counts double: it is the strongest signal
            doc = tokenize(t) * 2
            for c in cols:
                doc += tokenize(c[1])
                if (c[2] or "TEXT").upper().startswith(("TEXT", "VARCHAR", "CHAR")):
                    try:
                        vals = cur.execute(
                            f"SELECT DISTINCT {_ident(c[1])} FROM (SELECT {_ident(c[1])} FROM {_ident(t)} LIMIT 2000) "
                            f"WHERE {_ident(c[1])} IS NOT NULL LIMIT {values_per_col}"
                        ).fetchall()
                    except sqlite3.Error:
                        vals = []
                    for (v,) in vals:
                        if isinstance(v, str) and len(v) <= 40:
                            doc += tokenize(v)
            docs.append(doc or ["_"])
            fks = cur.execute(f"PRAGMA foreign_key_list({_ident(t)})").fetchall()
            # junction table: mostly foreign keys (e.g. singer_in_concert)
            fk_cols = {fk[3].lower() for fk in fks}
            if len(fks) >= 2 and len(cols) - len(fk_cols) <= 1:
                junction[t.lower()] = {fk[2].lower() for fk in fks}
            for fk in fks:
                a, b = t.lower(), fk[2].lower()
                if b in adj and a != b:
                    adj[a].add(b)
                    adj[b].add(a)
        return tables, docs, adj, junction
    finally:
        conn.close()


def link_tables(db_path: str, question: str, evidence: str | None = None, k: int = 4) -> tuple[str, ...]:
    """Tables to show the model, in schema order."""
    tables, docs, adj, junction = _db_index(db_path)
    if len(tables) <= k:
        return tuple(tables)
    query = tokenize(f"{question} {evidence or ''}")
    scores = BM25Okapi(docs).get_scores(query) if query else [0.0] * len(tables)
    ranked = sorted(range(len(tables)), key=lambda i: -scores[i])
    keep = {tables[i].lower() for i in ranked[:k] if scores[i] > 0} or {tables[ranked[0]].lower()}
    # a kept junction table is useless without the tables it links
    for j in [t for t in keep if t in junction]:
        keep |= {x for x in junction[j] if x in adj}
    # bridge tables: connect kept tables that are two FK hops apart
    for a in list(keep):
        for b in list(keep):
            if a < b and b not in adj[a]:
                keep |= adj[a] & adj[b]
    return tuple(t for t in tables if t.lower() in keep)


def linked_schema(db_path: str, question: str, evidence: str | None = None, k: int = 4, sample_rows: int = 3) -> str:
    return serialize_schema(db_path, sample_rows, link_tables(db_path, question, evidence, k))


def gold_tables(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table)} - ctes


def main() -> None:
    from data.prepare import load_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()
    data = load_jsonl(args.data)
    print(f"{len(data)} examples")
    for k in args.k:
        hit = kept = total = chars_full = chars_linked = 0
        for r in data:
            db = str(REPO / r["db_path"])
            tabs = link_tables(db, r["question"], r.get("evidence"), k)
            gold = gold_tables(r["gold_sql"])
            hit += gold <= {t.lower() for t in tabs}
            kept += len(tabs)
            total += len(_db_index(db)[0])
            chars_full += len(serialize_schema(db, 3))
            chars_linked += len(serialize_schema(db, 3, tabs))
        n = len(data)
        print(f"k={k}: all gold tables kept {100 * hit / n:.1f}% | tables kept {kept / n:.1f} of {total / n:.1f} "
              f"| schema size {100 * chars_linked / chars_full:.0f}% of full")


if __name__ == "__main__":
    main()
