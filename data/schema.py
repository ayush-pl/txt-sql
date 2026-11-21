"""Serialize a SQLite database schema into compact DDL (+ optional sample rows) for prompting."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

_SQL_KEYWORDS = {
    "order", "group", "by", "select", "from", "where", "table", "index", "key", "limit", "join",
    "union", "values", "default", "check", "references", "primary", "foreign", "case", "when",
    "end", "to", "as", "on", "in", "is", "not", "null", "and", "or", "set", "desc", "asc", "all",
}


def _ident(name: str) -> str:
    """Always-quoted identifier, used in queries we execute."""
    return '"' + name.replace('"', '""') + '"'


def _q(name: str) -> str:
    """Quote an identifier for display only when needed (spaces, keywords, leading digit)."""
    if name.replace("_", "").isalnum() and not name[0].isdigit() and name.lower() not in _SQL_KEYWORDS:
        return name
    return _ident(name)


def _fmt_value(v: object, max_len: int = 40) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        return "<blob>"
    s = str(v).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


@lru_cache(maxsize=512)
def serialize_schema(db_path: str, sample_rows: int = 3, tables: tuple[str, ...] | None = None) -> str:
    """Return CREATE TABLE statements for every table (or only `tables`, e.g. from the schema
    linker), each followed by sample rows as comments."""
    keep = tables
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = conn.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY rowid"
            )
        ]
        if keep is not None:
            wanted = {k.lower() for k in keep}
            tables = [t for t in tables if t.lower() in wanted]
        blocks: list[str] = []
        for t in tables:
            cols = cur.execute(f"PRAGMA table_info({_ident(t)})").fetchall()
            fks = cur.execute(f"PRAGMA foreign_key_list({_ident(t)})").fetchall()
            lines = [f"  {_q(c[1])} {c[2] or 'TEXT'}".rstrip() for c in cols]
            pks = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5] > 0]
            if pks:
                lines.append(f"  PRIMARY KEY ({', '.join(_q(p) for p in pks)})")
            for fk in fks:
                # (id, seq, table, from, to, ...)
                to_col = fk[4] or "rowid"
                lines.append(f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(to_col)})")
            block = f"CREATE TABLE {_q(t)} (\n" + ",\n".join(lines) + "\n);"
            if sample_rows > 0:
                try:
                    rows = cur.execute(f"SELECT * FROM {_ident(t)} LIMIT {sample_rows}").fetchall()
                except sqlite3.Error:
                    rows = []
                if rows:
                    header = " | ".join(c[1] for c in cols)
                    body = "\n".join("-- " + " | ".join(_fmt_value(v) for v in r) for r in rows)
                    block += f"\n-- sample rows ({header}):\n{body}"
            blocks.append(block)
        return "\n\n".join(blocks)
    finally:
        conn.close()


def db_path_for(db_root: Path, db_id: str) -> Path:
    return db_root / db_id / f"{db_id}.sqlite"
