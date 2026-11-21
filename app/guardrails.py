"""SQL guardrails: decide whether model-generated SQL may run against a user's database.

Defense in depth. The model's output is treated as untrusted input:
  1. Input checks  - question length, control characters.
  2. Static checks - parse with sqlglot; exactly one statement; read-only statement types only
                     (SELECT / UNION / INTERSECT / EXCEPT / WITH ... SELECT); no DML, DDL,
                     PRAGMA, ATTACH, transactions or writable CTEs; no dangerous functions;
                     every table must exist in the target schema (allowlist).
  3. Rewrites      - a LIMIT is added (or tightened) so a query can't dump a whole table.
  4. Execution     - the database is opened read-only (mode=ro) with a wall-clock timeout
                     (eval.metrics.execute_sql), so even a missed case can't write or hang.

The static layer is about failing fast with a clear reason; layer 4 is the hard guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

MAX_QUESTION_CHARS = 1000
DEFAULT_ROW_LIMIT = 1000

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    t for t in (
        getattr(exp, name, None)
        for name in ("Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable",
                     "TruncateTable", "Command", "Pragma", "Attach", "Detach", "Transaction",
                     "Commit", "Rollback", "Set", "Use", "Grant", "Copy", "LoadData", "Analyze")
    ) if isinstance(t, type)
)
_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
# SQLite functions with side effects or that read the filesystem / extensions
_FORBIDDEN_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile", "sqlar_compress"}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class GuardResult:
    ok: bool
    sql: str | None = None             # rewritten SQL to execute (with LIMIT), if ok
    reason: str | None = None          # why it was blocked
    rule: str | None = None            # machine-readable rule id, used for metrics
    warnings: list[str] = field(default_factory=list)


def check_question(question: str) -> GuardResult:
    q = (question or "").strip()
    if not q:
        return GuardResult(False, reason="empty question", rule="input_empty")
    if len(q) > MAX_QUESTION_CHARS:
        return GuardResult(False, reason=f"question longer than {MAX_QUESTION_CHARS} characters", rule="input_too_long")
    if _CONTROL.search(q):
        return GuardResult(False, reason="question contains control characters", rule="input_control_chars")
    return GuardResult(True)


def check_sql(sql: str, allowed_tables: set[str] | None = None, row_limit: int = DEFAULT_ROW_LIMIT) -> GuardResult:
    """Validate and rewrite one model-generated statement. `allowed_tables` = schema tables."""
    if not sql or not sql.strip():
        return GuardResult(False, reason="empty SQL", rule="sql_empty")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as e:
        return GuardResult(False, reason=f"SQL does not parse: {str(e).splitlines()[0][:200]}", rule="sql_parse_error")
    if len(statements) != 1:
        return GuardResult(False, reason=f"expected exactly one statement, got {len(statements)}", rule="sql_multi_statement")
    tree = statements[0]

    root = tree.this if isinstance(tree, exp.With) else tree  # older sqlglot wraps WITH at the root
    if not isinstance(root, _READ_ONLY_ROOTS):
        return GuardResult(False, reason=f"only read-only SELECT queries are allowed (got {type(root).__name__.upper()})",
                           rule="sql_not_select")
    bad = next((n for n in tree.walk() if isinstance(n, _FORBIDDEN_NODES)), None)
    if bad is not None:
        return GuardResult(False, reason=f"forbidden operation: {type(bad).__name__.upper()}", rule="sql_forbidden_op")
    for f in tree.find_all(exp.Anonymous, exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if name in _FORBIDDEN_FUNCS:
            return GuardResult(False, reason=f"forbidden function: {name}()", rule="sql_forbidden_func")

    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables}
        ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        used = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
        if any(t.startswith("sqlite_") for t in used):
            return GuardResult(False, reason="system tables are not accessible", rule="sql_system_table")
        unknown = sorted(used - allowed)
        if unknown:
            return GuardResult(False, reason=f"unknown table(s): {', '.join(unknown)}", rule="sql_unknown_table")

    warnings = []
    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.this) if limit.expression is not None else int(limit.this)
        except Exception:
            current = None
    if current is None or current > row_limit:
        tree = tree.limit(row_limit, copy=True) if hasattr(tree, "limit") else tree
        warnings.append(f"LIMIT {row_limit} applied")
    return GuardResult(True, sql=tree.sql(dialect="sqlite"), warnings=warnings)
