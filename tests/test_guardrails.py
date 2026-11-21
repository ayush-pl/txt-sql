import sqlite3

import pytest

from app.guardrails import check_question, check_sql
from eval.metrics import execute_sql

TABLES = {"singer", "concert"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM singer",
        "SELECT count(*) FROM singer AS s JOIN concert AS c ON s.id = c.singer_id",
        "WITH x AS (SELECT id FROM singer) SELECT * FROM x",
        "SELECT name FROM singer UNION SELECT venue FROM concert",
        "SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert)",
    ],
)
def test_allows_read_only(sql):
    r = check_sql(sql, TABLES)
    assert r.ok, r.reason


@pytest.mark.parametrize(
    "sql,rule",
    [
        ("DROP TABLE singer", "sql_not_select"),
        ("DELETE FROM singer", "sql_not_select"),
        ("UPDATE singer SET name = 'x'", "sql_not_select"),
        ("INSERT INTO singer VALUES (1, 'x')", "sql_not_select"),
        ("SELECT 1; DROP TABLE singer", "sql_multi_statement"),
        ("PRAGMA table_info(singer)", "sql_not_select"),
        ("ATTACH DATABASE '/tmp/x.db' AS x", "sql_not_select"),
        ("SELECT load_extension('evil')", "sql_forbidden_func"),
        ("SELECT * FROM sqlite_master", "sql_system_table"),
        ("SELECT * FROM users", "sql_unknown_table"),
        ("SELEC name FROM", "sql_parse_error"),
        ("", "sql_empty"),
    ],
)
def test_blocks(sql, rule):
    r = check_sql(sql, TABLES)
    assert not r.ok and r.rule == rule, (r.rule, r.reason)


def test_limit_injected_and_tightened():
    r = check_sql("SELECT name FROM singer", TABLES, row_limit=50)
    assert r.ok and r.sql.upper().endswith("LIMIT 50") and r.warnings
    r = check_sql("SELECT name FROM singer LIMIT 10", TABLES, row_limit=50)
    assert r.ok and "LIMIT 10" in r.sql.upper() and not r.warnings
    r = check_sql("SELECT name FROM singer LIMIT 100000", TABLES, row_limit=50)
    assert r.ok and "LIMIT 50" in r.sql.upper()


def test_question_checks():
    assert check_question("How many singers?").ok
    assert check_question("   ").rule == "input_empty"
    assert check_question("x" * 2000).rule == "input_too_long"
    assert check_question("hi\x00there").rule == "input_control_chars"


def test_execution_layer_is_read_only(tmp_path):
    """Even if a write slipped past static checks, the read-only connection refuses it."""
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE singer (id INT, name TEXT)")
    c.commit()
    c.close()
    res = execute_sql(str(p), "INSERT INTO singer VALUES (1, 'x')")
    assert not res.ok and "readonly" in res.error.replace(" ", "").lower()
