import sqlite3

import pytest

from data.prompt import extract_sql
from data.schema import serialize_schema
from eval.metrics import (
    classify_error,
    exact_match,
    execute_sql,
    has_top_level_order_by,
    results_match,
    spider_hardness,
)


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INT, country TEXT);
        CREATE TABLE song (id INTEGER PRIMARY KEY, singer_id INT, title TEXT,
                           FOREIGN KEY (singer_id) REFERENCES singer(id));
        INSERT INTO singer VALUES (1,'Ann',30,'US'),(2,'Bo',25,'UK'),(3,'Cy',41,'US');
        INSERT INTO song VALUES (1,1,'A'),(2,1,'B'),(3,3,'C');
        """
    )
    c.commit()
    c.close()
    return str(p)


def test_execute_ok_and_error(db):
    assert execute_sql(db, "SELECT count(*) FROM singer").rows == [(3,)]
    bad = execute_sql(db, "SELECT nope FROM singer")
    assert not bad.ok and classify_error(bad) == "unknown_column"
    assert classify_error(execute_sql(db, "SELEC x")) == "syntax_error"


def test_execute_is_read_only(db):
    r = execute_sql(db, "DELETE FROM singer")
    assert not r.ok
    assert execute_sql(db, "SELECT count(*) FROM singer").rows == [(3,)]


def test_timeout(db):
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    r = execute_sql(db, slow, timeout_s=0.5)
    assert r.error == "timeout"


def test_results_match_semantics():
    assert results_match([(1,), (2,)], [(2,), (1,)])                     # unordered
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)  # ordered
    assert results_match([("a", 1)], [(1, "a")])                         # column order tolerated
    assert not results_match([(1,), (1,)], [(1,)])                       # multiset (spider)
    assert results_match([(1,), (1,)], [(1,)], mode="bird")              # set (bird)
    assert results_match([(0.1 + 0.2,)], [(0.3,)])                       # float rounding


def test_exact_match_and_order_by():
    assert exact_match("select NAME from singer", "SELECT name FROM singer")
    assert not exact_match("SELECT age FROM singer", "SELECT name FROM singer")
    assert has_top_level_order_by("SELECT name FROM singer ORDER BY age")
    assert not has_top_level_order_by("SELECT name FROM (SELECT * FROM singer ORDER BY age)")


def test_hardness_buckets():
    assert spider_hardness("SELECT count(*) FROM singer") == "easy"
    assert spider_hardness(
        "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer) "
        "INTERSECT SELECT name FROM singer WHERE country = 'US' ORDER BY name LIMIT 1"
    ) in ("hard", "extra")


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```\nThis query...") == "SELECT 1"
    assert extract_sql("SELECT a\nFROM t;  -- done") == "SELECT a FROM t"
    assert extract_sql("```sql\nSELECT 2") == "SELECT 2"


def test_serialize_schema(db):
    s = serialize_schema(db, sample_rows=2)
    assert "CREATE TABLE singer" in s and "PRIMARY KEY (id)" in s
    assert "FOREIGN KEY (singer_id) REFERENCES singer(id)" in s
    assert "-- 1 | Ann | 30 | US" in s
