"""Tests for Week 3 components: error taxonomy, schema linking, self-correction hints."""

import sqlite3

import pytest

from data.schema import serialize_schema
from data.schema_linker import gold_tables, link_tables, tokenize
from eval.error_analysis import categorize
from eval.self_correct import error_hint, repair_messages


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "w3.sqlite"
    c = sqlite3.connect(p)
    c.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE concert (id INTEGER PRIMARY KEY, venue TEXT, year INT);
        CREATE TABLE singer_in_concert (singer_id INT, concert_id INT,
            FOREIGN KEY (singer_id) REFERENCES singer(id),
            FOREIGN KEY (concert_id) REFERENCES concert(id));
        CREATE TABLE stadium (id INTEGER PRIMARY KEY, capacity INT, "Stadium Name" TEXT);
        CREATE TABLE weather (day TEXT, rainfall REAL);
        INSERT INTO singer VALUES (1,'Ann','France'),(2,'Bo','Japan');
        INSERT INTO concert VALUES (1,'Arena',2014);
        INSERT INTO stadium VALUES (1,5000,'Big One');
        """
    )
    c.commit()
    c.close()
    return str(p)


def _row(pred, gold, error_type="wrong_result"):
    return {"ex": False, "error_type": error_type, "pred_sql": pred, "gold_sql": gold}


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("SELECT name FROM singer", "SELECT T1.name FROM singer T1 JOIN singer_in_concert T2 ON T1.id = T2.singer_id",
         "wrong_tables"),
        ("SELECT count(*) FROM singer", "SELECT max(id) FROM singer", "wrong_aggregation"),
        ("SELECT name FROM singer WHERE country = 'FR'", "SELECT name FROM singer WHERE country = 'France'",
         "wrong_filter_value"),
        ("SELECT name FROM singer WHERE id > 1", "SELECT name FROM singer WHERE country = 'France'", "wrong_filter"),
        ("SELECT name FROM singer ORDER BY id LIMIT 1", "SELECT name FROM singer ORDER BY id DESC LIMIT 1",
         "wrong_order_limit"),
        ("SELECT country FROM singer", "SELECT DISTINCT country FROM singer", "wrong_distinct"),
        ("SELECT name, country FROM singer", "SELECT name FROM singer", "wrong_select"),
    ],
)
def test_categorize_wrong_results(pred, gold, expected):
    assert categorize(_row(pred, gold)) == expected


def test_categorize_exec_error_and_correct():
    assert categorize(_row("SELECT x FROM singer", "SELECT name FROM singer", "unknown_column")) == "unknown_column"
    assert categorize({"ex": True, "pred_sql": "", "gold_sql": ""}) is None
    assert categorize({"ex": None, "pred_sql": "", "gold_sql": ""}) is None


def test_tokenize_splits_identifiers():
    assert tokenize("singerAge song_titles") == ["singer", "age", "song", "title"]
    assert tokenize("What is the name") == []  # stop words, incl. the ubiquitous "name"


def test_link_tables_keeps_relevant_and_bridge(db):
    tabs = link_tables(db, "Which singers performed in concerts at the Arena venue?", k=2)
    assert {"singer", "concert"} <= set(tabs)
    assert "singer_in_concert" in tabs  # bridge table between singer and concert
    assert "weather" not in tabs


def test_link_tables_small_db_returns_all(db):
    assert len(link_tables(db, "anything", k=10)) == 5


def test_serialize_schema_subset(db):
    s = serialize_schema(db, 0, ("singer",))
    assert "CREATE TABLE singer" in s and "concert" not in s


def test_gold_tables_ignores_ctes():
    sql = "WITH x AS (SELECT * FROM singer) SELECT * FROM x JOIN concert ON 1"
    assert gold_tables(sql) == {"singer", "concert"}


def test_error_hint_lists_columns_with_aliases(db):
    h = error_hint(db, "SELECT T2.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.id", "no such column: T2.name")
    assert "concert (alias T2): id, venue, year" in h
    assert "singer (alias T1): id, name, country" in h


def test_error_hint_unparseable_lists_quoted_names(db):
    h = error_hint(db, "SELECT Stadium Name( FROM", "syntax error")
    assert 'stadium."Stadium Name"' in h


def test_repair_messages_shape():
    msgs = repair_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], "SELECT 1", "boom", "hint\n")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "boom" in msgs[-1]["content"] and "hint" in msgs[-1]["content"]


def test_token_batches_respects_budget():
    from eval.generate import token_batches

    lens = [100, 200, 300, 1500, 3000]
    b = token_batches(list(range(5)), lens, max_batch=4, max_tokens=4096, new_tokens=256)
    assert b == [[0, 1, 2], [3], [4]]  # 3*(300+256) fits; 1500-token prompt alone; 3000 alone
    for batch in b:
        assert len(batch) * (max(lens[i] for i in batch) + 256) <= 4096 or len(batch) == 1
