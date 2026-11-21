"""The eval gate passes the committed predictions and fails broken ones (CI-safe: no model)."""

import json
import shutil

import pytest

from eval import gate


@pytest.fixture()
def preds(tmp_path):
    p, m = tmp_path / "preds.jsonl", tmp_path / "meta.json"
    shutil.copy(gate.PREDS, p)
    shutil.copy(gate.META, m)
    return p, m


def _rewrite(path, fn):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    path.write_text("".join(json.dumps(r) + "\n" for r in fn(rows)), encoding="utf-8")


def test_committed_predictions_pass(preds):
    assert gate.check(*preds) == 0


def test_broken_model_fails(preds):
    p, m = preds
    # a "broken model": a third of the answers become syntactically valid but wrong SQL
    _rewrite(p, lambda rows: [{**r, "pred_sql": "SELECT 1"} if i % 3 == 0 else r for i, r in enumerate(rows)])
    assert gate.check(p, m) == 1


def test_garbage_output_fails_valid_sql(preds):
    p, m = preds
    _rewrite(p, lambda rows: [{**r, "pred_sql": "Sure! Here is the query you asked for"} if i % 5 == 0 else r
                              for i, r in enumerate(rows)])
    assert gate.check(p, m) == 1


def test_missing_predictions_fail(preds):
    p, m = preds
    _rewrite(p, lambda rows: rows[:-10])
    assert gate.check(p, m) == 1


def test_stale_prompt_fails(preds):
    p, m = preds
    meta = json.loads(m.read_text())
    meta["prompt_fingerprint"] = "0000deadbeef"
    m.write_text(json.dumps(meta))
    assert gate.check(p, m) == 1
