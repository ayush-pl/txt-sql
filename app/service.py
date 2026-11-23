"""Text-to-SQL pipeline used by the API:

    question -> input guard -> schema (linked if wide) -> LLM -> SQL guard -> execute (read-only)
             -> on error: self-correct with schema hints (<= N rounds) -> result

The LLM is any OpenAI-compatible server (vLLM in WSL by default, or Ollama), called with the
standard library so it runs on Windows. Every step is timed and returned, so the API can export
metrics (Prometheus) and traces (Langfuse) without the pipeline knowing about either.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from app.guardrails import check_question, check_sql
from app.tracing import NoopTracer
from data.prompt import build_messages, extract_sql
from data.schema import serialize_schema
from data.schema_linker import _db_index, link_tables
from eval.metrics import execute_sql
from eval.self_correct import error_hint, fix_column_names, repair_messages

REPO = Path(__file__).resolve().parent.parent

# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 ::1 first and adds ~2 s per call
LLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.environ.get("MODEL_NAME", "sqlforge")
LINK_ABOVE_TABLES = int(os.environ.get("LINK_ABOVE_TABLES", "12"))  # link only genuinely wide schemas
LINK_K = int(os.environ.get("LINK_K", "6"))
MAX_REPAIRS = int(os.environ.get("MAX_REPAIRS", "2"))
QUERY_TIMEOUT_S = float(os.environ.get("QUERY_TIMEOUT_S", "15"))
ROW_LIMIT = int(os.environ.get("ROW_LIMIT", "1000"))


@dataclass
class Step:
    name: str
    ms: float
    detail: dict = field(default_factory=dict)


@dataclass
class Answer:
    ok: bool
    question: str
    db_id: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    error: str | None = None
    blocked_rule: str | None = None
    attempts: int = 0
    repaired: bool = False
    linked_tables: list[str] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: list[Step] = field(default_factory=list)
    total_ms: float = 0.0
    trace_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def databases() -> dict[str, str]:
    """db_id -> sqlite path, for every prepared Spider/BIRD dev database (demo catalogue)."""
    out = {}
    for name in ("spider_dev.jsonl", "bird_dev.jsonl"):
        p = REPO / "data" / "processed" / name
        if p.exists():
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                out.setdefault(r["db_id"], str(REPO / r["db_path"]))
    return dict(sorted(out.items()))


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> tuple[str, dict]:
    body = json.dumps({"model": LLM_MODEL, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"] or "", resp.get("usage") or {}


def _cols(db_path: str, sql: str) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [d[0] for d in conn.execute(f"SELECT * FROM ({sql}) LIMIT 0").description or []]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def answer(db_id: str, question: str, evidence: str | None = None, execute: bool = True, tracer=None) -> Answer:
    tracer = tracer or NoopTracer()
    dataset = "bird" if "/bird/" in databases().get(db_id, "").replace("\\", "/") else "spider"
    with tracer.attributes(trace_name="text2sql", tags=[db_id, dataset], metadata={"model": LLM_MODEL}),             tracer.observe("text2sql", input={"db_id": db_id, "question": question, "evidence": evidence},
                           metadata={"model": LLM_MODEL}) as root:
        tracer.set_trace_io(root, input={"db_id": db_id, "question": question, "evidence": evidence})
        ans = _answer(db_id, question, evidence, execute, tracer)
        ans.trace_id = root.trace_id
        out = {"ok": ans.ok, "sql": ans.sql, "rows": len(ans.rows), "error": ans.error,
               "blocked_rule": ans.blocked_rule, "attempts": ans.attempts}
        root.update(output=out, level="DEFAULT" if ans.ok else ("WARNING" if ans.blocked_rule else "ERROR"))
        tracer.set_trace_io(root, output=out)
    return ans


def _answer(db_id: str, question: str, evidence: str | None, execute: bool, tracer) -> Answer:
    t_start = time.perf_counter()
    ans = Answer(ok=False, question=question, db_id=db_id)

    def step(name: str, t0: float, **detail) -> None:
        ans.steps.append(Step(name, round(1000 * (time.perf_counter() - t0), 1), detail))

    def done() -> Answer:
        ans.total_ms = round(1000 * (time.perf_counter() - t_start), 1)
        return ans

    t = time.perf_counter()
    with tracer.observe("input_guard", as_type="guardrail", input=question) as o:
        g = check_question(question)
        o.update(output={"ok": g.ok, "rule": g.rule})
    step("input_guard", t, ok=g.ok)
    if not g.ok:
        ans.error, ans.blocked_rule = g.reason, g.rule
        return done()
    db_path = databases().get(db_id)
    if db_path is None:
        ans.error, ans.blocked_rule = f"unknown database '{db_id}'", "unknown_db"
        return done()

    t = time.perf_counter()
    with tracer.observe("schema", as_type="retriever", input={"db_id": db_id}) as o:
        all_tables = _db_index(db_path)[0]
        if len(all_tables) > LINK_ABOVE_TABLES:
            keep = link_tables(db_path, question, evidence, k=LINK_K)
            ans.linked_tables = list(keep)
            schema = serialize_schema(db_path, 3, keep)
        else:
            schema = serialize_schema(db_path, 3)
        o.update(output={"tables": len(all_tables), "linked_tables": ans.linked_tables, "schema_chars": len(schema)})
    step("schema", t, tables=len(all_tables), linked=ans.linked_tables is not None)
    messages = build_messages(schema, question, evidence)

    sql, err = None, None
    for attempt in range(1 + MAX_REPAIRS):
        t = time.perf_counter()
        convo = messages if attempt == 0 else repair_messages(
            messages, sql or "", err or "", error_hint(db_path, sql or "", err or ""))
        temp = 0.0 if attempt <= 1 else 0.7
        with tracer.observe("llm" if attempt == 0 else f"repair_{attempt}", as_type="generation", model=LLM_MODEL,
                            input=convo, model_parameters={"temperature": temp}) as o:
            raw, usage = chat(convo, temperature=temp)
            o.update(output=raw, usage_details={"input": usage.get("prompt_tokens", 0),
                                                "output": usage.get("completion_tokens", 0)})
        ans.prompt_tokens += usage.get("prompt_tokens", 0)
        ans.completion_tokens += usage.get("completion_tokens", 0)
        sql = extract_sql(raw)
        ans.attempts = attempt + 1
        step("llm" if attempt == 0 else f"repair_{attempt}", t, sql=sql)

        t = time.perf_counter()
        with tracer.observe("sql_guard", as_type="guardrail", input=sql) as o:
            gr = check_sql(sql, set(all_tables), ROW_LIMIT)
            o.update(output={"ok": gr.ok, "rule": gr.rule, "reason": gr.reason, "sql": gr.sql},
                     level="DEFAULT" if gr.ok else "WARNING")
        step("sql_guard", t, ok=gr.ok, rule=gr.rule)
        if not gr.ok:
            # a parse error or unknown table is worth repairing; a forbidden operation is not
            if gr.rule in ("sql_parse_error", "sql_unknown_table"):
                err = gr.reason
                continue
            ans.sql, ans.error, ans.blocked_rule = sql, gr.reason, gr.rule
            return done()
        ans.sql = gr.sql
        if not execute:
            ans.ok = True
            return done()

        run_sql = gr.sql
        for fix_round in range(4):  # deterministic column-name fixes are free: try them before the LLM
            t = time.perf_counter()
            with tracer.observe("execute", as_type="tool", input=run_sql) as o:
                res = execute_sql(db_path, run_sql, QUERY_TIMEOUT_S, max_rows=ROW_LIMIT)
                o.update(output={"ok": res.ok, "rows": len(res.rows or []), "error": res.error,
                                 "seconds": round(res.seconds, 4)}, level="DEFAULT" if res.ok else "ERROR")
            step("execute", t, ok=res.ok, rows=len(res.rows or []), error=res.error)
            if res.ok:
                ans.ok, ans.repaired = True, attempt > 0 or fix_round > 0
                ans.sql = run_sql
                ans.rows = [list(r) for r in res.rows]
                ans.columns = _cols(db_path, run_sql)
                return done()
            t = time.perf_counter()
            with tracer.observe("column_fix", as_type="tool", input={"sql": run_sql, "error": res.error}) as o:
                fixed = fix_column_names(db_path, run_sql, res.error)
                o.update(output=fixed)
            step("column_fix", t, fixed=fixed is not None, sql=fixed)
            if fixed is None:
                break
            run_sql = fixed
        sql, err = run_sql, res.error
    ans.error = f"query failed after {ans.attempts} attempts: {err}"
    return done()
