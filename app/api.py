"""SQLForge API.

    POST /v1/generate-sql   question -> guarded, executed SQL (+ result rows, trace id)
    POST /v1/feedback       thumbs up/down on an answer -> Langfuse score + metric
    GET  /v1/databases      demo database catalogue (Spider + BIRD dev)
    GET  /health            liveness + LLM backend reachability
    GET  /metrics           Prometheus metrics

Run (Windows .venv):  uvicorn app.api:app --port 8080
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app import service
from app.tracing import make_tracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sqlforge.api")
tracer = make_tracer()

# ---------------------------------------------------------------- metrics
LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
REQUESTS = Counter("sqlforge_requests_total", "generate-sql requests by outcome", ["outcome", "db_id"])
LATENCY = Histogram("sqlforge_request_seconds", "end-to-end generate-sql latency", buckets=LAT_BUCKETS)
STEP_LATENCY = Histogram("sqlforge_step_seconds", "pipeline step latency", ["step"], buckets=LAT_BUCKETS)
BLOCKED = Counter("sqlforge_guardrail_blocks_total", "requests blocked by a guardrail", ["rule"])
REPAIRS = Counter("sqlforge_repairs_total", "self-correction outcomes", ["result"])
TOKENS = Counter("sqlforge_llm_tokens_total", "LLM tokens", ["kind"])
ROWS = Histogram("sqlforge_result_rows", "rows returned", buckets=(0, 1, 5, 10, 50, 100, 500, 1000))
FEEDBACK = Counter("sqlforge_feedback_total", "user feedback", ["value"])
IN_FLIGHT = Gauge("sqlforge_in_flight_requests", "requests being processed")
LLM_UP = Gauge("sqlforge_llm_up", "1 if the LLM backend answered the last health probe")


class GenerateRequest(BaseModel):
    db_id: str = Field(..., examples=["concert_singer"])
    question: str = Field(..., examples=["How many singers are there?"])
    evidence: str | None = Field(None, description="optional domain hint (BIRD-style)")
    execute: bool = True


class FeedbackRequest(BaseModel):
    trace_id: str
    value: int = Field(..., ge=-1, le=1, description="1 = correct, -1 = wrong")
    comment: str | None = None


app = FastAPI(title="SQLForge", version="0.5.0",
              description="Fine-tuned text-to-SQL with guardrails, self-correction and tracing.")


@app.post("/v1/generate-sql")
async def generate_sql(req: GenerateRequest) -> dict:
    IN_FLIGHT.inc()
    t0 = time.perf_counter()
    try:
        ans = await run_in_threadpool(service.answer, req.db_id, req.question, req.evidence, req.execute, tracer)
    except Exception as e:  # LLM backend down, etc.
        REQUESTS.labels("backend_error", req.db_id).inc()
        log.exception("pipeline failed")
        raise HTTPException(status_code=502, detail=f"LLM backend error: {e}") from e
    finally:
        IN_FLIGHT.dec()
        LATENCY.observe(time.perf_counter() - t0)

    outcome = "ok" if ans.ok else ("blocked" if ans.blocked_rule else "failed")
    REQUESTS.labels(outcome, req.db_id).inc()
    if ans.blocked_rule:
        BLOCKED.labels(ans.blocked_rule).inc()
    if ans.attempts > 1:
        REPAIRS.labels("fixed" if ans.repaired else "not_fixed").inc()
    for s in ans.steps:
        STEP_LATENCY.labels(s.name.split("_")[0] if s.name.startswith("repair") else s.name).observe(s.ms / 1000)
    TOKENS.labels("prompt").inc(ans.prompt_tokens)
    TOKENS.labels("completion").inc(ans.completion_tokens)
    if ans.ok:
        ROWS.observe(len(ans.rows))
    out = ans.to_dict()
    out["trace_url"] = await run_in_threadpool(tracer.trace_url, ans.trace_id)
    return out


@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    FEEDBACK.labels({1: "up", -1: "down", 0: "neutral"}[req.value]).inc()
    # blocking HTTP call: run it off the event loop, or it stalls every concurrent request
    stored = await run_in_threadpool(tracer.score, req.trace_id, "user_feedback", req.value, req.comment)
    return {"ok": True, "stored_in_langfuse": stored}


@app.get("/v1/databases")
async def list_databases() -> dict:
    dbs = service.databases()
    return {"databases": [{"db_id": k, "dataset": "bird" if "/bird/" in v.replace("\\", "/") else "spider"}
                          for k in dbs for v in [dbs[k]]]}


def _llm_up() -> bool:
    try:
        with urllib.request.urlopen(service.LLM_BASE_URL.rstrip("/") + "/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


@app.get("/health")
async def health() -> dict:
    up = await run_in_threadpool(_llm_up)
    LLM_UP.set(1 if up else 0)
    return {"status": "ok", "llm_backend": service.LLM_BASE_URL, "llm_up": up, "model": service.LLM_MODEL,
            "tracing": tracer.enabled}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("shutdown")
def _flush() -> None:
    tracer.flush()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8080")))
