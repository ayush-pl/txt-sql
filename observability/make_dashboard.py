"""Generate the Grafana dashboard JSON (observability/grafana/dashboards/sqlforge.json).

Two rows: the SQLForge API (what users experience: requests, latency, guardrails, repairs,
feedback) and the vLLM engine underneath (queue, KV cache, TTFT, prefix-cache hit rate, tokens/s).

    python observability/make_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "grafana" / "dashboards" / "sqlforge.json"
DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def panel(title, targets, x, y, w=8, h=8, kind="timeseries", unit=None, desc="", stack=False, extra=None):
    global _id
    _id += 1
    p = {
        "id": _id, "type": kind, "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"datasource": DS, "expr": e, "legendFormat": lg, "refId": chr(65 + i)}
                    for i, (e, lg) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit or "short"}, "overrides": []},
        "options": {},
    }
    if kind == "timeseries":
        p["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 10,
                                                  "stacking": {"mode": "normal" if stack else "none"}}
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"}
    if extra:
        p["fieldConfig"]["defaults"].update(extra)
    return p


def row(title, y):
    global _id
    _id += 1
    return {"id": _id, "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}}


R = "[$__rate_interval]"
panels = [
    row("SQLForge API", 0),
    panel("Requests / s by outcome", [(f'sum by (outcome) (rate(sqlforge_requests_total{R}))', "{{outcome}}")],
          0, 1, stack=True, unit="reqps"),
    panel("End-to-end latency", [
        (f"histogram_quantile(0.5, sum by (le) (rate(sqlforge_request_seconds_bucket{R})))", "p50"),
        (f"histogram_quantile(0.95, sum by (le) (rate(sqlforge_request_seconds_bucket{R})))", "p95"),
    ], 8, 1, unit="s"),
    panel("Step latency p95", [
        (f"histogram_quantile(0.95, sum by (le, step) (rate(sqlforge_step_seconds_bucket{R})))", "{{step}}")],
        16, 1, unit="s", desc="input_guard, schema, llm, repair, sql_guard, execute"),
    panel("Success rate", [(
        f'sum(rate(sqlforge_requests_total{{outcome="ok"}}{R})) / sum(rate(sqlforge_requests_total{R}))', "ok")],
        0, 9, w=4, kind="stat", unit="percentunit", extra={"min": 0, "max": 1}),
    panel("In-flight requests", [("sum(sqlforge_in_flight_requests)", "in flight")], 4, 9, w=4, kind="stat"),
    panel("LLM backend up", [("max(sqlforge_llm_up)", "up")], 8, 9, w=4, kind="stat",
          extra={"mappings": [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"},
                                                           "1": {"text": "UP", "color": "green"}}}]}),
    panel("Guardrail blocks by rule", [(f"sum by (rule) (increase(sqlforge_guardrail_blocks_total{R}))", "{{rule}}")],
          12, 9, w=6, stack=True, desc="DROP/DELETE, multi-statement, unknown tables, bad input, ..."),
    panel("Self-correction outcomes", [(f"sum by (result) (increase(sqlforge_repairs_total{R}))", "{{result}}")],
          18, 9, w=6, stack=True),
    panel("LLM tokens / s", [(f"sum by (kind) (rate(sqlforge_llm_tokens_total{R}))", "{{kind}}")],
          0, 17, w=8, unit="short"),
    panel("User feedback", [(f"sum by (value) (increase(sqlforge_feedback_total{R}))", "{{value}}")],
          8, 17, w=8, stack=True),
    panel("Requests by database (top 8)", [("topk(8, sum by (db_id) (increase(sqlforge_requests_total[1h])))", "{{db_id}}")],
          16, 17, w=8, kind="bargauge"),
    row("vLLM engine", 25),
    panel("Running / waiting requests", [("sum(vllm:num_requests_running)", "running"),
                                         ("sum(vllm:num_requests_waiting)", "waiting")], 0, 26),
    panel("KV-cache usage", [("max(vllm:kv_cache_usage_perc)", "kv cache")], 8, 26, unit="percentunit",
          extra={"min": 0, "max": 1}),
    panel("Prefix-cache hit rate", [(
        f"sum(rate(vllm:prefix_cache_hits_total{R})) / sum(rate(vllm:prefix_cache_queries_total{R}))", "hit rate")],
        16, 26, unit="percentunit", extra={"min": 0, "max": 1},
        desc="schema prompts repeat per database, so most prefill comes from cache"),
    panel("Time to first token", [
        (f"histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{R})))", "p50"),
        (f"histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{R})))", "p95"),
    ], 0, 34, unit="s"),
    panel("Tokens / s", [(f"sum(rate(vllm:prompt_tokens_total{R}))", "prompt (prefill)"),
                         (f"sum(rate(vllm:generation_tokens_total{R}))", "generation")], 8, 34),
    panel("Queue time p95", [
        (f"histogram_quantile(0.95, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{R})))", "p95")],
        16, 34, unit="s"),
]

dashboard = {
    "uid": "sqlforge", "title": "SQLForge: text-to-SQL service", "tags": ["sqlforge", "llm", "vllm"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"}, "panels": panels, "templating": {"list": []},
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(panels)} panels)")
