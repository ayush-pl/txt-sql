"""SQLForge demo UI: pick a database, ask a question, see the SQL, the result, the pipeline
steps and a link to the Langfuse trace; rate the answer (feedback goes to Langfuse + Prometheus).

    streamlit run app/ui.py          (API must be running on :8080)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

API = os.environ.get("SQLFORGE_API", "http://127.0.0.1:8080")

EXAMPLES = {
    "concert_singer": ["How many singers are there?", "Which stadium hosted the most concerts?",
                       "Show the names of singers older than the average age."],
    "world_1": ["Which 5 countries have the largest population?", "What languages are spoken in Aruba?",
                "How many countries are in each continent?"],
    "california_schools": ["What is the highest eligible free rate for K-12 students in Alameda County?"],
    "pets_1": ["How many dogs are older than 3?", "Find the first names of students who own a cat."],
}


def call(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


st.set_page_config(page_title="SQLForge", page_icon="🗄️", layout="wide")
st.title("SQLForge")
st.caption("Fine-tuned Qwen2.5-Coder-3B (QLoRA, Spider + BIRD) · vLLM FP8 · guardrails · self-correction · Langfuse traces")

try:
    health = call("/health", timeout=5)
    dbs = [d["db_id"] for d in call("/v1/databases", timeout=10)["databases"]]
except (urllib.error.URLError, TimeoutError) as e:
    st.error(f"API not reachable at {API}: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Backend")
    st.write(("🟢" if health["llm_up"] else "🔴") + f" LLM `{health['model']}` at `{health['llm_backend']}`")
    st.write(("🟢" if health["tracing"] else "⚪") + " Langfuse tracing")
    st.markdown("[Langfuse](http://localhost:3000) · [Grafana](http://localhost:3001) · [API docs](http://127.0.0.1:8080/docs)")
    st.subheader("Database")
    default = dbs.index("concert_singer") if "concert_singer" in dbs else 0
    db_id = st.selectbox("Database", dbs, index=default, label_visibility="collapsed")
    st.caption("Spider and BIRD dev databases (never seen in training).")

examples = EXAMPLES.get(db_id, [])
if examples:
    pick = st.pills("Examples", examples, selection_mode="single") if hasattr(st, "pills") else None
else:
    pick = None
question = st.text_input("Question", value=pick or "", placeholder="Ask a question about this database")
evidence = st.text_input("Hint (optional)", placeholder="Domain knowledge, e.g. 'eligible free rate = free meals / enrollment'")

if st.button("Generate SQL", type="primary", disabled=not question.strip()):
    with st.spinner("Generating..."):
        try:
            st.session_state["last"] = call("/v1/generate-sql", {"db_id": db_id, "question": question,
                                                                  "evidence": evidence or None})
        except urllib.error.HTTPError as e:
            st.error(f"API error {e.code}: {e.read().decode()[:300]}")

ans = st.session_state.get("last")
if ans:
    if ans["blocked_rule"]:
        st.error(f"Blocked by guardrail `{ans['blocked_rule']}`: {ans['error']}")
    elif not ans["ok"]:
        st.warning(ans["error"])
    if ans["sql"]:
        st.code(ans["sql"], language="sql")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{ans['total_ms'] / 1000:.2f} s")
    c2.metric("Attempts", ans["attempts"], help="more than 1 = self-corrected after an execution error")
    c3.metric("Rows", len(ans["rows"]))
    c4.metric("Tokens", f"{ans['prompt_tokens']} / {ans['completion_tokens']}", help="prompt / completion")
    if ans["ok"]:
        st.dataframe(pd.DataFrame(ans["rows"], columns=ans["columns"] or None), width="stretch")
    if ans.get("linked_tables"):
        st.caption("Schema linking kept: " + ", ".join(ans["linked_tables"]))
    with st.expander("Pipeline steps"):
        st.dataframe(pd.DataFrame([{"step": s["name"], "ms": s["ms"], **{k: str(v)[:80] for k, v in s["detail"].items()}}
                                   for s in ans["steps"]]), width="stretch")
    if ans.get("trace_url"):
        st.markdown(f"[Open trace in Langfuse]({ans['trace_url']})")
    if ans.get("trace_id"):
        f1, f2, _ = st.columns([1, 1, 6])
        if f1.button("👍 Correct"):
            call("/v1/feedback", {"trace_id": ans["trace_id"], "value": 1})
            st.toast("Thanks: feedback recorded")
        if f2.button("👎 Wrong"):
            call("/v1/feedback", {"trace_id": ans["trace_id"], "value": -1})
            st.toast("Thanks: feedback recorded")
