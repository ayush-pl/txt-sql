"""SQLForge live demo for a free Hugging Face Space (2 vCPU, 16 GB RAM, no GPU).

Runs the real serving pipeline from `app/service.py` (input guard -> schema -> LLM -> SQL guard ->
read-only execution -> column-name fix / LLM self-correction), with two swaps for CPU hosting:
  - the LLM is the 4-bit GGUF (Q4_K_M) run in-process by llama.cpp instead of vLLM on a GPU;
  - the database catalogue is the 18 small Spider dev databases bundled with the eval gate.

`scripts/deploy_space.py` assembles this file, the pipeline modules and the databases into a
Space. To run it locally from the repo:

    MODEL_PATH=models/gguf/sqlforge-3b-Q4_K_M.gguf python space/demo.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Space: the pipeline packages sit next to this file. Repo checkout: they are one level up.
for p in (HERE, HERE.parent):
    if (p / "app" / "service.py").exists():
        sys.path.insert(0, str(p))
        break

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from app import service  # noqa: E402
from data.schema import serialize_schema  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "gguf/sqlforge-3b-Q4_K_M.gguf")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 2)))
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/ayush-pl/txt-sql")
VIDEO_URL = os.environ.get("VIDEO_URL", "")

DB_DIR = next(d for d in (HERE / "dbs", HERE.parent / "eval" / "gate" / "dbs") if d.exists())
CATALOGUE = {p.parent.name: str(p) for p in sorted(DB_DIR.glob("*/*.sqlite"))}

EXAMPLES = [
    ["concert_singer", "Which stadium hosted the most concerts?"],
    ["concert_singer", "Show the names of singers older than the average age."],
    ["world_1", "Which 5 countries have the largest population?"],
    ["world_1", "What languages are spoken in Aruba?"],
    ["pets_1", "Find the first names of students who own a cat."],
    ["car_1", "Which car maker has produced the most car models?"],
    ["flight_2", "Which airline has the most flights?"],
    ["employee_hire_evaluation", "Which employee received the biggest bonus? Give me the name."],
    ["concert_singer", "Delete all singers"],  # shows the guardrail: the model writes DELETE, it is blocked
]


def _model_path() -> str:
    if MODEL_PATH:
        return MODEL_PATH
    if not MODEL_REPO:
        raise SystemExit("set MODEL_REPO (Hub repo with the GGUF) or MODEL_PATH (local .gguf file)")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_REPO, MODEL_FILE)


print(f"loading {MODEL_PATH or MODEL_REPO + '/' + MODEL_FILE} with {N_THREADS} threads", flush=True)
LLM = Llama(model_path=_model_path(), n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
LLM_LOCK = threading.Lock()  # one llama.cpp context: one generation at a time
STOP = ["<|im_end|>", "<|endoftext|>"]


def llama_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> tuple[str, dict]:
    """Drop-in replacement for service.chat (same signature and return shape)."""
    with LLM_LOCK:
        resp = LLM.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens,
                                          stop=STOP, seed=0)
    return resp["choices"][0]["message"]["content"] or "", resp.get("usage") or {}


service.chat = llama_chat
service.databases = lambda: CATALOGUE
service.LLM_MODEL = "sqlforge-3b-Q4_K_M (llama.cpp, CPU)"


def warm(db_id: str) -> None:
    """Prefill the schema prompt for a database so the first question on it is faster.

    llama.cpp reuses the longest matching token prefix from the previous call, and the prompt is
    system + schema + question, so after this only the question has to be processed.
    """
    if db_id not in CATALOGUE or not LLM_LOCK.acquire(blocking=False):
        return
    try:
        from data.prompt import build_messages

        msgs = build_messages(serialize_schema(CATALOGUE[db_id], 3), "")
        LLM.create_chat_completion(messages=msgs, max_tokens=1, temperature=0.0)
    except Exception as e:  # warm-up is best effort
        print(f"warm-up failed for {db_id}: {e}", flush=True)
    finally:
        LLM_LOCK.release()


def schema_text(db_id: str) -> str:
    return serialize_schema(CATALOGUE[db_id], 0) if db_id in CATALOGUE else ""


def on_db_change(db_id: str) -> str:
    threading.Thread(target=warm, args=(db_id,), daemon=True).start()
    return schema_text(db_id)


def ask(db_id: str, question: str):
    question = (question or "").strip()
    if not question:
        return "", pd.DataFrame(), "Type a question first.", []
    t0 = time.perf_counter()
    try:
        ans = service.answer(db_id, question)
    except Exception as e:
        return "", pd.DataFrame(), f"**Error:** {type(e).__name__}: {e}", []
    secs = time.perf_counter() - t0

    if ans.ok:
        status = f"✅ **{len(ans.rows)} row(s)** in {secs:.1f} s"
        if ans.repaired:
            status += f" · 🔧 fixed by self-correction ({ans.attempts} LLM call(s))"
    elif ans.blocked_rule:
        status = f"🛡️ **Blocked by guardrail** `{ans.blocked_rule}`: {ans.error}"
    else:
        status = f"❌ **Failed:** {ans.error}"
    status += f"  \nTokens: {ans.prompt_tokens} prompt / {ans.completion_tokens} generated"

    df = pd.DataFrame(ans.rows[:200], columns=ans.columns or None) if ans.rows else pd.DataFrame()
    steps = [[s.name, s.ms, ", ".join(f"{k}={v}" for k, v in s.detail.items() if v is not None)]
             for s in ans.steps]
    return ans.sql or "", df, status, steps


links = f"[GitHub]({GITHUB_URL})"
if MODEL_REPO:
    links += f" · [Model](https://huggingface.co/{MODEL_REPO})"
if VIDEO_URL:
    links += f" · [Video of the GPU version]({VIDEO_URL})"

INTRO = f"""# SQLForge: text-to-SQL with a fine-tuned 3B model
{links}

Ask a question in plain English about a SQLite database. **SQLForge-3B** (Qwen2.5-Coder-3B fine-tuned with
QLoRA on Spider + BIRD: 80.5% execution accuracy on Spider dev, 54.1% on BIRD dev) writes the SQL. The query
then passes through guardrails (read-only, single statement, known tables only, row limit), runs against the
database, and is repaired automatically if it fails.

⏱️ This free Space runs a 4-bit GGUF on **2 CPU cores**, so an answer takes about 10–25 s. On a GPU
(vLLM FP8) the same pipeline answers in about 0.5 s.
"""

with gr.Blocks(title="SQLForge demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        db = gr.Dropdown(list(CATALOGUE), value="concert_singer", label="Database", scale=1)
        q = gr.Textbox(label="Question", placeholder="e.g. Which stadium hosted the most concerts?", scale=3)
    go = gr.Button("Generate SQL and run it", variant="primary")
    status = gr.Markdown()
    sql = gr.Code(label="Generated SQL (after guardrails)", language="sql", interactive=False)
    table = gr.Dataframe(label="Result (first 200 rows)", interactive=False, wrap=True)
    with gr.Accordion("Pipeline steps", open=False):
        steps = gr.Dataframe(headers=["step", "ms", "detail"], interactive=False, wrap=True)
    with gr.Accordion("Database schema", open=False):
        schema = gr.Code(value=schema_text("concert_singer"), language="sql", interactive=False)
    gr.Examples(EXAMPLES, inputs=[db, q], label="Try one")
    gr.Markdown("Databases: [Spider 1.0](https://yale-lily.github.io/spider) dev set (Yu et al., 2018, "
                "CC BY-SA 4.0), never seen in training.")

    db.change(on_db_change, db, schema)
    go.click(ask, [db, q], [sql, table, status, steps], concurrency_limit=1)
    q.submit(ask, [db, q], [sql, table, status, steps], concurrency_limit=1)
    demo.load(lambda: threading.Thread(target=warm, args=("concert_singer",), daemon=True).start())

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
