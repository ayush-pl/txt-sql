"""SQLForge live demo: the real serving pipeline behind a small Gradio page.

Runs `app/service.py` (input guard -> schema -> LLM -> SQL guard -> read-only execution ->
column-name fix / LLM self-correction) on the 18 small Spider dev databases bundled with the eval
gate, with one of two in-process LLM backends (SQLFORGE_BACKEND):
  - "llama_cpp" (default): the 4-bit GGUF on CPU, for a CPU-only host such as a Hugging Face Space;
  - "vllm": the merged model with FP8 weights on a GPU, as in production (used by deploy/modal_app.py).

Locally, from the repo:

    MODEL_PATH=models/gguf/sqlforge-3b-Q4_K_M.gguf python space/demo.py
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Space / Modal: the pipeline packages sit next to this file (or are installed). Repo checkout: one level up.
for p in (HERE, HERE.parent):
    if (p / "app" / "service.py").exists():
        sys.path.insert(0, str(p))
        break

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from app import service  # noqa: E402
from data.prompt import build_messages  # noqa: E402
from data.schema import serialize_schema  # noqa: E402

BACKEND = os.environ.get("SQLFORGE_BACKEND", "llama_cpp")
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "gguf/sqlforge-3b-Q4_K_M.gguf")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 2)))
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/ayush-pl/txt-sql")
VIDEO_URL = os.environ.get("VIDEO_URL", "")

DB_DIR = next(d for d in (Path(os.environ.get("DB_DIR", HERE / "dbs")), HERE.parent / "eval" / "gate" / "dbs")
              if d.exists())
CATALOGUE = {p.parent.name: str(p) for p in sorted(DB_DIR.glob("*/*.sqlite"))}

# all checked against the live model; used when space/suggestions.json is missing
EXAMPLES = [
    ["concert_singer", "Which stadium hosted the most concerts?"],
    ["pets_1", "Find the first names of students who own a cat."],
    ["cre_Doc_Template_Mgt", "Show all document ids, names and the number of paragraphs in each document."],
    ["world_1", "Which 5 countries have the largest population?"],
    ["pets_1", "How many dog pets are raised by female students?"],
    ["pets_1", "What is the average weight for each type of pet?"],
    ["real_estate_properties", "What is the feature type name of feature AirCon?"],
    ["concert_singer", "Delete all singers"],  # the model writes DELETE; the guardrail blocks it
]
LLM_LOCK = threading.Lock()  # one engine, one generation at a time


if BACKEND == "vllm":
    from vllm import LLM, SamplingParams

    print(f"loading {MODEL_PATH or MODEL_REPO} with vLLM", flush=True)
    # FP8 needs an Ada/Hopper GPU (L4, H100); on older GPUs (T4) set VLLM_QUANT="" and VLLM_DTYPE=half
    QUANT = os.environ.get("VLLM_QUANT", "fp8") or None
    ENGINE = LLM(model=MODEL_PATH or MODEL_REPO, quantization=QUANT, dtype=os.environ.get("VLLM_DTYPE", "auto"),
                 max_model_len=4096,
                 gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
                 enforce_eager=True,  # skip CUDA-graph capture: faster cold start, still well under 1 s/query
                 enable_prefix_caching=True)
    HARDWARE = f"an NVIDIA {os.environ.get('GPU_NAME', 'GPU')} with vLLM ({QUANT.upper() if QUANT else 'FP16'})"

    def engine_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> tuple[str, dict]:
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=0)
        with LLM_LOCK:
            out = ENGINE.chat(messages, params, use_tqdm=False)[0]
        return out.outputs[0].text, {"prompt_tokens": len(out.prompt_token_ids or []),
                                     "completion_tokens": len(out.outputs[0].token_ids)}

    def warm(db_id: str) -> None:  # vLLM's prefix cache handles repeated schemas on its own
        return None

    service.LLM_MODEL = "sqlforge-3b (vLLM FP8, GPU)"
else:
    from llama_cpp import Llama

    def _gguf_path() -> str:
        if MODEL_PATH:
            return MODEL_PATH
        if not MODEL_REPO:
            raise SystemExit("set MODEL_REPO (Hub repo with the GGUF) or MODEL_PATH (local .gguf file)")
        from huggingface_hub import hf_hub_download

        return hf_hub_download(MODEL_REPO, MODEL_FILE)

    print(f"loading {MODEL_PATH or MODEL_REPO + '/' + MODEL_FILE} with {N_THREADS} threads", flush=True)
    ENGINE = Llama(model_path=_gguf_path(), n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
    HARDWARE = f"{N_THREADS} CPU cores with a 4-bit GGUF (llama.cpp)"
    STOP = ["<|im_end|>", "<|endoftext|>"]

    def engine_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> tuple[str, dict]:
        with LLM_LOCK:
            resp = ENGINE.create_chat_completion(messages=messages, temperature=temperature,
                                                 max_tokens=max_tokens, stop=STOP, seed=0)
        return resp["choices"][0]["message"]["content"] or "", resp.get("usage") or {}

    def warm(db_id: str) -> None:
        """Prefill the schema prompt for a database so the first question on it is faster.

        llama.cpp reuses the longest matching token prefix from the previous call, and the prompt is
        system + schema + question, so after this only the question has to be processed.
        """
        if db_id not in CATALOGUE or not LLM_LOCK.acquire(blocking=False):
            return
        try:
            msgs = build_messages(serialize_schema(CATALOGUE[db_id], 3), "")
            ENGINE.create_chat_completion(messages=msgs, max_tokens=1, temperature=0.0)
        except Exception as e:  # warm-up is best effort
            print(f"warm-up failed for {db_id}: {e}", flush=True)
        finally:
            LLM_LOCK.release()

    service.LLM_MODEL = "sqlforge-3b-Q4_K_M (llama.cpp, CPU)"

# same signature and return shape as service.chat (an OpenAI-compatible HTTP call in production)
service.chat = engine_chat
service.databases = lambda: CATALOGUE


# ---------------------------------------------------------------- what can I ask?
# Plain-English summary of each bundled database, shown next to the question box.
DESCRIPTIONS = {
    "car_1": "Cars from the 1970s–80s: makers, models, countries and specs (mpg, cylinders, horsepower, weight, year).",
    "concert_singer": "Singers, the concerts they performed in, and the stadiums that hosted them.",
    "course_teach": "Teachers, courses, and which teacher teaches which course for which grade.",
    "cre_Doc_Template_Mgt": "A document system: templates and their types, documents built from them, and paragraphs.",
    "dog_kennels": "A dog kennel: dogs, owners, breeds, sizes, vets/staff, treatments and their costs.",
    "employee_hire_evaluation": "Employees, the shops that hired them (full or part time) and the bonuses they earned.",
    "flight_2": "Airlines, airports and 1,200 flights between US airports.",
    "museum_visit": "Museums, visitors with membership levels, and their visits (tickets and money spent).",
    "network_1": "High-school students, their grades, who is friends with whom and who likes whom.",
    "orchestra": "Orchestras, their conductors, record companies, performances and TV ratings.",
    "pets_1": "College students (age, sex, major) and the pets they own (type, age, weight).",
    "poker_player": "Poker players: earnings, best finishes and final tables, plus nationality and height.",
    "real_estate_properties": "Properties for sale: type, rooms, asking/selling prices and extra features.",
    "singer": "Singers (birth year, net worth, citizenship) and their songs' sales and chart positions.",
    "student_transcripts_tracking": "A university: students, addresses, degree programs, courses, semesters and transcripts.",
    "tvshow": "TV channels (country, language, content), TV series ratings and cartoons with their writers.",
    "voter_1": "A TV-show vote: contestants, phone votes by state, and US area codes.",
    "world_1": "Countries, cities and languages of the world: population, continent, GNP, life expectancy.",
}
# questions from the Spider dev set that this demo answers correctly (scripts/build_suggestions.py);
# without the file, fall back to the hand-checked EXAMPLES
SUGGESTIONS_FILE = next((d / "suggestions.json" for d in (HERE, HERE.parent / "space")
                         if (d / "suggestions.json").exists()), None)
if SUGGESTIONS_FILE:
    SUGGESTIONS: dict[str, list[str]] = {
        db: [x["question"] for x in qs]
        for db, qs in json.loads(SUGGESTIONS_FILE.read_text(encoding="utf-8")).items()}
else:
    SUGGESTIONS = {}
    for _db, _q in EXAMPLES:
        SUGGESTIONS.setdefault(_db, []).append(_q)
MAX_SUGGESTIONS = 8
DEFAULT_DB = "concert_singer"


def _tables(db_id: str) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{CATALOGUE[db_id]}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        return {t: [c[1] for c in conn.execute(f'PRAGMA table_info("{t}")')] for t in names}
    finally:
        conn.close()


def db_card(db_id: str) -> str:
    lines = [f"**{DESCRIPTIONS.get(db_id, db_id)}**", ""]
    for t, cols in _tables(db_id).items():
        lines.append(f"- `{t}`: " + ", ".join(cols[:7]) + (", …" if len(cols) > 7 else ""))
    return "\n".join(lines)


def preview(db_id: str, table_name: str | None) -> pd.DataFrame:
    if db_id not in CATALOGUE or not table_name:
        return pd.DataFrame()
    conn = sqlite3.connect(f"file:{CATALOGUE[db_id]}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 8', conn)
    finally:
        conn.close()


def suggestions_for(db_id: str) -> list[str]:
    qs = SUGGESTIONS.get(db_id, [])
    if len(qs) <= MAX_SUGGESTIONS:
        return qs
    # spread the picks over the list (sorted easy -> hard) so they cover several difficulties
    step = len(qs) / MAX_SUGGESTIONS
    return [qs[int(i * step)] for i in range(MAX_SUGGESTIONS)]


def schema_text(db_id: str) -> str:
    return serialize_schema(CATALOGUE[db_id], 0) if db_id in CATALOGUE else ""


def on_db_change(db_id: str):
    threading.Thread(target=warm, args=(db_id,), daemon=True).start()
    tables = list(_tables(db_id)) if db_id in CATALOGUE else []
    first = tables[0] if tables else None
    return (db_card(db_id), gr.Radio(choices=suggestions_for(db_id), value=None),
            gr.Dropdown(choices=tables, value=first), preview(db_id, first), schema_text(db_id))


def pick_suggestion(evt: gr.SelectData) -> str:
    return evt.value


def surprise():
    db_id = random.choice([d for d, qs in SUGGESTIONS.items() if qs and d in CATALOGUE])
    return db_id, random.choice(SUGGESTIONS[db_id])


def ask(db_id: str, question: str):
    question = (question or "").strip()
    if not question:
        return "", gr.Dataframe(visible=False), "Type a question first, or click a suggested one.", []
    t0 = time.perf_counter()
    try:
        ans = service.answer(db_id, question)
    except Exception as e:
        return "", gr.Dataframe(visible=False), f"**Error:** {type(e).__name__}: {e}", []
    secs = time.perf_counter() - t0

    if ans.ok:
        status = f"✅ **{len(ans.rows)} row(s)** in {secs:.1f} s"
        if ans.repaired:
            status += f" · 🔧 fixed by self-correction ({ans.attempts} LLM call(s))"
        if not ans.rows:
            status += "  \nThe query ran but found no rows: check the spelling of names and values."
    elif ans.blocked_rule:
        status = f"🛡️ **Blocked by guardrail** `{ans.blocked_rule}`: {ans.error}"
    else:
        status = (f"❌ **Couldn't answer this one:** {ans.error}  \nTry rephrasing it with the table and column "
                  "names listed on the left, or click a suggested question.")
    status += f"  \nTokens: {ans.prompt_tokens} prompt / {ans.completion_tokens} generated"

    df = (gr.Dataframe(pd.DataFrame(ans.rows[:200], columns=ans.columns or None), visible=True) if ans.rows
          else gr.Dataframe(visible=False))
    steps = [[s.name, s.ms, ", ".join(f"{k}={v}" for k, v in s.detail.items() if v is not None)]
             for s in ans.steps]
    return ans.sql or "", df, status, steps


links = f"[GitHub]({GITHUB_URL})"
if MODEL_REPO:
    links += f" · [Model](https://huggingface.co/{MODEL_REPO})"
if VIDEO_URL:
    links += f" · [Video of the GPU version]({VIDEO_URL})"

SPEED_NOTE = ("" if BACKEND == "vllm" else
              " An answer takes a few seconds to ~20 s (and the first visit after idle starts the server);"
              " on a GPU with vLLM the same pipeline answers in about 0.5 s.")
INTRO = f"""# SQLForge: text-to-SQL with a fine-tuned 3B model
{links}

Ask a question in plain English about a SQLite database. **SQLForge-3B** (Qwen2.5-Coder-3B fine-tuned with
QLoRA on Spider + BIRD: 80.5% execution accuracy on Spider dev, 54.1% on BIRD dev) writes the SQL. The query
then passes through guardrails (read-only, single statement, known tables only, row limit), runs against the
database, and is repaired automatically if it fails.

⚡ This demo runs on {HARDWARE}.{SPEED_NOTE}
"""
TIPS = (
    "**How to use it:** 1) pick a database, 2) click a suggested question or write your own about the tables "
    "listed under it, 3) run it. Ask one question at a time; it can only read data. To see the guardrail, "
    'ask it to *"Delete all singers"*.'
)

_first_tables = list(_tables(DEFAULT_DB))
_first = _first_tables[0] if _first_tables else None

with gr.Blocks(title="SQLForge demo") as demo:
    gr.Markdown(INTRO)
    gr.Markdown(TIPS)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            db = gr.Dropdown(list(CATALOGUE), value=DEFAULT_DB, label="1. Database")
            card = gr.Markdown(db_card(DEFAULT_DB))
        with gr.Column(scale=2):
            suggested = gr.Radio(suggestions_for(DEFAULT_DB), label="2. Suggested questions (click one)",
                                 interactive=True)  # only ever an event output, so Gradio would render it read-only
            q = gr.Textbox(label="...or write your own", lines=2,
                           placeholder="e.g. Which stadium hosted the most concerts?")
            with gr.Row():
                go = gr.Button("3. Generate SQL and run it", variant="primary", scale=3)
                rnd = gr.Button("🎲 Surprise me", scale=1)
    status = gr.Markdown()
    sql = gr.Code(label="Generated SQL (after guardrails)", language="sql", interactive=False)
    table = gr.Dataframe(label="Result (first 200 rows)", interactive=False, wrap=True, visible=False)
    with gr.Accordion("Pipeline steps", open=False):
        steps = gr.Dataframe(headers=["step", "ms", "detail"], interactive=False, wrap=True)
    with gr.Accordion("Browse the data", open=False):
        tbl = gr.Dropdown(_first_tables, value=_first, label="Table")
        rows = gr.Dataframe(preview(DEFAULT_DB, _first), label="First 8 rows", interactive=False, wrap=True)
    with gr.Accordion("Schema as SQL (what the model sees)", open=False):
        schema = gr.Code(value=schema_text(DEFAULT_DB), language="sql", interactive=False)
    gr.Markdown("Databases: [Spider 1.0](https://yale-lily.github.io/spider) dev set (Yu et al., 2018, "
                "CC BY-SA 4.0), never seen in training. The suggested questions come from the same dev set and "
                "were checked to be answered correctly by this demo.")

    db.change(on_db_change, db, [card, suggested, tbl, rows, schema])
    suggested.select(pick_suggestion, None, q)
    tbl.change(preview, [db, tbl], rows)
    rnd.click(surprise, None, [db, q])
    go.click(ask, [db, q], [sql, table, status, steps], concurrency_limit=1)
    q.submit(ask, [db, q], [sql, table, status, steps], concurrency_limit=1)
    demo.load(lambda: threading.Thread(target=warm, args=(DEFAULT_DB,), daemon=True).start())

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
