"""Prompt format shared by baselines, SFT training and serving. Change it in one place only."""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert SQLite engineer. Given a database schema and a question, "
    "write a single correct SQLite SELECT query that answers it. "
    "Return only the SQL, with no explanation."
)


def build_user_prompt(schema: str, question: str, evidence: str | None = None) -> str:
    parts = ["### Database schema", schema, "", "### Question", question.strip()]
    if evidence:
        parts += ["", "### Hint", evidence.strip()]
    parts += ["", "### SQL"]
    return "\n".join(parts)


def build_messages(schema: str, question: str, evidence: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(schema, question, evidence)},
    ]


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model completion (handles ``` fences and trailing chatter)."""
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    # unterminated fence
    sql = re.sub(r"^```(?:sql|sqlite)?", "", sql.strip(), flags=re.IGNORECASE).strip()
    # keep only the first statement
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return " ".join(sql.split())
