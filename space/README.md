---
title: SQLForge
emoji: 🗄️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: demo.py
pinned: false
license: apache-2.0
short_description: Fine-tuned 3B text-to-SQL with guardrails and self-correction
models:
  - __MODEL_REPO__
---

# SQLForge demo

Ask a question in plain English about a SQLite database. SQLForge-3B (Qwen2.5-Coder-3B fine-tuned with
QLoRA on Spider + BIRD) writes the SQL, guardrails check it, it runs read-only, and failures are repaired
automatically.

This Space runs the 4-bit GGUF with llama.cpp on 2 CPU cores. Code, training, evaluation and the
GPU serving stack: __GITHUB_URL__

Databases: Spider 1.0 dev set (Yu et al., 2018), CC BY-SA 4.0.
