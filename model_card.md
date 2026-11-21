---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: transformers
pipeline_tag: text-generation
language: [en]
tags: [text-to-sql, sql, sqlite, qlora, peft, spider, bird, code]
datasets: [spider, bird-bench]
model-index:
  - name: SQLForge-3B
    results:
      - task: { type: text-to-sql, name: Text-to-SQL }
        dataset: { type: spider, name: Spider 1.0 dev, split: dev }
        metrics:
          - { type: execution_accuracy, value: 78.5, name: Execution accuracy (greedy) }
          - { type: execution_accuracy, value: 80.5, name: Execution accuracy (+ column fix + self-correction) }
      - task: { type: text-to-sql, name: Text-to-SQL }
        dataset: { type: bird-bench, name: BIRD dev (500-question subset), split: dev }
        metrics:
          - { type: execution_accuracy, value: 53.1, name: Execution accuracy (greedy) }
          - { type: execution_accuracy, value: 54.1, name: Execution accuracy (+ self-correction) }
---

# SQLForge-3B

A 3B text-to-SQL model for SQLite: [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
fine-tuned with QLoRA on Spider and BIRD training data, built on a single 8 GB laptop GPU.

Part of the SQLForge project, which also includes an evaluation harness, serving with vLLM and
quantization, guardrails, self-correction and an observability stack. See the project README for
the full write-up.

## Results

Execution accuracy (EX): the predicted query's result set matches the gold query's.
Greedy decoding, schema given as `CREATE TABLE` DDL plus 3 sample rows per table.

| Model | Spider dev EX | BIRD dev EX* |
|---|---|---|
| Qwen2.5-Coder-3B-Instruct (base) | 71.8 | 41.7 |
| SQLForge-3B run 1 (Spider only) | 78.0 | 43.5 |
| **SQLForge-3B (this model)** | **78.5** | **53.1** |
| + deterministic column fix + execution-guided self-correction | **80.5** | **54.1** |

\*Fixed random 500-question subset of BIRD dev (seed 42); evidence hints included in the prompt.

**Quantized variants**:

| Variant | Size | Spider EX | BIRD EX |
|---|---|---|---|
| bf16 | 5.8 GB | 78.5 | 53.1 |
| FP8 (vLLM, online) | weights 3.2 GB in VRAM | 78.6 | 52.7 |
| GGUF Q8_0 | 3.3 GB | 78.4 | 52.5 |
| GGUF Q4_K_M | 1.8 GB | 76.6 | 52.7 |
| AWQ W4A16 | 2.0 GB | 75.2 | 51.9 |

## Prompt format

Use the chat template with this system prompt and user message (the model was trained on exactly this):

```text
[system] You are an expert SQLite engineer. Given a database schema and a question, write a single
correct SQLite SELECT query that answers it. Return only the SQL, with no explanation.

[user]
### Database schema
CREATE TABLE singer (
  Singer_ID INTEGER,
  Name TEXT,
  ...
);
-- sample rows (Singer_ID | Name | ...):
-- 1 | Joe Sharp | ...

### Question
How many singers are there?

### Hint            <- optional, BIRD-style domain knowledge
...

### SQL
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("<user>/sqlforge-3b")
model = AutoModelForCausalLM.from_pretrained("<user>/sqlforge-3b", dtype="bfloat16", device_map="auto")
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=256, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
```

With vLLM: `vllm serve <user>/sqlforge-3b --quantization fp8 --enable-prefix-caching`. Prefix
caching matters here: questions on the same database share the schema prefix, which gave 4.7×
throughput on BIRD.

## Training

- **Data:**
  - 16,218 examples: Spider train (8,140) and BIRD train (8,078), from 201 databases.
  - BIRD examples whose gold SQL fails to execute were dropped (478).
  - `formula_1` was excluded because it overlaps BIRD dev.
  - No dev databases or questions appear in training; a contamination check confirms this.
- **Method:**
  - QLoRA: NF4 4-bit base, LoRA r=16, α=32, dropout 0.05, on all attention and MLP projections (0.96% of parameters).
  - Loss on the SQL completion only.
- **Hyperparameters:** 1 epoch (1,014 steps), learning rate 2e-4 with cosine decay, effective batch 16, max length 2,048, paged AdamW 8-bit, gradient checkpointing.
- **Compute:** one RTX 5060 Laptop GPU (8 GB): 7.2 h, peak 3.5 GB.
- **Checkpoint:** the final checkpoint. On full Spider and BIRD it beat the checkpoint chosen by validation execution accuracy.

## Limitations

- **SQLite dialect only.** It was trained and evaluated on SQLite; other dialects will need adaptation.
- **It hallucinates column names.** This is the main remaining failure mode, with 66 of 1,034 Spider dev and 53 of 500 BIRD answers affected. On Spider most are near-misses (`pet_type` for `PetType`) that a deterministic post-processing step fixes. On BIRD they're mostly invented.
- **Harder questions and value semantics.** It's weaker on BIRD "challenging" questions (36% EX) and on questions that need value-level domain knowledge without a hint.
- **Schema size.** Prompts over about 8k tokens (very wide schemas) need schema linking, which is in the project's serving pipeline.
- **Not safe to run unchecked.** Like any LLM, it can be prompt-injected into emitting destructive SQL; this was observed with `DROP TABLE`. **Never run its output on a database without guardrails.** The project ships a read-only guardrail layer for this.

## License and attribution

The weights derive from Qwen2.5-Coder-3B-Instruct; check its license before redistribution. Training
data: [Spider](https://yale-lily.github.io/spider) (CC BY-SA 4.0) and [BIRD](https://bird-bench.github.io/) (CC BY-SA 4.0).
