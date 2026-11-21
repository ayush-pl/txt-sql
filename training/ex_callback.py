"""Execution-accuracy validation during training.

Run 1 showed validation *loss* picks the wrong checkpoint for text-to-SQL: loss rose from
0.34 to 0.50 while execution accuracy rose from 75.5 to 78.0 on Spider dev. Cross-entropy
penalizes a confident model for writing SQL differently from the reference, even when the
query returns the same rows.

This callback greedily decodes SQL for the validation set (held-out databases) at every
evaluation, executes it against the gold query, and adds `eval_ex` to the metrics, so the
Trainer can use `metric_for_best_model="eval_ex"`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from transformers import TrainerCallback

from data.prompt import extract_sql
from eval.metrics import execute_sql, has_top_level_order_by, results_match

REPO = Path(__file__).resolve().parent.parent


def load_ex_examples(path: Path, n: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [r for r in rows if r.get("db_path")]
    return rows[:n]


class ExecutionAccuracyCallback(TrainerCallback):
    def __init__(self, examples: list[dict], tokenizer, batch_size: int = 8, max_new_tokens: int = 256):
        self.examples = examples
        self.tok = copy.deepcopy(tokenizer)
        self.tok.padding_side = "left"
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.history: list[tuple[int, float]] = []
        # gold results never change: execute once
        self.gold = [
            execute_sql(str(REPO / r["db_path"]), r["completion"][0]["content"], 30) for r in examples
        ]

    def _generate(self, model) -> list[str]:
        import torch

        texts = [
            self.tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)
            for r in self.examples
        ]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        preds = [""] * len(texts)
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=self.tok.pad_token_id,
                )
            for i, txt in zip(idx, self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True), strict=True):
                preds[i] = extract_sql(txt)
            del enc, out
        # hand generation's KV-cache memory back before training resumes: on an 8 GB GPU, keeping
        # it cached pushed usage to 7.9 GB, where the driver starts spilling into system RAM
        torch.cuda.empty_cache()
        return preds

    def compute_ex(self, model) -> float:
        was_training = model.training
        model.eval()
        try:
            preds = self._generate(model)
        finally:
            if was_training:
                model.train()
        hits = scored = 0
        for r, gold, pred_sql in zip(self.examples, self.gold, preds, strict=True):
            if not gold.ok:
                continue
            scored += 1
            pred = execute_sql(str(REPO / r["db_path"]), pred_sql, 30)
            mode = "bird" if r.get("dataset") == "bird" else "spider"
            ordered = has_top_level_order_by(r["completion"][0]["content"])
            hits += bool(pred.ok and results_match(pred.rows, gold.rows, mode=mode, ordered=ordered))
        return 100 * hits / max(scored, 1)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if model is None or metrics is None:
            return
        ex = self.compute_ex(model)
        # metrics is the same dict the Trainer uses for best-checkpoint selection
        metrics["eval_ex"] = ex
        self.history.append((state.global_step, ex))
        print(f"[step {state.global_step}] eval_ex = {ex:.2f} on {len(self.examples)} held-out-DB examples")
        try:
            import mlflow

            if mlflow.active_run():
                mlflow.log_metric("eval_ex", ex, step=state.global_step)
        except Exception:
            pass
