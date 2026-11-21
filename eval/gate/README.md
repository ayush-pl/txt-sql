# Eval gate

Fixed accuracy check that runs in CI without a GPU (`python -m eval.gate check`).

- `gate_set.jsonl`: 150 Spider dev questions, stratified by difficulty (36 easy / 64 medium /
  26 hard / 24 extra), from 18 small databases. Built with `python -m eval.gate build`.
- `dbs/`: those SQLite databases (~1 MB), so CI can execute queries.
- `predictions.jsonl` + `predictions_meta.json`: the current model's answers. Regenerate with
  `python -m eval.gate predict` whenever the model, prompt or decoding changes, and commit them
  with the change. CI fails if they are missing, incomplete or made with another prompt format.
- `thresholds.yaml`: the floors.

| Model | Gate EX | Valid SQL | Result |
|---|---|---|---|
| SQLForge-3B run 2 (+ column fix) | 81.3 | 96.7 | pass |
| Qwen2.5-Coder-3B base | 68.0 | 90.7 | **fail** |
| Qwen2.5-Coder-1.5B base | 62.0 | 84.0 | **fail** |

Data: [Spider 1.0](https://yale-lily.github.io/spider) (Yu et al., 2018), CC BY-SA 4.0.
