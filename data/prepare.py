"""Turn raw Spider / BIRD files into unified JSONL used by training and evaluation.

Each record:
  {id, dataset, split, db_id, db_path, question, evidence, gold_sql, difficulty}

The schema text is NOT stored. It's rebuilt from db_path via data.schema at use time,
so the prompt format can change without re-preparing data.

Usage:
    python -m data.prepare
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import spider_hardness

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"


def _find(base: Path, name: str) -> Path | None:
    hits = sorted(p for p in base.glob(f"**/{name}") if "__MACOSX" not in p.parts)
    return hits[0] if hits else None


def _rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.parent).as_posix()


def prepare_spider() -> dict[str, int]:
    base = RAW / "spider"
    dev = _find(base, "dev.json")
    if dev is None:
        print("[spider] not found, run `python -m data.download` first")
        return {}
    sd = dev.parent
    db_root = sd / "database"
    counts = {}
    splits = {"train": ["train_spider.json", "train_others.json"], "dev": ["dev.json"]}
    for split, files in splits.items():
        recs = []
        for fname in files:
            fp = sd / fname
            if not fp.exists():
                continue
            for i, ex in enumerate(json.loads(fp.read_text(encoding="utf-8"))):
                db_path = db_root / ex["db_id"] / f"{ex['db_id']}.sqlite"
                if not db_path.exists():
                    continue
                recs.append(
                    {
                        "id": f"spider-{split}-{fname.split('.')[0]}-{i}",
                        "dataset": "spider",
                        "split": split,
                        "db_id": ex["db_id"],
                        "db_path": _rel(db_path),
                        "question": ex["question"],
                        "evidence": None,
                        "gold_sql": ex["query"].strip(),
                        "difficulty": spider_hardness(ex["query"]),
                    }
                )
        _write(OUT / f"spider_{split}.jsonl", recs)
        counts[f"spider_{split}"] = len(recs)
    return counts


def prepare_bird_dev() -> dict[str, int]:
    return _prepare_bird("dev", RAW / "bird")


def prepare_bird_train() -> dict[str, int]:
    if not (RAW / "bird_train").exists():
        return {}
    return _prepare_bird("train", RAW / "bird_train")


def _prepare_bird(split: str, base: Path) -> dict[str, int]:
    src = _find(base, f"{split}.json")
    if src is None:
        print(f"[bird] {split}.json not found, run `python -m data.download` first")
        return {}
    db_dir = next((p for p in base.glob(f"**/{split}_databases") if p.is_dir() and "__MACOSX" not in p.parts), None)
    if db_dir is None:
        print(f"[bird] {split}_databases folder missing")
        return {}
    recs = []
    for i, ex in enumerate(json.loads(src.read_text(encoding="utf-8"))):
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            continue
        recs.append(
            {
                "id": f"bird-{split}-{ex.get('question_id', i)}",
                "dataset": "bird",
                "split": split,
                "db_id": ex["db_id"],
                "db_path": _rel(db_path),
                "question": ex["question"],
                "evidence": ex.get("evidence") or None,
                "gold_sql": ex["SQL"].strip(),
                "difficulty": ex.get("difficulty", "unknown"),
            }
        )
    _write(OUT / f"bird_{split}.jsonl", recs)
    return {f"bird_{split}": len(recs)}


def _write(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs):>6} -> {path.relative_to(ROOT.parent)}")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    counts = {**prepare_spider(), **prepare_bird_dev(), **prepare_bird_train()}
    print(counts)


if __name__ == "__main__":
    main()
