"""Upload the merged model (+ optional GGUF / AWQ variants) and the model card to the Hugging Face Hub.
You run this yourself: it uses your HF token from .env (HF_TOKEN) and creates a repo under your account.

    python scripts/push_to_hub.py --repo <your-username>/sqlforge-3b              # private by default
    python scripts/push_to_hub.py --repo <your-username>/sqlforge-3b --public --with-gguf --with-awq
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<username>/<model-name>")
    ap.add_argument("--model-dir", default="models/sqlforge-3b-v2-step1014")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--with-gguf", action="store_true")
    ap.add_argument("--with-awq", action="store_true")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN in .env first (a write token from huggingface.co/settings/tokens)")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        card = (REPO / "model_card.md").read_text(encoding="utf-8").replace("<user>/sqlforge-3b", args.repo)
        (Path(tmp) / "README.md").write_text(card, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(Path(tmp) / "README.md"), path_in_repo="README.md", repo_id=args.repo)
    api.upload_folder(folder_path=str(REPO / args.model_dir), repo_id=args.repo, commit_message="SQLForge-3B weights")
    if args.with_gguf:
        for q in ("Q4_K_M", "Q8_0"):
            f = REPO / "models" / "gguf" / f"sqlforge-3b-{q}.gguf"
            if f.exists():
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f"gguf/{f.name}", repo_id=args.repo)
    if args.with_awq and (REPO / "models" / "sqlforge-3b-awq").exists():
        api.upload_folder(folder_path=str(REPO / "models" / "sqlforge-3b-awq"), path_in_repo="awq", repo_id=args.repo)
    shutil.rmtree(Path(tempfile.gettempdir()) / "hf_upload_unused", ignore_errors=True)
    print(f"done: https://huggingface.co/{args.repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
