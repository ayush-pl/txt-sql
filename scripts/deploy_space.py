"""Build the live demo as a Hugging Face Space and (optionally) upload it.

The Space is a self-contained folder: space/demo.py, the pipeline modules it imports (app/, data/,
eval/), the 18 Spider databases from the eval gate, and a README with the Space config. The model is
not copied; the Space downloads the GGUF from your model repo at startup, so push it first:

    python scripts/push_to_hub.py --repo <user>/sqlforge-3b --public --with-gguf

Then:
    python scripts/deploy_space.py --out outputs/space                                  # build only, to test locally
    python scripts/deploy_space.py --space <user>/sqlforge --model-repo <user>/sqlforge-3b   # build + upload

Uploading uses HF_TOKEN from .env (a write token) and creates a public Gradio Space on cpu-basic
hardware. Hugging Face requires a PRO account to host Gradio/Docker Spaces (only static Spaces are free).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# what demo.py imports from the pipeline, and nothing else (no training, no torch)
MODULES = [
    "app/__init__.py", "app/service.py", "app/guardrails.py", "app/tracing.py",
    "data/__init__.py", "data/prompt.py", "data/schema.py", "data/schema_linker.py", "data/prepare.py",
    "eval/__init__.py", "eval/metrics.py", "eval/self_correct.py", "eval/generate.py",
]


def build(out: Path, model_repo: str, github_url: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(REPO / "space" / "demo.py", out / "demo.py")
    shutil.copy2(REPO / "space" / "requirements.txt", out / "requirements.txt")
    shutil.copy2(REPO / "space" / "suggestions.json", out / "suggestions.json")
    readme = (REPO / "space" / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("__MODEL_REPO__", model_repo or "Qwen/Qwen2.5-Coder-3B-Instruct")
    readme = readme.replace("__GITHUB_URL__", github_url)
    (out / "README.md").write_text(readme, encoding="utf-8")
    for m in MODULES:
        dst = out / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / m, dst)
    shutil.copytree(REPO / "eval" / "gate" / "dbs", out / "dbs")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"built {out} ({n} files, {mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="<user>/<space-name>; omit to build only")
    ap.add_argument("--model-repo", default="", help="<user>/sqlforge-3b, the public repo holding gguf/*.gguf")
    ap.add_argument("--github-url", default="https://github.com/ayush-pl/txt-sql")
    ap.add_argument("--video-url", default="", help="optional link to a recorded demo of the GPU version")
    ap.add_argument("--out", help="build folder (default: a temp folder, deleted after upload)")
    args = ap.parse_args()
    if args.space and not args.model_repo:
        raise SystemExit("--model-repo is required with --space (the Space downloads the GGUF from it)")

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="sqlforge_space_")) / "space"
    build(out, args.model_repo, args.github_url)
    if not args.space:
        print("build only. Run it locally with:\n"
              f"  MODEL_PATH=models/gguf/sqlforge-3b-Q4_K_M.gguf python {out / 'demo.py'}")
        return

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(REPO / ".env")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN in .env first (a write token from huggingface.co/settings/tokens)")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    api.add_space_variable(args.space, "GITHUB_URL", args.github_url)
    if args.video_url:
        api.add_space_variable(args.space, "VIDEO_URL", args.video_url)
    api.upload_folder(folder_path=str(out), repo_id=args.space, repo_type="space",
                      commit_message="Deploy SQLForge demo")
    if not args.out:
        shutil.rmtree(out.parent, ignore_errors=True)
    print(f"done: https://huggingface.co/spaces/{args.space} (first build takes a few minutes)")


if __name__ == "__main__":
    main()
