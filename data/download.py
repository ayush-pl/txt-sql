"""Download Spider and BIRD (dev) into data/raw/.

Sources (official):
  - Spider 1.0: Yale LILY Google Drive release (spider_data.zip)
  - BIRD dev:   bird-bench.github.io release (dev.zip)

Archives are treated as data only: they're extracted into their own folder and
nothing inside them is executed.

Usage:
    python -m data.download            # both
    python -m data.download --only spider
    python -m data.download --only bird_train   # 8.9 GB, opt-in
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"  # ~8.9 GB


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            if "__MACOSX" in member.filename:
                continue
            zf.extract(member, dest)


def download_spider() -> Path:
    out = RAW / "spider"
    if (out / "spider_data" / "dev.json").exists():
        print("[spider] already present")
        return out
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "spider_data.zip"
    import gdown

    gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
    _safe_extract(zip_path, out)
    zip_path.unlink()
    print(f"[spider] extracted to {out}")
    return out


def download_bird_dev() -> Path:
    out = RAW / "bird"
    if (out / "dev" / "dev.json").exists() or any(out.glob("**/dev.json")):
        print("[bird] already present")
        return out
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "dev.zip"
    print(f"[bird] downloading {BIRD_DEV_URL}")
    with urllib.request.urlopen(BIRD_DEV_URL) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    _safe_extract(zip_path, out)
    zip_path.unlink()
    # dev.zip contains dev_databases.zip nested inside
    for inner in out.glob("**/dev_databases.zip"):
        _safe_extract(inner, inner.parent)
        inner.unlink()
    print(f"[bird] extracted to {out}")
    return out


def _download_resumable(url: str, dest: Path) -> None:
    """Stream to disk with HTTP Range resume, so a dropped connection doesn't restart 9 GB."""
    from tqdm import tqdm

    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    done = dest.stat().st_size if dest.exists() else 0
    while done < total:
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f, tqdm(
                total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        except (OSError, TimeoutError) as e:
            print(f"connection dropped ({e}), resuming")
        done = dest.stat().st_size


def download_bird_train() -> Path:
    """BIRD train (9.4k questions, 69 databases). Kept in its own folder: its DBs never overlap dev."""
    out = RAW / "bird_train"
    if any(out.glob("**/train.json")) and any(out.glob("**/train_databases/*/*.sqlite")):
        print("[bird_train] already present")
        return out
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "train.zip"
    print(f"[bird_train] downloading {BIRD_TRAIN_URL}")
    _download_resumable(BIRD_TRAIN_URL, zip_path)
    _safe_extract(zip_path, out)
    zip_path.unlink()
    for inner in out.glob("**/train_databases.zip"):
        _safe_extract(inner, inner.parent)
        inner.unlink()
    print(f"[bird_train] extracted to {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["spider", "bird", "bird_train"])
    args = ap.parse_args()
    if args.only in (None, "spider"):
        download_spider()
    if args.only in (None, "bird"):
        download_bird_dev()
    if args.only == "bird_train":  # large: opt-in only
        download_bird_train()


if __name__ == "__main__":
    main()
