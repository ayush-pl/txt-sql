"""Environment sanity check: Python, PyTorch, CUDA, GPU capability and a bf16 matmul.

Run in both environments (Windows 3.13 and WSL 3.12):
    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python      : {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    try:
        import torch
    except ImportError:
        print("PyTorch     : NOT INSTALLED")
        print("  -> pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"PyTorch     : {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("CUDA        : NOT AVAILABLE (CPU-only torch build, or driver issue)")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU         : {name}  |  sm_{cap[0]}{cap[1]}  |  {total_gb:.1f} GB")

    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        print(f"WARNING     : this torch build lacks {arch} kernels; reinstall from the cu128 index.")
        return 1

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    (a @ b).sum().item()  # warmup
    start.record()
    for _ in range(10):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 10
    tflops = 2 * 4096**3 / (ms / 1000) / 1e12
    print(f"bf16 matmul : OK  ({ms:.2f} ms per 4096^3, ~{tflops:.1f} TFLOPS)  checksum={c.float().mean().item():.4f}")

    for mod in ("bitsandbytes", "peft", "trl", "vllm"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'installed')}")
        except Exception:
            print(f"{mod:<12}: not installed (expected on Windows for vllm/bitsandbytes)")

    print("\nAll core checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
