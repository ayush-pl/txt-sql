"""Render the README result charts as standalone SVG (standard library only).

    python docs/make_charts.py        -> docs/img/{accuracy,quantization,throughput}.svg

Numbers are the measured results reported in the README (outputs/ is not committed).
Palette: validated categorical slots 1-3 (blue, orange, aqua) on a light surface; every mark is
direct-labeled (aqua is below 3:1 contrast, so values are never color-alone).
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / "img"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
FONT = "font-family='Inter, Segoe UI, Helvetica, Arial, sans-serif'"


def text(x, y, s, size=12, color=INK2, anchor="middle", weight=400):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{escape(str(s))}</text>")


def bar(x, y, w, h, color, r=4):
    """Bar with only the data end (top) rounded, anchored square on the baseline."""
    r = min(r, w / 2, h)
    return (f"<path d='M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z' fill='{color}'/>")


def frame(w, h, title, subtitle, body, legend):
    leg, lx = [], 24
    for name, color in legend:
        leg.append(f"<rect x='{lx}' y='62' width='12' height='12' rx='3' fill='{color}'/>")
        leg.append(text(lx + 18, 72, name, 12, INK2, "start"))
        lx += 26 + 7.2 * len(name)
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
            f"role='img' aria-label='{escape(title)}'>"
            f"<rect width='{w}' height='{h}' rx='10' fill='{SURFACE}'/>"
            + text(24, 30, title, 16, INK, "start", 600) + text(24, 50, subtitle, 12, MUTED, "start")
            + "".join(leg) + "".join(body) + "</svg>")


def grouped_bars(categories, series, *, title, subtitle, ymax, yticks, unit="", w=760, h=400, fmt="{:.1f}"):
    left, right, top, bottom = 56, 20, 96, 64
    pw, ph = w - left - right, h - top - bottom
    y = lambda v: top + ph * (1 - v / ymax)  # noqa: E731
    body = []
    for t in yticks:
        body.append(f"<line x1='{left}' x2='{w - right}' y1='{y(t):.1f}' y2='{y(t):.1f}' stroke='{GRID}' stroke-width='1'/>")
        body.append(text(left - 8, y(t) + 4, f"{t}{unit}", 11, MUTED, "end"))
    group_w = pw / len(categories)
    bw = min(44, (group_w * 0.72 - 2 * (len(series) - 1)) / len(series))
    for i, cat in enumerate(categories):
        gx = left + i * group_w + (group_w - (bw * len(series) + 2 * (len(series) - 1))) / 2
        for j, (_, color, vals) in enumerate(series):
            v = vals[i]
            bx = gx + j * (bw + 2)  # 2px surface gap between adjacent bars
            body.append(bar(bx, y(v), bw, y(0) - y(v), color))
            body.append(text(bx + bw / 2, y(v) - 6, fmt.format(v), 11, INK, "middle", 600))
        for k, line in enumerate(cat.split("\n")):
            body.append(text(left + i * group_w + group_w / 2, h - bottom + 20 + 14 * k, line, 12, INK2 if k == 0 else MUTED))
    body.append(f"<line x1='{left}' x2='{w - right}' y1='{y(0):.1f}' y2='{y(0):.1f}' stroke='{MUTED}' stroke-width='1'/>")
    return frame(w, h, title, subtitle, body, [(n, c) for n, c, _ in series])


def lines(xs, series, *, title, subtitle, ymax, yticks, xlabel, w=760, h=400, unit=""):
    left, right, top, bottom = 56, 140, 96, 64
    pw, ph = w - left - right, h - top - bottom
    y = lambda v: top + ph * (1 - v / ymax)  # noqa: E731
    x = lambda i: left + pw * i / (len(xs) - 1)  # noqa: E731
    body = []
    for t in yticks:
        body.append(f"<line x1='{left}' x2='{left + pw}' y1='{y(t):.1f}' y2='{y(t):.1f}' stroke='{GRID}' stroke-width='1'/>")
        body.append(text(left - 8, y(t) + 4, f"{t}{unit}", 11, MUTED, "end"))
    for i, xv in enumerate(xs):
        body.append(text(x(i), h - bottom + 20, xv, 12, INK2))
    body.append(text(left + pw / 2, h - bottom + 40, xlabel, 12, MUTED))
    ends = sorted(((vals[-1], name, color) for name, color, vals in series), reverse=True)
    placed = []
    for v, name, _color in ends:  # end labels, nudged apart so they never collide
        yy = y(v) + 4
        while any(abs(yy - p) < 15 for p in placed):
            yy += 15
        placed.append(yy)
        body.append(text(left + pw + 10, yy, f"{name}  {v:.1f}", 12, INK, "start", 600))
    for _name, color, vals in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        body.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
        for i, v in enumerate(vals):  # 8px markers with a 2px surface ring
            body.append(f"<circle cx='{x(i):.1f}' cy='{y(v):.1f}' r='4' fill='{color}' stroke='{SURFACE}' stroke-width='2'/>")
    body.append(f"<line x1='{left}' x2='{left + pw}' y1='{y(0):.1f}' y2='{y(0):.1f}' stroke='{MUTED}' stroke-width='1'/>")
    return frame(w, h, title, subtitle, body, [(n, c) for n, c, _ in series])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts = {
        "accuracy.svg": grouped_bars(
            ["Base model", "Run 1\nSpider SFT", "Run 2\nSpider + BIRD SFT", "Run 2 + column fix\n+ self-correction"],
            [("Spider dev", SERIES[0], [71.8, 78.0, 78.5, 80.5]), ("BIRD dev (500)", SERIES[1], [41.7, 43.5, 53.1, 54.1])],
            title="Execution accuracy (%), Qwen2.5-Coder-3B",
            subtitle="Fine-tuning on an 8 GB laptop GPU: +8.7 Spider, +12.4 BIRD over the base model",
            ymax=100, yticks=[0, 25, 50, 75, 100]),
        "quantization.svg": grouped_bars(
            ["bf16\n5.8 GB", "vLLM FP8\n3.2 GB in VRAM", "GGUF Q8_0\n3.3 GB", "GGUF Q4_K_M\n1.8 GB", "AWQ W4A16\n2.0 GB"],
            [("Spider dev", SERIES[0], [78.5, 78.6, 78.4, 76.6, 75.2]), ("BIRD dev (500)", SERIES[1], [53.1, 52.7, 52.5, 52.7, 51.9])],
            title="Accuracy vs. quantization (EX %)",
            subtitle="8-bit is lossless; at 4 bits llama.cpp Q4_K_M keeps more accuracy than AWQ",
            ymax=100, yticks=[0, 25, 50, 75, 100]),
        "throughput.svg": lines(
            ["1", "8", "32"],
            [("vLLM FP8", SERIES[0], [1.71, 10.01, 15.57]), ("vLLM AWQ", SERIES[1], [2.43, 12.73, 15.59]),
             ("Ollama Q4_K_M", SERIES[2], [2.01, 2.18, 2.17])],
            title="Throughput on BIRD prompts (requests / s)",
            subtitle="Mean prompt 2,173 tokens; vLLM continuous batching + prefix caching vs. Ollama's one-at-a-time serving",
            ymax=20, yticks=[0, 5, 10, 15, 20], xlabel="concurrent requests"),
    }
    for name, svg in charts.items():
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote docs/img/{name}")


if __name__ == "__main__":
    main()
