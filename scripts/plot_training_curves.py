"""把 run 的 metrics.jsonl 渲染为 SVG 训练曲线。

Render a run's metrics.jsonl as SVG training curves. This uses only the
standard library, so plotting does not add a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _series(rows: list[dict], key: str) -> list[tuple[float, float]]:
    return [
        (float(row["step"]), float(row[key]))
        for row in rows
        if key in row and "step" in row
    ]


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    span = high - low
    step = span / max(count - 1, 1)
    magnitude = 1.0
    while magnitude > step:
        magnitude /= 10
    while magnitude * 10 <= step:
        magnitude *= 10
    rounded = (step // magnitude) * magnitude
    if rounded <= 0:
        rounded = magnitude
    start = (low // rounded) * rounded
    ticks = []
    value = start
    while value <= high + rounded:
        ticks.append(value)
        value += rounded
    return ticks[:count]


def render_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, str, list[tuple[float, float]]]],
    width: int = 760,
    height: int = 340,
) -> str:
    left, right, top, bottom = 78, 18, 34, 46
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = [(x, y) for _, _, pairs in series for x, y in pairs]
    if len(points) < 2:
        raise ValueError(f"not enough points to plot {title}")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.06
    y_min -= pad
    y_max += pad

    def sx(value: float) -> float:
        if x_max <= x_min:
            return left
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-size="14" fill="#111111">{title}</text>',
    ]

    for tick in _nice_ticks(y_min, y_max):
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="#4b5563">{tick:g}</text>'
        )

    for tick in _nice_ticks(x_min, x_max, count=6):
        if tick < x_min or tick > x_max:
            continue
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" '
            f'text-anchor="middle" fill="#4b5563">{tick:g}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#9ca3af" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#9ca3af" stroke-width="1"/>'
    )

    colors = ("#2563eb", "#dc2626", "#059669", "#d97706")
    for index, (name, _key, pairs) in enumerate(series):
        if not pairs:
            continue
        color = colors[index % len(colors)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pairs)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linejoin="round"/>'
        )
        legend_x = left + 8 + index * 132
        legend_y = height - 12
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="11" height="3" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 17}" y="{legend_y - 4}" font-size="11" '
            f'fill="#374151">{name}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" font-size="11" '
        f'text-anchor="middle" fill="#4b5563">{x_label}</text>'
    )
    parts.append(
        f'<text x="14" y="{top + plot_h / 2:.1f}" font-size="11" fill="#4b5563" '
        f'transform="rotate(-90 14 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle">{y_label}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


CHARTS: tuple[tuple[str, str, str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "loss",
        "Training and dev loss",
        "optimizer step",
        "loss (nats/token)",
        (("train_loss", "train"), ("eval_loss", "dev")),
    ),
    (
        "learning_rate",
        "Learning rate",
        "optimizer step",
        "learning rate",
        (("learning_rate", "lr"),),
    ),
    (
        "memory",
        "CUDA peak memory",
        "optimizer step",
        "GiB",
        (("cuda_max_memory_allocated_gib", "peak allocated"),),
    ),
    (
        "throughput",
        "Training throughput",
        "optimizer step",
        "tokens/s",
        (("tokens_per_second", "tokens/s"),),
    ),
    (
        "bits_per_byte",
        "Training bits per byte",
        "optimizer step",
        "bits/byte",
        (("train_bits_per_byte", "train"),),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/plot_training_curves.py",
        description="Render training curves as SVG from metrics.jsonl.",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_path = args.run / "metrics.jsonl"
    if not metrics_path.is_file():
        print(f"error: missing {metrics_path}", file=sys.stderr)
        return 2

    try:
        rows = _load_rows(metrics_path)
        for row in rows:
            if "cuda_max_memory_allocated_bytes" in row:
                row["cuda_max_memory_allocated_gib"] = (
                    row["cuda_max_memory_allocated_bytes"] / 1024**3
                )

        args.output.mkdir(parents=True, exist_ok=True)
        written = []
        for name, title, x_label, y_label, keys in CHARTS:
            series = [
                (label, key, values)
                for key, label in keys
                if (values := _series(rows, key))
            ]
            if not series:
                continue
            target = args.output / f"{name}.svg"
            target.write_text(
                render_svg(
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    series=series,
                ),
                encoding="utf-8",
            )
            written.append(str(target))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"charts": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
