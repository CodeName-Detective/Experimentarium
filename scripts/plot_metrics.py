"""Plot metric histories from run JSONL logs as HTML/SVG and CSV."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_inspect import (
    DEFAULT_REGISTRY_PATH,
    latest_record_for_run,
    read_metric_points,
    read_registry_records,
)

COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#4f46e5', '#be123c']


def build_metric_series(
    run_ids: list[str], registry: Path, metrics: list[str] | None
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Collect metric series by metric and run id."""
    records = read_registry_records(registry)
    selected_metrics = set(metrics or [])
    series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for run_id in run_ids:
        record = latest_record_for_run(records, run_id)
        for point in read_metric_points(record):
            if selected_metrics and point.metric not in selected_metrics:
                continue
            step = int(point.step if point.step is not None else len(series[point.metric][run_id]))
            series[point.metric][run_id].append((step, point.value))
    return dict(series)


def write_csv(series: dict[str, dict[str, list[tuple[int, float]]]], path: Path) -> None:
    """Write metric series to a tidy CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run_id', 'metric', 'step', 'value'])
        writer.writeheader()
        for metric, by_run in sorted(series.items()):
            for run_id, points in sorted(by_run.items()):
                for step, value in points:
                    writer.writerow({'run_id': run_id, 'metric': metric, 'step': step, 'value': value})


def render_html(series: dict[str, dict[str, list[tuple[int, float]]]]) -> str:
    """Render all metric series as an HTML document with inline SVG."""
    sections = []
    for metric, by_run in sorted(series.items()):
        sections.append(f'<section><h2>{html.escape(metric)}</h2>{_render_svg(by_run)}</section>')
    body = '\n'.join(sections) if sections else '<p>No metrics found.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Metric History</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
section {{ margin-bottom: 2rem; }}
svg {{ border: 1px solid #ddd; max-width: 100%; height: auto; }}
.legend {{ font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Metric History</h1>
{body}
</body>
</html>
"""


def _render_svg(by_run: dict[str, list[tuple[int, float]]]) -> str:
    width = 900
    height = 320
    pad = 44
    all_points = [point for points in by_run.values() for point in points]
    if not all_points:
        return '<p>No data.</p>'
    min_x = min(step for step, _ in all_points)
    max_x = max(step for step, _ in all_points)
    min_y = min(value for _, value in all_points)
    max_y = max(value for _, value in all_points)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1.0

    def scale_x(step: int) -> float:
        return pad + ((step - min_x) / (max_x - min_x)) * (width - 2 * pad)

    def scale_y(value: float) -> float:
        return height - pad - ((value - min_y) / (max_y - min_y)) * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#555" />',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#555" />',
        f'<text x="{pad}" y="{height - 10}" font-size="12">step {min_x}</text>',
        f'<text x="{width - pad - 70}" y="{height - 10}" font-size="12">step {max_x}</text>',
        f'<text x="6" y="{pad}" font-size="12">{max_y:.4g}</text>',
        f'<text x="6" y="{height - pad}" font-size="12">{min_y:.4g}</text>',
    ]
    legend = []
    for index, (run_id, points) in enumerate(sorted(by_run.items())):
        color = COLORS[index % len(COLORS)]
        coords = ' '.join(f'{scale_x(step):.2f},{scale_y(value):.2f}' for step, value in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}" />')
        legend.append(f'<span style="color:{color}">{html.escape(run_id)}</span>')
    parts.append('</svg>')
    parts.append('<p class="legend">' + ' | '.join(legend) + '</p>')
    return '\n'.join(parts)


def build_parser() -> argparse.ArgumentParser:
    """Build the metric plotting CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='+')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--metrics', nargs='*', default=None, help='Metric names to plot. Defaults to all metrics.')
    parser.add_argument('--output', type=Path, default=Path('outputs/reports/metrics.html'))
    parser.add_argument('--csv-output', type=Path, help='Optional CSV output path. Defaults next to HTML output.')
    return parser


def main() -> None:
    """Run the metric plotting CLI."""
    args = build_parser().parse_args()
    series = build_metric_series(args.run_ids, args.registry, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(series), encoding='utf-8')
    csv_path = args.csv_output or args.output.with_suffix('.csv')
    write_csv(series, csv_path)
    print(f'wrote {args.output}')
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
