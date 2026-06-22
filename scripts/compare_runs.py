"""Generate comparison reports from run registry records and metrics logs."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import cfg_get
from src.utils.run_inspect import (
    DEFAULT_REGISTRY_PATH,
    best_metric_values,
    final_metric_values,
    latest_record_for_run,
    latest_records,
    read_metric_points,
    read_registry_records,
    run_status,
)

DEFAULT_CONFIG_FIELDS = [
    'model.name',
    'data.name',
    'task.name',
    'optimizer.name',
    'optimizer.lr',
    'trainer.max_epochs',
    'run.precision',
]


def build_rows(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    """Build comparison table columns and rows."""
    records = read_registry_records(args.registry)
    selected = (
        [latest_record_for_run(records, run_id) for run_id in args.run_ids] if args.run_ids else latest_records(records)
    )
    if args.limit and not args.run_ids:
        selected = selected[-args.limit :]
    metric_names = set(args.metrics or [])
    rows: list[dict[str, Any]] = []
    for record in selected:
        points = read_metric_points(record)
        final = final_metric_values(points)
        best = best_metric_values(points)
        if not args.metrics:
            metric_names.update(final)
        config = record.get('config', {})
        row: dict[str, Any] = {
            'run_id': record.get('run_id', ''),
            'status': run_status(record),
            'mode': cfg_get(config, 'run.mode', ''),
            'config_id': record.get('config_id', ''),
            'run_dir': record.get('run_dir', ''),
        }
        for field in args.config_fields:
            row[field] = cfg_get(config, field, '')
        for metric in sorted(metric_names):
            row[f'final/{metric}'] = final.get(metric, '')
            row[f'best/{metric}'] = best.get(metric, '')
        if args.include_command:
            row['command'] = record.get('command', '')
        rows.append(row)
    columns = ['run_id', 'status', 'mode', 'config_id', *args.config_fields]
    for metric in sorted(metric_names):
        columns.extend([f'final/{metric}', f'best/{metric}'])
    columns.append('run_dir')
    if args.include_command:
        columns.append('command')
    return columns, rows


def render_markdown(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Render rows as a Markdown table."""
    output = StringIO()
    output.write('| ' + ' | '.join(columns) + ' |\n')
    output.write('| ' + ' | '.join('---' for _ in columns) + ' |\n')
    for row in rows:
        output.write('| ' + ' | '.join(_format_cell(row.get(column, '')) for column in columns) + ' |\n')
    return output.getvalue()


def render_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Render rows as CSV."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def render_report(args: argparse.Namespace) -> str:
    """Render a comparison report in the requested format."""
    columns, rows = build_rows(args)
    if args.format == 'json':
        return json.dumps(rows, indent=2, sort_keys=True, default=str) + '\n'
    if args.format == 'csv':
        return render_csv(columns, rows)
    return render_markdown(columns, rows)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison-report CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='*', help='Run ids to compare. Defaults to latest record for every run.')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        '--metrics', nargs='*', default=None, help='Metric names to include. Defaults to all logged metrics.'
    )
    parser.add_argument('--config-fields', nargs='*', default=DEFAULT_CONFIG_FIELDS)
    parser.add_argument('--format', choices=('markdown', 'csv', 'json'), default='markdown')
    parser.add_argument('--output', type=Path, help='Optional output file. Prints to stdout when omitted.')
    parser.add_argument('--limit', type=int, default=0, help='Limit records when run_ids are omitted.')
    parser.add_argument('--include-command', action='store_true', help='Include original invocation command.')
    return parser


def main() -> None:
    """Run the comparison-report CLI."""
    args = build_parser().parse_args()
    report = render_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding='utf-8')
    else:
        print(report, end='')


if __name__ == '__main__':
    main()
