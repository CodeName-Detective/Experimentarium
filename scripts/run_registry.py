"""Inspect outputs/run_registry.jsonl and print replay commands."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path('outputs/run_registry.jsonl')


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read registry JSONL records from disk."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def latest_for_run(records: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Return the latest registry record for a run id."""
    matches = [record for record in records if str(record.get('run_id')) == run_id]
    if not matches:
        raise SystemExit(f'run_id not found in registry: {run_id}')
    return matches[-1]


def flatten(value: Any, prefix: str = '') -> dict[str, Any]:
    """Flatten a nested config dictionary into dotted keys."""
    if not isinstance(value, dict):
        return {prefix: value}
    items: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f'{prefix}.{key}' if prefix else str(key)
        items.update(flatten(child, child_prefix))
    return items


def command_list(args: argparse.Namespace) -> None:
    """Print recent registry records in a compact table."""
    records = read_records(args.registry)
    selected = records[-args.limit :] if args.limit else records
    for record in selected:
        config = record.get('config', {})
        mode = config.get('run', {}).get('mode', '') if isinstance(config, dict) else ''
        print(f'{record.get("run_id", "")}	{mode}	{record.get("config_id", "")}	{record.get("run_dir", "")}')


def command_show(args: argparse.Namespace) -> None:
    """Print one registry record as formatted JSON."""
    record = latest_for_run(read_records(args.registry), args.run_id)
    print(json.dumps(record, indent=2, sort_keys=True, default=str))


def command_latest_command(args: argparse.Namespace) -> None:
    """Print the original command for the newest registry record."""
    records = read_records(args.registry)
    if not records:
        raise SystemExit(f'no records found in {args.registry}')
    print(records[-1].get('command', ''))


def command_replay_command(args: argparse.Namespace) -> None:
    """Print a command that replays a saved resolved config."""
    record = latest_for_run(read_records(args.registry), args.run_id)
    config_path = record.get('config_path')
    if not config_path:
        raise SystemExit(f'registry record has no config_path for run_id={args.run_id}')
    parts = ['uv', 'run', 'python', 'src/main.py', '--config-file', str(config_path)]
    if args.new_run_id:
        parts.extend(['--run-id', args.new_run_id])
    if args.overrides:
        parts.extend(args.overrides)
    print(shlex.join(parts))


def command_diff(args: argparse.Namespace) -> None:
    """Print flattened config differences between two runs."""
    records = read_records(args.registry)
    left = flatten(latest_for_run(records, args.left).get('config', {}))
    right = flatten(latest_for_run(records, args.right).get('config', {}))
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            print(f'{key}: {left.get(key)!r} -> {right.get(key)!r}')


def build_parser() -> argparse.ArgumentParser:
    """Build the run-registry CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY, help='Path to run_registry.jsonl')
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser('list', help='List recent runs')
    list_parser.add_argument('--limit', type=int, default=20)
    list_parser.set_defaults(func=command_list)

    show_parser = subparsers.add_parser('show', help='Print one registry record as JSON')
    show_parser.add_argument('run_id')
    show_parser.set_defaults(func=command_show)

    latest_parser = subparsers.add_parser('latest-command', help='Print the original command for the latest record')
    latest_parser.set_defaults(func=command_latest_command)

    replay_parser = subparsers.add_parser('replay-command', help='Print a command that replays a saved resolved config')
    replay_parser.add_argument('run_id')
    replay_parser.add_argument('--new-run-id', help='Optional run id override for the replay')
    replay_parser.add_argument(
        'overrides', nargs='*', help='Optional key=value overrides appended to the replay command'
    )
    replay_parser.set_defaults(func=command_replay_command)

    diff_parser = subparsers.add_parser('diff', help='Print flattened config differences between two runs')
    diff_parser.add_argument('left')
    diff_parser.add_argument('right')
    diff_parser.set_defaults(func=command_diff)
    return parser


def main() -> None:
    """Run the run-registry CLI."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
