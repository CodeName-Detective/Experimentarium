"""List, archive, and clean up run artifacts, including failed or incomplete runs."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_inspect import (
    DEFAULT_REGISTRY_PATH,
    latest_record_for_run,
    latest_records,
    read_registry_records,
    run_dir_for_record,
    run_status,
)

UNSUCCESSFUL = {'failed', 'incomplete', 'missing'}


def selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Select registry records from ids and status filters."""
    records = read_registry_records(args.registry)
    selected = (
        [latest_record_for_run(records, run_id) for run_id in args.run_ids] if args.run_ids else latest_records(records)
    )
    statuses = set(args.statuses or [])
    if getattr(args, 'unsuccessful', False):
        statuses.update(UNSUCCESSFUL)
    if statuses:
        selected = [record for record in selected if run_status(record) in statuses]
    return selected


def command_list(args: argparse.Namespace) -> None:
    """List runs and cleanup status."""
    for record in selected_records(args):
        print(f'{record.get("run_id", "")}\t{run_status(record)}\t{run_dir_for_record(record)}')


def archive_record(record: dict[str, Any], output_dir: Path) -> Path:
    """Archive one run directory and registry record into a tar.gz file."""
    run_id = str(record.get('run_id', 'run'))
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f'{run_id}_{int(time.time())}.tar.gz'
    run_dir = run_dir_for_record(record)
    with tarfile.open(archive_path, 'w:gz') as archive:
        if run_dir.exists():
            archive.add(run_dir, arcname=f'{run_id}/run_dir')
        config_path = record.get('config_path')
        if config_path and Path(str(config_path)).exists():
            archive.add(str(config_path), arcname=f'{run_id}/config.yaml')
        payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode('utf-8')
        info = tarfile.TarInfo(f'{run_id}/registry_record.json')
        info.size = len(payload)
        archive.addfile(info, fileobj=io.BytesIO(payload))
    return archive_path


def command_archive(args: argparse.Namespace) -> None:
    """Archive selected runs."""
    for record in selected_records(args):
        print(f'archived {record.get("run_id", "")} -> {archive_record(record, args.output_dir)}')


def command_cleanup(args: argparse.Namespace) -> None:
    """Delete selected run artifacts after optional archive."""
    records = selected_records(args)
    if not records:
        print('no matching runs')
        return
    dry_run = not args.yes
    for record in records:
        run_id = str(record.get('run_id', ''))
        status = run_status(record)
        run_dir = run_dir_for_record(record)
        if args.archive_first and run_dir.exists() and not dry_run:
            print(f'archived {run_id} -> {archive_record(record, args.archive_dir)}')
        action = 'would remove' if dry_run else 'removing'
        print(f'{action} {run_id}\t{status}\t{run_dir}')
        if not dry_run and run_dir.exists():
            shutil.rmtree(run_dir)
        config_path = record.get('config_path')
        if args.delete_config and config_path and Path(str(config_path)).exists():
            print(f'{action} config\t{config_path}')
            if not dry_run:
                Path(str(config_path)).unlink()
    if dry_run:
        print('dry run only; pass --yes to delete')


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add common run-selection arguments."""
    parser.add_argument('run_ids', nargs='*')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--statuses', nargs='*', choices=('success', 'failed', 'incomplete', 'missing'))
    parser.add_argument('--unsuccessful', action='store_true', help='Select failed, incomplete, and missing runs')


def build_parser() -> argparse.ArgumentParser:
    """Build the cleanup/archive CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser('list', help='List runs and status')
    add_selection_args(list_parser)
    list_parser.set_defaults(func=command_list)

    archive_parser = subparsers.add_parser('archive', help='Archive selected runs')
    add_selection_args(archive_parser)
    archive_parser.add_argument('--output-dir', type=Path, default=Path('outputs/archives'))
    archive_parser.set_defaults(func=command_archive)

    cleanup_parser = subparsers.add_parser('cleanup', help='Delete selected run directories')
    add_selection_args(cleanup_parser)
    cleanup_parser.add_argument('--yes', action='store_true', help='Actually delete files. Omit for dry-run.')
    cleanup_parser.add_argument('--archive-first', action='store_true', help='Archive each run before deleting')
    cleanup_parser.add_argument('--archive-dir', type=Path, default=Path('outputs/archives'))
    cleanup_parser.add_argument('--delete-config', action='store_true', help='Also delete saved config snapshots')
    cleanup_parser.set_defaults(func=command_cleanup)
    return parser


def main() -> None:
    """Run the cleanup/archive CLI."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
