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
    config_path_for_record,
    read_registry_records,
    run_dir_for_record,
    run_status,
)

UNSUCCESSFUL = {'failed', 'incomplete', 'missing'}


def _latest_artifact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get('run_id', '')), str(run_dir_for_record(record)))
        if key[0]:
            latest[key] = record
    return list(latest.values())


def selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Select latest records per artifact directory from ids and status filters."""
    records = read_registry_records(args.registry)
    if args.run_ids:
        requested = set(args.run_ids)
        records = [record for record in records if str(record.get('run_id')) in requested]
    selected = _latest_artifact_records(records)
    statuses = set(args.statuses or [])
    if getattr(args, 'unsuccessful', False):
        statuses.update(UNSUCCESSFUL)
    if statuses:
        selected = [record for record in selected if run_status(record) in statuses]
    return selected


def command_list(args: argparse.Namespace) -> None:
    """List runs and cleanup status."""
    for record in selected_records(args):
        print(
            f'{record.get("run_id", "")}\ttrial={record.get("trial_id", 1)}'
            f'\t{run_status(record)}\t{run_dir_for_record(record)}'
        )


def archive_record(record: dict[str, Any], output_dir: Path) -> Path:
    """Archive one run directory and registry record into a tar.gz file."""
    run_id = str(record.get('run_id', 'run'))
    trial_id = int(record.get('trial_id', 1))
    artifact_name = f'{run_id}_trial_{trial_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f'{artifact_name}_{int(time.time())}.tar.gz'
    run_dir = run_dir_for_record(record)
    with tarfile.open(archive_path, 'w:gz') as archive:
        if run_dir.exists():
            archive.add(run_dir, arcname=f'{artifact_name}/run_dir')
        config_path = config_path_for_record(record)
        if config_path is not None and config_path.exists():
            archive.add(str(config_path), arcname=f'{artifact_name}/config.yaml')
        payload = json.dumps(record, indent=2, sort_keys=True, default=str).encode('utf-8')
        info = tarfile.TarInfo(f'{artifact_name}/registry_record.json')
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
        print(f'{action} {run_id}\ttrial={record.get("trial_id", 1)}\t{status}\t{run_dir}')
        if not dry_run and run_dir.exists():
            shutil.rmtree(run_dir)
        config_path = config_path_for_record(record)
        if args.delete_config and config_path is not None and config_path.exists():
            print(f'{action} config\t{config_path}')
            if not dry_run:
                config_path.unlink()
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
