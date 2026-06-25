"""Evaluate, test, or predict from a saved run id and checkpoint selector."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import shlex
import subprocess  # noqa: S404
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_inspect import DEFAULT_REGISTRY_PATH, checkpoint_path_for_run, config_path_for_run


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the Python command used to evaluate a saved run."""
    config_path = config_path_for_run(args.run_id, args.registry, mode='train')
    checkpoint_path = checkpoint_path_for_run(args.run_id, args.checkpoint, args.registry)
    command = [
        sys.executable,
        str(ROOT / 'src' / 'main.py'),
        '--config-file',
        str(config_path),
        f'run.mode={args.mode}',
        f'checkpoint.resume={checkpoint_path}',
    ]
    command.extend(args.overrides or [])
    return command


def printable_command(command: list[str]) -> str:
    """Return a shell-friendly uv command for display."""
    display = ['uv', 'run', 'python']
    script = str(ROOT / 'src' / 'main.py')
    with suppress(ValueError):
        script = str(Path(script).relative_to(Path.cwd()))
    display.append(script)
    display.extend(command[2:])
    return shlex.join(display)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluate-run CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--checkpoint', default='best', help='Checkpoint selector or .pt path. Default: best')
    parser.add_argument('--mode', choices=('eval', 'test', 'predict'), default='eval')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--print-only', action='store_true', help='Print the command without executing it')
    parser.add_argument('overrides', nargs='*', help='Additional key=value overrides')
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse options and trailing config overrides in any order."""
    return build_parser().parse_intermixed_args(argv)


def main() -> None:
    """Run the evaluate-run CLI."""
    args = parse_args()
    command = build_command(args)
    if args.print_only:
        print(printable_command(command))
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)  # noqa: S603


if __name__ == '__main__':
    main()
