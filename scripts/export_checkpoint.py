"""Export checkpoints as model-only weights, full checkpoint copies, or TorchScript."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.engine.evaluator import move_to_device
from src.utils.config import cfg_get
from src.utils.registry import MODEL_REGISTRY
from src.utils.run_inspect import DEFAULT_REGISTRY_PATH, checkpoint_path_for_run, config_path_for_run
from src.utils.sanity import bootstrap_registries


def resolve_checkpoint(source: str, selector: str, registry: Path) -> tuple[Path, str | None]:
    """Resolve source as either a checkpoint path or a run id."""
    path = Path(source).expanduser()
    if path.exists() or path.suffix == '.pt':
        return path, None
    return checkpoint_path_for_run(source, selector, registry), source


def default_output_path(checkpoint_path: Path, run_id: str | None, export_format: str) -> Path:
    """Return a default export path for a checkpoint."""
    stem = checkpoint_path.stem
    owner = run_id or checkpoint_path.parent.parent.name or 'checkpoint'
    suffix = '.pt' if export_format in {'state_dict', 'checkpoint'} else '.ts'
    return Path('outputs/exports') / owner / f'{stem}_{export_format}{suffix}'


def export_state_dict(checkpoint_path: Path, output: Path) -> None:
    """Export only model weights and minimal metadata."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    payload = {
        'model_state': state['model_state'],
        'checkpoint_meta': state.get('checkpoint_meta', {}),
        'epoch': state.get('epoch'),
        'global_step': state.get('global_step'),
        'best_metric': state.get('best_metric'),
        'metrics': state.get('metrics', {}),
        'cfg': state.get('cfg', {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def export_torchscript(checkpoint_path: Path, output: Path, run_id: str | None, registry: Path) -> None:
    """Trace and export a TorchScript model for compatible tensor-dict models."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg: Any
    if run_id is not None:
        cfg = OmegaConf.load(config_path_for_run(run_id, registry))
    else:
        cfg = OmegaConf.create(state.get('cfg', {}))
    bootstrap_registries()
    model = MODEL_REGISTRY.build(str(cfg_get(cfg, 'model.name', 'mlp')), cfg_get(cfg, 'model'))
    model.load_state_dict(state['model_state'])
    model.eval()
    loaders = build_dataloaders(cfg)
    sample = move_to_device(next(iter(loaders['train'])), 'cpu')
    traced = torch.jit.trace(model, (sample,), strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))


def build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint export CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Run id or checkpoint .pt path')
    parser.add_argument('--checkpoint', default='best', help='Selector used when source is a run id')
    parser.add_argument('--format', choices=('state_dict', 'checkpoint', 'torchscript'), default='state_dict')
    parser.add_argument('--output', type=Path, help='Output file path')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY_PATH)
    return parser


def main() -> None:
    """Run the checkpoint export CLI."""
    args = build_parser().parse_args()
    checkpoint_path, run_id = resolve_checkpoint(args.source, args.checkpoint, args.registry)
    if not checkpoint_path.exists():
        raise SystemExit(f'checkpoint not found: {checkpoint_path}')
    output = args.output or default_output_path(checkpoint_path, run_id, args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == 'checkpoint':
        shutil.copyfile(checkpoint_path, output)
    elif args.format == 'torchscript':
        export_torchscript(checkpoint_path, output, run_id, args.registry)
    else:
        export_state_dict(checkpoint_path, output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
