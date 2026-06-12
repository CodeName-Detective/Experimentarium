"""Generate simple tensor-file dataset splits for local smoke tests.

This module is intentionally small: it creates ``data/processed/train.pt``,
``val.pt``, and ``test.pt`` files that are compatible with ``TensorFileDataset``.
Use it when you want to validate file-backed data loading before replacing the
contents with a real project preprocessing pipeline.

CLI usage:
    uv run python src/data/preprocess.py
    uv run python src/data/preprocess.py --force --input-dim 32 --num-classes 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def make_split(path: Path, num_samples: int, input_dim: int, num_classes: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, input_dim, generator=generator)
    weights = torch.randn(input_dim, num_classes, generator=torch.Generator().manual_seed(9999))
    y = (x @ weights).argmax(dim=-1).long()
    samples = [{'input': x[i], 'label': y[i]} for i in range(num_samples)]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate tensor_file toy dataset splits.')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--input-dim', type=int, default=16)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--train-samples', type=int, default=256)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--test-samples', type=int, default=64)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    specs = {
        'train': (args.train_samples, 42),
        'val': (args.val_samples, 1042),
        'test': (args.test_samples, 2042),
    }
    for split, (num_samples, seed) in specs.items():
        path = output_dir / f'{split}.pt'
        if path.exists() and not args.force:
            print(f'skip existing {path} (use --force to overwrite)')
            continue
        make_split(path, num_samples, args.input_dim, args.num_classes, seed)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
