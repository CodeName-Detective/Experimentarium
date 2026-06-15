#!/usr/bin/env bash
# Profile a configurable CPU or CUDA training smoke workload.
# Usage: bash scripts/profile.sh
# Optional: PROFILE_CONFIG=configs/profiler.yaml PROFILE_CUDA=1 bash scripts/profile.sh
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PROFILE_CONFIG="${PROFILE_CONFIG:-configs/profiler.yaml}"

uv run python - <<'PROFILEPY'
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

import src.models  # noqa: F401 - import registers built-in models.
from src.data import build_dataloaders
from src.engine.evaluator import move_to_device
from src.engine.precision import precision_autocast
from src.tasks import build_task
from src.utils.config import cfg_get, config_to_dict, load_config
from src.utils.registry import MODEL_REGISTRY
from src.utils.seed import setup_reproducibility

_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'off'}


def env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f'{name} must be one of {_TRUE_VALUES | _FALSE_VALUES}, got {value!r}')


def set_cfg_value(cfg: dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split('.')
    for part in parts[:-1]:
        next_value = cur.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise TypeError(f'Cannot set {key}; {part} is not a mapping')
        cur = next_value
    cur[parts[-1]] = value


def infinite_batches(loader: Any) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def resolve_device(cfg: dict[str, Any]) -> tuple[torch.device, bool]:
    env_cuda = env_bool('PROFILE_CUDA')
    requested_device = str(cfg_get(cfg, 'run.device', 'cpu'))
    profile_cuda = bool(cfg_get(cfg, 'profiler.cuda', False)) if env_cuda is None else env_cuda

    if env_cuda is False:
        requested_device = 'cpu'
    elif requested_device.startswith('cuda') and env_cuda is None:
        profile_cuda = True
    elif profile_cuda and not requested_device.startswith('cuda'):
        requested_device = 'cuda'

    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA requested but unavailable; falling back to CPU.')
        device = torch.device('cpu')
        profile_cuda = False

    return device, profile_cuda and device.type == 'cuda'


def main() -> None:
    config_path = Path(os.environ['PROFILE_CONFIG'])
    if not config_path.exists():
        raise FileNotFoundError(f'Profiler config not found: {config_path}')

    cfg = config_to_dict(load_config(config_path))
    device, profile_cuda = resolve_device(cfg)
    set_cfg_value(cfg, 'run.device', str(device))

    seed = int(cfg_get(cfg, 'run.seed', 42))
    deterministic = bool(cfg_get(cfg, 'run.deterministic', False))
    setup_reproducibility(seed=seed, strict=deterministic, require_cuda=device.type == 'cuda')

    model_cfg = cfg_get(cfg, 'model')
    model_name = str(cfg_get(model_cfg, 'name', 'mlp'))
    model = MODEL_REGISTRY.build(model_name, model_cfg).to(device)
    model.train()

    task = build_task(cfg)
    split = str(cfg_get(cfg, 'profiler.split', 'train'))
    loaders = build_dataloaders(cfg)
    if split not in loaders:
        available = ', '.join(sorted(loaders))
        raise KeyError(f'Profiler split {split!r} not found. Available splits: {available}')
    batch_iter = infinite_batches(loaders[split])

    trace_dir = Path(str(cfg_get(cfg, 'profiler.trace_dir', 'outputs/profiles')))
    trace_dir.mkdir(parents=True, exist_ok=True)
    warmup_steps = max(0, int(cfg_get(cfg, 'profiler.warmup_steps', 0)))
    active_steps = max(1, int(cfg_get(cfg, 'profiler.active_steps', 1)))
    backward = bool(cfg_get(cfg, 'profiler.backward', True))
    precision = str(cfg_get(cfg, 'run.precision', 'fp32'))
    stage = str(cfg_get(cfg, 'profiler.stage', 'profile'))

    def run_step() -> None:
        batch = move_to_device(next(batch_iter), device)
        model.zero_grad(set_to_none=True)
        with precision_autocast(device, precision):
            result = task.step(model, batch, stage)
        if backward:
            if result.loss is None:
                raise RuntimeError('Profiler backward=true requires task.step(...) to return a loss')
            result.loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(warmup_steps):
        run_step()

    activities = [ProfilerActivity.CPU]
    if profile_cuda:
        activities.append(ProfilerActivity.CUDA)

    print(f'[profiler] config={config_path}')
    print(f'[profiler] device={device} precision={precision} cuda_activity={profile_cuda}')
    print(f'[profiler] split={split} warmup_steps={warmup_steps} active_steps={active_steps} trace_dir={trace_dir}')

    with profile(
        activities=activities,
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(cfg_get(cfg, 'profiler.record_shapes', True)),
        profile_memory=bool(cfg_get(cfg, 'profiler.profile_memory', False)),
        with_stack=bool(cfg_get(cfg, 'profiler.with_stack', False)),
        with_flops=bool(cfg_get(cfg, 'profiler.with_flops', False)),
    ) as prof:
        for _ in range(active_steps):
            run_step()
            prof.step()

    sort_by = str(cfg_get(cfg, 'profiler.sort_by', 'cpu_time_total'))
    row_limit = int(cfg_get(cfg, 'profiler.row_limit', 15))
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


if __name__ == '__main__':
    main()
PROFILEPY
