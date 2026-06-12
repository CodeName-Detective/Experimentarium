#!/usr/bin/env bash
# Profile a tiny CPU or CUDA training smoke workload.
# Usage: bash scripts/profile.sh
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
if [[ "${PROFILE_CUDA:-0}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi
uv run python - <<'PROFILEPY'
import os
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from src.data import build_dataloaders
from src.models import MLP
from src.tasks import build_task

cfg = {
    'run': {'seed': 42, 'device': 'cpu'},
    'model': {'name': 'mlp', 'input_dim': 16, 'hidden_dim': 32, 'num_layers': 1, 'num_classes': 2, 'dropout': 0.0},
    'data': {'name': 'toy_classification', 'seed': 42, 'input_dim': 16, 'num_classes': 2, 'batch_size': 4, 'num_workers': 0, 'pin_memory': False, 'splits': {'train': {'num_samples': 16}, 'val': {'num_samples': 8}, 'test': {'num_samples': 8}}},
    'task': {'name': 'classification', 'output_key': 'logits', 'target_key': 'label', 'loss': {'name': 'cross_entropy'}, 'metrics': ['accuracy']},
}
activities = [ProfilerActivity.CPU]
if os.environ.get('PROFILE_CUDA') == '1':
    activities.append(ProfilerActivity.CUDA)
model = MLP(cfg['model'])
task = build_task(cfg)
loader = build_dataloaders(cfg)['train']
batch = next(iter(loader))
with profile(activities=activities, on_trace_ready=tensorboard_trace_handler('./outputs/profiles'), record_shapes=True, with_stack=False) as prof:
    result = task.step(model, batch, 'profile')
    result.loss.backward()
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=15))
PROFILEPY
