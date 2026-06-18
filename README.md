# Experimentarium

A research-grade PyTorch experiment framework intended to be copied into a new ML project and extended. It is built around Hydra configs, registries, task-specific loss and metric logic, fault-tolerant checkpoints, sanity checks for new machines, and CPU-friendly tests.

Use `README.md` for day-to-day usage, `Run_commands.md` for train/eval command references, `sanity_check_commands.md` for sanity-check command references, `callback_tutorial.md` for callback hooks and examples, `profiler_tutorial.md` for profiler usage and result interpretation, `tutorial.md` for hands-on customization tutorials, `Description.md` for a folder-by-folder reference, and `Flowchart.md` for entrypoint-to-artifact flow diagrams.

## What This Template Provides

- Hydra config groups for models, data, tasks, optimizers, schedulers, trainer settings, logging, checkpoints, sanity checks, and experiments.
- Registry-based extension points for models, datasets, tasks, losses, metrics, optimizers, schedulers, callbacks, and loggers.
- A generic trainer and evaluator that keep model code pure and task logic explicit.
- Fault-tolerant checkpointing with atomic writes, manifest checksums, `last.pt`, `best.pt`, retention, resume, fallback loading, and exception checkpoints.
- A canonical sanity-check command for validating a machine before running expensive training.
- Toy classification, regression, sequence, and tensor-file workloads for smoke tests and examples.
- Local scripts for training, evaluation, preprocessing, profiling, sanity checks, and W&B sweeps.

## Quick Start

If you keep `pyproject.toml`, use `uv sync` as the primary install path:

```bash
uv sync --extra dev
```

If you also need experiment tracking or vision packages:

```bash
uv sync --extra dev --extra tracking --extra vision
```

## UV Dependency Install

The framework expects Python `>=3.10`. If you copy this template into a project where you will not keep `pyproject.toml`, create a `uv` environment and install the packages directly.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "torch>=2.2" "numpy>=1.24" "hydra-core>=1.3" "tqdm>=4.66" "rich>=13.0"
uv pip install "pytest>=7" "pytest-cov>=4" "ruff>=0.4" "mypy>=1.8"
```

Optional packages:

```bash
# experiment tracking
uv pip install "wandb>=0.16" "tensorboard>=2.15"

# vision workloads
uv pip install "torchvision>=0.17"
```

If your new project does keep a `pyproject.toml`, prefer `uv add` so requirements stay recorded:

```bash
uv add "torch>=2.2" "numpy>=1.24" "hydra-core>=1.3" "tqdm>=4.66" "rich>=13.0"
uv add --dev "pytest>=7" "pytest-cov>=4" "ruff>=0.4" "mypy>=1.8"
uv add --optional tracking "wandb>=0.16" "tensorboard>=2.15"
uv add --optional vision "torchvision>=0.17"
```

Choose the PyTorch `uv pip install` or `uv add` command that matches your CUDA, ROCm, or CPU-only machine. After installing, run the sanity check before launching real experiments.

### PyTorch CUDA Install With UV

For GPU machines, install the PyTorch wheel that matches the NVIDIA driver and GPU available on that machine. This example uses the CUDA 12.1 PyTorch wheel index:

```bash
uv init --python 3.10
```

[UV PyTorch Installation Guide](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch)

After installation, verify compatibility before training:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python scripts/run_sanity.py sanity.cuda.check_driver=true
uv run python scripts/run_sanity.py +experiment=sanity_gpu
uv run python src/main.py +experiment=sanity_gpu
```

Run the new-machine sanity check:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu
```

Run tests and lint:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src tests scripts/run_sanity.py scripts/run_registry.py scripts/verify_checkpoints.py
```

Train the default toy classification experiment:

```bash
uv run python src/main.py
```

Train with Hydra overrides:

```bash
uv run python src/main.py optimizer.lr=3e-4 trainer.max_epochs=20 model.hidden_dim=128
```

Use the shell wrappers if preferred:

```bash
make install
make sanity
make test
make lint
make train ARGS="+experiment=baseline trainer.max_epochs=10"
```

## Run Identity And Output Organization

Every entrypoint calls `src.utils.run.prepare_run` before building loggers or checkpoints. It computes a deterministic `run.config_id` hash from the effective config and derives a readable base run id from `run.name`, `model.name`, `data.name`, `task.name`, `run.trial`, and the hash.

Default artifact layout:

```text
outputs/
  run_configs/
    <run.id>.yaml              # resolved config snapshot for the run
  run_registry.jsonl           # run id, paths, repeat command, working directory, and resolved config
  runs/
    <run.id>/
      logs/
        train.log
        metrics.jsonl
        tensorboard/
      checkpoints/
        epoch_0001.pt
        last.pt
        best.pt
        manifest.json
      predictions/
        test_predictions.json
      profiles/
  evaluations/
    <run.id>/
      config.yaml
      logs/
        train.log
        metrics.jsonl
        tensorboard/
      predictions/
        test_predictions.json
      profiles/
```

Runs reuse the same run folder when the derived or manual id already exists. Fresh duplicate runs emit a capital warning and append new logs, metrics, predictions, and tracking events under the existing `outputs/runs/<run.id>/` directory. Intentional training resumes with `checkpoint.resume=...` do not emit that reuse warning; the trainer logs the resume epoch after loading the checkpoint. `eval`, `test`, and `predict` modes keep the source training `run.id`, load checkpoints from `outputs/runs/<run.id>/checkpoints/`, and write results plus their resolved config to `outputs/evaluations/<run.id>/`.

To create a planned repeat with a different base id, change a config value that is part of the run identity or provide a manual id:

```bash
uv run python src/main.py +experiment=baseline run.trial=2
uv run python src/main.py +experiment=baseline run.id=my_manual_run_id
```

W&B receives the resolved config through `wandb.init(config=...)`, uses `run.tracking_id` as the W&B id and default name, and logs the resolved config YAML as a W&B artifact. Training uses `run.tracking_id=<run.id>`; `eval`, `test`, and `predict` keep `run.id=<run.id>` for the filesystem while using `run.tracking_id=<run.id>_evaluation` to avoid merging tracking records.

Each `outputs/run_registry.jsonl` record includes a shell-safe `command` and `command_cwd`. Run that command from the recorded directory to repeat the same invocation, or use `uv run python scripts/run_registry.py replay-command <run_id>` to print a replay command from the saved resolved config. Sensitive CLI argument values are redacted in `command`, but avoid putting secrets in config values because the resolved config is also stored for reproducibility.

You can also replay a resolved output config directly:

```bash
uv run python src/main.py --config-file outputs/run_configs/<run_id>.yaml
uv run python src/main.py --config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run
```

`--config-file` is for fully resolved YAML snapshots such as `outputs/run_configs/<run_id>.yaml` or `outputs/evaluations/<run_id>/config.yaml`. Generated runtime paths and ids are regenerated before the run starts. Add `--run-id <id>`, `run.trial=...`, or `run.output_dir=...` after the file path when you want a distinct replay.

## Use This As A New Project Template

1. Copy or clone the repository into your new project directory.
2. If you keep `pyproject.toml`, update project name, description, dependencies, optional extras, and package metadata. If you remove it, use the `uv venv` and `uv pip install` commands above as your install reference.
3. Update `configs/logging/default.yaml`: W&B project, tags, and logging defaults.
4. Run `uv sync --extra dev` if using `pyproject.toml`, or create a `uv` environment and install the dependencies with `uv pip install`.
5. Run `uv run python scripts/run_sanity.py +experiment=sanity_cpu` before changing anything substantial.
6. Add your real dataset under `src/data/` and a matching config under `configs/data/`.
7. Add or adapt a model under `src/models/` and a matching config under `configs/model/`.
8. Add a task under `src/tasks/` if your workload differs from the built-in classification, regression, segmentation, detection, ranking, or language-modeling behavior.
9. Create an experiment file under `configs/experiment/` that selects the model, data, task, optimizer, scheduler, and overrides.
10. Add focused tests under `tests/`, then run the validation checklist below.

Keep the template boundary clean: source, configs, scripts, tests, and docs should be versioned; generated outputs, checkpoints, processed data, caches, local virtual environments, and editor folders should not be committed.

## Sanity Check On A New Machine

The sanity system is canonical in `src/utils/sanity/`, and the CLI entrypoint is `scripts/run_sanity.py`. Run it immediately after moving the code to a new machine or before launching a long experiment.

```bash
uv run python scripts/run_sanity.py sanity.torch_install.recommend=true
uv run python scripts/run_sanity.py +experiment=sanity_cpu
uv run python scripts/run_sanity.py +experiment=sanity_gpu
uv run python src/main.py +experiment=sanity_gpu
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true
```

It checks the Python version and package versions declared in `pyproject.toml`, expected config keys, runtime device visibility, PyTorch CUDA build and NVIDIA driver compatibility, output directory writability, disk space, registry contents, tensor-file paths, and a tiny data/model/loss/backward/optimizer/scheduler smoke pass. Package severity comes from the TOML structure: `[project].dependencies` are required checks, while `[project.optional-dependencies]` are warning-only checks. If `pyproject.toml` is removed, sanity warns and skips package version checks because there is no dependency source of truth.

CUDA compatibility is checked automatically when `run.device=cuda` or when your installed PyTorch build includes CUDA. In CPU mode, a PyTorch/driver mismatch is reported as a warning by default; with `+experiment=sanity_gpu`, it becomes a hard failure and the tiny smoke pass runs on CUDA. The report includes `torch.__version__`, `torch.version.cuda`, `nvidia-smi` driver details when available, the known minimum NVIDIA driver for that CUDA build, and captured PyTorch CUDA initialization warnings. Use `sanity.torch_install.recommend=true` to print recommended PyTorch install commands based on the Python requirement in `pyproject.toml`, detected GPU/driver, and known official PyTorch wheel indexes. This recommendation path is designed to work before torch is installed; it gets GPU and driver information from `nvidia-smi`.

W&B readiness checks run automatically when `logging.wandb.enabled=true`, or explicitly with `sanity.wandb.check=true`. They verify that `wandb` imports, an API key is available from `WANDB_API_KEY` or `uv run wandb login`, and the machine can reach the W&B host unless `logging.wandb.mode=offline` or `sanity.wandb.check_connectivity=false`.

If a copied project raises `Key 'wandb' is not in struct` for `sanity.wandb.check=true`, its `configs/sanity/default.yaml` is missing the W&B sanity block. Copy the `wandb:` block from this template's `configs/sanity/default.yaml`, or run a one-time appended override with `+sanity.wandb.check=true`.

Python usage:

```python
from src.utils.sanity import run_sanity_checks

report = run_sanity_checks(cfg, strict=True)
assert report.passed
```

## Common Commands

```bash
# Train default config
uv run python src/main.py

# Train an experiment config
uv run python src/main.py +experiment=baseline
uv run python src/main.py +experiment=regression
uv run python src/main.py +experiment=ablation_heads

# Evaluate/test/predict from a checkpoint
uv run python src/main.py run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
uv run python src/main.py run.mode=test checkpoint.resume=best
uv run python src/main.py run.mode=predict checkpoint.resume=epoch_0005
bash scripts/eval.sh outputs/runs/<run_id>/checkpoints/best.pt

# Resume training from latest checkpoint in this config-derived run directory
uv run python src/main.py checkpoint.resume=latest

# Replay a saved resolved run config
uv run python src/main.py --config-file outputs/run_configs/<run_id>.yaml
uv run python src/main.py --config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run

# Generate toy tensor-file data, then train from files
bash scripts/preprocess.sh --force
uv run python src/main.py data=tensor_file scheduler=none

# Profile a tiny workload using configs/profiler.yaml
bash scripts/profile.sh
PROFILE_CUDA=1 bash scripts/profile.sh
PROFILE_CONFIG=configs/profiler.yaml bash scripts/profile.sh

# Profile the configured trainer stack through src/main.py
uv run python src/main.py run.mode=profile profiler.active_steps=3

# Launch a W&B sweep after installing tracking extras
uv sync --extra tracking          # if keeping pyproject.toml
# or: uv pip install "wandb>=0.16" "tensorboard>=2.15"
bash scripts/sweep.sh
```

## Configuration Pattern

The root config is `configs/config.yaml`. It selects one file from each config group:

```yaml
defaults:
  - model: mlp
  - data: toy_classification
  - task: classification
  - optimizer: adamw
  - scheduler: cosine
  - trainer: default
  - logging: default
  - checkpoint: default
  - sanity: default
  - _self_
```

Use Hydra overrides to change any value:

```bash
uv run python src/main.py model=small_transformer data=toy_sequence task=classification
uv run python src/main.py scheduler=onecycle scheduler.max_lr=3e-3 scheduler.interval=step
uv run python src/main.py run.device=cuda run.precision=amp data.batch_size=128
```

Experiment presets are opt-in: files under `configs/experiment/` are not composed by default and only apply when selected with `+experiment=<name>`. Their values override the root defaults and selected config-group settings.

Standalone profiler defaults live in `configs/profiler.yaml` and are loaded by `bash scripts/profile.sh`; use `PROFILE_CONFIG=<path>` for an alternate profiler config. Trainer-level profiling through `src/main.py run.mode=profile` uses the top-level `profiler:` block in `configs/config.yaml` plus any CLI overrides.

Use experiment files for repeatable research runs:

```bash
uv run python src/main.py +experiment=baseline
```

## Add New Components

Add a model:

```python
import torch.nn as nn
from src.utils.registry import register_model


@register_model('my_model')
class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...

    def forward(self, batch):
        return {'logits': ...}
```

Then add `configs/model/my_model.yaml` and run:

```bash
uv run python src/main.py model=my_model
```

Add a dataset:

```python
from torch.utils.data import Dataset
from src.utils.registry import register_dataset


@register_dataset('my_dataset')
class MyDataset(Dataset):
    def __init__(self, cfg, split='train'): ...

    def __getitem__(self, idx):
        return {'x': ..., 'label': ...}
```

Then add `configs/data/my_dataset.yaml`. `src/data/dataloader.py` will build train, val, and test loaders from the registry.

File-backed paths belong in the data config. The built-in `tensor_file` dataset reads `data.splits.train.path`, `data.splits.val.path`, and `data.splits.test.path`; relative paths resolve from the repository working directory. A custom key such as `data.root` only has an effect when the registered dataset class reads it.

Add a task when the model output, target format, loss, metrics, or prediction records differ from the built-in classification, regression, segmentation, detection, ranking, or language-modeling behavior. Subclass `BaseTask`, register it with `@register_task('my_task')`, and reference it from `configs/task/my_task.yaml`.

Add losses and metrics with `@register_loss('name')` and `@register_metric('name')`, then reference them from task YAML. Add schedulers with `@register_scheduler('name')` and a matching config under `configs/scheduler/`.

## Scheduler Behavior

- `none`: Leaves LR unchanged; use for debugging and tiny sanity runs.
- `constant`: Keeps LR at a fixed factor for `total_iters`, then returns to base LR behavior.
- `linear`: Linearly interpolates LR from `start_factor` to `end_factor`.
- `step`: Drops LR by `gamma` every fixed `step_size` epochs or steps.
- `multistep`: Drops LR by `gamma` at explicit milestone epochs or steps.
- `exponential`: Multiplies LR by `gamma` each step for smooth monotonic decay.
- `cosine`: Smoothly anneals LR toward `eta_min` over the training horizon.
- `cosine_restart`: Runs cosine cycles with warm restarts controlled by `T_0` and `T_mult`.
- `plateau`: Reduces LR when a monitored metric stops improving.
- `polynomial`: Decays LR polynomially over the configured training horizon.
- `onecycle`: Ramps LR up and then down in one run; normally use `interval: step`.

## Fault Tolerance And Resume

Checkpointing is managed by `CheckpointManager` in `src/utils/checkpoint.py`.

Important settings live in `configs/checkpoint/default.yaml`:

```yaml
checkpoint:
  dir: outputs/checkpoints
  save_every: 1
  keep_last_k: 5
  save_last: true
  save_top_k: 1
  save_on_exception: true
  monitor: val/loss
  mode: min
  resume: null
```

Resume options:

```bash
uv run python src/main.py checkpoint.resume=latest
uv run python src/main.py checkpoint.resume=outputs/checkpoints/best.pt
uv run python src/main.py run.mode=eval checkpoint.resume=latest
uv run python src/main.py run.mode=eval checkpoint.resume=best
uv run python src/main.py run.mode=eval checkpoint.resume=epoch_0005
uv run python src/main.py run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
```

The checkpoint state includes model, optimizer, scheduler, scaler, RNG state, epoch, global step, best metric, and the composed config.

## Distributed And Mixed Precision

The framework includes distributed runtime helpers, rank-zero logging/checkpointing, distributed dataloader samplers, metric reduction, run-id broadcast, and automatic `torch.nn.parallel.DistributedDataParallel` wrapping when launched with `torchrun`.

A multi-process launch looks like:

```bash
uv run torchrun --nproc_per_node=2 src/main.py run.device=cuda data.batch_size=64
```

Useful precision overrides:

```bash
uv run python src/main.py run.precision=fp32
uv run python src/main.py run.precision=amp
uv run python src/main.py run.precision=bf16
```

`run.precision=fp32` uses the normal non-autocast path for training, validation, test evaluation, and prediction export. `amp`, `fp16`, and `bf16` use the same CUDA autocast policy across training and evaluator/prediction paths.

`src/runtime/distributed.py` handles rank helpers, DDP wrapping/unwrapping, small-object broadcast, and distributed metric reduction. `src/main.py` initializes distributed runtime from the `torchrun` environment when present.

## Validation Checklist

Run this before treating changes as reliable:

```bash
python3 -m compileall src scripts tests
uv run ruff check src tests scripts/run_sanity.py scripts/run_registry.py scripts/verify_checkpoints.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run python scripts/run_sanity.py +experiment=sanity_cpu
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true
uv run python src/main.py +experiment=sanity_cpu
uv run python src/main.py +experiment=sanity_gpu  # on CUDA machines only
```

For broader smoke coverage:

```bash
uv run python src/main.py +experiment=baseline trainer.max_epochs=1
uv run python src/main.py +experiment=regression trainer.max_epochs=1
uv run python src/main.py +experiment=ablation_heads trainer.max_epochs=1
```

## Generated Files

These are intentionally ignored and can be deleted at any time:

- `outputs/`: run registry, resolved config snapshots, run-scoped checkpoints, logs, profiler traces, and prediction samples.
- `data/processed/`: generated tensor-file toy splits from `scripts/preprocess.sh`.
- `.venv/`: local virtual environment.
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `__pycache__/`: tool and Python caches.
- `.vscode/`, `.idea/`: local editor settings.
- `wandb/`: local W&B run metadata.
