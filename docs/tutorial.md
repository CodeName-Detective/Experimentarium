# Framework Tutorial And Customization Cookbook

Use this file when you want to learn the framework by changing it. It is organized as hands-on tutorials: each section explains the goal, the files to edit, the minimum code/config shape, how to run it, and what to validate.

For quick commands, use `README.md`. For file-by-file reference, use `Description.md`. For execution flow diagrams, use `Flowchart.md`.

## Learning Path

If you are new to this repo, follow this order:

1. Run the new-machine sanity check.
2. Run the toy classification, regression, and sequence experiments.
3. Inspect the generated run folder, config snapshot, logs, checkpoints, and predictions.
4. Learn Hydra config groups and CLI overrides.
5. Add one small dataset.
6. Add one small model.
7. Add one metric or loss.
8. Add one experiment preset that combines your changes.
9. Run tests and sanity checks after each customization.

The framework is intentionally registry-based. Most customizations follow this pattern:

1. Add implementation under `src/...`.
2. Register it with `@register_*('name')`.
3. Add a matching config under `configs/...`.
4. Add or update an experiment under `configs/experiment/...`.
5. Add a focused test under `tests/...`.
6. Run sanity, tests, and lint.

## Tutorial 1: Install With UV

Use this when you clone or copy the template to a new machine. The `dev` dependency group is included by default:

```bash
uv sync
```

Add the optional tracking packages when needed:

```bash
uv sync --extra tracking
```

If you removed `pyproject.toml` in your copied project, install directly:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "torch>=2.2" "numpy>=1.24" "hydra-core>=1.3" "omegaconf>=2.3" "tqdm>=4.66" "rich>=13.0"
uv pip install "pytest>=7" "pytest-cov>=4" "ruff>=0.4" "mypy>=1.8"
```

Optional tracking and vision packages:

```bash
uv pip install "wandb>=0.16" "tensorboard>=2.15"
uv pip install "torchvision>=0.17"
```

Validate:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src tests scripts
```

## Tutorial 2: Run The First Training Job

Use this to confirm the normal training, validation, test, checkpoint, and prediction path works.

```bash
uv run python src/main.py
```

Equivalent shell wrapper:

```bash
bash scripts/train.sh
```

Useful overrides:

```bash
uv run python src/main.py trainer.max_epochs=3 optimizer.lr=3e-4 model.hidden_dim=128
uv run python src/main.py +experiment=baseline trainer.max_epochs=10
```

What happens:

1. Hydra composes `configs/config.yaml` plus overrides.
2. `src.utils.run.prepare_run` derives `run.id`, `run.config_id`, and run-scoped artifact paths.
3. Registries are bootstrapped.
4. Loggers, dataloaders, model, task, optimizer, scheduler, and checkpoint manager are built.
5. The trainer runs train, validation, checkpointing, test evaluation, and prediction export.

## Tutorial 3: Understand Run IDs And Outputs

Fresh training uses a stable experiment id plus a code-managed trial:

```text
outputs/
  run_configs/<run.id>/trial_<n>.yaml
  run_registry.jsonl
  runs/<run.id>/trial_<n>/
    logs/
    checkpoints/
    predictions/
    profiles/
```

Important behavior:

- Fresh `run.id` is derived from `run.name`, model/data/task names, and `run.config_id`, unless you provide a manual stable id.
- `run.trial_id` is not configured by the user. Repeating the same fresh config allocates the next trial automatically.
- Fresh trials never append logs or checkpoints into an older fresh trial.
- `run.tracking_id` and the W&B run name are generated from the same run/trial identity.
- Every invocation begins with a bold cyan line showing run id, trial id, mode, and output path.

Try two identical runs:

```bash
uv run python src/main.py +experiment=sanity_cpu
uv run python src/main.py +experiment=sanity_cpu
```

They share one stable run id and create `trial_1`, then `trial_2`.

Use a manual stable id for fresh work when helpful:

```bash
uv run python src/main.py +experiment=baseline run.id=my_manual_run
```

Replay and resume are different operations:

```bash
# Fresh replay: receives the next trial automatically.
uv run python src/main.py --from-run <run.id>
uv run python src/main.py --from-run <run.id> --run-id replayed_run

# Intentional resume: continues the checkpoint's existing trial.
uv run python src/main.py --resume-run <run.id>
```

Do not pass `--run-id` with `--resume-run`; resume identity is recovered from the checkpoint path.

## Tutorial 4: Use Hydra Overrides Correctly

Hydra lets you modify config values from the command line.

Change existing values:

```bash
uv run python src/main.py optimizer.lr=1e-4 trainer.max_epochs=20
```

Switch a config group:

```bash
uv run python src/main.py data=toy_regression model=regression_mlp task=regression
```

Load an experiment preset:

```bash
uv run python src/main.py +experiment=regression
```

Add a new field from CLI only when you really need it:

```bash
uv run python src/main.py +my_new_key=123
```

Preferred ownership:

- Global defaults: `configs/config.yaml`
- Reusable model settings: `configs/model/*.yaml`
- Reusable data settings: `configs/data/*.yaml`
- Workload semantics: `configs/task/*.yaml`
- Repeatable run recipes: `configs/experiment/*.yaml`
- One-off values: CLI overrides

## Tutorial 5: Create A New Experiment Preset

Use this when you want one named run recipe that is easy to repeat.

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_
# Short description of this experiment.

defaults:
  - override /model: mlp
  - override /data: toy_classification
  - override /task: classification
  - override /optimizer: adamw
  - override /scheduler: cosine

trainer:
  max_epochs: 10

optimizer:
  lr: 3.0e-4

model:
  hidden_dim: 128
  dropout: 0.1

run:
  name: my_experiment
  trial: 1
```

Run it:

```bash
uv run python src/main.py +experiment=my_experiment
```

Validate:

```bash
uv run python scripts/run_sanity.py +experiment=my_experiment sanity.check_all_experiments=true
```

## Tutorial 6: Run Existing Example Workloads

Classification baseline:

```bash
uv run python src/main.py +experiment=baseline
```

Regression:

```bash
uv run python src/main.py +experiment=regression
```

Sequence transformer ablation:

```bash
uv run python src/main.py +experiment=ablation_heads
```

Tiny CPU sanity workload:

```bash
uv run python src/main.py +experiment=sanity_cpu
```

Use these before adding your own dataset/model so you know the framework is healthy.

## Tutorial 7: Add A New Dataset

Use this when your batch can be represented as a dictionary containing tensors or nested tensor structures. Models usually consume `batch['input']`; target keys are task-specific. Classification and regression use:

```python
{'input': tensor, 'label': tensor}
```

Segmentation uses `mask`, ranking uses `relevance`, language modeling uses `labels`, and detection commonly uses nested `targets`. Keep the selected data, model, and task contracts aligned.

Edit `src/data/dataset.py` or add a new module under `src/data/` and import it from `src/data/__init__.py`.

Minimal dataset:

```python
from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.utils.config import cfg_get
from src.utils.registry import register_dataset


@register_dataset('my_dataset')
class MyDataset(Dataset):
    def __init__(self, cfg, split: str = 'train') -> None:
        self.split = split
        root = Path(str(cfg_get(cfg, 'root', 'data/my_dataset')))
        self.path = root / f'{split}.pt'
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.samples = torch.load(self.path, map_location='cpu', weights_only=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {'input': sample['input'], 'label': sample['label']}
```

Create `configs/data/my_dataset.yaml`:

```yaml
# Dataset selected by DATASET_REGISTRY.
name: my_dataset
# Project-relative root containing train.pt, val.pt, and test.pt.
root: data/my_dataset
batch_size: 32
num_workers: 4
pin_memory: true
drop_last: false
splits:
  train: {}
  val: {}
  test: {}
```

Run:

```bash
uv run python src/main.py data=my_dataset
```

Validate:

```bash
uv run python scripts/run_sanity.py data=my_dataset sanity.run_model_smoke=true
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_dataset.py -q
```

Best practices:

- Validate files and sample schema inside the dataset constructor.
- Keep expensive preprocessing out of `__getitem__`.
- Return CPU tensors; the trainer moves batches to `run.device`.
- Keep split-specific randomness deterministic through config seeds.

## Tutorial 8: Use Tensor-File Data

This template includes a simple file-backed dataset named `tensor_file`.

Generate toy tensor files:

```bash
bash scripts/preprocess.sh --force
```

Run from files:

```bash
uv run python src/main.py data=tensor_file scheduler=none
```

Expected default files:

```text
data/processed/train.pt
data/processed/val.pt
data/processed/test.pt
```

`TensorFileDataset` reads `data.splits.train.path`, `data.splits.val.path`, and `data.splits.test.path`. Relative paths resolve from the repository working directory. Customize them in `configs/data/tensor_file.yaml` or from the CLI:

```bash
uv run python src/main.py data=tensor_file data.splits.train.path=/path/train.pt data.splits.val.path=/path/val.pt data.splits.test.path=/path/test.pt
```

## Tutorial 9: Add A New Model

Model contract:

- Constructor receives `cfg.model`.
- `forward(batch)` receives a batch dictionary.
- `forward` returns a dictionary.
- Classification/regression tasks read `outputs[task.output_key]`, usually `outputs['logits']`.
- Dataset and task code own batch structure; models should not create dummy batches.

Edit `src/models/model.py` or add a module under `src/models/` and import it from `src/models/__init__.py`.

Minimal model:

```python
import torch
import torch.nn as nn

from src.utils.config import cfg_get
from src.utils.registry import register_model


@register_model('my_mlp')
class MyMLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        input_dim = int(cfg_get(cfg, 'input_dim', 16))
        hidden_dim = int(cfg_get(cfg, 'hidden_dim', 64))
        output_dim = int(cfg_get(cfg, 'output_dim', cfg_get(cfg, 'num_classes', 2)))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'logits': self.net(batch['input'].float())}
```

Create `configs/model/my_mlp.yaml`. Keep dataset-owned dimensions as interpolations from `data` so the model follows the selected dataset shape:

```yaml
name: my_mlp
input_dim: ${data.input_dim}
hidden_dim: 64
num_classes: ${data.num_classes}
dropout: 0.0
```

Run:

```bash
uv run python src/main.py model=my_mlp
```

Validate:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_model.py -q
uv run python scripts/run_sanity.py model=my_mlp
```

## Tutorial 10: Add A New Task

Add a task when your workload changes loss semantics, target keys, model-output interpretation, or prediction formatting. Built-ins already cover segmentation, detection, pointwise ranking, and autoregressive language modeling. Add another task for semantics such as multi-task or contrastive learning.

The built-in modules are `src/tasks/segmentation.py`, `detection.py`, `ranking.py`, and `language_modeling.py`. They still require compatible models and datasets. Detection models must accept the framework batch dictionary and return model losses and/or a `detections` collection; variable-size detection datasets can expose `collate_fn` for `build_dataloaders` to use.

Built-in task contracts:

| Task config | Batch target | Model output | Built-in behavior |
| --- | --- | --- | --- |
| `task=segmentation` | `mask` with shape `[B, ...]` | `logits` with shape `[B, C, ...]` | Cross-entropy, ignore-index filtering, pixel accuracy, and mask export |
| `task=detection` | `targets`, usually one mapping per image | `loss`, `losses`, or `loss_*` values for training; `detections` for prediction | Loss aggregation, score filtering, NMS, and JSON-safe export |
| `task=ranking` | `relevance` with the same shape as scores | `scores` | Pointwise MSE/MAE and descending ranking export |
| `task=language_modeling` | `labels` with shape `[B, T]` | `logits` with shape `[B, T, V]` | Token cross-entropy, ignore-index filtering, token accuracy, perplexity, and token export |

Selecting a task config changes task semantics only. Provide a model and dataset whose output and batch keys satisfy the selected contract; the four new task configs do not automatically select matching models or datasets.

Edit `src/tasks/task.py` or add a module under `src/tasks/` and import it from `src/tasks/__init__.py`.

Minimal task:

```python
from typing import Any
from torch import Tensor, nn

from src.losses import build_loss
from src.tasks.task import BaseTask, StepResult
from src.utils.config import cfg_get
from src.utils.registry import register_task


@register_task('my_task')
class MyTask(BaseTask):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.loss_fn = build_loss(cfg_get(cfg, 'loss', {'name': 'cross_entropy'}))

    def step(self, model: nn.Module, batch: dict[str, Tensor], stage: str) -> StepResult:
        outputs = model(batch)
        logits = outputs[self.output_key]
        targets = batch[self.target_key].long()
        loss = self.loss_fn(logits, targets)
        self.metrics.update(logits.detach(), targets.detach(), n=targets.shape[0])
        return StepResult(loss=loss, outputs=outputs, targets=targets)

    def predict_records(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> list[dict[str, Any]]:
        preds = outputs[self.output_key].argmax(dim=-1).detach().cpu()
        return [{'pred': int(value)} for value in preds]
```

Create `configs/task/my_task.yaml`:

```yaml
name: my_task
output_key: logits
target_key: label
loss:
  name: cross_entropy
metrics:
  - accuracy
```

Run:

```bash
uv run python src/main.py task=my_task
```

Validate:

```bash
uv run python scripts/run_sanity.py task=my_task
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_training.py -q
```

## Tutorial 11: Add A New Loss

Use this when the task contract stays the same but the loss formula changes.

Edit `src/losses/losses.py`:

```python
import torch
import torch.nn as nn

from src.utils.config import cfg_get
from src.utils.registry import register_loss


@register_loss('weighted_mse')
class WeightedMSELoss(nn.Module):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.weight = float(cfg_get(cfg, 'weight', 1.0))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.nn.functional.mse_loss(preds.float(), targets.float())
```

Use it in a task config:

```yaml
loss:
  name: weighted_mse
  weight: 2.0
```

Run:

```bash
uv run python src/main.py +experiment=regression task.loss.name=weighted_mse task.loss.weight=2.0
```

Validate with a focused unit test for shape, dtype, and expected numeric behavior.

## Tutorial 12: Add A New Metric

Metrics are functions that receive predictions and targets and return a float.

Edit `src/metrics/metrics.py`:

```python
import torch
from torch import Tensor

from src.utils.registry import register_metric


@register_metric('rmse')
def rmse(preds: Tensor, targets: Tensor) -> float:
    return torch.sqrt(torch.nn.functional.mse_loss(preds.float(), targets.float())).item()
```

Add it to `configs/task/regression.yaml` or override it:

```yaml
metrics:
  - mse
  - mae
  - rmse
```

Run:

```bash
uv run python src/main.py +experiment=regression task.metrics='[mse,mae,rmse]' checkpoint.monitor=val/rmse
```

If you monitor it for checkpoints, ensure `checkpoint.mode` is correct:

```bash
uv run python src/main.py +experiment=regression checkpoint.monitor=val/rmse checkpoint.mode=min
```

## Tutorial 13: Add A New Optimizer

Use this when changing optimizer algorithm, not just optimizer hyperparameters.

Edit `src/optim/optimizers.py`:

```python
import torch
from torch.optim import Optimizer

from src.utils.config import cfg_get
from src.utils.registry import register_optimizer


@register_optimizer('rmsprop')
def build_rmsprop(params, cfg) -> Optimizer:
    return torch.optim.RMSprop(
        params,
        lr=float(cfg_get(cfg, 'lr', 1e-3)),
        momentum=float(cfg_get(cfg, 'momentum', 0.0)),
        weight_decay=float(cfg_get(cfg, 'weight_decay', 0.0)),
    )
```

Create `configs/optimizer/rmsprop.yaml`:

```yaml
name: rmsprop
lr: 1.0e-3
momentum: 0.0
weight_decay: 0.0
no_decay_norm_bias: true
```

Run:

```bash
uv run python src/main.py optimizer=rmsprop
```

The existing optimizer factory already handles optional no-decay parameter groups for biases and normalization parameters.

## Tutorial 14: Choose Or Add A Scheduler

Existing scheduler behavior:

| Scheduler | Behavior |
| --- | --- |
| `none` | Keeps LR unchanged; best for debugging and tiny sanity runs. |
| `constant` | Holds LR at a fixed factor for `total_iters`. |
| `linear` | Moves LR from `start_factor` to `end_factor`. |
| `step` | Drops LR by `gamma` every `step_size`. |
| `multistep` | Drops LR at configured milestones. |
| `exponential` | Multiplies LR by `gamma` each scheduler step. |
| `cosine` | Smoothly anneals LR toward `eta_min`. |
| `cosine_restart` | Cosine schedule with warm restarts. |
| `plateau` | Reduces LR when a monitored metric stops improving. |
| `polynomial` | Polynomial decay over the configured horizon. |
| `onecycle` | LR rises then falls during one run; usually batch-step based. |

For horizon-based cosine and polynomial schedules, warmup steps are subtracted from the main schedule horizon. Schedulers with explicit durations keep their configured parameters. Warmup is not supported with `plateau`; scheduler construction rejects that combination. The explicit sanity command also reports it.

Switch scheduler:

```bash
uv run python src/main.py scheduler=none
uv run python src/main.py scheduler=plateau scheduler.monitor=val/loss
uv run python src/main.py scheduler=onecycle scheduler.interval=step scheduler.max_lr=1e-3
```

Add a scheduler in `src/optim/schedulers.py`:

```python
from torch.optim.lr_scheduler import LambdaLR
from src.utils.registry import register_scheduler


@register_scheduler('inverse_sqrt')
def build_inverse_sqrt(optimizer, cfg, total_steps: int, total_epochs: int):
    warmup_steps = int(cfg_get(cfg, 'warmup_steps', 1000))

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        return min(step**-0.5, step * warmup_steps**-1.5)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
```

Also add a description to `SCHEDULER_DESCRIPTIONS` and create `configs/scheduler/inverse_sqrt.yaml`.

## Tutorial 15: Configure Training Behavior

Edit `configs/trainer/default.yaml` for global trainer defaults or override per experiment.

Common controls:

```yaml
trainer:
  max_epochs: 20
  accumulate_grad_batches: 4
  grad_clip: 1.0
  log_every_n_steps: 10
  val_every_n_epochs: 1
  check_finite_loss: true
  detect_anomaly: false
  early_stopping:
    patience: 5
```

Runtime precision/device controls live under `run`:

```yaml
run:
  device: cuda
  precision: amp
  deterministic: false
```

CLI examples:

```bash
uv run python src/main.py run.device=cuda run.precision=amp
uv run python src/main.py trainer.accumulate_grad_batches=8 trainer.grad_clip=0.5
uv run python src/main.py trainer.detect_anomaly=true trainer.check_finite_loss=true
```

Notes:

- `run.precision=fp32` uses the normal non-autocast path for train, validation, test, and prediction. `fp16`, `bf16`, and `amp` use CUDA autocast across both trainer and evaluator paths.
- `detect_anomaly=true` is useful for debugging but slows training.
- A final accumulation window with fewer batches is rescaled by its actual size before clipping and stepping.
- Step-based validation restores model training mode and the in-progress training metric state before the next batch; logger steps remain monotonic because epoch summaries also use `global_step`.
- `check_finite_loss=true` fails fast on NaN or Inf losses.

## Tutorial 16: Resume And Use Fault-Tolerant Checkpoints

Checkpoint files live under one code-managed training trial:

```text
outputs/runs/<run.id>/trial_<n>/checkpoints/epoch_0001.pt
outputs/runs/<run.id>/trial_<n>/checkpoints/last.pt
outputs/runs/<run.id>/trial_<n>/checkpoints/best.pt
outputs/runs/<run.id>/trial_<n>/checkpoints/manifest.json
```

Resume by saved run id or exact model path:

```bash
uv run python src/main.py --resume-run <run.id>
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run.id>/trial_<n>/checkpoints/last.pt
```

The explicit path supplies `run.id` and `trial_id`; the config hash does not choose resume identity. `--resume-run` resolves the latest training registry record to an explicit checkpoint path. It cannot be combined with `--run-id`.

Selectors remain useful when the matching config identifies the desired run:

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=latest
uv run python src/main.py +experiment=baseline checkpoint.resume=best
uv run python src/main.py +experiment=baseline checkpoint.resume=epoch_0005
```

Checkpoint policy:

```yaml
checkpoint:
  save_every: 1
  keep_last_k: 5
  save_last: true
  save_top_k: 1
  save_on_exception: true
  monitor: val/loss
  mode: min
```

`keep_last_k: 0` keeps all epoch files. `save_top_k: 0` disables `best.pt`. `save_last: false` disables `last.pt`. Training checkpoints use atomic writes, checksums, retention, exception saves, and full optimizer/scheduler/scaler/RNG/epoch/step state.

## Tutorial 17: Enable Logging, TensorBoard, And W&B

Local logging for a training trial:

```text
outputs/runs/<run.id>/trial_<n>/logs/train.log
outputs/runs/<run.id>/trial_<n>/logs/metrics.jsonl
```

Enable TensorBoard or W&B:

```bash
uv sync --extra tracking
uv run python src/main.py logging.tensorboard.enabled=true
uv run tensorboard --logdir outputs/runs

uv run python src/main.py logging.wandb.enabled=true logging.wandb.project=my-project
```

W&B project/entity/tags/notes/mode remain configurable. W&B run ids and names do not: the framework generates `<run.id>-trial-<n>` for train/profile and `<run.id>-trial-<source_trial>-<mode>-<checkpoint>` for eval/test/predict. W&B receives the resolved config and config artifact; all logging is rank-zero only.

## Tutorial 18: Run Evaluation And Export Predictions

Use the saved training config and a checkpoint selector:

```bash
uv run ml-evaluate-run <run.id> --checkpoint best
uv run ml-evaluate-run <run.id> --checkpoint last --mode test
uv run ml-evaluate-run <run.id> --checkpoint epoch_0005 --mode predict
```

The helper resolves the selected checkpoint to an explicit path. A direct equivalent is:

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run.id>/trial_<n>/checkpoints/best.pt
```

For source trial 3 and `best.pt`, prediction output is:

```text
outputs/evaluations/<run.id>/trial_3/eval_best/predictions/test_predictions.json
```

`last.pt` and `epoch_0005.pt` use `eval_last/` and `eval_epoch_0005/`. `test` and `predict` use corresponding `test_*` and `predict_*` folders. Repeating the same source-trial/mode/checkpoint evaluation removes and recreates that exact folder and logs a bold red warning; other evaluated checkpoints remain untouched.

Limit exported records:

```bash
uv run ml-evaluate-run <run.id> --checkpoint best run.prediction_limit=20
```

`run.mode=test` logs held-out metrics, `predict` exports predictions, and `eval` does both.

## Tutorial 19: Run Sanity Checks On A New Machine

Canonical command:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu
```

CUDA compatibility checks:

```bash
# Strict GPU check: fails if PyTorch CUDA cannot use the installed NVIDIA driver.
uv run python scripts/run_sanity.py run.device=cuda sanity.strict=true

# Driver compatibility audit even while staying in CPU mode. Mismatches are warnings by default.
uv run python scripts/run_sanity.py sanity.cuda.check_driver=true

# Print recommended PyTorch UV and pip install commands based on pyproject Python, GPU, and driver.
# This works before torch is installed because it reads GPU/driver information from nvidia-smi.
uv run python scripts/run_sanity.py sanity.torch_install.recommend=true
```

Compose all experiments:

```bash
uv run python scripts/run_sanity.py sanity.check_all_experiments=true
```

What sanity checks:

- Python version.
- Required package versions from `pyproject.toml` when present.
- Optional packages from `[project.optional-dependencies]` as warnings; in this repository that is the `tracking` extra.
- Required config keys from `configs/sanity/default.yaml`.
- Runtime CPU/CUDA/DDP visibility, including PyTorch CUDA build and NVIDIA driver compatibility.
- Output/checkpoint/log/prediction directory writability.
- Free disk space.
- Registry entries for configured components.
- Tensor-file paths when `data=tensor_file`.
- Tiny data/model/loss/backward/optimizer/scheduler smoke pass.

CUDA compatibility report fields include `torch.__version__`, `torch.version.cuda`, `nvidia-smi` driver details when available, the known minimum NVIDIA driver for the PyTorch CUDA build, `torch.cuda.is_available()`, and any captured PyTorch CUDA initialization warning. If you enable `sanity.torch_install.recommend=true`, sanity also prints recommended UV and pip commands using the Python requirement in `pyproject.toml`, the detected GPU/driver from `nvidia-smi`, and the known official PyTorch wheel indexes. This mode is intended to work before torch is installed. If you see a warning like `driver on your system is too old`, either update the NVIDIA driver or install a PyTorch build compiled for an older CUDA version supported by your driver.

Customize `configs/sanity/default.yaml` when your project needs additional required config keys or stricter machine checks. Set `sanity.cuda.check_driver=false` to skip CUDA driver checks, or `sanity.cuda.fail_on_cpu_mismatch=true` when you want CUDA/PyTorch mismatches to fail even in CPU-mode sanity.
Strict mode is enabled by default for the explicit sanity command. Normal fresh training, resume, profile, eval, test, and predict workflows do not run this suite. The sanity smoke test uses `run.precision` and performs one optimizer and scheduler step; it reports invalid precision names, invalid plateau scheduler intervals, and plateau-plus-warmup configurations.

## Tutorial 20: Profile A Tiny Workload

Run CPU profiler with the default profiler config:

```bash
bash scripts/profile.sh
```

The workload is defined in `configs/profiler.yaml`. Edit that file to change model, data, task, precision, warmup steps, recorded steps, or trace options.

Run CUDA profiler:

```bash
PROFILE_CUDA=1 bash scripts/profile.sh
```

Run an alternate profiler config:

```bash
PROFILE_CONFIG=configs/profiler.yaml bash scripts/profile.sh
```

Outputs are written under `profiler.trace_dir`, which defaults to:

```text
outputs/profiles/
```

Open with TensorBoard:

```bash
uv run tensorboard --logdir outputs/profiles
```

Use this when a model or dataloader change becomes slow and you need operator-level traces. To profile the composed trainer stack instead of the standalone wrapper, run `uv run python src/main.py run.mode=profile profiler.active_steps=3`. See `profiler_tutorial.md` for how to interpret the table, inspect TensorBoard traces, and decide what to optimize.

## Tutorial 21: Run A W&B Sweep

Install tracking extras:

```bash
uv sync --extra tracking
```

Edit `configs/sweep.yaml` to define search space and metric.

Launch:

```bash
bash scripts/sweep.sh
```

Current sweep explores:

- `optimizer.lr`
- `model.dropout`
- `model.hidden_dim`
- `data.batch_size`

Make sure the swept metric exists in logs. For example, if the sweep metric is `val/loss`, the task and trainer must produce `val/loss`.

## Tutorial 22: Add A Callback

Use callbacks when you need lifecycle behavior without editing the trainer: extra logging, EMA, visualization, custom artifacts, profiler ranges, or diagnostics.

Create a callback class. See `callback_tutorial.md` for the complete hook reference, wiring options, and additional examples:

```python
from src.callbacks import Callback
from src.utils.config import cfg_get
from src.utils.registry import register_callback


@register_callback('project_lr_logger')
class ProjectLearningRateLogger(Callback):
    def __init__(self, cfg) -> None:
        self.every_n_steps = max(1, int(cfg_get(cfg, 'every_n_steps', 1)))

    def on_batch_end(self, trainer, batch_idx: int, metrics: dict[str, float]) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        learning_rate = float(trainer.optimizer.param_groups[0]['lr'])
        trainer.loggers.log_metrics({'train/lr': learning_rate}, step=trainer.global_step)
```

`src/main.py` already builds callbacks from the top-level `callbacks:` list through `src.callbacks.build_callbacks(cfg)`. For normal experiments, register the callback and add it to config:

```yaml
callbacks:
  - name: project_lr_logger
    every_n_steps: 10
```

Use a registry key that is not already registered by `src/callbacks/common.py`.

Minimal extension path:

1. Add the callback class under `src/callbacks/` and decorate it with `@register_callback`.
2. Import the module from `src/callbacks/__init__.py` so registration happens.
3. Add an entry under `callbacks:` in `configs/config.yaml`, an experiment YAML, or the CLI.
4. Add a focused test that verifies the hook runs.

## Tutorial 23: Add A Custom Logger Backend

Use this when JSONL, TensorBoard, and W&B are not enough.

Implement the `LoggerBackend` protocol in `src/utils/logger.py`:

```python
class MyLogger:
    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg_get(cfg, 'logging.my_logger.enabled', False))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            ...

    def log_artifact(self, path, name: str, artifact_type: str = 'artifact', metadata=None) -> None:
        if self.enabled:
            ...

    def finish(self) -> None: ...
```

Add it in `build_loggers(cfg)` when enabled. Then add config under `configs/logging/default.yaml`:

```yaml
my_logger:
  enabled: false
```

Validate with a tiny run and a focused logger test.

## Tutorial 24: Add A New Shell Entrypoint

Use this when you want a repeatable command for a workflow.

Create `scripts/my_workflow.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
uv run python src/main.py +experiment=baseline "$@"
```

Make executable:

```bash
chmod +x scripts/my_workflow.sh
```

Run:

```bash
bash scripts/my_workflow.sh trainer.max_epochs=3
```

Optionally add a Makefile target:

```make
my-workflow: bash scripts/my_workflow.sh $(ARGS)
```

## Tutorial 25: Add A Console Script

If you keep `pyproject.toml`, add a script entrypoint. Hydra-decorated script entrypoints should use the existing wrappers in `src/cli.py`:

```toml
[project.scripts]
ml-train = "src.cli:train"
ml-sanity = "src.cli:sanity"
```

Ordinary utility modules with a plain `main()` function can point directly to that function, for example `ml-run-registry = "scripts.run_registry:main"`.

Then sync:

```bash
uv sync
uv run ml-sanity +experiment=sanity_cpu
```

If you remove `pyproject.toml` from your final template copy, use `uv run python ...` and shell scripts instead.

## Tutorial 26: Work With DDP And Distributed Runtime

Current distributed support includes:

- `setup_from_env` in `src/runtime/distributed.py`.
- Rank helpers such as `is_rank0`, `rank`, and `world_size`.
- Rank-zero-only checkpointing and logging.
- Distributed dataloader samplers when a process group is initialized.
- Weighted distributed metric reduction across trainer/evaluator task metrics.
- Broadcast of the rank-zero selected run id.

The standard entrypoint wraps the model with `torch.nn.parallel.DistributedDataParallel` when launched with `torchrun`, and checkpoints unwrap the model state before saving/loading.

CPU smoke launch with `gloo`:

```bash
uv run torchrun --nproc_per_node=2 src/main.py +experiment=sanity_cpu run.distributed_backend=gloo run.device=cpu
```

CUDA launch with `nccl`:

```bash
uv run torchrun --nproc_per_node=2 src/main.py +experiment=baseline run.device=cuda data.batch_size=64
```

Validation and test ranks process disjoint, non-padded dataset shards. Scalar metrics use weighted global reductions, prediction records are gathered, and only rank zero writes shared logs and prediction JSON.

Keep custom callbacks rank-aware if they write files or call external services.

## Tutorial 27: Adapt The Template To A Real Project

Use this checklist when copying the repo for a new research project.

1. Rename project metadata in `pyproject.toml` if you keep it.
2. Update package install docs in `README.md` if your project uses different dependencies.
3. Update W&B project/tags in `configs/logging/default.yaml`.
4. Replace toy configs with project-specific defaults only after sanity passes.
5. Add real datasets under `src/data/` and `configs/data/`.
6. Add real models under `src/models/` and `configs/model/`.
7. Add or extend tasks under `src/tasks/` for your problem semantics.
8. Add losses and metrics under `src/losses/` and `src/metrics/`.
9. Add experiment presets under `configs/experiment/`.
10. Keep `scripts/run_sanity.py` working on CPU.
11. Keep `+experiment=sanity_cpu` small and fast.
12. Add tests before large refactors.

Do not commit generated paths:

```text
outputs/
data/processed/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
wandb/
```

## Tutorial 28: Add Tests For A Custom Component

Use focused tests to keep the framework reliable.

Dataset test idea:

```python
def test_my_dataset_sample_shape(tmp_path):
    cfg = {'name': 'my_dataset', 'root': str(tmp_path), 'batch_size': 2}
    dataset = MyDataset(cfg, split='train')
    sample = dataset[0]
    assert 'input' in sample
    assert 'label' in sample
```

Model test idea:

```python
def test_my_model_forward(tiny_batch):
    model = MODEL_REGISTRY.build('my_mlp', {'input_dim': 16, 'hidden_dim': 8, 'num_classes': 2})
    outputs = model(tiny_batch)
    assert outputs['logits'].shape == (4, 2)
```

Task/training smoke:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_training.py -q
```

Full validation:

```bash
python3 -m compileall src scripts tests
uv run ruff check src tests scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run python scripts/run_sanity.py +experiment=sanity_cpu
```

## Tutorial 29: Debug Common Failures

Registry error:

```text
'my_name' not in model registry. Available: [...]
```

Fix:

- Confirm the component uses `@register_model('my_name')` or the right `@register_*` decorator.
- Confirm its module is imported during bootstrap. For a new file, import it in the package `__init__.py`.
- Confirm config `name` matches the registry key.

Missing config error:

- Confirm the YAML file is under the correct config group.
- Use `model=my_model`, not `+model=my_model`, when selecting an existing group option.
- Use `+experiment=my_experiment` for experiment presets.

Shape error:

- Check dataset batch keys and tensor shapes.
- Check model `forward(batch)` output keys.
- Check `task.output_key` and `task.target_key`.
- Run `uv run python scripts/run_sanity.py sanity.run_model_smoke=true`.

Resume does not find checkpoint:

- Use the same config base id when running `checkpoint.resume=latest`.
- Check `outputs/run_registry.jsonl` for the run id, run directory, repeat command, and command working directory.
- Use an explicit checkpoint path if the config changed.

W&B not logging:

- Install tracking extras.
- Set `logging.wandb.enabled=true`.
- Check W&B login and `logging.wandb.mode`.
- Look for a warning in `outputs/runs/<run.id>/trial_<n>/logs/train.log`.

## Tutorial 30: Decide Where A Change Belongs

Use this table before editing.

| Goal | Primary files |
| --- | --- |
| Change default run/device/output behavior | `configs/config.yaml` |
| Add a model architecture | `src/models/`, `configs/model/` |
| Add a dataset | `src/data/`, `configs/data/` |
| Change loss/metrics/prediction semantics | `src/tasks/`, `configs/task/` |
| Add a loss function | `src/losses/losses.py`, `configs/task/*.yaml` |
| Add a metric | `src/metrics/metrics.py`, `configs/task/*.yaml` |
| Add optimizer algorithm | `src/optim/optimizers.py`, `configs/optimizer/` |
| Add LR scheduler | `src/optim/schedulers.py`, `configs/scheduler/` |
| Change train loop behavior | `src/engine/trainer.py`, `configs/trainer/` |
| Change eval/prediction loop behavior | `src/engine/evaluator.py`, task `predict_records` |
| Change checkpoint policy | `configs/checkpoint/default.yaml`, `src/utils/checkpoint.py` |
| Change run-id/artifact layout | `src/utils/run.py`, `configs/config.yaml` |
| Change logging backends | `src/utils/logger.py`, `configs/logging/default.yaml` |
| Change sanity checks | `src/utils/sanity/`, `configs/sanity/default.yaml` |
| Add repeatable run recipe | `configs/experiment/*.yaml` |
| Add shell workflow | `scripts/*.sh`, `Makefile` |
| Add docs | `README.md`, `Description.md`, `Flowchart.md`, `tutorial.md` |

## Final Validation Checklist

Before trusting a customized repo, run:

```bash
python3 -m compileall src scripts tests
uv run ruff check src tests scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run python scripts/run_sanity.py +experiment=sanity_cpu
uv run python src/main.py +experiment=sanity_cpu
```

For a new dataset/model/task, also run one tiny end-to-end experiment with that exact stack:

```bash
uv run python src/main.py +experiment=my_experiment trainer.max_epochs=1
```

Then inspect:

```text
outputs/run_configs/<run.id>/trial_<n>.yaml
outputs/run_registry.jsonl
outputs/runs/<run.id>/trial_<n>/logs/train.log
outputs/runs/<run.id>/trial_<n>/logs/metrics.jsonl
outputs/runs/<run.id>/trial_<n>/checkpoints/manifest.json
outputs/runs/<run.id>/trial_<n>/predictions/test_predictions.json
```
