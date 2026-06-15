# Repository Description

This document is the detailed reference for the cleaned `ml-template` repository. It explains what each folder and file is for, when to use it, and how it fits into the framework.

For command-oriented usage, start with `README.md`.

## Root Files

- `README.md`: Day-to-day usage guide. Use it when you want to install, sanity-check, train, evaluate, resume, extend, or validate the template.
- `Run_commands.md`: Command cookbook for training, evaluation, resume, config overrides, shell wrappers, Makefile targets, profiling, preprocessing, and W&B sweeps.
- `sanity_check_commands.md`: Command cookbook for machine/environment sanity checks, CUDA diagnostics, PyTorch install recommendations, and smoke-test controls.
- `callback_tutorial.md`: Callback lifecycle reference with hook timing, direct and registry-based wiring, examples, distributed-training caveats, and testing guidance.
- `profiler_tutorial.md`: Profiler guide covering when to profile, CPU/CUDA usage, TensorBoard traces, result interpretation, and optimization workflow.
- `Description.md`: This file. Use it when you need to understand the repository layout or decide where a new feature belongs.
- `tutorial.md`: Hands-on cookbook for learning the framework and customizing datasets, models, tasks, losses, metrics, optimizers, schedulers, logging, checkpoints, sanity checks, and scripts.
- `Flowchart.md`: Mermaid diagrams covering every entrypoint, train/eval/sanity flow, artifact layout, registries, and customization edit points.
- `pyproject.toml`: Python project metadata, dependency/version requirements used by sanity checks, optional extras, console scripts, and pytest settings. Edit this when renaming the project, adding dependencies, or changing package entrypoints.
- `uv.lock`: Locked dependency graph for reproducible installs with `uv`. Update it after dependency changes if you keep the `uv` workflow. Don't change manually.
- `ruff.toml`: Dedicated Ruff lint/format configuration. Edit this for line length, target Python version, selected rule families, and ignored rules.
- `Makefile`: Shortcuts for common local commands: `install`, `sanity`, `train`, `eval`, `test`, `test-cov`, `lint`, `fmt`, and `clean`.
- `.gitignore`: Ignore rules for generated outputs, checkpoints, processed data, virtual environments, caches, local editor settings, and W&B metadata.

## Configs

`configs/` contains Hydra configuration. The framework uses grouped configs so each parameter family has one owner. Prefer adding a new file under the right group instead of adding unrelated keys to existing files.

### `configs/config.yaml`

Root Hydra config. It selects default groups for model, data, task, optimizer, scheduler, trainer, logging, checkpoint, and sanity settings. It also owns `run.*` settings such as mode, seed, device, precision, output directories, and prediction limits.

Use it when you need to change global defaults for the whole template. For project-specific or experiment-specific changes, prefer `configs/experiment/*.yaml` or CLI overrides.

### `configs/model/`

Model architecture configs. These should contain architecture-only parameters, not optimizer or trainer parameters.

- `configs/model/mlp.yaml`: Default MLP for toy classification. Use for CPU smoke tests and simple tabular/vector classification examples.
- `configs/model/regression_mlp.yaml`: MLP variant for regression. Its input and output dimensions resolve from the selected dataset. Use with `task=regression` and `data=toy_regression`.
- `configs/model/cnn.yaml`: Small image CNN config. Class count resolves from the selected dataset; use it as a reference when adding image workloads.
- `configs/model/small_transformer.yaml`: Small sequence classifier config. Vocab size, sequence length, and class count resolve from the selected dataset; hidden size, heads, and layers stay model-owned.

### `configs/data/`

Dataset and dataloader configs. These select a registered dataset and define data-owned shape/label properties, split sizes or split paths, and dataloader options. Models resolve input/output dimensions from these dataset fields when appropriate.

File locations are dataset-owned configuration. `TensorFileDataset` reads `data.splits.<split>.path` from `configs/data/tensor_file.yaml`; relative paths are resolved from the stable repository working directory because Hydra does not change directories. Custom datasets may define a key such as `data.root`, but the key is effective only if that dataset class reads it.

- `configs/data/toy_classification.yaml`: Synthetic vector classification splits. Use for fast trainer, metric, checkpoint, and sanity checks.
- `configs/data/toy_regression.yaml`: Synthetic vector regression splits. Use with `model=regression_mlp` and `task=regression`.
- `configs/data/toy_sequence.yaml`: Synthetic token sequence classification splits. Use with `model=small_transformer`.
- `configs/data/tensor_file.yaml`: File-backed tensor dataset using `data/processed/train.pt`, `val.pt`, and `test.pt`. Generate those files with `bash scripts/preprocess.sh --force` before using it.

### `configs/task/`

Task behavior configs. Tasks own the loss, metric list, model output key, and target key.

- `configs/task/classification.yaml`: Single-label classification with `cross_entropy` and `accuracy`.
- `configs/task/detection.yaml`: Object detection with model-provided loss aggregation, score filtering, NMS, and prediction export.
- `configs/task/language_modeling.yaml`: Autoregressive token prediction with cross-entropy, token accuracy, and perplexity.
- `configs/task/ranking.yaml`: Point-wise learning-to-rank with MSE, MAE, and ranked prediction export.
- `configs/task/regression.yaml`: Regression with `mse` loss plus `mse` and `mae` metrics.
- `configs/task/segmentation.yaml`: Semantic segmentation with per-pixel cross-entropy and pixel accuracy.

Use this group when your workload changes prediction semantics. Classification, regression, segmentation, detection, ranking, and language modeling have registered task implementations. Detection mAP, segmentation IoU/Dice, and ranking NDCG/MRR remain optional metric extensions.

### `configs/optimizer/`

Optimizer configs.

- `configs/optimizer/adamw.yaml`: AdamW with weight decay and optional no-decay grouping for norm and bias parameters. Good default for most research experiments.
- `configs/optimizer/sgd.yaml`: SGD with momentum and optional Nesterov. Useful for classic vision baselines and controlled comparisons.

### `configs/scheduler/`

Learning-rate scheduler configs. `interval` controls whether the scheduler steps by epoch or batch, and `monitor` is used by metric-driven schedulers such as plateau.

- `configs/scheduler/none.yaml`: Leaves LR unchanged; use for debugging and tiny sanity runs.
- `configs/scheduler/constant.yaml`: Keeps LR at a fixed factor for `total_iters`.
- `configs/scheduler/linear.yaml`: Linearly moves LR from `start_factor` to `end_factor`.
- `configs/scheduler/step.yaml`: Drops LR by `gamma` every `step_size` epochs or steps.
- `configs/scheduler/multistep.yaml`: Drops LR at explicit milestones.
- `configs/scheduler/exponential.yaml`: Multiplies LR by `gamma` each step for monotonic decay.
- `configs/scheduler/cosine.yaml`: Cosine anneals LR toward `eta_min` over the training horizon.
- `configs/scheduler/cosine_restart.yaml`: Cosine schedule with warm restarts using `T_0` and `T_mult`.
- `configs/scheduler/plateau.yaml`: Reduces LR when the monitored metric stops improving.
- `configs/scheduler/polynomial.yaml`: Polynomial LR decay over the configured horizon.
- `configs/scheduler/onecycle.yaml`: LR ramps up and down in one run; normally batch-step based.

### `configs/trainer/default.yaml`

Trainer control settings: max epochs, gradient accumulation, gradient clipping, logging cadence, validation cadence, finite-loss checks, anomaly detection, and early-stopping patience.

Use it for defaults that should apply to many runs. Override values in experiment YAML or at the CLI for a specific run.

### `configs/logging/default.yaml`

Logging settings for JSONL metrics, TensorBoard, and W&B. JSONL logging is enabled by default; TensorBoard and W&B are disabled by default.

Enable tracking only when needed:

```bash
uv sync --extra tracking
uv run python src/main.py logging.wandb.enabled=true logging.wandb.project=my-project
```

### `configs/checkpoint/default.yaml`

Checkpoint policy. Controls checkpoint directory, save frequency, retention, `last.pt`, `best.pt`, monitored metric, metric direction, resume source, and exception checkpointing.

Use `checkpoint.resume=latest`, `last`, `best`, an epoch selector such as `5` or `epoch_0005`, or a specific `.pt` path to choose the checkpoint.

### `configs/sanity/default.yaml`

Sanity-check policy. Controls strictness, whether to run model smoke tests, whether to compose all experiments, minimum disk space, W&B readiness checks, and required config keys. Python and package version requirements come from `pyproject.toml`, not this YAML file.

Use this when adapting the template to a new project so sanity checks validate the config keys and smoke-test behavior your project truly depends on.

### `configs/profiler.yaml`

Profiler wrapper config loaded by `scripts/profile.sh`. It owns the profiler smoke workload model, data, task, device, precision, warmup steps, recorded steps, trace directory, and `torch.profiler` options. Edit this file for repeatable profiler customization, or run `PROFILE_CONFIG=<path> bash scripts/profile.sh` for an alternate profiler config.

### `configs/experiment/`

Experiment presets. These are Hydra global overrides used for repeatable research runs. They are not included in the default config composition. An experiment is applied only when explicitly selected with `+experiment=<name>`, at which point it overrides the existing root settings and selected config-group defaults.

- `configs/experiment/baseline.yaml`: Default MLP classification baseline with explicit trainer, optimizer, and model overrides.
- `configs/experiment/sanity_cpu.yaml`: Tiny CPU run for fast sanity checks and validation.
- `configs/experiment/sanity_gpu.yaml`: Tiny CUDA run for GPU setup validation and full trainer smoke testing on CUDA machines.
- `configs/experiment/ablation_heads.yaml`: Small transformer on toy sequence data; useful as an ablation example.
- `configs/experiment/regression.yaml`: Regression workload selecting regression model, data, task, and checkpoint monitor.

Run an experiment with:

```bash
uv run python src/main.py +experiment=baseline
```

### `configs/sweep.yaml`

W&B sweep definition. It currently sweeps learning rate, MLP dropout, hidden dimension, and batch size against `val/loss`. Use with `bash scripts/sweep.sh` after installing tracking extras.

## Scripts

`scripts/` contains command-line helpers. They set `PYTHONPATH` to the repository root and call the canonical Python entrypoints.

- `scripts/run_sanity.py`: Canonical machine validation script. Use after cloning, moving machines, changing dependencies, or before expensive training.
- `scripts/train.sh`: Wrapper around `uv run python src/main.py`. Pass Hydra overrides after the script name.
- `scripts/eval.sh`: Wrapper for checkpoint evaluation. First argument is the checkpoint path; extra arguments are Hydra overrides.
- `scripts/preprocess.sh`: Generates toy tensor-file splits by calling `src/data/preprocess.py`.
- `scripts/profile.sh`: Loads `configs/profiler.yaml`, runs the configured profiler workload, and writes traces to `profiler.trace_dir` (`outputs/profiles/` by default). Set `PROFILE_CUDA=1` or `profiler.cuda: true` to include CUDA profiling. See `profiler_tutorial.md` for interpreting traces and using profiler output to optimize code.
- `scripts/sweep.sh`: Creates and runs a W&B sweep from `configs/sweep.yaml`.
- `scripts/__init__.py`: Makes `scripts.run_sanity:main` importable as the `ml-sanity` console script from `pyproject.toml`.

## Source Package

`src/` is the framework package. The trainer is intentionally generic; task classes define workload-specific loss, metric, and prediction behavior.

### `src/main.py`

Hydra entrypoint for training and evaluation. It composes config, initializes runtime, seeds reproducibility, runs sanity checks, builds data/model/task/optimizer/scheduler/loggers/checkpoints, handles resume, and dispatches train or eval mode.

Use this as the primary executable for experiments.

### `src/__init__.py`

Package marker and top-level project docstring. Use it only for package metadata or narrow public exports.

### `src/models/`

Model definitions and architecture helpers.

- `src/models/model.py`: Registered model classes: `MLP`, `SmallCNN`, and `SmallTransformer`. Models consume a batch dictionary and return an output dictionary.
- `src/models/layers.py`: Reusable neural-network layers: multi-head attention, feed-forward block, and positional encoding.
- `src/models/__init__.py`: Public model exports.

Add new architectures here and register them with `@register_model('name')`.

### `src/data/`

Datasets, dataloaders, preprocessing, and transforms.

- `src/data/dataset.py`: Registered datasets: toy classification, toy regression, toy sequence, and tensor-file dataset.
- `src/data/dataloader.py`: Builds train, val, and test dataloaders from the dataset registry. Handles distributed samplers when DDP is active and uses a dataset-provided `collate_fn` for variable-size batches such as detection targets.
- `src/data/preprocess.py`: Generates file-backed toy tensor splits for `data=tensor_file`.
- `src/data/transforms.py`: Placeholder transform helpers for train and validation transforms.
- `src/data/__init__.py`: Public data exports.

Add real datasets here and keep schema validation close to dataset code.

### `src/tasks/`

Task contracts and task implementations.

- `src/tasks/task.py`: `BaseTask`, `ClassificationTask`, `RegressionTask`, `StepResult`, and `build_task`. Tasks own model-output interpretation, loss calls, metric updates, and prediction record formatting.
- `src/tasks/segmentation.py`: Per-pixel cross-entropy, ignore-index handling, flattened pixel metrics, and mask prediction export.
- `src/tasks/detection.py`: Model-provided detection loss aggregation, nested target support, score filtering, NMS, and JSON-safe detection export.
- `src/tasks/ranking.py`: Point-wise ranking loss and score metrics plus descending ranked-item export.
- `src/tasks/language_modeling.py`: Flattened next-token loss, ignore-index handling, token metrics, perplexity, and token prediction export.
- `src/tasks/__init__.py`: Public task exports.

Add a task when the trainer should stay generic but the workload semantics change.

### `src/losses/`

Loss functions.

- `src/losses/losses.py`: Registered losses: cross entropy, MSE, BCE-with-logits, focal loss, and label smoothing.
- `src/losses/__init__.py`: Public loss exports.

Add losses with `@register_loss('name')` and reference them from `configs/task/*.yaml`.

### `src/metrics/`

Metrics and metric accumulation.

- `src/metrics/metrics.py`: Registered metrics: accuracy, MSE, MAE, plus `MetricAccumulator`, `MetricCollection`, and `compute_all_metrics`.
- `src/metrics/__init__.py`: Public metric exports.

Add metrics with `@register_metric('name')` and include them in a task config.

### `src/optim/`

Optimizer and scheduler factories.

- `src/optim/optimizers.py`: Parameter grouping plus registered AdamW, Adam, and SGD builders.
- `src/optim/schedulers.py`: Registered schedulers, `SchedulerBundle`, warmup wrapping, scheduler stepping, and one-line scheduler behavior descriptions.
- `src/optim/__init__.py`: Public optimizer and scheduler exports.

Use this package for training optimization logic. Do not keep optimizer code inside model classes.

### `src/engine/`

Training and evaluation orchestration.

- `src/engine/trainer.py`: Generic trainer with train/validation/test loops, AMP support, accumulation, clipping, finite-loss checks, callbacks, checkpoint save/resume, early stopping, and exception checkpointing.
- `src/engine/evaluator.py`: Evaluation and prediction helper that recursively moves nested tensor batches to the target device, supports tasks that return no evaluation loss, and returns metrics or prediction records.
- `src/engine/__init__.py`: Public engine exports.

Modify this package only for framework-level behavior. For workload-specific behavior, prefer a task, callback, metric, or loss.

### `src/callbacks/`

Trainer extension hooks.

- `src/callbacks/base.py`: `Callback` hook interface and `CallbackList` dispatcher.
- `src/callbacks/__init__.py`: Public callback exports.

Use callbacks for optional side effects around training events without changing the core trainer. See `callback_tutorial.md` for hook timing and examples. Callbacks currently work when passed directly to `Trainer`; `src/main.py` does not yet build them from Hydra config.

### `src/runtime/`

Distributed runtime helpers.

- `src/runtime/distributed.py`: DDP environment detection, setup, cleanup, rank helpers, barriers, small-object broadcast, and distributed metric averaging.
- `src/runtime/__init__.py`: Public runtime exports.

Use this package for process/rank concerns instead of scattering `torch.distributed` checks through training code.

### `src/utils/`

Shared framework utilities.

- `src/utils/config.py`: Safe config access, config existence checks, conversion to dictionaries, loading, and merging.
- `src/utils/registry.py`: Generic registry implementation and decorators for models, datasets, losses, metrics, optimizers, schedulers, tasks, callbacks, and loggers.
- `src/utils/seed.py`: Reproducibility helpers for Python, NumPy, PyTorch, CUDA, workers, deterministic mode, and rank-aware seeds.
- `src/utils/types.py`: Shared type aliases and protocols for batches, config dictionaries, model-like objects, datasets, metrics, and schedulers.
- `src/utils/checkpoint.py`: Fault-tolerant checkpoint manager, RNG state capture/restore, atomic saves, manifest checksums, latest/best lookup, fallback loading, and legacy save/load helpers.
- `src/utils/run.py`: Config-derived run identity helper. Computes `run.id`, writes resolved config snapshots, appends repeat-command metadata to `outputs/run_registry.jsonl`, and rewrites artifact paths under `outputs/runs/<run.id>/` for training or `outputs/evaluations/<run.id>/` for evaluation.
- `src/utils/paths.py`: Output, checkpoint, log, and prediction path helpers.
- `src/utils/logger.py`: Console, JSONL, TensorBoard, and W&B logger backends plus logger collection setup.
- `src/utils/__init__.py`: Utility package marker.

### `src/utils/sanity/`

Canonical sanity-check package. Use this package when moving to a new machine or validating framework health.

- `src/utils/sanity/__init__.py`: Public sanity exports: `CheckResult`, `SanityReport`, CUDA diagnostics, `bootstrap_registries`, and `run_sanity_checks`.
- `src/utils/sanity/core.py`: Main sanity check implementation. Validates Python and package versions from `pyproject.toml`, config keys, runtime, PyTorch CUDA/NVIDIA driver compatibility, output directories, disk space, registries, tensor-file paths, data/model smoke pass, and optional experiment composition.
- `src/utils/sanity/cuda.py`: CUDA diagnostics helper. Captures PyTorch CUDA build/version, `torch.cuda.is_available()` warnings, `nvidia-smi` driver details, known minimum NVIDIA driver compatibility for CUDA builds, and UV/pip PyTorch install recommendations that can run before torch is installed.
- `src/utils/sanity/ddp.py`: Lightweight DDP helper functions for sanity-related checks.
- `src/utils/sanity/model.py`: Model smoke helper for checking backward behavior.
- `src/utils/sanity/packages.py`: Package import check helper.

The old top-level `src/utils/sanity_check.py` file was removed. New and existing code should import from `src.utils.sanity`.

## Tests

`tests/` contains CPU-friendly unit and smoke tests. Use it as the first safety net before running expensive experiments.

- `tests/conftest.py`: Shared pytest fixtures, including tiny configs.
- `tests/test_config.py`: Hydra/config composition checks, including all built-in task config options.
- `tests/test_dataset.py`: Dataset and dataloader behavior checks, including dataset-provided collation for variable-size samples.
- `tests/test_model.py`: Model forward-shape and model behavior checks.
- `tests/test_training.py`: Trainer smoke tests.
- `tests/test_checkpoint.py`: Checkpoint save/load/resume/fault-tolerance tests.
- `tests/test_run_identity.py`: Stable run-id, training/evaluation output layout, config snapshot, selector, and tracking-id checks.
- `tests/test_metrics.py`: Metric correctness tests.
- `tests/test_tasks.py`: Registration, loss, metric, prediction, detection filtering, and nested-device-transfer checks for task implementations.
- `tests/test_schedulers.py`: Scheduler factory and behavior tests.
- `tests/test_distributed.py`: Distributed runtime helper checks.
- `tests/test_sanity.py`: Sanity check smoke test.
- `tests/__init__.py`: Test package marker.

Run tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
```

## Notebooks

- `notebooks/01_eda.ipynb`: EDA scratch notebook. Use for dataset inspection and exploratory analysis.
- `notebooks/02_results.ipynb`: Results scratch notebook. Use for plots, metric summaries, and experiment analysis.

Keep notebooks lightweight. Large generated outputs should stay under ignored output or artifact folders.

## Output Organization

Run artifacts are isolated by a config-derived id instead of by timestamp. At startup, `prepare_run(cfg)` sets:

- `run.id`: readable id derived from `run.name`, model/data/task names, `run.trial`, and `run.config_id`, unless explicitly provided. Existing ids are reused instead of suffixed; fresh duplicate runs warn, while intentional training resumes do not.
- `run.config_id`: stable hash of the effective experiment config after excluding generated path fields.
- `run.run_dir`: `outputs/runs/<run.id>` for training or `outputs/evaluations/<run.id>` for evaluation.
- `checkpoint.dir`: `<run.run_dir>/checkpoints`.
- `run.log_dir`: `<run.run_dir>/logs`.
- `logging.jsonl.path`: `<run.run_dir>/logs/metrics.jsonl`.
- `run.prediction_dir`: `<run.run_dir>/predictions`.
- `run.profile_dir`: `<run.run_dir>/profiles`.
- `run.tracking_id`: `<run.id>` for training and `<run.id>_evaluation` for evaluation.

Training config snapshots are stored at `outputs/run_configs/<run.id>.yaml`. Evaluation snapshots are stored at `outputs/evaluations/<run.id>/config.yaml`, which prevents evaluation from overwriting the training config. Both modes append metadata to `outputs/run_registry.jsonl`, including `command` and `command_cwd` for repeating the invocation. Sensitive CLI values are redacted in `command`, but resolved config values are still stored for reproducibility. If a fresh invocation reuses an existing artifact directory or config snapshot, `prepare_run` keeps that id and emits a warning; intentional training resumes suppress that warning. In eval mode, checkpoint selectors such as `latest`, `best`, and `epoch_0005` load from the training checkpoint folder while all generated evaluation artifacts go to `outputs/evaluations/<run.id>/`.

## Generated And Ignored Paths

These paths are not part of the framework source and can be deleted safely:

- `outputs/`: Generated run registry, resolved config snapshots, run-scoped logs, checkpoints, profiler traces, and predictions.
- `data/processed/`: Generated tensor-file data from `scripts/preprocess.sh`.
- `data/raw/`: Optional raw data location, ignored by default.
- `.venv/`, `venv/`, `env/`: Local virtual environments.
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `__pycache__/`: Tool and Python caches.
- `.vscode/`, `.idea/`: Local editor settings.
- `wandb/`: Local W&B metadata.

## Removed Legacy Files

The cleanup removed files that duplicated canonical locations or were generated artifacts:

- Top-level compatibility configs: `configs/base.yaml`, `configs/model.yaml`, `configs/data.yaml`, `configs/scheduler.yaml`, and `configs/sanity_check.yaml`.
- Deprecated source shims: `src/utils/sanity_check.py`, `src/utils/schedulers.py`, `src/utils/sanity/register.py`, and `src/utils/sanity/registry.py`.
- Redundant uppercase lint pointer: `RUFF.toml`; active Ruff config now lives in dedicated `ruff.toml`.
- Generated artifacts and caches: `outputs/`, `data/processed/`, Python bytecode folders, pytest/Ruff/mypy caches, and stale snapshot files if present.

## Where To Put New Work

- New architecture: `src/models/` plus `configs/model/`.
- New dataset: `src/data/` plus `configs/data/`.
- New task semantics: `src/tasks/` plus `configs/task/`.
- New loss: `src/losses/` and a task config reference.
- New metric: `src/metrics/` and a task config reference.
- New optimizer or scheduler: `src/optim/` plus `configs/optimizer/` or `configs/scheduler/`.
- New training side effect: `src/callbacks/`.
- New experiment preset: `configs/experiment/`.
- New run identity behavior: `src/utils/run.py` plus `configs/config.yaml` run identity fields.
- New machine/environment validation: `src/utils/sanity/core.py` or an extra check passed to `run_sanity_checks`.
- New CLI workflow: `scripts/` plus documentation in `README.md`.
- New safety coverage: `tests/`.
