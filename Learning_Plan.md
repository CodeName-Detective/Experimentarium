# Learning Plan

This is the current learning path for understanding this repository.

Use this as a guided reading order. Do not read files alphabetically. Follow one
real command from Hydra config composition to run outputs, then learn each
extension point.

## Quick Start

Start with the smallest full run:

```bash
uv run python src/main.py +experiment=sanity_cpu trainer.max_epochs=1
```

This composes a Hydra config, prepares a run directory, builds data/model/task/optimizer/scheduler/loggers/checkpoints, trains, tests, logs metrics, and writes predictions. It does not run sanity checks.

After CUDA sanity passes, try:

```bash
uv run python src/main.py +experiment=sanity_gpu
```

Keep these references open:

- `README.md`: main usage and behavior overview.
- `Run_commands.md`: command cookbook with explanations.
- `sanity_check_commands.md`: environment and sanity-check commands.
- `Flowchart.md`: visual entrypoint-to-artifact flow.
- `Description.md`: file-by-file repository map.
- `tutorial.md`: customization examples.
- `callback_tutorial.md`: callback lifecycle and registry wiring.
- `profiler_tutorial.md`: profiler usage and interpretation.

## Phase 1: Big Picture

Read:

1. `README.md`
2. `Flowchart.md`
3. `Description.md`
4. `Run_commands.md`
5. `sanity_check_commands.md`

Learn:

- The framework is driven by Hydra configs under `configs/`.
- `src/main.py` is the main train/eval/test/predict/profile entrypoint.
- `scripts/run_sanity.py` validates the environment and config without training.
- Config groups select model, data, task, optimizer, scheduler, trainer,
  logging, checkpoint, and sanity behavior.
- Train/profile outputs live under `outputs/runs/<run.id>/trial_<n>/`; checkpoint-backed eval/test/predict outputs use `outputs/evaluations/<run.id>/trial_<source_trial>/<mode>_<checkpoint>/`.

Mental model:

```text
command
  -> Hydra config
  -> prepare_run
  -> registries build components
  -> train, eval, test, predict, or profile
  -> mode-specific logs/checkpoints/predictions/profiles
```

## Phase 2: Config System

Read:

1. `configs/config.yaml`
2. `configs/experiment/baseline.yaml`
3. `configs/experiment/regression.yaml`
4. `configs/experiment/ablation_heads.yaml`
5. `configs/experiment/sanity_cpu.yaml`
6. `configs/experiment/sanity_gpu.yaml`
7. `configs/data/*.yaml`
8. `configs/model/*.yaml`
9. `configs/task/*.yaml`
10. `configs/optimizer/*.yaml`
11. `configs/scheduler/*.yaml`
12. `configs/trainer/default.yaml`
13. `configs/checkpoint/default.yaml`
14. `configs/logging/default.yaml`
15. `configs/sanity/default.yaml`
16. `configs/profiler.yaml`

Learn:

- `configs/config.yaml` chooses the default config groups.
- Experiment configs override groups and selected values.
- Dataset configs own data shape and label/target properties.
- Model configs own architecture parameters and resolve input/output dimensions
  from `data.*` when appropriate.
- Task configs own loss, metrics, output key, and target key.
- Optimizer/scheduler/trainer configs own update-loop behavior.
- Checkpoint config owns save, retention, and resume selectors.
- Logging config owns JSONL, TensorBoard, and W&B settings.
- Sanity config owns validation policy.
- The top-level `callbacks:` list owns optional trainer hooks, while the top-level `profiler:` block owns trainer-level profiling behavior. `configs/profiler.yaml` separately configures the standalone profiler wrapper.

Important current convention:

```text
data.num_classes -> model.num_classes
data.input_dim   -> model.input_dim
data.output_dim  -> model.output_dim
```

Examples:

```bash
uv run python src/main.py +experiment=baseline optimizer.lr=3e-4
uv run python src/main.py +experiment=regression checkpoint.monitor=val/mse
uv run python src/main.py model=small_transformer data=toy_sequence task=classification
```

## Phase 3: Main Entrypoint

Read:

1. `src/main.py`
2. `src/cli.py`
3. `scripts/train.sh`
4. `scripts/eval.sh`
5. `Makefile`

Trace:

```bash
uv run python src/main.py +experiment=sanity_cpu
```

Follow `src/main.py` in this order:

1. Hydra composes `cfg`.
2. Distributed runtime is initialized.
3. `prepare_run(cfg)` derives `run.id` and rewrites output paths.
4. Reproducibility is configured.
5. Output directories are created.
6. Registries are bootstrapped.
7. Loggers are built.
8. Dataloaders, model, task, optimizer, and scheduler are built.
9. `CheckpointManager` and config-driven callbacks are created.
10. `Trainer` is created.
11. Train mode calls `trainer.fit()`, then optionally logs held-out test metrics and exports predictions.
12. Eval/test/predict modes call `trainer.resume()`; eval logs metrics and predictions, test logs metrics only, and predict exports predictions only.
13. Profile mode calls `trainer.resume()` and then `_profile_training(...)`; checkpoint resume is model-only in profile mode.
14. Resume runs with zero new epochs skip post-train test logging and prediction export.
15. Loggers finish and distributed runtime cleans up.

`src/main.py` never executes `run_sanity_checks`. Use `scripts/run_sanity.py`, `ml-sanity`, `make sanity`, or the sanity Python API explicitly.

Key detail:

```text
src/main.py orchestrates. It should not contain dataset-specific, model-specific,
loss-specific, or metric-specific logic.
```

## Phase 4: Run Identity And Output Paths

Read:

1. `src/utils/run.py`
2. `src/utils/run_inspect.py`
3. `src/utils/paths.py`
4. `scripts/run_registry.py`
5. `configs/config.yaml`
6. `README.md`

Learn:

- `prepare_run(cfg)` computes the stable `run.config_id` and derived `run.id` for fresh runs.
- Users may supply a stable `run.id`, but `run.trial_id`, `run.tracking_id`, and the W&B run name are generated by code.
- Fresh duplicate configs allocate `outputs/runs/<run.id>/trial_<next>/`; they never append to an older fresh trial.
- Trial numbers remain monotonic through the registry even if an unsuccessful trial directory is cleaned up.
- Train/profile configs use `outputs/run_configs/<run.id>/trial_<n>.yaml`.
- The startup log highlights run id, trial id, mode, and output directory in bold cyan.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu
uv run python src/main.py +experiment=sanity_cpu
uv run ml-run-registry list
```

The two fresh invocations share a stable run id and use consecutive trial folders.

## Phase 5: Resume And Evaluation Behavior

Training resume examples:

```bash
uv run python src/main.py --resume-run <run_id>
uv run python src/main.py --resume-run <run_id> --registry /path/to/run_registry.jsonl
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/last.pt
```

Evaluation examples:

```bash
uv run ml-evaluate-run <run_id> --checkpoint best
uv run ml-evaluate-run <run_id> --checkpoint epoch_0005
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
```

Learn:

- An explicit checkpoint path encodes the source `run.id` and `trial_id`; those values override config-derived identity.
- `--resume-run` selects the newest training registry record and converts the requested selector to an explicit checkpoint path.
- `--resume-run` cannot be combined with `--run-id`.
- Intentional training resume continues the original training trial and restores full training state.
- Train, resume, profile, eval, test, and predict do not run sanity checks automatically. Evaluation-style modes load model-only state plus checkpoint counters/metadata.
- Evaluation output is `outputs/evaluations/<run.id>/trial_<source_trial>/<mode>_<checkpoint>/`.
- Repeating the same evaluation deletes and recreates that exact folder, emitting a bold red overwrite warning; another checkpoint gets another folder.
- A resume that runs zero new epochs skips post-train test/prediction logging.

## Behavior Exceptions To Remember

| Situation | Behavior | Why |
| --- | --- | --- |
| Fresh duplicate config | Allocates the next code-managed trial. | Keeps logs and checkpoints isolated. |
| Explicit checkpoint resume | Recovers run/trial identity from the path. | The model artifact is the identity source. |
| `--resume-run` | Resolves the registry record to an explicit checkpoint path. | Avoids hash-based or manually supplied trial selection. |
| Repeated same evaluation | Replaces the same `<mode>_<checkpoint>` folder with a red warning. | Keeps one clean local result per evaluation target. |
| Different evaluated checkpoints | Use separate `eval_best`, `eval_last`, or `eval_epoch_...` folders. | Prevents model-result collisions. |
| Normal experiment entrypoints | Never run sanity checks automatically. | Sanity validation is an explicit operation. |
| Resume with zero new epochs | Skips duplicate post-train test/prediction output. | Avoids duplicate tracking points. |

## Phase 6: Registries

Read:

1. `src/utils/registry.py`
2. `src/models/__init__.py`
3. `src/data/__init__.py`
4. `src/tasks/__init__.py`
5. `src/losses/__init__.py`
6. `src/metrics/__init__.py`
7. `src/optim/__init__.py`

Then read implementations:

1. `src/models/model.py`
2. `src/data/dataset.py`
3. `src/tasks/task.py`
4. `src/losses/losses.py`
5. `src/metrics/metrics.py`
6. `src/optim/optimizers.py`
7. `src/optim/schedulers.py`

Learn:

- Registries map config names to Python implementations.
- Config selects `model.name`, `data.name`, `task.name`, etc.
- Adding a component usually means adding a registered class/function and a
  matching config file.

Mental model:

```text
configs/model/mlp.yaml -> model.name: mlp -> MODEL_REGISTRY.build(...)
configs/data/toy_classification.yaml -> data.name -> DATASET_REGISTRY
configs/task/classification.yaml -> task.name -> TASK_REGISTRY
```

## Phase 7: Data Flow

Read:

1. `src/data/dataset.py`
2. `src/data/dataloader.py`
3. `configs/data/toy_classification.yaml`
4. `configs/data/toy_regression.yaml`
5. `configs/data/toy_sequence.yaml`
6. `configs/data/tensor_file.yaml`
7. `src/data/preprocess.py`
8. `scripts/preprocess.sh`

Learn:

- Every dataset returns dictionary samples.
- The trainer and tasks operate on batch dictionaries.
- Toy datasets are deterministic synthetic datasets.
- `tensor_file` reads generated `.pt` files.
- `build_dataloaders(cfg)` creates train/val/test dataloaders.
- Dataset configs own properties such as feature dimension, target dimension,
  sequence length, vocabulary size, and number of classes.

Try:

```bash
bash scripts/preprocess.sh --force
uv run python src/main.py data=tensor_file scheduler=none trainer.max_epochs=1
```

Question to answer:

```text
What keys does each dataset return, and does the task config use the same
target key?
```

## Phase 8: Models

Read:

1. `src/models/model.py`
2. `src/models/layers.py`
3. `configs/model/mlp.yaml`
4. `configs/model/regression_mlp.yaml`
5. `configs/model/small_transformer.yaml`
6. `configs/model/cnn.yaml`

Learn:

- Every model consumes a batch dictionary.
- Every model returns an output dictionary.
- Tasks decide which output key to read.
- MLP supports classification and regression through config.
- Models resolve dataset-owned dimensions from `data.*`.
- Architecture-only parameters remain model-owned, such as hidden size, layer
  count, dropout, attention heads, and convolution width.

Trace:

```text
batch -> model(batch) -> {"logits": tensor} -> task.step(...)
```

## Phase 9: Tasks, Losses, And Metrics

Read:

1. `src/tasks/task.py`
2. `src/tasks/segmentation.py`, `detection.py`, `ranking.py`, and `language_modeling.py`
3. `configs/task/*.yaml`
4. `src/losses/losses.py`
5. `src/metrics/metrics.py`

Learn:

- Trainer is generic because task classes own workload semantics.
- A task knows the model output key and target key.
- A task calls the selected loss.
- A task updates selected metrics.
- A task formats prediction records.

Important classes:

- `BaseTask`
- `ClassificationTask`
- `RegressionTask`
- `SegmentationTask`
- `DetectionTask`
- `RankingTask`
- `LanguageModelingTask`
- `StepResult`
- `MetricAccumulator`
- `MetricCollection`

Question to answer:

```text
For a segmentation project, what changes: model, data, task, loss, metric, or
trainer?
```

Expected answer:

```text
model + data + task + loss/metrics, usually not trainer.
```

## Phase 10: Optimizers And Schedulers

Read:

1. `src/optim/optimizers.py`
2. `src/optim/schedulers.py`
3. `configs/optimizer/adamw.yaml`
4. `configs/optimizer/sgd.yaml`
5. `configs/scheduler/*.yaml`

Learn:

- Optimizer builders are registered.
- AdamW supports no-decay grouping for norm and bias parameters.
- Scheduler builders return `SchedulerBundle`.
- `interval` controls epoch-step vs batch-step behavior.
- Plateau scheduling uses a monitored metric.
- Warmup can wrap compatible schedulers.

Try:

```bash
uv run python src/main.py +experiment=baseline scheduler=none trainer.max_epochs=1
uv run python src/main.py +experiment=baseline scheduler=onecycle scheduler.interval=step trainer.max_epochs=1
uv run python src/main.py +experiment=baseline optimizer=sgd optimizer.lr=0.05 trainer.max_epochs=1
```

## Phase 11: Trainer And Evaluator

Read:

1. `src/engine/trainer.py`
2. `src/engine/evaluator.py`
3. `src/callbacks/base.py`
4. `callback_tutorial.md`
5. `configs/trainer/default.yaml`

Learn:

- Device placement.
- AMP and precision handling.
- `fp32` stays on the normal train/validation/test/predict path; `amp`, `fp16`, and `bf16` share CUDA autocast across trainer and evaluator paths.
- Gradient accumulation.
- Gradient clipping.
- Finite-loss checks.
- Training loop.
- Validation loop.
- Recursive device transfer for nested batches and evaluation without a task loss.
- Callback hook timing, top-level `callbacks:` config construction, registry wiring, and direct `Trainer` wiring for tests/custom scripts.
- Scheduler stepping.
- Metric logging.
- Checkpoint saving.
- Resume handling.
- Early stopping.
- Exception checkpointing after training has started.
- `trained_epochs`, used to avoid duplicate post-resume test logs.

Learn in `Evaluator`:

- Eval-only metric loop.
- Prediction export.
- Task-driven prediction records.

Key idea:

```text
Trainer owns orchestration. Task owns semantics. Model owns architecture.
```

## Phase 12: Checkpoints And Fault Tolerance

Read:

1. `src/utils/checkpoint.py`
2. `configs/checkpoint/default.yaml`
3. `tests/test_checkpoint.py`
4. README checkpoint section

Learn:

- Checkpoints include model, optimizer, scheduler, scaler, RNG state, epoch,
  global step, best metric, metrics, and config.
- Saves are atomic.
- Manifest stores checksum metadata.
- Checksum validation uses the latest manifest entry for a checkpoint path.
- Manifest entries are replaced on same-path overwrite and removed on retention rotation. Exception saves do not invalidate `last.pt`; disabled selectors are absent from both disk and manifest.
- `last.pt` is the latest checkpoint pointer.
- `best.pt` is the best monitored checkpoint pointer.
- `latest`, `last`, `best`, and epoch selectors are supported.
- `load_latest` falls back across candidates when a candidate is corrupt.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu checkpoint.save_every=1
uv run python src/main.py +experiment=sanity_cpu checkpoint.resume=latest
uv run python src/main.py +experiment=sanity_cpu checkpoint.resume=best
```

## Phase 13: Logging And Tracking

Read:

1. `src/utils/logger.py`
2. `configs/logging/default.yaml`
3. README W&B/output organization sections

Learn:

- JSONL logging is enabled by default.
- TensorBoard logging is optional.
- W&B logging is optional.
- W&B ids and names are generated as `<run.id>-trial-<n>` for train/profile and `<run.id>-trial-<source_trial>-<mode>-<checkpoint>` for eval/test/predict.
- W&B uses `resume='allow'`.
- W&B receives the resolved config.
- W&B logs the resolved config YAML as an artifact.
- Resume with zero new epochs skips post-train test logging to avoid duplicate
  tracking points.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu
cat outputs/runs/<run_id>/trial_<n>/logs/metrics.jsonl
```

Optional tracking:

```bash
uv sync --extra tracking
uv run python src/main.py +experiment=baseline logging.tensorboard.enabled=true
uv run tensorboard --logdir outputs/runs
uv run python src/main.py +experiment=baseline logging.wandb.enabled=true logging.wandb.project=my-project
```

## Phase 14: Reproducibility And Runtime

Read:

1. `src/utils/seed.py`
2. `src/runtime/distributed.py`
3. `src/runtime/__init__.py`
4. `configs/config.yaml` runtime section

Learn:

- Python, NumPy, and PyTorch seeds are set together.
- CUDA seed setup happens when CUDA is required.
- Deterministic settings are configurable.
- Distributed helpers handle rank, world size, barriers, broadcast, weighted metric
  reductions, tensor sums, and object gathering.
- `src/main.py` automatically wraps the model with `torch.nn.parallel.DistributedDataParallel` when launched under an initialized `torchrun` process group.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu run.seed=123
uv run python src/main.py +experiment=sanity_cpu run.deterministic=true
```

## Phase 15: Sanity Checks

Read:

1. `scripts/run_sanity.py`
2. `src/utils/sanity/core.py`
3. `src/utils/sanity/cuda.py`
4. `src/utils/sanity/packages.py`
5. `src/utils/sanity/model.py`
6. `src/utils/sanity/ddp.py`
7. `configs/sanity/default.yaml`
8. `tests/test_sanity.py`
9. `sanity_check_commands.md`

Learn:

- `scripts/run_sanity.py` composes config and runs checks without training.
- Python/package checks come from `pyproject.toml`.
- Required packages are `[project].dependencies`.
- Optional warning-only packages are `[project.optional-dependencies]`.
- CUDA diagnostics can run before torch is installed.
- Driver compatibility uses `nvidia-smi` plus known CUDA minimum driver mapping.
- Registry checks confirm selected component names exist.
- Smoke checks run a tiny data/model/loss/backward/optimizer/scheduler path.
- Experiment composition checks catch broken experiment YAML.
- Normal train/resume/profile/eval/test/predict workflows do not execute the sanity suite; invoke the standalone sanity command explicitly.

Try:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu
uv run python scripts/run_sanity.py sanity.torch_install.recommend=true
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true
uv run python scripts/run_sanity.py +experiment=sanity_gpu
```

## Phase 16: Tests As Executable Documentation

Read:

1. `tests/conftest.py`
2. `tests/test_config.py`
3. `tests/test_dataset.py`
4. `tests/test_data.py`
5. `tests/test_model.py`
6. `tests/test_precision.py`
7. `tests/test_tasks.py`
8. `tests/test_callbacks.py`
9. `tests/test_training.py`
10. `tests/test_checkpoint.py`
11. `tests/test_metrics.py`
12. `tests/test_schedulers.py`
13. `tests/test_distributed.py`
14. `tests/test_run_identity.py`
15. `tests/test_run_tools.py`
16. `tests/test_sanity.py`

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_run_identity.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_checkpoint.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_training.py -q
```

Learn:

- Tests show the expected component contracts.
- Run identity tests document reuse, evaluation layout, tracking ids, and resume selectors.
- Run-tool tests document replay/resume lookup, comparison, plotting, evaluation command construction, export, and unsuccessful-run selection.
- Checkpoint tests document selector paths, fallback loading, and manifest checksum behavior.
- Trainer regressions cover step-validation state restoration, monotonic logging steps, partial accumulation normalization, scheduler state at checkpoint time, and model-only evaluation loading.
- Distributed and task regressions cover duplicate-free evaluation shards, dataset confusion-matrix segmentation metrics, and dataset-level detection AP@50.
- Precision, distributed, callback, task, and training tests document their respective runtime contracts.

## Phase 17: Customization Path

When adapting this template to a real project, work in this order:

1. Add or replace dataset code in `src/data/`.
2. Add a matching data config in `configs/data/`.
3. Put dataset-owned dimensions and class/target properties in the data config.
4. Add or adapt model code in `src/models/`.
5. Add a model config that resolves data-owned dimensions from `data.*`.
6. Add or adapt task code in `src/tasks/` if loss/metrics/predictions differ.
7. Add losses and metrics only when existing ones do not fit.
8. Create an experiment config under `configs/experiment/`.
9. Run sanity composition checks.
10. Run a one-epoch smoke train.
11. Run eval and resume commands.

Minimal validation sequence:

```bash
uv run python scripts/run_sanity.py +experiment=<your_experiment> sanity.check_all_experiments=true
uv run python src/main.py +experiment=<your_experiment> trainer.max_epochs=1
uv run python src/main.py +experiment=<your_experiment> checkpoint.resume=latest trainer.max_epochs=2
uv run python src/main.py +experiment=<your_experiment> run.mode=eval checkpoint.resume=best
```

## Where To Edit

Use this table when you know what you want to change but not where it belongs:

| Goal | Edit Code | Edit Config | Notes |
| --- | --- | --- | --- |
| Add a dataset | `src/data/` | `configs/data/` | Put data shape, class count, target dimension, paths, and dataloader settings in data config. |
| Add a model | `src/models/` | `configs/model/` | Keep architecture parameters in model config; resolve dataset-owned dimensions from `data.*`. |
| Add task behavior | `src/tasks/` | `configs/task/` | Use this for new target formats, losses, metrics, prediction records, or output keys. |
| Add a loss | `src/losses/` | referenced by task config | Register it and select it through `task.loss.name`. |
| Add a metric | `src/metrics/` | referenced by task config | Register it and include it in `task.metrics`. |
| Add optimizer behavior | `src/optim/optimizers.py` | `configs/optimizer/` | Use for new optimizer types or parameter grouping. |
| Add scheduler behavior | `src/optim/schedulers.py` | `configs/scheduler/` | Return a `SchedulerBundle` with the right interval. |
| Add an experiment preset | usually no code | `configs/experiment/` | Override groups and values for a repeatable run. |
| Change output/run identity behavior | `src/utils/run.py` | `configs/config.yaml` run section | Keep logger/checkpoint/prediction paths consistent. |
| Change checkpoint policy | `src/utils/checkpoint.py` | `configs/checkpoint/default.yaml` | Preserve atomic writes and manifest validation. |
| Change logging/tracking | `src/utils/logger.py` | `configs/logging/default.yaml` | Keep JSONL, TensorBoard, and W&B behavior aligned. |
| Change sanity checks | `src/utils/sanity/`, `scripts/run_sanity.py` | `configs/sanity/default.yaml` | Sanity should catch broken machines/configs before expensive runs. |

## Final Mental Model

Keep these ownership boundaries clear:

```text
Data config: dataset identity, data shape, class/target dimensions, dataloader settings.
Model config: architecture parameters; input/output dimensions resolve from data when needed.
Task config: loss, metrics, model output key, target key, prediction formatting.
Trainer config: loop controls only.
Checkpoint config: save/resume policy.
Logging config: JSONL, TensorBoard, W&B.
Run config: identity, device, precision, output roots, reproducibility.
```

The trainer should stay generic. Most real project changes should happen in
data, model, task, loss, metric, and config files.
