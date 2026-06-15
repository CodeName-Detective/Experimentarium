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

This composes a Hydra config, prepares a run directory, runs sanity checks,
builds data/model/task/optimizer/scheduler/loggers/checkpoints, trains, tests,
logs metrics, and writes predictions.

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

## Phase 1: Big Picture

Read:

1. `README.md`
2. `Flowchart.md`
3. `Description.md`
4. `Run_commands.md`
5. `sanity_check_commands.md`

Learn:

- The framework is driven by Hydra configs under `configs/`.
- `src/main.py` is the main train/eval entrypoint.
- `scripts/run_sanity.py` validates the environment and config without training.
- Config groups select model, data, task, optimizer, scheduler, trainer,
  logging, checkpoint, and sanity behavior.
- Run outputs live under `outputs/runs/<run.id>/`.

Mental model:

```text
command
  -> Hydra config
  -> prepare_run
  -> sanity checks when needed
  -> registries build components
  -> train or eval
  -> logs/checkpoints/predictions
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
2. `scripts/train.sh`
3. `scripts/eval.sh`
4. `Makefile`

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
8. Sanity checks run unless the command is eval or training resume.
9. Dataloaders, model, task, optimizer, and scheduler are built.
10. `CheckpointManager` is created.
11. `Trainer` is created.
12. Train mode calls `trainer.fit()`.
13. Eval mode calls `trainer.resume()` and `Evaluator.evaluate(...)`.
14. Test metrics are logged and predictions are exported for normal train/eval runs.
15. Resume runs with zero new epochs skip post-train test logging and prediction export.
16. Loggers finish and distributed runtime cleans up.

Key detail:

```text
src/main.py orchestrates. It should not contain dataset-specific, model-specific,
loss-specific, or metric-specific logic.
```

## Phase 4: Run Identity And Output Paths

Read:

1. `src/utils/run.py`
2. `src/utils/paths.py`
3. `configs/config.yaml` run section
4. `README.md` run identity section

Learn:

- `prepare_run(cfg)` computes `run.config_id`.
- It derives `run.id` from run/model/data/task/trial/config hash unless
  `run.id` is manually set.
- Fresh duplicate runs reuse the existing run directory and emit a warning.
- Intentional training resumes reuse the existing run directory without the
  duplicate-run warning.
- Eval from a training checkpoint keeps the training run id and writes to `outputs/evaluations/<run.id>/`.
- Training configs are stored in `outputs/run_configs/<run.id>.yaml`; evaluation configs use `outputs/evaluations/<run.id>/config.yaml`.
- Run metadata, a shell-safe repeat command, and its working directory are appended to `outputs/run_registry.jsonl`.
- Checkpoint/log/prediction/profile paths are rewritten under `run.run_dir`.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu
ls outputs/runs
ls outputs/run_configs
tail -n 3 outputs/run_registry.jsonl
```

## Phase 5: Resume And Evaluation Behavior

Read:

1. `src/utils/run.py`
2. `src/engine/trainer.py`
3. `src/utils/checkpoint.py`
4. `src/main.py`
5. `configs/checkpoint/default.yaml`

Training resume examples:

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=latest
uv run python src/main.py +experiment=baseline checkpoint.resume=best
uv run python src/main.py +experiment=baseline checkpoint.resume=last
uv run python src/main.py +experiment=baseline checkpoint.resume=5
uv run python src/main.py +experiment=baseline checkpoint.resume=epoch_0005
```

Learn:

- `latest` loads the newest valid checkpoint.
- `last` loads `checkpoints/last.pt`.
- `best` loads `checkpoints/best.pt`.
- `5` and `epoch_0005` load `checkpoints/epoch_0005.pt`.
- Training resume skips sanity checks.
- Training resume logs a bold green resume message.
- If resume does not run any new epochs, post-train test logging and prediction
  export are skipped to avoid duplicate W&B/JSONL/TensorBoard points.

Evaluation examples:

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=best
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=epoch_0005
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
```

Learn:

- Eval mode skips sanity checks.
- Eval selector paths are resolved against the training run.
- Eval artifacts and the resolved eval config are written to `outputs/evaluations/<run_id>/`.

## Behavior Exceptions To Remember

These are intentional deviations from the simple train/eval mental model:

| Situation | Behavior | Why |
| --- | --- | --- |
| Fresh duplicate run id | Reuses the existing run directory and emits a warning. | Keeps run ids stable instead of creating `_2`, `_3`, etc. |
| Training resume with `checkpoint.resume=...` | Reuses the training run directory without the duplicate-run warning. | Resume is intentional and should continue the same run. |
| Training resume | Skips sanity checks. | Resume should start quickly from a known run/config. |
| Eval mode | Skips sanity checks. | Eval should only load and score a checkpoint. |
| Eval from training checkpoint | Keeps `run.id` and writes to `outputs/evaluations/<run_id>/`. | Keeps evaluation artifacts separate from training artifacts. |
| Resume with zero new epochs | Skips post-train test metrics and prediction export. | Avoids duplicate JSONL/TensorBoard/W&B points. |

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
- `fp32` stays on the normal train/eval/predict path; `amp`, `fp16`, and `bf16` share CUDA autocast across trainer and evaluator paths.
- Gradient accumulation.
- Gradient clipping.
- Finite-loss checks.
- Training loop.
- Validation loop.
- Recursive device transfer for nested batches and evaluation without a task loss.
- Callback hook timing, direct `Trainer` wiring, and current config-integration limitations.
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
- W&B uses `run.tracking_id`: `run.id` for training and `<run.id>_evaluation` for evaluation.
- W&B uses `resume='allow'`.
- W&B receives the resolved config.
- W&B logs the resolved config YAML as an artifact.
- Resume with zero new epochs skips post-train test logging to avoid duplicate
  tracking points.

Try:

```bash
uv run python src/main.py +experiment=sanity_cpu
cat outputs/runs/<run_id>/logs/metrics.jsonl
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
- Distributed helpers handle rank, world size, barriers, broadcast, and metric
  averaging.
- True DDP model wrapping is documented as a future extension.

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
- Main eval and training resume skip sanity checks by design.

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
4. `tests/test_model.py`
5. `tests/test_training.py`
6. `tests/test_checkpoint.py`
7. `tests/test_metrics.py`
8. `tests/test_schedulers.py`
9. `tests/test_run_identity.py`
10. `tests/test_sanity.py`

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_run_identity.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_checkpoint.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_training.py -q
```

Learn:

- Tests show the expected component contracts.
- Run identity tests document reuse, eval suffixing, and resume selectors.
- Checkpoint tests document selector paths, fallback loading, and manifest
  checksum behavior.
- Training tests document fit/resume behavior.

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
