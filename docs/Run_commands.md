# Run Commands

This file is the command reference for running the framework after the
environment sanity checks pass. It covers training, evaluation, test, prediction,
resume, config overrides, shell wrappers, Makefile shortcuts, profiling,
preprocessing, run-registry inspection, run comparison, metric plotting, checkpoint export, run cleanup, checkpoint verification, and tracking workflows.

Note: Evaluation computes metrics and exports predictions, test computes only metrics, and prediction exports only predictions.

Use `sanity_check_commands.md` before this file when validating a new machine.

## Entry Points

The main training/evaluation entrypoint is:

```bash
uv run python src/main.py
```

Runs the default Hydra config through `src/main.py`, including run setup, training, checkpointing, held-out test evaluation, and prediction export. Sanity checks are not run automatically. Set `run.mode=eval`, `test`, `predict`, or `profile` for the other entrypoint modes.

If the project is installed from `pyproject.toml`, the console-script entrypoint is:

```bash
uv run ml-train
```

Calls the same application as `src/main.py` through the package console script. Use it when you want the shorter installed-command form.

The shell training wrapper is:

```bash
bash scripts/train.sh
```

Sets the project import path and forwards all arguments to the main training entrypoint.

The shell evaluation wrapper is:

```bash
bash scripts/eval.sh <checkpoint_path>
```

Runs `run.mode=eval` for the checkpoint path passed as the first argument; extra arguments are forwarded as Hydra overrides.

Makefile shortcuts are also available:

```bash
make train ARGS="+experiment=baseline trainer.max_epochs=1"
make eval CHECKPOINT=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
```

`make train` forwards `ARGS` to the training entrypoint. `make eval` evaluates the checkpoint supplied through `CHECKPOINT`.

Prefer the direct `uv run python src/main.py ...` form when learning the repo.
Use `uv run ml-train`, shell scripts, or Makefile targets when you want shorter
repeatable commands.

## Basic Training

### Train The Default Config

```bash
uv run python src/main.py
```

Uses `configs/config.yaml`, which defaults to MLP classification on synthetic
toy classification data with AdamW and cosine scheduling.

Console-script equivalent:

```bash
uv run ml-train
```

Runs the same default training workflow through the installed console script.

Shell equivalent:

```bash
bash scripts/train.sh
```

Runs default training through the shell wrapper, which forwards arguments to `src/main.py`.

Make equivalent:

```bash
make train
```

Runs the default training command through the Makefile shortcut.

### Train A Named Experiment

```bash
uv run python src/main.py +experiment=baseline
uv run python src/main.py +experiment=regression
uv run python src/main.py +experiment=ablation_heads
```

Each command composes a named experiment preset from `configs/experiment/` and trains that workload with its selected model, data, task, and checkpoint monitor.

Experiment files are not invoked by the default config. `+experiment=<name>` explicitly composes the selected preset, whose group selections and values override the existing defaults.

Console-script equivalents:

```bash
uv run ml-train +experiment=baseline
uv run ml-train +experiment=regression
uv run ml-train +experiment=ablation_heads
```

These run the same named experiments through the installed `ml-train` command.

Shell equivalents:

```bash
bash scripts/train.sh +experiment=baseline
bash scripts/train.sh +experiment=regression
bash scripts/train.sh +experiment=ablation_heads
```

These run the same named experiments through the shell wrapper.

Make equivalent:

```bash
make train ARGS="+experiment=baseline"
```

Runs the baseline experiment through `make train`, forwarding the Hydra override via `ARGS`.

Available experiment presets:

- `baseline`: MLP classification baseline.
- `regression`: regression model/data/task preset.
- `ablation_heads`: transformer-head ablation on toy sequence data.
- `sanity_cpu`: tiny CPU run.
- `sanity_gpu`: tiny CUDA run.

### Tiny Smoke Training

```bash
uv run python src/main.py +experiment=sanity_cpu
```

Runs one tiny CPU epoch and verifies the full train/validation/test/checkpoint
path.

On CUDA machines:

```bash
uv run python src/main.py +experiment=sanity_gpu
```

This forces `run.device=cuda` and runs a tiny full trainer workflow on the GPU. Run `uv run python scripts/run_sanity.py +experiment=sanity_gpu` separately when CUDA diagnostics are needed.

## Evaluation

Evaluation derives filesystem and tracking identity from the selected checkpoint. The recommended command loads the saved training config and resolves the checkpoint selector to an explicit path:

```bash
uv run ml-evaluate-run <run_id> --checkpoint best
uv run ml-evaluate-run <run_id> --checkpoint last --mode test
uv run ml-evaluate-run <run_id> --checkpoint epoch_0005 --mode predict
uv run ml-evaluate-run <run_id> --checkpoint best --print-only
```

For a source checkpoint such as `outputs/runs/<run_id>/trial_3/checkpoints/best.pt`, output goes to:

```text
outputs/evaluations/<run_id>/trial_3/eval_best/
```

Different modes and checkpoints remain separate, for example `test_last/`, `predict_epoch_0005/`, and `eval_best/`. Repeating the same source-trial/mode/checkpoint evaluation deletes and recreates only that exact directory and emits a bold red warning.

Direct checkpoint-path equivalents:

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
uv run ml-train +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
bash scripts/eval.sh outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt +experiment=baseline
make eval CHECKPOINT=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
```

The composed model/data/task config must match the checkpoint. `ml-evaluate-run` avoids that mismatch by loading `outputs/run_configs/<run_id>/trial_<n>.yaml` from the latest training registry record.

Checkpoint selectors also work with a matching composed config:

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=latest
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=best
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=epoch_0005
```

Here the matching config derives the stable run id, and the framework selects the newest checkpoint-bearing training trial. Prefer an explicit checkpoint path or `ml-evaluate-run` when more than one trial exists.

## Resume Training

Resume reuses the original training trial. It never allocates a new trial and does not use a newly computed hash to choose identity when an explicit checkpoint path is supplied.

Resume by registry id:

```bash
uv run python src/main.py --resume-run <run_id>
uv run python src/main.py --resume-run <run_id> --registry /path/to/run_registry.jsonl
uv run ml-run-registry resume-command <run_id>
```

`--resume-run` selects the latest training record, resolves `last.pt` (or an overridden selector) to an explicit checkpoint path, loads the saved config, and defaults to `run.mode=train`. Do not combine it with `--run-id`; the checkpoint path owns run and trial identity.

Resume an exact model:

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/last.pt
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/trial_<n>/checkpoints/epoch_0005.pt
```

The path is parsed as `<run_id>/trial_<n>/checkpoints/<model>.pt`; `prepare_run` continues that exact folder. The startup identity line highlights the recovered trial id. Full optimizer, scheduler, scaler, RNG, epoch, and global-step state are restored in train mode.

Selectors remain available when the current config identifies the intended run:

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=latest
uv run python src/main.py +experiment=baseline checkpoint.resume=best
uv run python src/main.py +experiment=baseline checkpoint.resume=epoch_0005
```

If resume produces zero new epochs, post-train test metrics and prediction export are skipped to avoid duplicate local and W&B points.

## Config Override Patterns

Hydra lets you override any config value from the command line.

### Trainer Overrides

```bash
uv run python src/main.py trainer.max_epochs=20
uv run python src/main.py trainer.max_steps=100
uv run python src/main.py trainer.limit_train_batches=5 trainer.limit_val_batches=2
uv run python src/main.py trainer.val_every_n_steps=50
uv run python src/main.py trainer.log_gradient_norm=true trainer.log_learning_rate=true
uv run python src/main.py trainer.grad_clip=0.5
uv run python src/main.py trainer.early_stopping.patience=3
```

These override training loop behavior: total epochs or optimizer steps, debug batch limits, validation cadence, optional step diagnostics, gradient clipping threshold, and early-stopping patience.
Step-based validation is side-effect free: the trainer restores training mode and the in-progress training metric state before continuing. Epoch and step metrics both use `global_step`, so logger step values never move backward.

### Optimizer Overrides

```bash
uv run python src/main.py optimizer.lr=3e-4
uv run python src/main.py optimizer.weight_decay=1e-2
uv run python src/main.py optimizer=sgd optimizer.lr=0.05 optimizer.momentum=0.9
```

These change optimizer hyperparameters or swap the optimizer config group from AdamW to SGD.

### Scheduler Overrides

```bash
uv run python src/main.py scheduler=none
uv run python src/main.py scheduler=cosine scheduler.eta_min=1e-6
uv run python src/main.py scheduler=onecycle scheduler.max_lr=3e-3 scheduler.interval=step
uv run python src/main.py scheduler=plateau scheduler.monitor=val/loss scheduler.patience=2
```

These disable scheduling or select a learning-rate schedule and tune its key parameters. For horizon-based cosine and polynomial schedules, warmup steps are subtracted from the main schedule horizon; fixed-parameter schedulers keep their configured durations. Do not enable `scheduler.warmup.enabled` with `scheduler=plateau`; preflight and scheduler construction reject that unsupported combination.

### Model Overrides

```bash
uv run python src/main.py model=mlp model.hidden_dim=128 model.num_layers=3 model.dropout=0.2
uv run python src/main.py model=small_transformer data=toy_sequence task=classification
uv run python src/main.py model=regression_mlp data=toy_regression task=regression
```

These resize the MLP or swap model/data/task groups together so the model output matches the workload.

### Data Overrides

```bash
uv run python src/main.py data.batch_size=64
uv run python src/main.py data.splits.train.batch_size=32 data.splits.val.batch_size=128
uv run python src/main.py data.num_workers=4
uv run python src/main.py data.pin_memory=true run.device=cuda
uv run python src/main.py data.transforms.train.name=identity
uv run python src/main.py data.splits.train.num_samples=1024 data.splits.val.num_samples=256
```

These tune global and split-specific dataloader settings, optional input transforms, and synthetic split sizes; `pin_memory=true` is useful when training on CUDA.

`data.pin_memory=true` stores DataLoader batches in pinned CPU memory, enabling faster asynchronous CPU-to-GPU transfers when training on CUDA; it uses extra host RAM and generally provides no benefit for CPU-only runs.

### Runtime Device And Precision

CPU:

```bash
uv run python src/main.py run.device=cpu run.precision=fp32
```

Forces deterministic CPU-friendly fp32 execution for debugging or machines without CUDA.

CUDA fp32:

```bash
uv run python src/main.py run.device=cuda run.precision=fp32
```

Runs on the GPU while keeping the normal non-autocast fp32 path for training, validation, test evaluation, and prediction export.

CUDA AMP:

```bash
uv run python src/main.py run.device=cuda run.precision=amp
```

Runs on CUDA with automatic mixed precision for training, validation, test evaluation, and prediction export.

BF16, if supported by your GPU:

```bash
uv run python src/main.py run.device=cuda run.precision=bf16
```

Uses bfloat16 autocast on compatible GPUs for training, validation, test evaluation, and prediction export, usually with better numerical range than fp16.

fp16 has greater numerical precision but a smaller value range and often needs loss scaling, while bf16 has a much larger range and better stability but requires newer hardware support.

### Checkpoint Overrides

```bash
uv run python src/main.py checkpoint.save_every=1 checkpoint.keep_last_k=3
uv run python src/main.py checkpoint.monitor=val/loss checkpoint.mode=min
uv run python src/main.py checkpoint.save_top_k=2
```

These control checkpoint frequency, retention, and which validation metric decides the best checkpoint.
`checkpoint.keep_last_k=0` disables epoch-file rotation and keeps all saved epoch files. `checkpoint.save_top_k=0` disables both top-k retention and `best.pt`. `checkpoint.save_last=false` disables `last.pt`. The manifest contains only currently retained epoch files, so `scripts/verify_checkpoints.py` remains valid after rotation or overwrites.

For regression:

```bash
uv run python src/main.py +experiment=regression checkpoint.monitor=val/mse checkpoint.mode=min
```

For regression, this saves the best checkpoint according to lowest validation MSE.

### Logging Overrides

JSONL logging is enabled by default.

Enable TensorBoard:

```bash
uv run python src/main.py logging.tensorboard.enabled=true
```

Adds TensorBoard scalar logging under the run's `logs/tensorboard/` directory.

Then view logs:

```bash
uv run tensorboard --logdir outputs/runs
```

Starts TensorBoard over all run directories so you can compare experiments.

Enable W&B:

```bash
uv sync --extra tracking
uv run wandb login
uv run python scripts/run_sanity.py logging.wandb.enabled=true logging.wandb.project=my-project
uv run python src/main.py logging.wandb.enabled=true logging.wandb.project=my-project
```

Installs tracking dependencies, authenticates W&B, validates W&B readiness, then starts a run that logs online to the chosen project.

The sanity command checks that W&B imports, credentials are present, and the W&B
host is reachable before you start a run that logs online.

Run W&B offline:

```bash
uv run python src/main.py logging.wandb.enabled=true logging.wandb.mode=offline
```

Records W&B data locally without requiring network access; sync it later with W&B tooling.

### Run Identity

Use a manual stable id only for fresh work:

```bash
uv run python src/main.py +experiment=baseline run.id=my_baseline_debug
```

Fresh invocations with the same effective config and stable id automatically allocate consecutive trials. `run.trial_id`, `run.tracking_id`, and the W&B run name are generated by the framework and are not user-controlled. Every invocation prints a bold cyan startup line with the run id, trial id, mode, and output directory.

Fresh replays may receive another stable id with `--run-id`:

```bash
uv run python src/main.py --from-run <run_id> --run-id replayed_run
```

Resume cannot use `--run-id`; identity is recovered from the checkpoint path.

## Full Example Runs

### Short Baseline Debug Run

```bash
uv run python src/main.py +experiment=baseline trainer.max_epochs=1 trainer.log_every_n_steps=1
```

Runs a one-epoch baseline with frequent logging, useful for quick debugging.

### Larger MLP Classification Run

```bash
uv run python src/main.py +experiment=baseline model.hidden_dim=256 model.num_layers=4 data.batch_size=128 optimizer.lr=3e-4 trainer.max_epochs=30
```

Runs a larger MLP classification experiment with a deeper/wider model and longer training.

### Regression Run

```bash
uv run python src/main.py +experiment=regression trainer.max_epochs=10 optimizer.lr=1e-3
```

Runs the regression preset for 10 epochs with an explicit learning rate.

### Sequence Transformer Run

```bash
uv run python src/main.py +experiment=ablation_heads model.num_heads=4 model.d_model=128 trainer.max_epochs=5
```

Runs the transformer-head ablation with a specific head count and model width.

### OneCycle Schedule Run

```bash
uv run python src/main.py +experiment=baseline scheduler=onecycle scheduler.max_lr=3e-3 scheduler.interval=step
```

Trains the baseline with a step-wise OneCycle learning-rate schedule.

### GPU AMP Run

```bash
uv run python src/main.py +experiment=baseline run.device=cuda run.precision=amp data.batch_size=128
```

Runs the baseline on GPU with mixed precision and a larger batch size.

### Tensor-File Dataset Run

Generate toy tensor-file data:

```bash
bash scripts/preprocess.sh --force
```

Regenerates processed tensor files under `data/processed/`, overwriting existing generated data.

Train from generated `.pt` files:

```bash
uv run python src/main.py data=tensor_file scheduler=none
```

Switches the data config to file-backed tensors and disables the scheduler for a simple run.

Shell equivalent:

```bash
bash scripts/train.sh data=tensor_file scheduler=none
```

Runs the same tensor-file training command through the shell wrapper.

## Shell Script Workflows

### Training Wrapper

```bash
bash scripts/train.sh +experiment=baseline trainer.max_epochs=10 optimizer.lr=3e-4
```

Runs the baseline through the shell wrapper for 10 epochs with the selected learning rate.

This simply sets `PYTHONPATH` and calls:

```bash
uv run python src/main.py "$@"
```

This is the exact command run by `scripts/train.sh`; `"$@"` forwards every argument you passed to the wrapper.

### Evaluation Wrapper

```bash
bash scripts/eval.sh outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt +experiment=baseline
```

Evaluates the provided best checkpoint while composing the baseline experiment config.

The first argument is the checkpoint path. Remaining arguments are Hydra
overrides passed to `src/main.py`.

### Preprocess Wrapper

```bash
bash scripts/preprocess.sh --force
```

Generates `data/processed/train.pt`, `val.pt`, and `test.pt` for
`data=tensor_file`. Use it before training with file-backed tensor data.

### Profile Wrapper

CPU profile using `configs/profiler.yaml`:

```bash
bash scripts/profile.sh
```

Runs the profiling wrapper with the default profiler config and writes traces under `outputs/profiles/`. Customize model, data, task, precision, step counts, and trace options in `configs/profiler.yaml`.

CUDA profile:

```bash
PROFILE_CUDA=1 bash scripts/profile.sh
```

Runs the profiling wrapper with CUDA profiling enabled. You can also set `profiler.cuda: true` and `run.device: cuda` in the profiler config.

Alternate profiler config:

```bash
PROFILE_CONFIG=configs/profiler.yaml bash scripts/profile.sh
```

Traces are written to `profiler.trace_dir`, which defaults to `outputs/profiles/`.
See `profiler_tutorial.md` for when to profile, how to inspect the terminal table and TensorBoard trace, and how to turn profiler output into optimization steps.

Trainer-level profiling through the main entrypoint:

```bash
uv run python src/main.py run.mode=profile profiler.active_steps=3 profiler.warmup_steps=1
```

This composes the normal experiment config, builds callbacks/loggers/checkpoints like a regular run, loads model weights and checkpoint counters/metadata when `checkpoint.resume` is configured, profiles a short training-style workload, and writes traces under `outputs/runs/<run.id>/trial_<n>/profiles/`. Profile mode does not restore optimizer, scheduler, scaler, or RNG state. Use an explicit checkpoint path when profiling weights from a different run id.

### W&B Sweep Wrapper

```bash
uv sync --extra tracking
bash scripts/sweep.sh
```

Installs tracking dependencies and launches the W&B sweep defined in `configs/sweep.yaml`.

This creates a W&B sweep from `configs/sweep.yaml` and starts a local agent.

## Makefile Workflows

Install the project, development group, and optional tracking dependencies:

```bash
make install
```

Runs `uv sync`, `uv sync --all-extras`, and `uv sync --extra tracking` in sequence, matching the current Makefile target. The repository currently defines only the `tracking` optional extra, so the final command is redundant after `--all-extras` but documents the target exactly.

Train:

```bash
make train
make train ARGS="+experiment=baseline trainer.max_epochs=1"
```

Runs default training, or forwards the supplied Hydra overrides through `ARGS`.

Run CPU sanity:

```bash
make sanity
```

Runs the CPU sanity workflow to validate the environment and core pipeline.

Evaluate default-config checkpoint:

```bash
make eval CHECKPOINT=outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt
```

Evaluates the checkpoint path supplied via `CHECKPOINT` using the Makefile eval target.

Test and lint:

```bash
make test
make test-cov
make lint
make fmt
```

Runs tests, coverage, lint checks, and formatter commands through the Makefile. The current `lint` and `fmt` targets cover `src`, `tests`, and `scripts/run_sanity.py`; use `uv run ruff check src tests scripts` and `uv run ruff format src tests scripts` when you want every Python utility included.

Clean generated files:

```bash
make clean
```

Deletes `outputs/`, generated `data/processed/`, Python bytecode directories, and pytest/Ruff/mypy caches. This removes run logs, checkpoints, reports, exports, and archives; do not run it while those artifacts are still needed.

## Distributed And Multi-GPU

The repository includes distributed runtime helpers, rank-zero process and backend logging, rank-aware checkpointing, duplicate-free evaluation samplers, weighted metric reduction, rank-gathered prediction export, and automatic `torch.nn.parallel.DistributedDataParallel` wrapping when launched with `torchrun`.

Multi-process CUDA launch:

```bash
uv run torchrun --nproc_per_node=2 src/main.py +experiment=baseline run.device=cuda data.batch_size=64
```

Use the same config you would use for a single-process run. Checkpoint saving and artifact logging are rank-zero gated; dataloaders use `DistributedSampler` when the process group is active.
Training metrics and losses are reduced using global weighted numerators and denominators. Validation/test samplers do not pad with duplicate examples, and prediction shards are gathered before rank zero writes the JSON export.

## Run Outputs

Fresh training/profile artifacts:

```text
outputs/runs/<run.id>/trial_<n>/
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
  profiles/
```

Evaluation/test/predict artifacts are keyed by the source training trial, mode, and checkpoint:

```text
outputs/evaluations/<run.id>/trial_<source_trial>/
  eval_best/
    config.yaml
    logs/
    predictions/test_predictions.json
  eval_last/
  eval_epoch_0005/
  test_best/
  predict_best/
```

Rerunning one exact evaluation target replaces its existing folder. Other checkpoint/mode folders are untouched.

Training configs and registry:

```text
outputs/run_configs/<run.id>/trial_<n>.yaml
outputs/run_registry.jsonl
```

Each registry record stores run id, trial id, checkpoint label when applicable, artifact paths, resolved config, redacted command, and command working directory.

Inspect and replay:

```bash
uv run ml-run-registry list
uv run ml-run-registry show <run_id>
uv run ml-run-registry latest-command
uv run ml-run-registry replay-command <run_id> --new-run-id replayed_run
uv run ml-run-registry resume-command <run_id>
uv run ml-run-registry diff <run_id_a> <run_id_b>

uv run python src/main.py --config-file outputs/run_configs/<run.id>/trial_<n>.yaml
uv run python src/main.py --from-run <run.id>
uv run python src/main.py --from-run <run.id> --run-id replayed_run
```

A config-file/from-run replay is fresh and receives a new code-managed trial. A resume command resolves an existing checkpoint and continues its original trial.

## Post-Run Analysis And Cleanup

Compare runs from registry records and JSONL metrics:

```bash
uv run python scripts/compare_runs.py <run_id_a> <run_id_b> --metrics val/loss val/accuracy
uv run python scripts/compare_runs.py --limit 10 --format markdown --output outputs/reports/compare.md
uv run ml-compare-runs --limit 10 --format csv --output outputs/reports/compare.csv
```

The comparison report includes status, selected config fields, final metric values, and best metric values. Best values use metric-name heuristics: losses and errors are minimized, accuracy-like metrics are maximized.

Plot metric history as an HTML/SVG report plus a tidy CSV:

```bash
uv run python scripts/plot_metrics.py <run_id_a> <run_id_b> --metrics train/loss val/loss --output outputs/reports/loss.html
uv run ml-plot-metrics <run_id> --metrics val/accuracy
```

Evaluate a saved run id from its best checkpoint:

```bash
uv run python scripts/evaluate_run.py <run_id> --checkpoint best
uv run python scripts/evaluate_run.py <run_id> --checkpoint best --print-only
uv run ml-evaluate-run <run_id> --checkpoint best
```

Export a checkpoint for sharing or deployment:

```bash
uv run python scripts/export_checkpoint.py <run_id> --checkpoint best --format state_dict
uv run python scripts/export_checkpoint.py <run_id> --checkpoint best --format checkpoint
uv run python scripts/export_checkpoint.py outputs/runs/<run_id>/trial_<n>/checkpoints/best.pt --format state_dict --output outputs/exports/<run_id>/model.pt
uv run ml-export-checkpoint <run_id> --checkpoint best --format state_dict
```

`state_dict` writes model weights plus minimal metadata. `checkpoint` copies the full training checkpoint. `torchscript` attempts to build the saved model/config and trace one train batch; use it only for model/data combinations that accept tensor-dict tracing cleanly.

List, archive, or clean failed and incomplete runs:

```bash
uv run python scripts/cleanup_runs.py list --unsuccessful
uv run python scripts/cleanup_runs.py cleanup --unsuccessful
uv run python scripts/cleanup_runs.py cleanup --unsuccessful --archive-first --yes
uv run python scripts/cleanup_runs.py archive <run_id> --output-dir outputs/archives
uv run ml-cleanup-runs list --statuses failed incomplete missing
```

`cleanup` is dry-run by default. Pass `--yes` to delete selected run directories. `--unsuccessful` selects failed, incomplete, and missing runs; failed runs include exception checkpoints or train logs with tracebacks/errors. Add `--delete-config` only when you also want to remove the saved config snapshot.
Cleanup status is evaluated per artifact directory, so a newer successful evaluation record cannot hide an older failed training directory with the same run id. A later successful training resume supersedes exception files and traceback text from earlier sessions.

Verify a checkpoint directory and selector checksums:

```bash
uv run python scripts/verify_checkpoints.py outputs/runs/<run_id>/trial_<n>/checkpoints
uv run ml-verify-checkpoints outputs/runs/<run_id>/trial_<n>/checkpoints
```

Config-file mode expects a fully resolved output config, not a Hydra group preset. It removes generated runtime paths and ids before `prepare_run`, then applies `--run-id` and any trailing `key=value` dotlist overrides. `--from-run <run_id>` resolves the saved config through `outputs/run_registry.jsonl`; `--resume-run <run_id>` selects the latest training record and supplies its checkpoint as an explicit path; `--run-id` is rejected for resume. Add `--registry <path>` when using a nondefault registry; generated replay/resume commands include it automatically.

## Common Problems

- `Could not find 'experiment/name'`: check the filename under
  `configs/experiment/` and use `+experiment=<filename_without_yaml>`.
- Checkpoint shape mismatch during eval: use the same model/data/task config
  that created the checkpoint.
- `checkpoint.resume=latest` does not find the intended checkpoint: use `--resume-run <run_id>` or pass an explicit path under `outputs/runs/<run_id>/trial_<n>/checkpoints/`.
- CUDA unavailable: run `uv run python scripts/run_sanity.py +experiment=sanity_gpu`
  or see `sanity_check_commands.md`.
- Missing W&B or TensorBoard package: install tracking extras with
  `uv sync --extra tracking`.
