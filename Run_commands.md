# Run Commands

This file is the command reference for running the framework after the
environment sanity checks pass. It covers training, evaluation, resume, config
overrides, shell wrappers, Makefile shortcuts, profiling, preprocessing, and
tracking workflows.

Use `sanity_check_commands.md` before this file when validating a new machine.

## Entry Points

The main training/evaluation entrypoint is:

```bash
uv run python src/main.py
```

Runs the default Hydra config through `src/main.py`, including run setup, sanity checks, training, checkpointing, test evaluation, and prediction export.

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
make eval CHECKPOINT=outputs/runs/<run_id>/checkpoints/best.pt
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

This forces `run.device=cuda`, strict sanity checks, CUDA driver checks, and a
tiny full trainer run on the GPU.

## Evaluation

Evaluation uses the same `src/main.py` entrypoint with `run.mode=eval`. Evaluation mode skips sanity checks and writes artifacts to `outputs/runs/<run_id>_evaluation/` when using a checkpoint path or selector for an existing training run.

### Evaluate A Specific Checkpoint

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
```

Loads the specified checkpoint, evaluates the matching baseline config, and writes evaluation artifacts under `<run_id>_evaluation`.

The composed model/data/task config must match the checkpoint. If the checkpoint
was trained with `+experiment=baseline`, evaluate with `+experiment=baseline`.

Console-script equivalent:

```bash
uv run ml-train +experiment=baseline run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
```

Performs the same checkpoint evaluation through the installed console script.

Shell equivalent:

```bash
bash scripts/eval.sh outputs/runs/<run_id>/checkpoints/best.pt +experiment=baseline
```

Passes the checkpoint as the first wrapper argument and forwards `+experiment=baseline` to Hydra.

Make equivalent for the default config:

```bash
make eval CHECKPOINT=outputs/runs/<run_id>/checkpoints/best.pt
```

Evaluates the checkpoint through the Makefile target using the default config unless the target is extended with extra overrides.

The current Makefile `eval` target does not pass arbitrary experiment overrides.
Use `bash scripts/eval.sh ... +experiment=...` for experiment-specific eval.

### Evaluate By Checkpoint Selector

Use a selector when the run identity resolves to the training run directory you want. If you used `run.id` or `run.trial` during training, use the same override during evaluation. The evaluation output still goes to `<run_id>_evaluation`.

```bash
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=latest
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=best
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=5
uv run python src/main.py +experiment=baseline run.mode=eval checkpoint.resume=epoch_0005
```

These evaluate the latest/last checkpoint, best checkpoint, or epoch-5 checkpoint from the resolved training run and write results to the `_evaluation` run directory.

Example with a manual training run id:

```bash
uv run python src/main.py +experiment=baseline run.id=my_baseline run.mode=eval checkpoint.resume=best
```

Evaluates the best checkpoint from the manually named training run `my_baseline` and writes artifacts to `my_baseline_evaluation`.

## Resume Training

### Resume From Latest Checkpoint

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=latest
```

Loads the latest valid checkpoint from the baseline run directory and continues training from the next epoch.

This resumes from the latest valid checkpoint in the run directory derived from
the same config identity. Use the same `+experiment`, `run.trial`, and `run.id`
values that created the original run.

### Resume From A Specific Checkpoint

```bash
uv run python src/main.py +experiment=baseline checkpoint.resume=latest
uv run python src/main.py +experiment=baseline checkpoint.resume=best
uv run python src/main.py +experiment=baseline checkpoint.resume=last
uv run python src/main.py +experiment=baseline checkpoint.resume=5
uv run python src/main.py +experiment=baseline checkpoint.resume=epoch_0005
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/checkpoints/last.pt
uv run python src/main.py +experiment=baseline checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
```

These resume training from the selected checkpoint: latest/last, best, epoch 5, or an explicit checkpoint file. If no new epochs run, test logging is skipped to avoid duplicate tracking points.

Use explicit checkpoint paths when you know exactly which run artifact you want.

## Config Override Patterns

Hydra lets you override any config value from the command line.

### Trainer Overrides

```bash
uv run python src/main.py trainer.max_epochs=20
uv run python src/main.py trainer.log_every_n_steps=1
uv run python src/main.py trainer.grad_clip=0.5
uv run python src/main.py trainer.early_stopping.patience=3
```

These override training loop behavior: total epochs, scalar logging frequency, gradient clipping threshold, and early-stopping patience.

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

These disable scheduling or select a learning-rate schedule and tune its key parameters.

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
uv run python src/main.py data.num_workers=4
uv run python src/main.py data.pin_memory=true run.device=cuda
uv run python src/main.py data.splits.train.num_samples=1024 data.splits.val.num_samples=256
```

These tune dataloader throughput and synthetic split sizes; `pin_memory=true` is useful when training on CUDA.

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

Runs on the GPU while keeping full fp32 precision.

CUDA AMP:

```bash
uv run python src/main.py run.device=cuda run.precision=amp
```

Runs on CUDA with automatic mixed precision for faster training and lower memory use.

BF16, if supported by your GPU:

```bash
uv run python src/main.py run.device=cuda run.precision=bf16
```

Uses bfloat16 autocast on compatible GPUs, usually with better numerical range than fp16.

### Checkpoint Overrides

```bash
uv run python src/main.py checkpoint.save_every=1 checkpoint.keep_last_k=3
uv run python src/main.py checkpoint.monitor=val/loss checkpoint.mode=min
uv run python src/main.py checkpoint.save_top_k=2
```

These control checkpoint frequency, retention, and which validation metric decides the best checkpoint.

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

### Run Identity Overrides

Create planned repeats:

```bash
uv run python src/main.py +experiment=baseline run.trial=2
uv run python src/main.py +experiment=baseline run.trial=3
```

Creates separate planned repeat run identities while keeping the rest of the baseline config unchanged.

Use a manual run id:

```bash
uv run python src/main.py +experiment=baseline run.id=my_baseline_debug
```

Uses a manually chosen run id, making output folders and tracking runs easier to find.

If the final run id already exists, the framework reuses that run directory. Fresh duplicate runs emit a warning; intentional training resumes with `checkpoint.resume=...` do not.

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
bash scripts/eval.sh outputs/runs/<run_id>/checkpoints/best.pt +experiment=baseline
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

CPU profile:

```bash
bash scripts/profile.sh
```

Runs the profiling wrapper in CPU mode and writes traces under `outputs/profiles/`.

CUDA profile:

```bash
PROFILE_CUDA=1 bash scripts/profile.sh
```

Runs the profiling wrapper with CUDA profiling enabled.

Traces are written under `outputs/profiles/`.

### W&B Sweep Wrapper

```bash
uv sync --extra tracking
bash scripts/sweep.sh
```

Installs tracking dependencies and launches the W&B sweep defined in `configs/sweep.yaml`.

This creates a W&B sweep from `configs/sweep.yaml` and starts a local agent.

## Makefile Workflows

Install dev dependencies:

```bash
make install
```

Installs the project dependencies needed for local development.

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
make eval CHECKPOINT=outputs/runs/<run_id>/checkpoints/best.pt
```

Evaluates the checkpoint path supplied via `CHECKPOINT` using the Makefile eval target.

Test and lint:

```bash
make test
make test-cov
make lint
make fmt
```

Runs tests, coverage, lint checks, and formatter commands through the Makefile.

Clean generated files:

```bash
make clean
```

Deletes generated outputs and caches, including run artifacts if the Makefile target is configured that way.

`make clean` deletes generated outputs and caches. Do not run it if you still
need checkpoints or run logs.

## Distributed And Multi-GPU

The repository includes distributed runtime helpers, rank-aware logging, and
rank-aware checkpointing. The current trainer does not yet wrap models in
`torch.nn.parallel.DistributedDataParallel`, so treat multi-process training as
an extension task before relying on `torchrun` for gradient synchronization.

After DDP wrapping is implemented, the launch shape should be:

```bash
uv run torchrun --nproc_per_node=2 src/main.py +experiment=baseline run.device=cuda data.batch_size=64
```

Shows the intended future multi-process launch shape after DDP wrapping is implemented.

For now, prefer single-process CUDA runs:

```bash
uv run python src/main.py +experiment=baseline run.device=cuda
```

Runs the baseline on a single CUDA process, which is the currently supported GPU path.

## Run Outputs

Every run writes artifacts under a run-specific directory:

```text
outputs/runs/<run.id>/
  logs/
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
```

Resolved configs are stored in:

```text
outputs/run_configs/<run.id>.yaml
outputs/run_registry.jsonl
```

Use those files to recover the exact command/config for a previous run.

## Common Problems

- `Could not find 'experiment/name'`: check the filename under
  `configs/experiment/` and use `+experiment=<filename_without_yaml>`.
- Checkpoint shape mismatch during eval: use the same model/data/task config
  that created the checkpoint.
- `checkpoint.resume=latest` does not find the intended checkpoint: use the same
  `+experiment`, `run.trial`, and `run.id` values from the original run, or pass
  an explicit checkpoint path under `outputs/runs/<run_id>/checkpoints/`.
- CUDA unavailable: run `uv run python scripts/run_sanity.py +experiment=sanity_gpu`
  or see `sanity_check_commands.md`.
- Missing W&B or TensorBoard package: install tracking extras with
  `uv sync --extra tracking`.
