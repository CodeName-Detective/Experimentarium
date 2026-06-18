# Repository Flowchart

This file is a visual map of how the framework runs from each entrypoint to final artifacts. Use it when you want to customize the template and need to know which files must change together.

## Entrypoint Map

```mermaid
flowchart TD
    User[User / Researcher]

    User --> UVTrain[uv run python src/main.py]
    User --> ReplayTrain[uv run python src/main.py --config-file outputs/run_configs/run.yaml --run-id replay]
    User --> UVSanity[uv run python scripts/run_sanity.py]
    User --> ShellTrain[bash scripts/train.sh]
    User --> ShellEval[bash scripts/eval.sh CHECKPOINT]
    User --> ShellSanity[bash scripts/run_sanity.py not used directly]
    User --> ShellPreprocess[bash scripts/preprocess.sh]
    User --> ShellProfile[bash scripts/profile.sh]
    User --> ShellSweep[bash scripts/sweep.sh]
    User --> MakeTargets["make install / train / eval / sanity / test / lint"]
    User --> ConsoleScripts[ml-train / ml-sanity]

    ShellTrain --> Main[src/main.py]
    ShellEval --> Main
    UVTrain --> Main
    ReplayTrain --> ReplayConfig[Load resolved YAML and scrub generated paths]
    ReplayConfig --> Main
    ConsoleScripts --> Main
    MakeTargets --> Main

    UVSanity --> SanityScript[scripts/run_sanity.py]
    ConsoleScripts --> SanityScript
    MakeTargets --> SanityScript

    ShellPreprocess --> Preprocess[src/data/preprocess.py]
    ShellProfile --> ProfileConfig[Load configs/profiler.yaml]
    ProfileConfig --> ProfileInline[Profiler workload in scripts/profile.sh]
    ShellSweep --> WandBSweep[wandb sweep configs/sweep.yaml]

    SanityScript --> SanityFlow[Sanity-only flow]
    Main --> MainFlow[Train / Eval flow]
    Preprocess --> TensorFiles[data/processed/*.pt]
    ProfileInline --> ProfileOutputs["profiler.trace_dir, default outputs/profiles"]
    WandBSweep --> Main
```

Notes:

- `scripts/train.sh` is a thin wrapper around `uv run python src/main.py "$@"`; it can pass `--config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run` for replay.
- `scripts/eval.sh` calls `src/main.py` with `run.mode=eval` and `checkpoint.resume=<path>`.
- `scripts/run_sanity.py` only performs environment/config/smoke validation; it does not train.
- `scripts/preprocess.sh` generates toy tensor-file data for `data=tensor_file`.
- `scripts/profile.sh` loads `configs/profiler.yaml` and runs a small profiler workload outside the main trainer.
- `scripts/sweep.sh` creates a W&B sweep from `configs/sweep.yaml`, then each W&B agent run calls `src/main.py`.
- `Makefile` targets mostly forward to the same Python or shell entrypoints.

## Main Training And Evaluation Flow

```mermaid
flowchart TD
    Start[src/main.py Hydra entrypoint]

    Start --> Compose[Hydra composes configs/config.yaml plus overrides]
    Compose --> DDP[setup_from_env: initialize DDP if torchrun env exists]
    DDP --> RunIdentity[src.utils.run.prepare_run]

    RunIdentity --> RunID[derive config_id and stable run_id]
    RunIdentity --> Paths[rewrite mode-specific artifact paths]
    RunIdentity --> ConfigSnapshot["write training config under run_configs or eval config under evaluations"]
    RunIdentity --> RegistryMap[append outputs/run_registry.jsonl with repeat command and cwd]

    Paths --> Seed[setup_reproducibility]
    Seed --> Dirs[make_output_dirs]
    Dirs --> RegistryBootstrap[bootstrap_registries]
    RegistryBootstrap --> Loggers[build_loggers]

    Loggers --> Sanity[run_sanity_checks]
    Sanity --> Data[build_dataloaders]
    Sanity --> RegistryChecks[check registries, packages, disk, paths]

    Data --> Model[MODEL_REGISTRY.build model]
    Model --> Task[build_task]
    Task --> Optimizer[build_optimizer]
    Optimizer --> Scheduler[build_scheduler]
    Scheduler --> CheckpointManager[CheckpointManager]
    CheckpointManager --> Trainer[Trainer]

    Trainer --> Mode{"run.mode"}
    Mode -->|train| Fit[trainer.fit]
    Mode -->|eval| ResumeEval[trainer.resume + Evaluator]

    Fit --> TrainEpoch[train_epoch]
    TrainEpoch --> BatchLoop[forward / loss / backward / optimizer step]
    BatchLoop --> StepScheduler[step scheduler if interval=step]
    TrainEpoch --> Validation[Evaluator on val loader]
    Validation --> EpochScheduler[step scheduler if interval=epoch]
    Validation --> CheckpointSave[save last/best/epoch checkpoint]
    CheckpointSave --> Test[trainer.test]

    ResumeEval --> LoadCheckpoint[CheckpointManager.load or load_latest]
    LoadCheckpoint --> EvalTest[Evaluator on test loader]

    Test --> Predictions[write predictions/test_predictions.json]
    EvalTest --> Predictions
    Predictions --> Finish[loggers.finish and cleanup_distributed]

    Loggers --> ConsoleLog[console + logs/train.log]
    Loggers --> JsonlLog[logs/metrics.jsonl]
    Loggers --> TensorBoard[logs/tensorboard if enabled]
    Loggers --> WandB[W&B if enabled]

    WandB --> WandBConfig["wandb.init config=resolved config"]
    WandB --> WandBArtifact["log resolved config YAML artifact"]
```

## Sanity Check Flow

```mermaid
flowchart TD
    Start[scripts/run_sanity.py]
    Start --> Compose[Hydra compose config]
    Compose --> RunIdentity[prepare_run]
    RunIdentity --> Dirs[make_output_dirs]
    Dirs --> Sanity[run_sanity_checks]

    Sanity --> Python[Python version from pyproject.toml]
    Sanity --> Packages[package versions from pyproject.toml]
    Sanity --> ConfigKeys[required_config_keys from configs/sanity/default.yaml]
    Sanity --> Runtime[CPU/CUDA/DDP visibility plus PyTorch CUDA driver compatibility]
    Sanity --> Writable[run/checkpoint/log/prediction dirs writable]
    Sanity --> Disk[min_disk_gb]
    Sanity --> Registries[model/data/task/optimizer/scheduler registry entries]
    Sanity --> TensorPaths[tensor_file paths if data=tensor_file]
    Sanity --> Smoke[data/model/loss/backward/optimizer/scheduler smoke]
    Sanity --> Experiments[optional compose all experiment configs]

    Python --> Report[SANITY CHECK REPORT]
    Packages --> Report
    ConfigKeys --> Report
    Runtime --> Report
    Writable --> Report
    Disk --> Report
    Registries --> Report
    TensorPaths --> Report
    Smoke --> Report
    Experiments --> Report
```

## Artifact Layout

```mermaid
flowchart TD
    Config[Resolved config] --> Hash[config hash: run.config_id]
    Config --> RunID[stable run.id]
    RunID --> Mode{run.mode}
    RunID --> Registry[outputs/run_registry.jsonl]

    Mode -->|train| ConfigFile["outputs/run_configs/<run_id>.yaml"]
    Mode -->|train| RunDir["outputs/runs/<run_id>/"]
    Mode -->|eval/test/predict| EvalDir["outputs/evaluations/<run_id>/"]
    EvalDir --> EvalConfig[config.yaml]

    RunDir --> Logs[logs/]
    EvalDir --> Logs
    Logs --> TrainLog[train.log]
    Logs --> Metrics[metrics.jsonl]
    Logs --> TB[tensorboard/ if enabled]

    RunDir --> Checkpoints[checkpoints/]
    Checkpoints --> Epoch[epoch_0001.pt]
    Checkpoints --> Last[last.pt]
    Checkpoints --> Best[best.pt]
    Checkpoints --> Manifest[manifest.json]

    RunDir --> Predictions[predictions/test_predictions.json]
    EvalDir --> EvalPredictions[predictions/test_predictions.json]
    RunDir --> Profiles[profiles/]
    EvalDir --> EvalProfiles[profiles/]

    RunID --> TrackingID["run.tracking_id"]
    TrackingID --> WandB["W&B id and default name if enabled"]
    ConfigFile --> WandBArtifact["W&B config artifact if enabled"]
    EvalConfig --> WandBArtifact
```

Key behavior:

- The framework disables Hydra timestamp output folders in `configs/config.yaml`.
- `run.id` starts from a deterministic base id for the same effective config.
- Fresh-run collisions reuse the stable id and emit a warning; intentional training resumes suppress that warning.
- `run.config_id` stays stable for identical configs, so repeated runs remain traceable to the same effective setup.
- Use `run.trial=2` or an explicit `run.id` when you want a different planned base id.
- `checkpoint.resume=latest` keeps the base id so checkpoint lookup targets the existing training run directory.
- Evaluation preserves that id while changing `run.run_dir` to `outputs/evaluations/<run.id>/`.
- `outputs/run_registry.jsonl` records train/profile/eval/test/predict configs, artifact directories, repeat commands, and command working directories.
- `src/main.py --config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run` replays a saved resolved config after regenerating runtime artifact paths.

## Registry And Component Flow

```mermaid
flowchart TD
    Config[Hydra config groups]

    Config --> ModelName[model.name]
    Config --> DataName[data.name]
    Config --> TaskName[task.name]
    Config --> LossName[task.loss.name]
    Config --> MetricNames[task.metrics]
    Config --> OptimName[optimizer.name]
    Config --> SchedName[scheduler.name]

    ModelName --> ModelRegistry[MODEL_REGISTRY]
    DataName --> DatasetRegistry[DATASET_REGISTRY]
    TaskName --> TaskRegistry[TASK_REGISTRY]
    LossName --> LossRegistry[LOSS_REGISTRY]
    MetricNames --> MetricRegistry[METRIC_REGISTRY]
    OptimName --> OptimRegistry[OPTIMIZER_REGISTRY]
    SchedName --> SchedulerRegistry[SCHEDULER_REGISTRY]

    ModelRegistry --> Model[src/models/]
    DatasetRegistry --> Dataset[src/data/]
    TaskRegistry --> Task[src/tasks/]
    LossRegistry --> Loss[src/losses/]
    MetricRegistry --> Metrics[src/metrics/]
    OptimRegistry --> Optim[src/optim/optimizers.py]
    SchedulerRegistry --> Scheduler[src/optim/schedulers.py]

    Model --> Trainer[Trainer]
    Dataset --> Dataloaders[build_dataloaders]
    Task --> Trainer
    Loss --> Task
    Metrics --> Task
    Optim --> Trainer
    Scheduler --> Trainer
    Dataloaders --> Trainer
```

## What To Edit For Common Customizations

### Add A New Dataset

Edit or add:

- `src/data/dataset.py` or a new file under `src/data/`
- `src/data/__init__.py` if the new module must be imported to register itself
- `configs/data/<your_dataset>.yaml`
- `tests/test_dataset.py` or a new focused dataset test
- `README.md` and `Description.md` if it becomes a public template example

Expected path through the framework:

```text
configs/data/<name>.yaml -> data.name -> DATASET_REGISTRY -> build_dataloaders -> Trainer/Evaluator
```

### Add A New Model

Edit or add:

- `src/models/model.py` or a new model file under `src/models/`
- `src/models/__init__.py` if the new module must be imported to register itself
- `configs/model/<your_model>.yaml`
- `tests/test_model.py`

Expected path:

```text
configs/model/<name>.yaml -> model.name -> MODEL_REGISTRY -> src/main.py -> Trainer/Evaluator
```

### Add A New Task Type

Edit or add:

- `src/tasks/task.py` or a new task file under `src/tasks/`
- `src/tasks/__init__.py` if the new module must be imported to register itself
- `configs/task/<your_task>.yaml`
- losses or metrics if the task needs new ones
- task/training tests

Expected path:

```text
configs/task/<name>.yaml -> task.name -> TASK_REGISTRY -> task.step / task.predict_records
```

### Add A New Loss Or Metric

Edit or add:

- `src/losses/losses.py` and `configs/task/*.yaml` for losses
- `src/metrics/metrics.py` and `configs/task/*.yaml` for metrics
- corresponding tests in `tests/test_metrics.py` or a new file

Expected path:

```text
task.loss.name -> LOSS_REGISTRY -> BaseTask.step
task.metrics -> METRIC_REGISTRY -> MetricCollection -> validation/test metrics
```

### Add A New Scheduler Or Optimizer

Edit or add:

- `src/optim/schedulers.py` plus `configs/scheduler/<name>.yaml`
- `src/optim/optimizers.py` plus `configs/optimizer/<name>.yaml`
- `tests/test_schedulers.py` or optimizer-specific tests

Expected path:

```text
configs/scheduler/<name>.yaml -> SCHEDULER_REGISTRY -> Trainer scheduler_step
configs/optimizer/<name>.yaml -> OPTIMIZER_REGISTRY -> Trainer optimizer step
```

### Change Run Output Organization

Edit:

- `src/utils/run.py` for run id generation, hashing exclusions, config snapshots, and registry mapping
- `configs/config.yaml` for run identity defaults and output directory defaults
- `src/utils/paths.py` if new artifact directories should be created automatically
- `src/utils/logger.py` if log file, TensorBoard, JSONL, or W&B behavior changes
- `tests/test_run_identity.py`

Expected path:

```text
Hydra config -> prepare_run -> run.id/config_id/run_dir -> loggers/checkpoints/predictions/W&B
```

### Change Sanity Checks

Edit:

- `src/utils/sanity/core.py` for new checks
- `configs/sanity/default.yaml` for check toggles and required config keys
- `scripts/run_sanity.py` only if the CLI behavior changes
- `tests/test_sanity.py`

Expected path:

```text
scripts/run_sanity.py or src/main.py -> prepare_run -> run_sanity_checks -> SANITY CHECK REPORT
```

### Change W&B Behavior

Edit:

- `configs/logging/default.yaml` for default project/entity/tags/mode
- `src/utils/logger.py` for W&B initialization, config logging, and artifacts
- `src/utils/run.py` if W&B names should derive from different config fields

Expected path:

```text
logging.wandb.* + run.id -> WandBLogger -> wandb.init(config=resolved_config) -> config artifact
```

## Minimal Debug Checklist

When something breaks, follow this order:

1. Check composed config: `uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true`.
2. Check the run id and config snapshot under `outputs/run_configs/<run_id>.yaml`.
3. Check `outputs/run_registry.jsonl` for the config-to-run mapping, repeat command, and command working directory.
4. Check local logs under `outputs/runs/<run_id>/logs/`.
5. Check checkpoints under `outputs/runs/<run_id>/checkpoints/`.
6. Run focused tests for the component you changed.
7. Run the full validation checklist from `README.md`.
