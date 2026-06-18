# Sanity Check Commands

This file is a command reference for validating the framework after cloning it,
copying it to a new machine, changing dependencies, editing configs, or before
starting a long experiment.

The canonical CLI entrypoint is:

```bash
uv run python scripts/run_sanity.py
```

If the project is installed from `pyproject.toml`, the console script is also
available:

```bash
uv run ml-sanity
```

Both commands compose the Hydra config, prepare a run directory, validate the
environment, check registries and paths, and run a tiny data/model/loss/backward
smoke test unless you disable it.

## What The Report Means

- `PASS`: the check passed.
- `WARN`: the check found a non-blocking issue.
- `FAIL`: the check found a blocking issue. With `sanity.strict=true`, failures
  raise an error and the command exits unsuccessfully.

Package severity comes from `pyproject.toml`:

- `[project].dependencies` are required.
- `[project.optional-dependencies]` are warning-only optional checks.
- If `pyproject.toml` is missing, package version checks are skipped with a
  warning because there is no dependency source of truth.

## Why Some Reports Look Similar

Most commands in this file call the same core `run_sanity_checks()` pipeline:
Python/package checks, config-key checks, runtime checks, writable output paths,
registry checks, data/model/task smoke checks, and optimizer/scheduler smoke
checks. When your machine is healthy, many lines will therefore look identical.

The commands differ by what config they compose and what extra policy they turn
on:

- `sanity.torch_install.recommend=true`: prints install recommendations; useful before or after installing PyTorch.
- `+experiment=sanity_cpu`: CPU-only framework smoke test.
- `sanity.cuda.check_driver=true`: CUDA diagnostics only, without changing experiment preset.
- `+experiment=sanity_gpu`: CUDA diagnostics plus a tiny GPU smoke check because it sets `run.device=cuda`.
- `src/main.py +experiment=sanity_gpu`: full trainer path on GPU, not just the sanity checker.
- `sanity.check_all_experiments=true`: config composition coverage for every experiment YAML.
- `sanity.wandb.check=true`: W&B import, login/API-key, and network readiness checks.

You do not need to run both `sanity.cuda.check_driver=true` and
`+experiment=sanity_gpu` every time. Run the standalone CUDA diagnostic when you
only want driver/PyTorch compatibility. Run `+experiment=sanity_gpu` when you
want CUDA compatibility plus a tiny GPU model smoke test.

## Everyday Commands

### Default Sanity Check

```bash
uv run python scripts/run_sanity.py

uv run python scripts/run_sanity.py run.id=my_debug_sanity_run
```

Uses the default composed config. Run this after dependency changes or before a
normal training session.

### Tiny CPU New-Machine Check

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu
```

Uses `configs/experiment/sanity_cpu.yaml`. This is the safest first command on a
fresh machine because it avoids GPU requirements and uses a tiny synthetic
classification workload.

### Strict CPU Sanity Check

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.strict=true
```

Treats non-warning failures as hard errors. Use this in CI or before expensive
runs when you want a failing exit code for broken required checks.

### Check Every Experiment Config

```bash
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true
```

Composes every file in `configs/experiment/` to catch broken config references,
missing groups, or invalid override paths. This does not train every experiment.

## CUDA And PyTorch Checks

### Show CUDA Driver Compatibility

```bash
uv run python scripts/run_sanity.py sanity.cuda.check_driver=true
```

Forces the CUDA/NVIDIA driver compatibility check. The report shows the PyTorch
version, `torch.version.cuda`, `torch.cuda.is_available()`, `nvidia-smi` driver
version, GPU name, and the minimum known driver for the installed PyTorch CUDA
build.

### GPU Sanity Check With Hard Failure

```bash
uv run python scripts/run_sanity.py +experiment=sanity_gpu
```

Uses `configs/experiment/sanity_gpu.yaml`. CUDA or driver problems become
failures because the experiment requests `run.device=cuda`, enables strict
sanity checks, forces CUDA driver compatibility checks, and runs the tiny model
smoke pass on the CUDA device.

### Ad Hoc GPU Sanity Check

```bash
uv run python scripts/run_sanity.py run.device=cuda sanity.strict=true
```

Use this when you want to validate CUDA with the default experiment instead of
the dedicated tiny GPU preset.

### CPU Mode But Fail On CUDA Mismatch

```bash
uv run python scripts/run_sanity.py sanity.cuda.check_driver=true sanity.cuda.fail_on_cpu_mismatch=true sanity.strict=true
```

By default, CUDA/driver mismatches in CPU mode are warnings. Use this command
when you want a CPU-mode sanity check to fail if the installed PyTorch CUDA build
does not work with the local NVIDIA driver.

### Disable CUDA Driver Check

```bash
uv run python scripts/run_sanity.py sanity.cuda.check_driver=false
```

Skips CUDA driver compatibility checks unless torch-install recommendation mode
is enabled. Use this on CPU-only machines where `nvidia-smi` is unavailable and
you only care about non-GPU checks.

### Recommend PyTorch Install Command

```bash
uv run python scripts/run_sanity.py sanity.torch_install.recommend=true
```

Prints recommended UV and pip commands for the detected machine. This path is
designed to work before PyTorch is installed: it reads Python requirements from
`pyproject.toml` and GPU/driver information from `nvidia-smi`.

## W&B Checks

### Check W&B Readiness Explicitly

```bash
uv run python scripts/run_sanity.py sanity.wandb.check=true
```

Checks that `wandb` imports, credentials are available, and the machine can
reach the W&B host. Credentials are detected from `WANDB_API_KEY` or the
`.netrc` entry created by `uv run wandb login`. The network check opens a TCP
connection to `WANDB_BASE_URL` when set, otherwise `api.wandb.ai:443`. It does
not print or store the API key.

If Hydra reports `Key 'wandb' is not in struct`, the copied project has an older
`configs/sanity/default.yaml` that does not define `sanity.wandb` yet. Add this
block under `configs/sanity/default.yaml`:

```yaml
wandb:
  # auto checks only when logging.wandb.enabled=true; use true to force checks or false to disable.
  check: auto
  # auto checks network reachability when W&B checks are active and mode is online; false skips internet probing.
  check_connectivity: auto
  # TCP connection timeout in seconds for api.wandb.ai or WANDB_BASE_URL.
  timeout_seconds: 5.0
```

For a one-time check before updating the copied config, append the missing keys
with Hydra's `+` syntax:

```bash
uv run python scripts/run_sanity.py +sanity.wandb.check=true +sanity.wandb.check_connectivity=auto +sanity.wandb.timeout_seconds=5.0
```

### Check W&B As Part Of A Run Config

```bash
uv run python scripts/run_sanity.py logging.wandb.enabled=true logging.wandb.project=my-project
```

When `logging.wandb.enabled=true`, W&B readiness checks run automatically. Missing
credentials or blocked network are failures because the configured run would try
to log online.

### Offline W&B Mode

```bash
uv run python scripts/run_sanity.py logging.wandb.enabled=true logging.wandb.mode=offline
```

Offline mode still checks that `wandb` can be imported, but it does not require
an API key or internet access.

### Skip W&B Network Probe

```bash
uv run python scripts/run_sanity.py logging.wandb.enabled=true sanity.wandb.check_connectivity=false
```

Use this when credentials should be checked but the current machine intentionally
has no internet access.

## Smoke-Test Controls

### Skip Model/Data Smoke Test

```bash
uv run python scripts/run_sanity.py sanity.run_model_smoke=false
```

Skips data construction, model forward pass, loss computation, backward pass,
optimizer step, and scheduler step. Use only when dependency/config checks are
needed but model code is intentionally incomplete.

### Increase Disk-Space Requirement

```bash
uv run python scripts/run_sanity.py sanity.min_disk_gb=20
```

Fails or warns if the output location has less than the requested free space.
Use this before long runs that will write checkpoints, predictions, TensorBoard
logs, or W&B artifacts.

## Profiling Usage

Profiling is a separate diagnostic workflow and is not part of the
`run_sanity_checks()` report. The profiling wrapper loads `configs/profiler.yaml`,
runs the configured forward/backward workload, prints the most expensive
operators by CPU time, and writes a TensorBoard-compatible trace.

### Profile On CPU

```bash
bash scripts/profile.sh
```

CPU profiling is the default in `configs/profiler.yaml`. Customize the workload
there when you want a different model, data config, task, precision, or number
of recorded steps.

### Enable CUDA Profiling

```bash
PROFILE_CUDA=1 bash scripts/profile.sh
```

This enables CUDA profiler activity in addition to CPU activity and moves the
profiled model/batch to CUDA when available. You can also set
`profiler.cuda: true` and `run.device: cuda` in `configs/profiler.yaml`. Run
the GPU sanity check first if CUDA compatibility has not already been validated:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_gpu
```

### Inspect Profiler Traces

By default, profiler traces are written under:

```text
outputs/profiles/
```

Open them with TensorBoard:

```bash
uv run tensorboard --logdir outputs/profiles
```

For result interpretation and optimization workflow, see `profiler_tutorial.md`.

Use the wrapper for a quick profiler/toolchain check. For profiler coverage of
the normal composed trainer stack, use the main entrypoint:

```bash
uv run python src/main.py run.mode=profile profiler.active_steps=3 profiler.warmup_steps=1
```

This writes traces under `outputs/runs/<run.id>/profiles/`. For alternate standalone wrapper settings, either edit `configs/profiler.yaml` or run `PROFILE_CONFIG=<path> bash scripts/profile.sh`.

## Run-Specific Overrides

### Validate A Different Experiment

```bash
uv run python scripts/run_sanity.py +experiment=baseline
```

Composes the selected experiment and runs the sanity checks against its model,
data, task, optimizer, scheduler, output paths, and smoke-test behavior.

### Validate A Specific Device And Precision

```bash
uv run python scripts/run_sanity.py run.device=cuda run.precision=amp sanity.strict=true
```

Checks that the selected runtime settings are compatible with the machine before
using the same overrides for training.

### Validate A Manual Run ID

```bash
uv run python scripts/run_sanity.py run.id=my_debug_sanity_run
```

Uses a predictable run ID so the generated sanity output directory is easy to
find. If the ID already exists, the framework reuses that directory and emits a
warning before appending new artifacts there.

## Training Entry Sanity

### Run Main With The CPU Sanity Experiment

```bash
uv run python src/main.py +experiment=sanity_cpu
```

This is different from `scripts/run_sanity.py`: it runs the normal training
entrypoint after pre-flight checks. Use it when you want to verify that the full
trainer, logger, checkpoint, and evaluator path works end to end on CPU.

### Run Main With The GPU Sanity Experiment

```bash
uv run python src/main.py +experiment=sanity_gpu
```

Runs the full trainer, validation, test, checkpoint, logger, and prediction path
on CUDA using a tiny synthetic workload. Use this after the CUDA driver sanity
check passes.

### Train On CUDA After Sanity Checks

```bash
uv run python src/main.py +experiment=baseline run.device=cuda sanity.strict=true
```

The main entrypoint runs sanity checks before training. Use strict mode so broken
required checks stop the training run immediately.

## Recommended New-Machine Sequence

Run these in order on a fresh machine:

```bash
uv run python scripts/run_sanity.py sanity.torch_install.recommend=true
uv run python scripts/run_sanity.py +experiment=sanity_cpu
uv run python scripts/run_sanity.py +experiment=sanity_gpu
uv run python src/main.py +experiment=sanity_gpu
uv run python scripts/run_sanity.py +experiment=sanity_cpu sanity.check_all_experiments=true
```

If the first command recommends a different PyTorch wheel than the one currently
installed, fix the environment first, then rerun the sequence. If you only want
to inspect driver compatibility without the GPU smoke preset, run
`uv run python scripts/run_sanity.py sanity.cuda.check_driver=true`.

## Python API

Use the Python API when you want to call sanity checks from a custom script:

```python
from src.utils.sanity import run_sanity_checks

report = run_sanity_checks(cfg, strict=True)
if not report.passed:
    raise RuntimeError('sanity checks failed')
```

You can also pass extra checks:

```python
from src.utils.sanity import SanityReport, run_sanity_checks


def check_my_dataset(report: SanityReport, cfg) -> None:
    report.add('custom.dataset_contract', True, 'dataset contract looks valid')


report = run_sanity_checks(cfg, strict=True, extra_checks=[check_my_dataset])
```

## Common Fixes

- Missing required package: add it under `[project].dependencies`, run `uv sync`,
  then rerun sanity.
- Missing optional package: install the relevant extra, for example
  `uv sync --extra tracking`, or leave it as a warning if you do not need it.
- W&B API key missing: run `uv run wandb login` or set `WANDB_API_KEY`; use
  `logging.wandb.mode=offline` when you intentionally do not want online logging.
- CUDA unavailable: run `sanity.torch_install.recommend=true`, install the
  recommended PyTorch wheel, then rerun `sanity.cuda.check_driver=true`.
- Broken experiment config: run `sanity.check_all_experiments=true`, fix the
  referenced config group or override, then rerun the command.
- Smoke-test failure: check the selected dataset, model dimensions, task loss,
  target keys, optimizer, and scheduler config.
