# Profiler Tutorial

Use this guide when training feels slower than expected, GPU utilization is low, memory use is surprising, or a model/data change needs performance investigation. The profiler answers a specific question: where does time go during a representative workload?

There are two profiler entrypoints. `scripts/profile.sh` loads `configs/profiler.yaml` and profiles a standalone smoke workload, writing traces under `outputs/profiles/` by default. `uv run python src/main.py run.mode=profile` composes a normal experiment config, builds the trainer stack, and profiles a short training-style workload under `outputs/runs/<run.id>/profiles/`.

## When To Use It

Use profiling after the code is correct and reproducible. A profiler run is most useful when you can compare two versions: before and after a model, dataset, transform, precision, or batch-size change.

Good reasons to profile:

- A training step became slower after a code change.
- GPU utilization is low while CPU is busy.
- Data loading, transforms, or collation may be the bottleneck.
- A model has many small tensor operations and poor throughput.
- Mixed precision did not speed up training as expected.
- You need evidence before optimizing code.

Do not start with profiling when the run is failing, non-deterministic, or numerically unstable. Fix correctness first, then profile.

## Quick Start

Run the default CPU profiler smoke workload from `configs/profiler.yaml`:

```bash
bash scripts/profile.sh
```

Run with CUDA profiler activity enabled:

```bash
PROFILE_CUDA=1 bash scripts/profile.sh
```

Run with an alternate profiler config:

```bash
PROFILE_CONFIG=configs/profiler.yaml bash scripts/profile.sh
```

Profile the normal `src/main.py` trainer stack:

```bash
uv run python src/main.py run.mode=profile profiler.warmup_steps=1 profiler.active_steps=3
```

Profile a named experiment with the same model/data/task config you train with:

```bash
uv run python src/main.py +experiment=baseline run.mode=profile run.device=cuda run.precision=amp profiler.active_steps=5
```

Open traces with TensorBoard. Use `outputs/profiles` for the standalone wrapper and `outputs/runs/<run.id>/profiles` for `run.mode=profile`:

```bash
uv run tensorboard --logdir outputs/profiles
```

By default, the wrapper writes traces to `profiler.trace_dir`:

```text
outputs/profiles/
```

The terminal also prints a table similar to:

```text
Name                         Self CPU %      Self CPU      CPU total %     CPU total      # of Calls
aten::linear                 ...             ...           ...             ...            ...
aten::addmm                  ...             ...           ...             ...            ...
```

## What Each Entrypoint Profiles

`scripts/profile.sh` profiles the workload defined in `configs/profiler.yaml`, not your full experiment. The default config is intentionally tiny:

- Model: `mlp` from `MODEL_REGISTRY`
- Data: `toy_classification`
- Task: `classification`
- Batch size: `4`
- Work: one `task.step(...)` plus `loss.backward()`
- Trace destination: `outputs/profiles/`
- CPU profiling: always enabled
- CUDA profiling: enabled with `PROFILE_CUDA=1`, `profiler.cuda: true`, or `run.device: cuda`

This is useful for checking that profiler tooling works and for understanding operator-level output. To optimize a real project, edit `configs/profiler.yaml` or pass `PROFILE_CONFIG=<path>` so the profiler uses your model, dataset, task, batch size, precision, and several representative training steps.

`run.mode=profile` profiles the trainer stack built from the composed Hydra config. It uses the configured model, task, optimizer, precision policy, dataloader split, and profiler settings from the top-level `profiler:` block in `configs/config.yaml` or CLI overrides. It is the better option when you want the profile to match a real experiment config without duplicating that config into `configs/profiler.yaml`.

## Read The Terminal Table

Start with the printed table because it is fast to inspect.

Important columns:

- `Name`: PyTorch operator or profiler event.
- `Self CPU`: time spent inside that operator itself.
- `CPU total`: time spent in that operator plus child calls.
- `CPU time avg`: average time per call.
- `# of Calls`: how often the operator ran.
- CUDA columns, when present: GPU-side kernel time captured for CUDA activity.

How to interpret common patterns:

- High `CPU total`, low `Self CPU`: the parent operation is expensive because its child operations are expensive. Expand the trace in TensorBoard to find the child work.
- High `Self CPU`: that operator itself is consuming host time. Look for inefficient tensor operations, conversions, Python-side work, or synchronization.
- Very high `# of Calls`: the code may be launching many small ops. Fusing operations, batching work, or simplifying Python loops can help.
- CPU-heavy table with low CUDA activity: the GPU may be waiting on data loading, collation, CPU transforms, or small kernel launches.
- CUDA kernels dominate while CPU is low: the model math is probably the bottleneck; optimize architecture, shapes, precision, or batch size.

## Read The TensorBoard Trace

Open TensorBoard and inspect the profiler trace:

```bash
uv run tensorboard --logdir outputs/profiles
```

Useful views:

- Trace timeline: shows CPU work, CUDA kernel launches, gaps, and synchronization points.
- Operator table: aggregates expensive ops across the profiled step.
- Kernel view, when CUDA is enabled: shows GPU kernels and their duration.

What to look for:

- Large blank gaps before GPU kernels: CPU or input pipeline may be starving the GPU.
- Repeated tiny kernels: operations may be too fragmented.
- Long `aten::to`, `copy_`, or transfer events: device transfer or dtype conversion may be costly.
- Expensive loss or metric operations: task code may need flattening, masking, or aggregation cleanup.
- Unexpected shape-heavy ops: data layout or tensor reshaping may be inefficient.

## Optimization Workflow

1. Establish a baseline.

Run the profiler before making performance changes and keep the terminal table or TensorBoard trace directory.

2. Form one hypothesis.

Examples: data transfer is slow, many small ops are hurting throughput, matrix multiplies dominate, CPU transforms are starving the GPU.

3. Make one change.

Examples:

- Increase `data.num_workers` for real datasets.
- Enable `data.pin_memory=true` when training on CUDA.
- Increase batch size if memory allows.
- Use `run.precision=amp` or `run.precision=bf16` on compatible CUDA hardware.
- Move expensive preprocessing out of `__getitem__`.
- Vectorize Python loops into tensor operations.
- Reduce repeated `.to(...)`, `.cpu()`, `.item()`, or dtype conversions inside the step.
- Combine many tiny tensor ops when possible.

4. Profile again.

Compare the same workload. If the bottleneck moved or total step time improved, keep the change. If not, revert or test the next hypothesis.

5. Validate correctness.

After performance changes, rerun focused tests and at least one short training run. Faster incorrect code is not useful.

## CPU Bottleneck Clues

Likely CPU-side issues:

- High time in data transforms, collation, `aten::to`, Python wrappers, or many tiny ops.
- GPU activity has gaps in the TensorBoard timeline.
- Increasing batch size does not increase GPU utilization.

Possible fixes:

- Use more dataloader workers for real datasets.
- Keep `pin_memory=true` for CUDA training.
- Avoid expensive work in `Dataset.__getitem__` when it can be precomputed.
- Prefer tensorized batch operations over per-sample Python loops.
- Use a dataset `collate_fn` for variable-size samples instead of ad hoc work in the training step.

## GPU Bottleneck Clues

Likely GPU-side issues:

- CUDA kernels dominate the trace.
- Large matmul, convolution, attention, or normalization kernels consume most time.
- CPU timeline has little idle time and GPU stays busy.

Possible fixes:

- Try `run.precision=amp` or `run.precision=bf16` on supported hardware.
- Increase batch size to improve GPU occupancy if memory allows.
- Use model dimensions that align well with GPU kernels.
- Remove unnecessary reshapes, transposes, or dtype conversions.
- Reduce model width/depth if the architecture is simply too expensive.

## Data Transfer And Synchronization Clues

Watch for expensive events such as `aten::to`, `copy_`, `.cpu()`, `.item()`, or synchronization-like gaps.

Possible fixes:

- Move tensors to device once through the framework batch transfer path.
- Avoid calling `.item()` inside hot loops except for logging after the step.
- Avoid frequent CPU/GPU round trips in metrics or callbacks.
- Keep prediction serialization out of training hot paths.

## Adapting The Profiler Config For A Real Experiment

The wrapper config is intentionally tiny. For real optimization, either edit `configs/profiler.yaml` and run `PROFILE_CONFIG=<path> bash scripts/profile.sh`, or use `run.mode=profile` with the same Hydra overrides as the experiment you care about. Keep workload-specific profiler choices in YAML or explicit CLI overrides.

Recommended changes:

- Copy the relevant `model`, `data`, and `task` settings from your real experiment into the profiler config.
- Profile the same precision and batch size as the slow run.
- Set `profiler.warmup_steps` to a few unrecorded steps to avoid one-time initialization costs.
- Set `profiler.active_steps` to record several representative steps, not just one.
- For CUDA profiling, set `run.device: cuda` and either `profiler.cuda: true` or `PROFILE_CUDA=1`, then run a GPU sanity check first:

```bash
uv run python scripts/run_sanity.py +experiment=sanity_gpu
```

A more realistic profiler should answer: what is slow in the run I actually care about?

## Limits And Caveats

- The standalone wrapper does not profile `src/main.py`; use `run.mode=profile` when you want the composed trainer stack.
- The default wrapper config profiles one tiny step, so results may not match large real workloads until you customize `configs/profiler.yaml` or use experiment overrides with `run.mode=profile`.
- CUDA work is asynchronous; timeline interpretation is more reliable than isolated CPU timing for GPU-heavy code.
- Profiling adds overhead. Compare profiled runs to profiled runs, not profiled runs to normal training logs.
- `record_shapes=True` helps explain shape-related cost but adds overhead.
- The default config does not record memory history or Python stacks; enable `profiler.profile_memory` or `profiler.with_stack` only when you need that extra detail.

## What To Save In Notes

When investigating performance, record:

- Command used to profile.
- Git commit or working-tree note.
- Device and precision.
- Batch size and dataset.
- Top 5 expensive operators from the terminal table.
- TensorBoard trace path.
- Hypothesis and change tested.
- Before/after step time or throughput.

This keeps profiling evidence actionable instead of becoming a one-off trace that is hard to interpret later.
