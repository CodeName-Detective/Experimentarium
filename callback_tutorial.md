# Callbacks

Callbacks add lifecycle behavior to training without putting optional side effects into `Trainer`. Typical uses include extra metric logging, timing, artifact uploads, profiler control, diagnostics, and exception reporting.

The callback API is defined in `src/callbacks/base.py`. The trainer accepts callback objects through its `callbacks` constructor argument.

## Current Support

Callbacks work when passed directly to `Trainer`:

```python
trainer = Trainer(
    cfg,
    model,
    task,
    loaders,
    optimizer,
    scheduler,
    loggers,
    checkpoint_manager,
    callbacks=[LearningRateLogger(), ValidationReporter()],
)
```

`src/main.py` does not currently build callbacks from Hydra configuration. Registering a callback makes it available through `CALLBACK_REGISTRY`, but does not instantiate or attach it to the trainer automatically. See [Configurable Callbacks](#configurable-callbacks) for the required integration.

## Step By Step: Add A Callback

Prefer this pattern for project callbacks: implement direct hook methods on a small `Callback` subclass, register the class with `@register_callback`, and build it through `CALLBACK_REGISTRY`. That keeps callback code consistent with the rest of the framework registries instead of scattering ad hoc callback construction through training scripts.

1. Pick the hook that matches the lifecycle event.

Use `on_train_start` for setup, `on_batch_end` for optimizer-step logging, `on_validation_end` for validation summaries, `on_checkpoint_saved` for artifact work, `on_exception` for failure reporting, and `on_train_end` for final cleanup.

2. Create a concrete callback module.

Example file: `src/callbacks/learning_rate_logger.py`

```python
from src.callbacks.base import Callback
from src.utils.config import cfg_get
from src.utils.registry import register_callback


@register_callback('learning_rate_logger')
class LearningRateLogger(Callback):
    def __init__(self, cfg) -> None:
        self.every_n_steps = int(cfg_get(cfg, 'every_n_steps', 1))

    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        trainer.loggers.log_metrics(
            {'train/lr': float(trainer.optimizer.param_groups[0]['lr'])},
            step=trainer.global_step,
        )
```

3. Import the module so registration happens.

Example edit: `src/callbacks/__init__.py`

```python
from src.callbacks.base import Callback, CallbackList
from src.callbacks.learning_rate_logger import LearningRateLogger

__all__ = ['Callback', 'CallbackList', 'LearningRateLogger']
```

The decorator runs at import time. If the module is never imported, `CALLBACK_REGISTRY` will not know about the callback.

4. Build the callback through the registry where `Trainer` is constructed.

For the standard training entrypoint, add this wiring in `src/main.py` after the core components are built and before `trainer = Trainer(...)`. For a one-off experiment script or a test, put the same wiring in that script at the point where it constructs `Trainer`.

If you put this in `src/main.py`, import the callback registry near the other registry imports:

```python
from src.utils.registry import CALLBACK_REGISTRY, MODEL_REGISTRY
```

Then, in the body of `src/main.py`, build the callbacks before `Trainer` is created:

```python
callbacks = [
    CALLBACK_REGISTRY.build(
        'learning_rate_logger',
        {'name': 'learning_rate_logger', 'every_n_steps': 10},
    )
]
```

Then pass it to `Trainer` in the same file:

```python
trainer = Trainer(
    cfg,
    model,
    task,
    loaders,
    optimizer,
    scheduler,
    loggers,
    checkpoint_manager,
    callbacks=callbacks,
)
```

5. For Hydra-driven callbacks, add a config shape and builder.

One simple root config shape is:

```yaml
callbacks:
  - name: learning_rate_logger
    every_n_steps: 10
```

Builder:

```python
from src.utils.config import cfg_get
from src.utils.registry import CALLBACK_REGISTRY


def build_callbacks(cfg):
    callback_configs = list(cfg_get(cfg, 'callbacks', []) or [])
    return [
        CALLBACK_REGISTRY.build(
            str(cfg_get(callback_cfg, 'name')),
            callback_cfg,
        )
        for callback_cfg in callback_configs
    ]
```

Then call `build_callbacks(cfg)` in `src/main.py` before constructing `Trainer` and pass `callbacks=callbacks`. Until this wiring exists, YAML callback settings are inert.

6. Add a focused test.

Use a tiny recording callback or a fake logger and assert the expected hook fired. Keep the test small; callback tests should verify lifecycle wiring, not retrain a real model.

```python
from src.callbacks import Callback


class RecordingCallback(Callback):
    def __init__(self) -> None:
        self.events = []

    def on_train_start(self, trainer) -> None:
        self.events.append('train_start')

    def on_train_end(self, trainer) -> None:
        self.events.append('train_end')
```

Expected validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_training.py -q
uv run ruff check src tests scripts/run_sanity.py
```

## Available Hooks

Subclass `Callback` and override only the hooks you need:

| Hook | Called when | Arguments |
| --- | --- | --- |
| `on_train_start` | After checkpoint resume and before the epoch loop | `trainer` |
| `on_epoch_start` | After `trainer.current_epoch` is set and before the epoch trains | `trainer`, `epoch` |
| `on_batch_end` | After an optimizer step, scheduler step, and gradient clearing | `trainer`, `batch_idx`, `metrics` |
| `on_validation_end` | After validation metrics are computed and before checkpoint selection | `trainer`, `metrics` |
| `on_checkpoint_saved` | After a checkpoint file is successfully saved | `trainer`, `path` |
| `on_exception` | When training raises an exception, before the exception checkpoint is saved | `trainer`, `exc` |
| `on_train_end` | After successful completion of the training loop | `trainer` |

Important timing details:

- `on_batch_end` means optimization-step end, not dataloader-batch end. With gradient accumulation, it runs only when the optimizer steps.
- Gradients have already been cleared with `zero_grad(set_to_none=True)` before `on_batch_end`; this hook cannot inspect the gradients from that step.
- `on_validation_end` runs only when a validation loader exists and validation is scheduled for that epoch.
- `trainer.best_metric` is updated after `on_validation_end`.
- `on_train_end` does not run when training exits through an exception.

## Trainer State

Callbacks receive the live trainer object. Commonly useful attributes are:

- `trainer.model`
- `trainer.task`
- `trainer.optimizer`
- `trainer.scheduler`
- `trainer.loggers`
- `trainer.checkpoints`
- `trainer.loaders`
- `trainer.cfg`
- `trainer.device`
- `trainer.current_epoch`
- `trainer.global_step`
- `trainer.best_metric`

Prefer callbacks for observation and optional side effects. Mutating optimizer, model, or trainer state is possible, but creates lifecycle coupling and should be covered by focused tests.

## Example: Learning-Rate Logging

This callback logs optimizer learning rates after each optimizer step:

```python
from src.callbacks import Callback


class LearningRateLogger(Callback):
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        values = {
            f'train/lr_group_{index}': float(group['lr']) for index, group in enumerate(trainer.optimizer.param_groups)
        }
        trainer.loggers.log_metrics(values, step=trainer.global_step)
```

The `metrics` argument currently contains `train/loss_step`. It can be combined with callback metrics:

```python
values = dict(metrics)
values['train/lr'] = float(trainer.optimizer.param_groups[0]['lr'])
trainer.loggers.log_metrics(values, step=trainer.global_step)
```

## Example: Total Training Time

```python
import time

from src.callbacks import Callback


class TrainingTimer(Callback):
    def on_train_start(self, trainer) -> None:
        self.started_at = time.perf_counter()

    def on_train_end(self, trainer) -> None:
        elapsed = time.perf_counter() - self.started_at
        trainer.loggers.log_metrics(
            {'runtime/train_seconds': elapsed},
            step=trainer.global_step,
        )
```

Implement `on_exception` too if failed-run timing is required.

## Example: Validation Monitoring

The trainer updates `best_metric` after `on_validation_end`, so a callback that needs immediate best-value detection should maintain its own value:

```python
import logging

from src.callbacks import Callback


class ValidationReporter(Callback):
    def __init__(self, monitor: str = 'val/loss') -> None:
        self.monitor = monitor
        self.best = float('inf')
        self.logger = logging.getLogger('ml_template')

    def on_validation_end(
        self,
        trainer,
        metrics: dict[str, float],
    ) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return
        if value < self.best:
            self.best = value
            self.logger.info(
                'New callback best: %s=%.6f at epoch=%s',
                self.monitor,
                value,
                trainer.current_epoch,
            )
```

Reverse the comparison for metrics that should be maximized.

## Example: Log Every Saved Checkpoint

The trainer already logs its selected best checkpoint as a model artifact. This callback logs every checkpoint emitted by the checkpoint manager:

```python
from pathlib import Path

from src.callbacks import Callback


class CheckpointArtifactLogger(Callback):
    def on_checkpoint_saved(self, trainer, path: Path) -> None:
        trainer.loggers.log_artifact(
            path,
            name=f'checkpoint-epoch-{trainer.current_epoch}',
            artifact_type='checkpoint',
            metadata={
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
            },
        )
```

## Example: Exception Reporting

```python
import logging

from src.callbacks import Callback


class ExceptionReporter(Callback):
    def on_exception(self, trainer, exc: BaseException) -> None:
        logging.getLogger('ml_template').exception(
            'Training failed at epoch=%s global_step=%s: %s',
            trainer.current_epoch,
            trainer.global_step,
            exc,
        )
```

The original exception is re-raised after callback handling and exception-checkpoint handling.

## Registering A Callback

Registration is useful when callback construction will be driven by config:

```python
from src.callbacks import Callback
from src.utils.config import cfg_get
from src.utils.registry import register_callback


@register_callback('learning_rate_logger')
class LearningRateLogger(Callback):
    def __init__(self, cfg) -> None:
        self.every_n_steps = int(cfg_get(cfg, 'every_n_steps', 1))

    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        trainer.loggers.log_metrics(
            {'train/lr': float(trainer.optimizer.param_groups[0]['lr'])},
            step=trainer.global_step,
        )
```

The module containing the decorated class must be imported before calling `CALLBACK_REGISTRY.build(...)`. A common approach is to import concrete callback modules from `src/callbacks/__init__.py`.

## Configurable Callbacks

One possible root-config structure is:

```yaml
callbacks:
  - name: learning_rate_logger
    every_n_steps: 10
  - name: checkpoint_artifact_logger
```

Add a builder:

```python
from src.utils.config import cfg_get
from src.utils.registry import CALLBACK_REGISTRY


def build_callbacks(cfg):
    callback_configs = list(cfg_get(cfg, 'callbacks', []) or [])
    return [
        CALLBACK_REGISTRY.build(
            str(cfg_get(callback_cfg, 'name')),
            callback_cfg,
        )
        for callback_cfg in callback_configs
    ]
```

Then update `src/main.py`:

```python
callbacks = build_callbacks(cfg)

trainer = Trainer(
    cfg,
    model,
    task,
    loaders,
    optimizer,
    scheduler,
    loggers,
    checkpoint_manager,
    callbacks=callbacks,
)
```

Until this builder is added to `src/main.py`, callback YAML has no effect.

## Distributed Training

Trainer hooks run independently in every distributed process. Framework logger backends suppress duplicate output on non-zero ranks, but callbacks that write files, call external services, or print directly should guard the action:

```python
from src.runtime.distributed import is_rank0

if is_rank0():
    ...
```

Collective operations must be called consistently by all ranks or training can deadlock.

## Error Handling And State

- `CallbackList` does not isolate callback failures. An exception raised by a callback fails the training run.
- Callback-owned state is not included in checkpoints automatically.
- A resumed run creates new callback objects unless the application restores their state explicitly.
- Keep callbacks small and independent. Put loss, target, and prediction semantics in tasks rather than callbacks.

## Testing A Callback

Use a recording callback in a focused trainer test:

```python
from src.callbacks import Callback


class RecordingCallback(Callback):
    def __init__(self) -> None:
        self.events = []

    def on_train_start(self, trainer) -> None:
        self.events.append('train_start')

    def on_validation_end(self, trainer, metrics) -> None:
        self.events.append(('validation_end', dict(metrics)))

    def on_train_end(self, trainer) -> None:
        self.events.append('train_end')
```

Pass it through `callbacks=[callback]`, run one short epoch, and assert the expected event order and metric values.
