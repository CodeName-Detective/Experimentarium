# Callbacks

Callbacks add lifecycle behavior to training without putting optional side effects into `Trainer`. Typical uses include extra metric logging, timing, artifact uploads, profiler control, diagnostics, and exception reporting.

The callback API is defined in `src/callbacks/base.py`. The trainer accepts callback objects through its `callbacks` constructor argument.

## Current Support

Callbacks can be attached in two ways:

- Preferred for normal experiments: add callback configs under the top-level `callbacks:` list. `src/main.py` calls `build_callbacks(cfg)` from `src/callbacks/__init__.py` and passes the resulting objects to `Trainer`.
- Useful for tests or custom scripts: instantiate callback objects directly and pass `callbacks=[...]` to `Trainer`.

Built-in registered callbacks live in `src/callbacks/common.py`:

- `learning_rate_logger`
- `grad_norm_logger`
- `training_timer`
- `checkpoint_artifact_logger`

Example config-driven usage:

```yaml
callbacks:
  - name: learning_rate_logger
    every_n_steps: 10
  - name: training_timer
  - name: checkpoint_artifact_logger
```

Equivalent CLI form:

```bash
uv run python src/main.py callbacks='[{name: learning_rate_logger, every_n_steps: 10}]'
```

## Step By Step: Add A Callback

Prefer this pattern for project callbacks: implement direct hook methods on a small `Callback` subclass, register the class with `@register_callback`, and build it through the registry. That keeps callback construction consistent with models, datasets, tasks, losses, metrics, and optimizers.

1. Pick the hook that matches the lifecycle event.

    Use `on_train_start` for setup, `on_batch_end` for optimizer-step logging, `on_validation_end` for validation summaries, `on_checkpoint_saved` for artifact work, `on_exception` for failure reporting, and `on_train_end` for final cleanup.

2. Create a concrete callback module.

    Example file: `src/callbacks/custom_lr_logger.py`

    ```python
    from src.callbacks.base import Callback
    from src.utils.config import cfg_get
    from src.utils.registry import register_callback


    @register_callback('custom_lr_logger')
    class CustomLearningRateLogger(Callback):
        def __init__(self, cfg) -> None:
            self.every_n_steps = max(1, int(cfg_get(cfg, 'every_n_steps', 1)))

        def on_batch_end(self, trainer, batch_idx: int, metrics: dict[str, float]) -> None:
            if trainer.global_step % self.every_n_steps != 0:
                return
            metrics['train/lr'] = float(trainer.optimizer.param_groups[0]['lr'])
    ```

    `on_batch_end` receives the mutable step `metrics` dictionary. Adding values there lets the trainer log them at its normal `trainer.log_every_n_steps` cadence. If the callback has its own `every_n_steps`, a mutated value is written only on steps where both cadences align; call `trainer.loggers.log_metrics(...)` directly when the callback cadence must be independent.

3. Import the module so registration happens.

    Example edit: `src/callbacks/__init__.py`

    ```python
    from src.callbacks.custom_lr_logger import CustomLearningRateLogger
    ```

    The decorator runs at import time. If the module is never imported, `CALLBACK_REGISTRY` will not know about the callback.

4. Add it to config.

    Use the root `callbacks:` list in `configs/config.yaml`, an experiment YAML, or a CLI override:

    ```yaml
    callbacks:
    - name: custom_lr_logger
        every_n_steps: 10
    ```

    Use a registry key that is not already registered by `src/callbacks/common.py`.

5. Know where the registry builder is wired.

    For the standard training entrypoint, the wiring belongs in `src/main.py`. It is already present there:

    ```python
    from src.callbacks import build_callbacks

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

    That means you usually do not need to edit `src/main.py` for a new callback. You only edit `src/main.py` if you are changing the callback construction policy itself.

6. Direct Python wiring for tests or custom scripts.

    When you are not using `src/main.py`, build through the registry in the file that constructs `Trainer`:

    ```python
    import src.callbacks  # noqa: F401 - imports built-ins so their decorators run.
    from src.utils.registry import CALLBACK_REGISTRY

    callbacks = [
        CALLBACK_REGISTRY.build(
            'learning_rate_logger',
            {'name': 'learning_rate_logger', 'every_n_steps': 10},
        )
    ]

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

7. Add a focused test.

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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_callbacks.py tests/test_training.py -q
uv run ruff check src tests scripts
```

## Available Hooks

Subclass `Callback` and override only the hooks you need:

| Hook | Called when | Arguments |
| --- | --- | --- |
| `on_train_start` | After checkpoint resume and before the epoch loop | `trainer` |
| `on_epoch_start` | After `trainer.current_epoch` is set and before the epoch trains | `trainer`, `epoch` |
| `on_batch_end` | After an optimizer step, scheduler step, and gradient clearing | `trainer`, `batch_idx`, `metrics` |
| `on_validation_end` | After scheduled validation metrics are computed; epoch validation occurs before checkpoint selection | `trainer`, `metrics` |
| `on_checkpoint_saved` | After a checkpoint file is successfully saved | `trainer`, `path` |
| `on_exception` | When training raises an exception, before the exception checkpoint is saved | `trainer`, `exc` |
| `on_train_end` | After successful completion of the training loop | `trainer` |

Important timing details:

- `on_batch_end` means optimization-step end, not dataloader-batch end. With gradient accumulation, it runs only when the optimizer steps.
- Gradients have already been cleared with `zero_grad(set_to_none=True)` before `on_batch_end`; this hook cannot inspect the gradients from that step.
- `on_validation_end` runs when a validation loader exists and either epoch-based or step-based validation is scheduled.
- Epoch-end validation is followed by best-metric updates, checkpoint selection, and early stopping. Step-triggered validation is logged but does not update `trainer.best_metric` or save a checkpoint in the current trainer.
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

For epoch-end validation, the trainer updates `best_metric` after `on_validation_end`. Step-triggered validation does not update it. A callback that needs immediate best-value detection across either cadence should therefore maintain its own value:

```python
import logging

from src.callbacks import Callback


class ValidationReporter(Callback):
    def __init__(self, monitor: str = 'val/loss') -> None:
        self.monitor = monitor
        self.best = float('inf')
        self.logger = logging.getLogger(__name__)

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
        logging.getLogger(__name__).exception(
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


@register_callback('project_lr_logger')
class ProjectLearningRateLogger(Callback):
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

The root-config structure is active today:

```yaml
callbacks:
  - name: learning_rate_logger
    every_n_steps: 10
  - name: grad_norm_logger
    every_n_steps: 10
  - name: training_timer
  - name: checkpoint_artifact_logger
```

`src/callbacks/factory.py` owns `build_callbacks(cfg)`, and `src/main.py` calls it before constructing `Trainer`. Each entry requires `name`; `enabled: false` skips an entry without deleting it from an experiment file. The built-in `grad_norm_logger` only has a value to emit when `trainer.log_gradient_norm=true`; the trainer also includes that metric in its normal step logs, so use the callback only when you need its separate cadence.

```yaml
callbacks:
  - name: learning_rate_logger
    every_n_steps: 10
  - name: checkpoint_artifact_logger
    enabled: false
```

## Distributed Training

Trainer hooks run independently in every distributed process. Framework logger backends suppress duplicate output on non-zero ranks, but callbacks that write files, call external services, or print directly should guard the action:

```python
from src.runtime.distributed import is_rank0

if is_rank0():
    ...
```

Collective operations must be called consistently by all ranks or training can deadlock.

## Error Handling And State

- `CallbackList` does not isolate normal lifecycle-hook failures. Those exceptions fail the training run. Failures inside `on_exception` are logged separately while the original training exception is re-raised.
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
