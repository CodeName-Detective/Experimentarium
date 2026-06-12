"""Generic, fault-tolerant training engine for research experiments.

The trainer owns orchestration only: epochs, optimization, precision, validation,
checkpointing, resume, callbacks, and logging. Task-specific behavior lives in
``src.tasks``. This separation lets you add segmentation, detection, NLP, or
multimodal workloads without editing the engine.

Typical usage:
    trainer = Trainer(cfg, model, task, loaders, optimizer, scheduler, loggers, checkpoints)
    metrics = trainer.fit()
    test_metrics = trainer.test()

Fault tolerance:
    - Saves full state on normal checkpoint intervals.
    - Saves an ``*_exception.pt`` checkpoint on exceptions when enabled.
    - Detects non-finite losses before optimizer steps.
    - Restores model, optimizer, scheduler, scaler, RNG, epoch, and step.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from src.callbacks import Callback, CallbackList
from src.engine.evaluator import Evaluator, move_to_device
from src.optim.schedulers import SchedulerBundle, scheduler_step
from src.tasks import BaseTask
from src.utils.checkpoint import CheckpointManager, get_rng_state
from src.utils.config import cfg_get, config_to_dict
from src.utils.logger import LoggerCollection

_LOG = logging.getLogger('ml_template')


class Trainer:
    """Training loop with AMP, accumulation, clipping, validation, resume, and checkpoints."""

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        task: BaseTask,
        loaders: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: SchedulerBundle | None,
        loggers: LoggerCollection,
        checkpoint_manager: CheckpointManager,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg_get(cfg, 'run.device', cfg_get(cfg, 'device', 'cpu')))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            _LOG.warning('CUDA requested but unavailable; falling back to CPU')
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.task = task
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler or SchedulerBundle(None)
        self.loggers = loggers
        self.checkpoints = checkpoint_manager
        self.callbacks = CallbackList(callbacks)
        self.max_epochs = int(cfg_get(cfg, 'trainer.max_epochs', cfg_get(cfg, 'trainer.epochs', 1)))
        self.accumulate_grad_batches = max(1, int(cfg_get(cfg, 'trainer.accumulate_grad_batches', 1)))
        self.grad_clip = cfg_get(cfg, 'trainer.grad_clip', None)
        self.log_every_n_steps = max(1, int(cfg_get(cfg, 'trainer.log_every_n_steps', 50)))
        self.val_every_n_epochs = max(1, int(cfg_get(cfg, 'trainer.val_every_n_epochs', 1)))
        self.early_patience = int(cfg_get(cfg, 'trainer.early_stopping.patience', cfg_get(cfg, 'trainer.early_stopping_patience', 0)))
        self.check_finite_loss = bool(cfg_get(cfg, 'trainer.check_finite_loss', True))
        self.detect_anomaly = bool(cfg_get(cfg, 'trainer.detect_anomaly', False))
        self.save_on_exception = bool(cfg_get(cfg, 'checkpoint.save_on_exception', True))
        self.precision = str(cfg_get(cfg, 'run.precision', 'fp32'))
        self.use_amp = self.device.type == 'cuda' and self.precision in {'fp16', 'bf16', 'amp'}
        self.amp_dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.precision != 'bf16')
        self.global_step = 0
        self.start_epoch = 1
        self.current_epoch = 0
        self.best_metric: float | None = None
        self.epochs_without_improvement = 0
        self.trained_epochs = 0
        self.monitor = str(cfg_get(cfg, 'checkpoint.monitor', 'val/loss'))

    def fit(self) -> dict[str, float]:
        """Run training and return final train/validation metrics."""

        final_metrics: dict[str, float] = {}
        training_started = False
        self.trained_epochs = 0
        self.resume()
        try:
            training_started = True
            self.callbacks.call('on_train_start', self)
            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                for epoch in range(self.start_epoch, self.max_epochs + 1):
                    self.trained_epochs += 1
                    self.current_epoch = epoch
                    self.callbacks.call('on_epoch_start', self, epoch)
                    self._set_epoch(epoch)
                    train_metrics = self.train_epoch(epoch)
                    metrics = dict(train_metrics)
                    if epoch % self.val_every_n_epochs == 0 and 'val' in self.loaders:
                        evaluator = Evaluator(self.model, self.task, self.device)
                        metrics.update(evaluator.evaluate(self.loaders['val'], prefix='val'))
                        self.callbacks.call('on_validation_end', self, metrics)
                    self.loggers.log_metrics(metrics, step=epoch)
                    final_metrics = metrics
                    monitor_value = metrics.get(self.monitor)
                    improved = self._update_best(monitor_value)
                    self._save_checkpoint(epoch, improved, metrics)
                    if self._should_stop(improved):
                        _LOG.info('Early stopping at epoch %s', epoch)
                        break
                    if self.scheduler.scheduler is not None and self.scheduler.interval == 'epoch':
                        scheduler_step(self.scheduler, monitor_value)
            self.callbacks.call('on_train_end', self)
            return final_metrics
        except BaseException as exc:
            self.callbacks.call('on_exception', self, exc)
            if self.save_on_exception and training_started:
                self._save_exception_checkpoint(final_metrics)
            raise

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return prefixed metrics."""

        self.model.train()
        self.task.reset_metrics()
        total_loss = 0.0
        total_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(self.loaders['train'])
        for batch_idx, batch in enumerate(self.loaders['train'], start=1):
            batch = move_to_device(batch, self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                result = self.task.step(self.model, batch, stage='train')
                if result.loss is None:
                    raise RuntimeError('Task returned no loss during training')
                if self.check_finite_loss and not torch.isfinite(result.loss.detach()).all().item():
                    raise FloatingPointError(f'Non-finite training loss at epoch={epoch} batch={batch_idx}: {result.loss.detach().cpu()}')
                loss = result.loss / self.accumulate_grad_batches
            self.scaler.scale(loss).backward()
            should_step = batch_idx % self.accumulate_grad_batches == 0 or batch_idx == num_batches
            batch_size = int(result.targets.shape[0]) if result.targets is not None else 1
            total_loss += float(result.loss.detach().cpu()) * batch_size
            total_count += batch_size
            if should_step:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.scheduler.scheduler is not None and self.scheduler.interval == 'step':
                    scheduler_step(self.scheduler)
                step_metrics = {'train/loss_step': float(result.loss.detach().cpu())}
                self.callbacks.call('on_batch_end', self, batch_idx, step_metrics)
                if self.global_step % self.log_every_n_steps == 0:
                    self.loggers.log_metrics(step_metrics, step=self.global_step)
        metrics = self.task.compute_metrics()
        metrics['loss'] = total_loss / max(1, total_count)
        return {f'train/{key}': value for key, value in metrics.items()}

    def test(self) -> dict[str, float]:
        """Run test evaluation with the current model state."""

        evaluator = Evaluator(self.model, self.task, self.device)
        return evaluator.evaluate(self.loaders['test'], prefix='test')

    def resume(self) -> None:
        """Resume from ``checkpoint.resume`` when configured."""

        resume = cfg_get(self.cfg, 'checkpoint.resume', None)
        if not resume:
            return
        if str(resume).lower() == 'latest':
            state = self.checkpoints.load_latest(self.model, self.optimizer, self.scheduler.scheduler, self.scaler)
            if state is None:
                return
        else:
            resume_path = self.checkpoints.resolve_resume_path(resume)
            state = self.checkpoints.load(resume_path, self.model, self.optimizer, self.scheduler.scheduler, self.scaler)
        self.start_epoch = int(state.get('epoch', 0)) + 1
        self.current_epoch = int(state.get('epoch', 0))
        self.global_step = int(state.get('global_step', 0))
        self.best_metric = state.get('best_metric')
        if str(cfg_get(self.cfg, 'run.mode', 'train')).lower() == 'train':
            _LOG.info(
                '\033[1;32mRESUMING TRAINING FROM EPOCH %s (checkpoint_epoch=%s global_step=%s)\033[0m',
                self.start_epoch,
                self.current_epoch,
                self.global_step,
            )
        else:
            _LOG.info('Loaded checkpoint from epoch=%s global_step=%s', self.current_epoch, self.global_step)

    def _state_dict(self, epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
        scheduler_state = self.scheduler.scheduler.state_dict() if self.scheduler.scheduler is not None else None
        return {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': scheduler_state,
            'scaler_state': self.scaler.state_dict(),
            'rng_state': get_rng_state(seed_cuda=self.device.type == 'cuda'),
            'metrics': metrics,
            'cfg': config_to_dict(self.cfg),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool, metrics: dict[str, float]) -> None:
        monitor_value = metrics.get(self.monitor)
        path = self.checkpoints.save(self._state_dict(epoch, metrics), epoch=epoch, metric=monitor_value, is_best=is_best)
        if path is not None:
            self.callbacks.call('on_checkpoint_saved', self, path)
            if is_best:
                self.loggers.log_artifact(path, name='best_checkpoint', artifact_type='model', metadata=metrics)

    def _save_exception_checkpoint(self, metrics: dict[str, float]) -> None:
        try:
            state = self._state_dict(epoch=max(0, self.current_epoch), metrics=metrics)
            path = self.checkpoints.save_exception(state, epoch=max(0, self.current_epoch))
            if path is not None:
                _LOG.warning('Saved exception checkpoint: %s', path)
        except Exception as save_exc:
            _LOG.exception('Failed to save exception checkpoint: %s', save_exc)

    def _update_best(self, metric: float | None) -> bool:
        if metric is None or math.isnan(float(metric)):
            return False
        improved = self.checkpoints.is_better(float(metric), self.best_metric)
        if improved:
            self.best_metric = float(metric)
        return improved

    def _should_stop(self, improved: bool) -> bool:
        if self.early_patience <= 0:
            return False
        self.epochs_without_improvement = 0 if improved else self.epochs_without_improvement + 1
        return self.epochs_without_improvement >= self.early_patience

    def _set_epoch(self, epoch: int) -> None:
        sampler = getattr(self.loaders['train'], 'sampler', None)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

    def run(self) -> None:
        """Backward-compatible alias for older code paths."""

        self.fit()
