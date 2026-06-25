import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.engine import Trainer
from src.models import MLP
from src.optim import build_optimizer, build_scheduler
from src.optim.schedulers import SchedulerBundle
from src.tasks import build_task
from src.tasks.task import BaseTask, StepResult
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import LoggerCollection


class NoopLogger:
    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path, name, artifact_type='artifact', metadata=None):
        pass

    def finish(self):
        pass


def test_trainer_runs_one_epoch(tmp_path, tiny_cfg):
    tiny_cfg['checkpoint']['dir'] = str(tmp_path)
    loaders = build_dataloaders(tiny_cfg)
    model = MLP(tiny_cfg['model'])
    task = build_task(tiny_cfg)
    optimizer = build_optimizer(model, tiny_cfg)
    scheduler = build_scheduler(tiny_cfg, optimizer, steps_per_epoch=len(loaders['train']))
    trainer = Trainer(
        tiny_cfg,
        model,
        task,
        loaders,
        optimizer,
        scheduler,
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )
    metrics = trainer.fit()
    assert 'val/loss' in metrics
    assert Path(tmp_path, 'best.pt').exists()


def test_trainer_resume_at_max_epoch_runs_no_new_epochs(tmp_path, tiny_cfg):
    tiny_cfg['checkpoint']['dir'] = str(tmp_path)
    loaders = build_dataloaders(tiny_cfg)
    model = MLP(tiny_cfg['model'])
    task = build_task(tiny_cfg)
    optimizer = build_optimizer(model, tiny_cfg)
    scheduler = build_scheduler(tiny_cfg, optimizer, steps_per_epoch=len(loaders['train']))
    manager = CheckpointManager(tmp_path)
    trainer = Trainer(tiny_cfg, model, task, loaders, optimizer, scheduler, LoggerCollection([NoopLogger()]), manager)
    trainer.fit()

    resume_cfg = dict(tiny_cfg)
    resume_cfg['checkpoint'] = dict(tiny_cfg['checkpoint'])
    resume_cfg['checkpoint']['resume'] = 'latest'
    reloaded = MLP(resume_cfg['model'])
    optimizer = build_optimizer(reloaded, resume_cfg)
    scheduler = build_scheduler(resume_cfg, optimizer, steps_per_epoch=len(loaders['train']))
    resumed = Trainer(
        resume_cfg, reloaded, task, loaders, optimizer, scheduler, LoggerCollection([NoopLogger()]), manager
    )
    resumed.fit()

    assert resumed.trained_epochs == 0


def test_trainer_honors_max_steps_and_logs_optional_step_metrics(tmp_path, tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['checkpoint']['dir'] = str(tmp_path)
    cfg['trainer']['max_steps'] = 1
    cfg['trainer']['log_gradient_norm'] = True
    cfg['trainer']['log_learning_rate'] = True
    loaders = build_dataloaders(cfg)
    model = MLP(cfg['model'])
    task = build_task(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loaders['train']))
    trainer = Trainer(
        cfg,
        model,
        task,
        loaders,
        optimizer,
        scheduler,
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )

    trainer.fit()

    assert trainer.global_step == 1
    assert trainer.trained_epochs == 1


def test_trainer_integer_limit_train_batches_means_exact_batches(tmp_path, tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['checkpoint']['dir'] = str(tmp_path)
    cfg['trainer']['limit_train_batches'] = 1
    loaders = build_dataloaders(cfg)
    model = MLP(cfg['model'])
    task = build_task(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loaders['train']))
    trainer = Trainer(
        cfg,
        model,
        task,
        loaders,
        optimizer,
        scheduler,
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )

    trainer.fit()

    assert trainer.global_step == 1


class CaptureLogger(NoopLogger):
    def __init__(self):
        self.events = []

    def log_metrics(self, metrics, step=None):
        self.events.append((step, dict(metrics)))


def test_step_validation_restores_training_mode_metrics_and_monotonic_steps(tmp_path, tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['checkpoint']['dir'] = str(tmp_path)
    cfg['trainer']['val_every_n_steps'] = 1
    cfg['trainer']['limit_train_batches'] = 2
    cfg['trainer']['limit_val_batches'] = 1
    loaders = build_dataloaders(cfg)

    class RecordingMLP(MLP):
        def __init__(self, model_cfg):
            super().__init__(model_cfg)
            self.forward_modes = []

        def forward(self, batch):
            self.forward_modes.append(self.training)
            return super().forward(batch)

    model = RecordingMLP(cfg['model'])
    task = build_task(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loaders['train']))
    capture = CaptureLogger()
    trainer = Trainer(
        cfg, model, task, loaders, optimizer, scheduler, LoggerCollection([capture]), CheckpointManager(tmp_path)
    )

    metrics = trainer.fit()

    assert model.forward_modes[:3] == [True, False, True]
    assert trainer.model.training
    assert 'train/accuracy' in metrics
    steps = [step for step, _ in capture.events if step is not None]
    assert steps == sorted(steps)


def test_partial_gradient_accumulation_uses_actual_window_size(tmp_path):
    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class ScalarTask(BaseTask):
        def __init__(self):
            super().__init__({'metrics': []})

        def step(self, model, batch, stage):
            return StepResult(loss=model.weight, outputs={}, targets=torch.ones(1), loss_weight=1)

    cfg = {
        'run': {'device': 'cpu', 'precision': 'fp32', 'mode': 'train'},
        'trainer': {
            'max_epochs': 1,
            'accumulate_grad_batches': 4,
            'log_every_n_steps': 10,
            'val_every_n_epochs': 1,
            'val_every_n_steps': 0,
            'early_stopping': {'patience': 0},
        },
        'checkpoint': {'monitor': 'train/loss', 'mode': 'min', 'save_on_exception': False},
    }
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = Trainer(
        cfg,
        model,
        ScalarTask(),
        {'train': [{}, {}]},
        optimizer,
        SchedulerBundle(None),
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )

    trainer.fit()

    torch.testing.assert_close(model.weight.detach(), torch.tensor(0.0))


def test_epoch_scheduler_state_is_saved_after_advancement(tmp_path, tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['checkpoint']['dir'] = str(tmp_path)
    cfg['trainer']['limit_train_batches'] = 1
    cfg['scheduler'] = {'name': 'step', 'interval': 'epoch', 'step_size': 1, 'gamma': 0.1, 'monitor': 'val/loss'}
    loaders = build_dataloaders(cfg)
    model = MLP(cfg['model'])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loaders['train']))
    trainer = Trainer(
        cfg,
        model,
        build_task(cfg),
        loaders,
        optimizer,
        scheduler,
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )

    trainer.fit()
    saved = torch.load(tmp_path / 'last.pt', map_location='cpu', weights_only=False)

    assert saved['optimizer_state']['param_groups'][0]['lr'] == trainer.optimizer.param_groups[0]['lr']
    assert saved['scheduler_state']['last_epoch'] == trainer.scheduler.scheduler.last_epoch


def test_logger_collection_finishes_remaining_backends_after_failure():
    events = []

    class FailingBackend(NoopLogger):
        def finish(self):
            raise RuntimeError('finish failed')

    class RecordingBackend(NoopLogger):
        def finish(self):
            events.append('finished')

    LoggerCollection([FailingBackend(), RecordingBackend()]).finish()

    assert events == ['finished']


def test_evaluation_resume_loads_model_without_optimizer_state(tmp_path, tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['checkpoint']['dir'] = str(tmp_path)
    loaders = build_dataloaders(cfg)
    trained_model = MLP(cfg['model'])
    trained_optimizer = build_optimizer(trained_model, cfg)
    trained = Trainer(
        cfg,
        trained_model,
        build_task(cfg),
        loaders,
        trained_optimizer,
        build_scheduler(cfg, trained_optimizer, len(loaders['train'])),
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )
    trained.fit()

    eval_cfg = deepcopy(cfg)
    eval_cfg['run']['mode'] = 'eval'
    eval_cfg['checkpoint']['resume'] = 'latest'
    eval_model = MLP(eval_cfg['model'])
    incompatible_optimizer = torch.optim.SGD([next(eval_model.parameters())], lr=0.1)
    evaluator = Trainer(
        eval_cfg,
        eval_model,
        build_task(eval_cfg),
        loaders,
        incompatible_optimizer,
        SchedulerBundle(None),
        LoggerCollection([NoopLogger()]),
        CheckpointManager(tmp_path),
    )

    evaluator.resume()

    assert all(
        torch.equal(left, right)
        for left, right in zip(trained_model.state_dict().values(), eval_model.state_dict().values(), strict=True)
    )


def test_main_entrypoint_does_not_run_sanity_checks(tmp_path):
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            'src/main.py',
            '+experiment=sanity_cpu',
            f'run.output_dir={tmp_path / "outputs"}',
            'run.id=no-automatic-sanity',
            'trainer.limit_train_batches=1',
            'trainer.limit_val_batches=1',
            'trainer.limit_test_batches=1',
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'SANITY CHECK REPORT' not in output
    assert 'INFO __main__:' in output
    assert 'RUN ID: no-automatic-sanity' in output
