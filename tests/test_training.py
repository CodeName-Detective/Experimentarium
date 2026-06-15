from pathlib import Path

from src.data import build_dataloaders
from src.engine import Trainer
from src.models import MLP
from src.optim import build_optimizer, build_scheduler
from src.tasks import build_task
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
