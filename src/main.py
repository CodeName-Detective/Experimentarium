"""Hydra entrypoint for training, evaluation, and prediction export.

This file is intentionally thin. It composes config, bootstraps registries,
constructs data/model/task/optimizer/scheduler/checkpoint/logging objects, and
then delegates training or evaluation to ``Trainer`` and ``Evaluator``.

Typical usage:
    uv run python src/main.py
    uv run python src/main.py +experiment=sanity_cpu
    uv run python src/main.py run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

from src.data import build_dataloaders
from src.engine import Evaluator, Trainer
from src.optim import build_optimizer, build_scheduler
from src.runtime.distributed import cleanup as cleanup_distributed, setup_from_env
from src.tasks import build_task
from src.utils.checkpoint import CheckpointManager
from src.utils.config import cfg_get
from src.utils.logger import build_loggers
from src.utils.paths import make_output_dirs
from src.utils.registry import MODEL_REGISTRY
from src.utils.run import prepare_run
from src.utils.sanity import bootstrap_registries, run_sanity_checks
from src.utils.seed import setup_reproducibility


@hydra.main(config_path='../configs', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    """Run the configured training or evaluation workflow."""
    setup_from_env(str(cfg_get(cfg, 'run.distributed_backend', 'nccl')))
    run_info = prepare_run(cfg)
    setup_reproducibility(
        int(cfg_get(cfg, 'run.seed', 42)),
        strict=bool(cfg_get(cfg, 'run.deterministic', False)),
        require_cuda=str(cfg_get(cfg, 'run.device', 'cpu')).startswith('cuda'),
    )
    make_output_dirs(cfg)
    bootstrap_registries()
    loggers = build_loggers(cfg)
    logger = logging.getLogger('ml_template')
    if run_info.warning:
        logger.warning('\033[1;33m%s\033[0m', run_info.warning)
    logger.info('run_id=%s config_id=%s run_dir=%s', run_info.run_id, run_info.config_id, run_info.run_dir)
    mode = str(cfg_get(cfg, 'run.mode', 'train'))
    is_checkpoint_resume = bool(cfg_get(cfg, 'checkpoint.resume', None))
    try:
        if mode != 'eval' and not (mode == 'train' and is_checkpoint_resume):
            run_sanity_checks(cfg, strict=bool(cfg_get(cfg, 'sanity.strict', False)))
        loaders = build_dataloaders(cfg)
        model = MODEL_REGISTRY.build(str(cfg_get(cfg, 'model.name', 'mlp')), cfg_get(cfg, 'model'))
        task = build_task(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loaders['train']))

        checkpoint_manager = CheckpointManager(
            Path(str(cfg_get(cfg, 'checkpoint.dir', 'outputs/checkpoints'))),
            save_every=int(cfg_get(cfg, 'checkpoint.save_every', 1)),
            keep_last_k=int(cfg_get(cfg, 'checkpoint.keep_last_k', 5)),
            monitor=str(cfg_get(cfg, 'checkpoint.monitor', 'val/loss')),
            mode=str(cfg_get(cfg, 'checkpoint.mode', 'min')),
            save_last=bool(cfg_get(cfg, 'checkpoint.save_last', True)),
            save_top_k=int(cfg_get(cfg, 'checkpoint.save_top_k', 1)),
        )

        trainer = Trainer(cfg, model, task, loaders, optimizer, scheduler, loggers, checkpoint_manager)
        if mode == 'train':
            trainer.fit()
            if is_checkpoint_resume and trainer.trained_epochs == 0:
                logger.info(
                    'No new training epochs ran after resume; skipping test metrics and prediction export to avoid duplicate logs.'
                )
                return
            test_metrics = trainer.test()
        elif mode == 'eval':
            trainer.resume()
            evaluator = Evaluator(trainer.model, task, trainer.device)
            test_metrics = evaluator.evaluate(loaders['test'], prefix='test')
        else:
            raise ValueError(f'Unknown run.mode={mode}')

        loggers.log_metrics(test_metrics, step=trainer.global_step)
        prediction_limit = int(cfg_get(cfg, 'run.prediction_limit', 100))
        pred_path = Path(str(cfg_get(cfg, 'run.prediction_dir', 'outputs/predictions'))) / 'test_predictions.json'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        records = Evaluator(trainer.model, task, trainer.device).predict(loaders['test'], limit=prediction_limit)
        pred_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    finally:
        loggers.finish()
        cleanup_distributed()


if __name__ == '__main__':
    main()
