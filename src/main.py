"""Hydra entrypoint for training, evaluation, and prediction export.

This file is intentionally thin. It composes config, bootstraps registries,
constructs data/model/task/optimizer/scheduler/checkpoint/logging objects, and
then delegates training or evaluation to ``Trainer`` and ``Evaluator``.

Typical usage:
    uv run python src/main.py
    uv run python src/main.py +experiment=sanity_cpu
    uv run python src/main.py run.mode=eval checkpoint.resume=outputs/runs/<run_id>/checkpoints/best.pt
    uv run python src/main.py --config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import build_dataloaders
from src.engine import Evaluator, Trainer
from src.optim import build_optimizer, build_scheduler
from src.runtime.distributed import cleanup as cleanup_distributed, setup_from_env
from src.tasks import build_task
from src.utils.checkpoint import CheckpointManager
from src.utils.config import cfg_get, config_to_dict, load_config
from src.utils.logger import build_loggers
from src.utils.paths import make_output_dirs
from src.utils.registry import MODEL_REGISTRY
from src.utils.run import prepare_run
from src.utils.sanity import bootstrap_registries, run_sanity_checks
from src.utils.seed import setup_reproducibility

_REPLAY_CONFIG_FLAGS = {'--config-file', '--run-config', '--replay-config'}
_REPLAY_RUN_ID_FLAGS = {'--run-id', '--replay-run-id'}
_GENERATED_REPLAY_PATHS = (
    'run.id',
    'run.config_id',
    'run.run_dir',
    'run.runs_dir',
    'run.evaluations_dir',
    'run.config_dir',
    'run.config_path',
    'run.config_registry',
    'run.tracking_id',
    'run.log_dir',
    'run.prediction_dir',
    'run.profile_dir',
    'checkpoint.dir',
    'logging.jsonl.path',
    'logging.tensorboard.log_dir',
)


def run(cfg: Any) -> None:
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
            evaluator = Evaluator(trainer.model, task, trainer.device, precision=trainer.precision)
            test_metrics = evaluator.evaluate(loaders['test'], prefix='test')
        else:
            raise ValueError(f'Unknown run.mode={mode}')

        loggers.log_metrics(test_metrics, step=trainer.global_step)
        prediction_limit = int(cfg_get(cfg, 'run.prediction_limit', 100))
        pred_path = Path(str(cfg_get(cfg, 'run.prediction_dir', 'outputs/predictions'))) / 'test_predictions.json'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        records = Evaluator(trainer.model, task, trainer.device, precision=trainer.precision).predict(
            loaders['test'], limit=prediction_limit
        )
        pred_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    finally:
        loggers.finish()
        cleanup_distributed()


@hydra.main(config_path='../configs', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    """Run from the standard Hydra-composed config."""
    run(cfg)


def run_from_config_file(argv: list[str] | None = None) -> None:
    """Run from a resolved YAML config file plus optional dotlist overrides."""
    config_file, overrides = _extract_config_file_args(list(sys.argv[1:] if argv is None else argv))
    if config_file is None:
        raise ValueError('Config-file mode requires --config-file <path>')
    cfg = load_replay_config(config_file, overrides)
    run(cfg)


def load_replay_config(config_file: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a resolved run config for replaying an experiment.

    Generated runtime paths and ids from saved output configs are removed before
    overrides are applied so prepare_run can derive fresh artifact paths. If you
    want a specific replay id, pass ``--run-id <id>`` or ``run.id=<id>``.

    Returns:
        A DictConfig ready to pass to the shared run workflow.
    """
    cfg = config_to_dict(load_config(config_file))
    _clear_generated_replay_fields(cfg)
    cfg = OmegaConf.create(cfg)
    normalized_overrides = [_normalize_replay_override(override) for override in (overrides or [])]
    _validate_replay_overrides(normalized_overrides)
    if normalized_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalized_overrides))
    _sync_legacy_aliases(cfg)
    return cfg


def _has_config_file_arg(argv: list[str]) -> bool:
    return any(
        arg in _REPLAY_CONFIG_FLAGS or any(arg.startswith(f'{flag}=') for flag in _REPLAY_CONFIG_FLAGS) for arg in argv
    )


def _extract_config_file_args(argv: list[str]) -> tuple[str | None, list[str]]:
    config_file: str | None = None
    replay_run_id: str | None = None
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        matched_config_flag = _match_flag(arg, _REPLAY_CONFIG_FLAGS)
        matched_run_id_flag = _match_flag(arg, _REPLAY_RUN_ID_FLAGS)
        if matched_config_flag is not None:
            if config_file is not None:
                raise ValueError('Only one replay config file can be supplied')
            config_file, idx = _consume_flag_value(argv, idx, matched_config_flag, value_name='path')
            continue
        if matched_run_id_flag is not None:
            if replay_run_id is not None:
                raise ValueError('Only one replay run id can be supplied')
            replay_run_id, idx = _consume_flag_value(argv, idx, matched_run_id_flag, value_name='run id')
            continue
        remaining.append(arg)
        idx += 1
    if replay_run_id is not None:
        remaining.append(f'run.id={replay_run_id}')
    return config_file, remaining


def _match_flag(argument: str, flags: set[str]) -> str | None:
    return next((flag for flag in flags if argument == flag or argument.startswith(f'{flag}=')), None)


def _consume_flag_value(argv: list[str], idx: int, flag: str, value_name: str) -> tuple[str, int]:
    argument = argv[idx]
    if argument == flag:
        if idx + 1 >= len(argv):
            raise ValueError(f'{flag} requires a {value_name} argument')
        value = argv[idx + 1]
        next_idx = idx + 2
    else:
        value = argument.split('=', maxsplit=1)[1]
        next_idx = idx + 1
    if not value:
        raise ValueError(f'{flag} requires a non-empty {value_name}')
    return value, next_idx


def _normalize_replay_override(override: str) -> str:
    if override.startswith('++'):
        return override[2:]
    if override.startswith('+'):
        return override[1:]
    return override


def _validate_replay_overrides(overrides: list[str]) -> None:
    invalid = [override for override in overrides if '=' not in override]
    if invalid:
        joined = ', '.join(invalid)
        raise ValueError(f'Config-file mode accepts key=value dotlist overrides only; invalid: {joined}')


def _clear_generated_replay_fields(cfg: dict[str, Any]) -> None:
    run_id = cfg_get(cfg, 'run.id', None)
    tracking_id = cfg_get(cfg, 'run.tracking_id', None)
    wandb_run_name = cfg_get(cfg, 'logging.wandb.run_name', None)
    generated_wandb_names = {
        str(value) for value in (run_id, tracking_id, f'{run_id}_evaluation' if run_id else None) if value
    }
    if wandb_run_name is not None and str(wandb_run_name) in generated_wandb_names:
        _drop_replay_path(cfg, 'logging.wandb.run_name')
    for path in _GENERATED_REPLAY_PATHS:
        _drop_replay_path(cfg, path)


def _drop_replay_path(cfg: dict[str, Any], path: str) -> None:
    cur: Any = cfg
    parts = path.split('.')
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _sync_legacy_aliases(cfg: DictConfig) -> None:
    if cfg_get(cfg, 'run.seed', None) is not None:
        OmegaConf.update(cfg, 'seed', cfg_get(cfg, 'run.seed'), merge=False, force_add=True)
    if cfg_get(cfg, 'run.device', None) is not None:
        OmegaConf.update(cfg, 'device', cfg_get(cfg, 'run.device'), merge=False, force_add=True)


if __name__ == '__main__':
    if _has_config_file_arg(sys.argv[1:]):
        run_from_config_file()
    else:
        main()
