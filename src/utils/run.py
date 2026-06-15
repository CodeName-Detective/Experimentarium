"""Config-derived run identity and artifact path preparation.

Use this module at process startup before building loggers or checkpoints. It
computes a deterministic config hash, derives a readable run id, writes a
resolved config snapshot, appends a run-to-config registry record, and rewrites
artifact paths so every run has an auditable directory.

Typical usage:
    from src.utils.run import prepare_run
    info = prepare_run(cfg)
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.runtime.distributed import broadcast_object, is_rank0
from src.utils.config import cfg_get, config_to_dict

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - fallback for minimal environments
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


_HASH_EXCLUDED_PATHS = (
    ('run', 'id'),
    ('run', 'config_id'),
    ('run', 'run_dir'),
    ('run', 'runs_dir'),
    ('run', 'config_dir'),
    ('run', 'config_path'),
    ('run', 'config_registry'),
    ('run', 'log_dir'),
    ('run', 'prediction_dir'),
    ('run', 'profile_dir'),
    ('run', 'output_dir'),
    ('run', 'mode'),
    ('checkpoint', 'dir'),
    ('checkpoint', 'resume'),
    ('logging', 'jsonl', 'path'),
    ('logging', 'tensorboard', 'log_dir'),
    ('logging', 'wandb', 'run_name'),
)


@dataclass(frozen=True)
class RunInfo:
    """Resolved run identity and artifact locations."""

    run_id: str
    config_id: str
    run_dir: str
    config_path: str
    config_registry: str
    reused_existing: bool = False
    warning: str | None = None


def prepare_run(cfg: Any) -> RunInfo:
    """Derive run identity from config and rewrite artifact paths.

    The base id is deterministic for the same effective experiment config.
    Reusing an existing id writes artifacts to the same run directory and emits
    a warning instead of selecting ``run_2``, ``run_3``, and so on. Evaluation
    from a checkpoint under ``runs/<run_id>/checkpoints`` writes to
    ``runs/<run_id>_evaluation`` so evaluation artifacts are grouped separately.

    Returns:
        Resolved run identity and artifact paths.
    """
    raw_config = config_to_dict(cfg)
    config_id = _config_hash(raw_config)
    output_dir = Path(str(cfg_get(cfg, 'run.output_dir', 'outputs')))
    runs_dir = Path(str(cfg_get(cfg, 'run.runs_dir', output_dir / 'runs')))
    config_dir = Path(str(cfg_get(cfg, 'run.config_dir', output_dir / 'run_configs')))
    registry_path = Path(str(cfg_get(cfg, 'run.config_registry', output_dir / 'run_registry.jsonl')))
    explicit_id = cfg_get(cfg, 'run.id', None)
    derived_run_id = _derived_run_id(cfg, config_id)
    eval_run_id = _prepare_eval_resume(cfg, runs_dir, explicit_id, derived_run_id)
    if eval_run_id:
        base_run_id = eval_run_id
    elif explicit_id:
        base_run_id = _slug(str(explicit_id)) or 'run'
    else:
        base_run_id = derived_run_id

    run_id, reused_existing = _select_run_id(base_run_id, runs_dir, config_dir)
    run_dir = runs_dir / run_id
    config_path = config_dir / f'{run_id}.yaml'
    warning = _reuse_warning(run_id, run_dir, config_path) if reused_existing and not _is_training_resume(cfg) else None

    _cfg_set(cfg, 'run.id', run_id)
    _cfg_set(cfg, 'run.config_id', config_id)
    _cfg_set(cfg, 'run.runs_dir', str(runs_dir))
    _cfg_set(cfg, 'run.config_dir', str(config_dir))
    _cfg_set(cfg, 'run.config_path', str(config_path))
    _cfg_set(cfg, 'run.config_registry', str(registry_path))
    _cfg_set(cfg, 'run.run_dir', str(run_dir))
    _cfg_set(cfg, 'run.log_dir', str(run_dir / 'logs'))
    _cfg_set(cfg, 'run.prediction_dir', str(run_dir / 'predictions'))
    _cfg_set(cfg, 'run.profile_dir', str(run_dir / 'profiles'))
    _cfg_set(cfg, 'checkpoint.dir', str(run_dir / 'checkpoints'))
    _cfg_set(cfg, 'logging.jsonl.path', str(run_dir / 'logs' / 'metrics.jsonl'))
    _cfg_set(cfg, 'logging.tensorboard.log_dir', str(run_dir / 'logs' / 'tensorboard'))
    if not cfg_get(cfg, 'logging.wandb.run_name', None):
        _cfg_set(cfg, 'logging.wandb.run_name', run_id)

    info = RunInfo(
        run_id=run_id,
        config_id=config_id,
        run_dir=str(run_dir),
        config_path=str(config_path),
        config_registry=str(registry_path),
        reused_existing=reused_existing,
        warning=warning,
    )
    if is_rank0():
        _persist_run_mapping(cfg, info)
    return info


def _derived_run_id(cfg: Any, config_id: str) -> str:
    name = cfg_get(cfg, 'run.name', None) or 'run'
    trial = cfg_get(cfg, 'run.trial', None)
    parts = [str(name)]
    for key in ('model.name', 'data.name', 'task.name'):
        value = cfg_get(cfg, key, None)
        if value:
            parts.append(str(value))
    if trial is not None:
        parts.append(f'trial{trial}')
    stem = _slug('-'.join(parts)) or 'run'
    return f'{stem}-{config_id}'


def _select_run_id(base_run_id: str, runs_dir: Path, config_dir: Path) -> tuple[str, bool]:
    if is_rank0():
        run_id, reused_existing = _resolve_run_id(base_run_id, runs_dir, config_dir)
    else:
        run_id = base_run_id
        reused_existing = False
    payload = broadcast_object((run_id, reused_existing))
    return str(payload[0]), bool(payload[1])


def _resolve_run_id(base_run_id: str, runs_dir: Path, config_dir: Path) -> tuple[str, bool]:
    run_dir = runs_dir / base_run_id
    config_path = config_dir / f'{base_run_id}.yaml'
    reused_existing = run_dir.exists() or config_path.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    return base_run_id, reused_existing


def _is_training_resume(cfg: Any) -> bool:
    return str(cfg_get(cfg, 'run.mode', 'train')).lower() == 'train' and bool(cfg_get(cfg, 'checkpoint.resume', None))


def _prepare_eval_resume(cfg: Any, runs_dir: Path, explicit_id: Any, derived_run_id: str) -> str | None:
    if str(cfg_get(cfg, 'run.mode', 'train')).lower() != 'eval':
        return None

    resume = str(cfg_get(cfg, 'checkpoint.resume', '') or '').strip()
    if not resume:
        return None

    checkpoint_path = Path(resume).expanduser()
    training_run_id = _training_run_id_from_checkpoint_path(checkpoint_path, runs_dir)
    if training_run_id is None:
        training_run_id = _slug(str(explicit_id)) if explicit_id else derived_run_id

    selected_path = _resolve_eval_checkpoint_selector(resume, runs_dir / training_run_id)
    if selected_path is not None:
        _cfg_set(cfg, 'checkpoint.resume', str(selected_path))

    return f'{training_run_id}_evaluation' if training_run_id else None


def _training_run_id_from_checkpoint_path(checkpoint_path: Path, runs_dir: Path) -> str | None:
    try:
        checkpoint_path = checkpoint_path.resolve()
        resolved_runs_dir = runs_dir.expanduser().resolve()
    except OSError:
        return None
    parts = checkpoint_path.parts
    run_parts = resolved_runs_dir.parts
    if parts[: len(run_parts)] != run_parts or len(parts) <= len(run_parts) + 2:
        return None
    if parts[len(run_parts) + 1] != 'checkpoints':
        return None
    return _slug(parts[len(run_parts)]) or None


def _resolve_eval_checkpoint_selector(resume: str, training_run_dir: Path) -> Path | None:
    checkpoint_dir = training_run_dir / 'checkpoints'
    selector = resume.strip().lower()
    if selector in {'latest', 'last'}:
        return checkpoint_dir / 'last.pt'
    if selector == 'best':
        return checkpoint_dir / 'best.pt'
    epoch = _parse_epoch_selector(selector)
    if epoch is not None:
        return checkpoint_dir / f'epoch_{epoch:04d}.pt'
    return None


def _parse_epoch_selector(selector: str) -> int | None:
    match = re.fullmatch(r'(?:epoch[_-]?)?(\d+)(?:\.pt)?', selector)
    if match is None:
        return None
    return int(match.group(1))


def _reuse_warning(run_id: str, run_dir: Path, config_path: Path) -> str:
    return (
        f'WARNING: RUN ID {run_id} ALREADY EXISTS. REUSING EXISTING RUN DIRECTORY '
        f'{run_dir} AND CONFIG SNAPSHOT {config_path}; NEW LOGS, METRICS, PREDICTIONS, '
        'AND TRACKING EVENTS WILL BE WRITTEN TO THIS SAME RUN ID.'
    )


def _config_hash(cfg: dict[str, Any]) -> str:
    normalized = _normalized_for_hash(cfg)
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def _normalized_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(cfg)
    for path in _HASH_EXCLUDED_PATHS:
        _drop_path(normalized, path)
    return normalized


def _drop_path(data: dict[str, Any], path: tuple[str, ...]) -> None:
    cur: Any = data
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if OmegaConf is not None and DictConfig is not None and isinstance(cfg, DictConfig):
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
        return
    if isinstance(cfg, dict):
        cur = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            next_value = cur.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                cur[part] = next_value
            cur = next_value
        cur[parts[-1]] = value
        return
    parts = key.split('.')
    cur = cfg
    for part in parts[:-1]:
        if not hasattr(cur, part) or getattr(cur, part) is None:
            setattr(cur, part, type('ConfigNode', (), {})())
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _persist_run_mapping(cfg: Any, info: RunInfo) -> None:
    config = config_to_dict(cfg)
    config_path = Path(info.config_path)
    registry_path = Path(info.config_registry)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    Path(info.run_dir).mkdir(parents=True, exist_ok=True)
    _write_config_yaml(config_path, config)
    record = {**asdict(info), 'config': config}
    with registry_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + '\n')


def _write_config_yaml(path: Path, config: dict[str, Any]) -> None:
    if OmegaConf is not None:
        OmegaConf.save(config=OmegaConf.create(config), f=str(path))
    else:
        path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding='utf-8')


def _slug(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', value.strip().lower())
    slug = re.sub(r'-+', '-', slug).strip('-_.')
    return slug[:96]
