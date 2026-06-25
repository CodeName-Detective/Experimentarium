"""Config-derived run identity and artifact path preparation.

Fresh training runs receive code-managed trial ids. Resume and evaluation
identity is recovered from checkpoint paths so logs, checkpoints, configs, and
external tracking use the same run/trial naming.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.runtime.distributed import broadcast_object, is_rank0
from src.utils.config import cfg_get, config_to_dict

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - fallback for minimal environments
    DictConfig = None  # type: ignore[misc, assignment]
    OmegaConf = None  # type: ignore[misc, assignment]


_HASH_EXCLUDED_PATHS = (
    ('sanity',),
    ('run', 'id'),
    ('run', 'trial'),
    ('run', 'trial_id'),
    ('run', 'source_trial_id'),
    ('run', 'checkpoint_label'),
    ('run', 'config_id'),
    ('run', 'run_dir'),
    ('run', 'runs_dir'),
    ('run', 'evaluations_dir'),
    ('run', 'config_dir'),
    ('run', 'config_path'),
    ('run', 'config_registry'),
    ('run', 'tracking_id'),
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

_SENSITIVE_ARGUMENT_PATTERN = re.compile(
    r'(?:^|[._-])(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)(?:$|[._-])',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunInfo:
    """Resolved run identity and artifact locations."""

    run_id: str
    trial_id: int
    config_id: str
    run_dir: str
    config_path: str
    config_registry: str
    checkpoint_label: str | None = None
    reused_existing: bool = False
    warning: str | None = None


@dataclass(frozen=True)
class _TrainingTarget:
    run_id: str
    trial_id: int
    run_dir: Path
    runs_dir: Path


def prepare_run(cfg: Any) -> RunInfo:
    """Resolve code-owned identity and rewrite all runtime artifact paths.

    Fresh train/profile invocations atomically allocate the next ``trial_<n>``.
    Training resume reuses the run id and trial encoded in the checkpoint path.
    Evaluation uses a deterministic directory keyed by source trial, mode, and
    checkpoint name; rerunning that exact evaluation replaces its old outputs.

    Returns:
        Resolved run identity and artifact paths.

    Raises:
        FileNotFoundError: If an evaluation checkpoint does not exist.
    """
    raw_config = config_to_dict(cfg)
    config_id = _config_hash(raw_config)
    mode = str(cfg_get(cfg, 'run.mode', 'train')).lower()
    output_dir = _configured_path(cfg, 'run.output_dir', Path('outputs'))
    runs_dir = _configured_path(cfg, 'run.runs_dir', output_dir / 'runs')
    evaluations_dir = _configured_path(cfg, 'run.evaluations_dir', output_dir / 'evaluations')
    config_dir = _configured_path(cfg, 'run.config_dir', output_dir / 'run_configs')
    registry_path = _configured_path(cfg, 'run.config_registry', output_dir / 'run_registry.jsonl')

    explicit_id = cfg_get(cfg, 'run.id', None)
    base_run_id: str | None = _slug(str(explicit_id)) if explicit_id else None
    resume = str(cfg_get(cfg, 'checkpoint.resume', '') or '').strip()
    checkpoint_label: str | None = None
    warning: str | None = None

    if mode == 'train' and resume:
        target = _resolve_training_target(
            resume,
            cfg,
            runs_dir,
            registry_path,
            base_run_id or _derived_run_id(cfg, config_id),
        )
        if _is_explicit_checkpoint_path(resume):
            output_dir, runs_dir, evaluations_dir, config_dir, registry_path = _adopt_checkpoint_root(
                cfg, target, registry_path
            )
        run_id = target.run_id
        trial_id = target.trial_id
        run_dir = target.run_dir
        reused_existing = True
    elif mode in {'eval', 'test', 'predict'} and resume:
        target = _resolve_training_target(
            resume,
            cfg,
            runs_dir,
            registry_path,
            base_run_id or _derived_run_id(cfg, config_id),
        )
        if _is_explicit_checkpoint_path(resume):
            output_dir, runs_dir, evaluations_dir, config_dir, registry_path = _adopt_checkpoint_root(
                cfg, target, registry_path
            )
        checkpoint_path = _resolve_checkpoint_path(resume, target.run_dir)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Evaluation checkpoint does not exist: {checkpoint_path}')
        checkpoint_label = _checkpoint_label(checkpoint_path)
        _cfg_set(cfg, 'checkpoint.resume', str(checkpoint_path))
        run_id = target.run_id
        trial_id = target.trial_id
        run_dir = evaluations_dir / run_id / _trial_dir_name(trial_id) / f'{mode}_{checkpoint_label}'
        reused_existing = _replace_evaluation_dir(run_dir)
        if reused_existing:
            warning = _evaluation_overwrite_warning(run_id, trial_id, mode, checkpoint_label, run_dir)
    else:
        run_id = base_run_id or _derived_run_id(cfg, config_id)
        artifact_root = evaluations_dir if mode in {'eval', 'test', 'predict'} else runs_dir
        trial_id, trial_dir = _select_fresh_trial(artifact_root, run_id, registry_path, mode)
        checkpoint_label = 'uninitialized' if mode in {'eval', 'test', 'predict'} else None
        run_dir = trial_dir / f'{mode}_{checkpoint_label}' if checkpoint_label is not None else trial_dir
        if is_rank0():
            run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(str(broadcast_object(str(run_dir))))
        reused_existing = False

    config_path = (
        run_dir / 'config.yaml'
        if mode in {'eval', 'test', 'predict'}
        else config_dir / run_id / f'{_trial_dir_name(trial_id)}.yaml'
    )
    tracking_id = _tracking_id(run_id, trial_id, mode, checkpoint_label)

    _cfg_set(cfg, 'run.output_dir', str(output_dir))
    _cfg_set(cfg, 'run.id', run_id)
    _cfg_set(cfg, 'run.trial_id', trial_id)
    _cfg_set(cfg, 'run.source_trial_id', trial_id if mode in {'eval', 'test', 'predict'} and resume else None)
    _cfg_set(cfg, 'run.checkpoint_label', checkpoint_label)
    _cfg_set(cfg, 'run.config_id', config_id)
    _cfg_set(cfg, 'run.runs_dir', str(runs_dir))
    _cfg_set(cfg, 'run.evaluations_dir', str(evaluations_dir))
    _cfg_set(cfg, 'run.config_dir', str(config_dir))
    _cfg_set(cfg, 'run.config_path', str(config_path))
    _cfg_set(cfg, 'run.config_registry', str(registry_path))
    _cfg_set(cfg, 'run.tracking_id', tracking_id)
    _cfg_set(cfg, 'run.run_dir', str(run_dir))
    _cfg_set(cfg, 'run.log_dir', str(run_dir / 'logs'))
    _cfg_set(cfg, 'run.prediction_dir', str(run_dir / 'predictions'))
    _cfg_set(cfg, 'run.profile_dir', str(run_dir / 'profiles'))
    _cfg_set(cfg, 'checkpoint.dir', str(run_dir / 'checkpoints'))
    _cfg_set(cfg, 'logging.jsonl.path', str(run_dir / 'logs' / 'metrics.jsonl'))
    _cfg_set(cfg, 'logging.tensorboard.log_dir', str(run_dir / 'logs' / 'tensorboard'))
    _cfg_set(cfg, 'logging.wandb.run_name', tracking_id)

    info = RunInfo(
        run_id=run_id,
        trial_id=trial_id,
        config_id=config_id,
        run_dir=str(run_dir),
        config_path=str(config_path),
        config_registry=str(registry_path),
        checkpoint_label=checkpoint_label,
        reused_existing=reused_existing,
        warning=warning,
    )
    if is_rank0():
        _persist_run_mapping(cfg, info)
    return info


def _configured_path(cfg: Any, key: str, default: Path) -> Path:
    value = cfg_get(cfg, key, None)
    if value is None or str(value).lower() in {'none', 'null'}:
        return default
    return Path(str(value))


def _derived_run_id(cfg: Any, config_id: str) -> str:
    parts = [str(cfg_get(cfg, 'run.name', None) or 'run')]
    for key in ('model.name', 'data.name', 'task.name'):
        value = cfg_get(cfg, key, None)
        if value:
            parts.append(str(value))
    stem = _slug('-'.join(parts)) or 'run'
    return f'{stem}-{config_id}'


def _resolve_training_target(
    resume: str,
    cfg: Any,
    runs_dir: Path,
    registry_path: Path,
    fallback_run_id: str,
) -> _TrainingTarget:
    if _is_explicit_checkpoint_path(resume):
        target = _training_target_from_checkpoint_path(Path(resume).expanduser(), runs_dir)
        if target is None:
            raise ValueError(
                'Explicit checkpoint paths must be under '
                f'{runs_dir}/<run_id>/trial_<n>/checkpoints/ so run identity can be recovered'
            )
        return target
    return _existing_training_trial(runs_dir, registry_path, fallback_run_id)


def _is_explicit_checkpoint_path(value: str) -> bool:
    if '/' in value or '\\' in value or Path(value).is_absolute():
        return True
    return _selector_checkpoint_name(value) is None


def _training_target_from_checkpoint_path(checkpoint_path: Path, runs_dir: Path) -> _TrainingTarget | None:
    del runs_dir  # The checkpoint path is the source of truth for explicit resume/evaluation.
    try:
        resolved = checkpoint_path.resolve()
    except OSError:
        return None
    parts = resolved.parts
    try:
        checkpoint_index = len(parts) - 1 - tuple(reversed(parts)).index('checkpoints')
    except ValueError:
        return None
    if checkpoint_index < 2 or checkpoint_index >= len(parts) - 1:
        return None

    trial_match = re.fullmatch(r'trial_(\d+)', parts[checkpoint_index - 1])
    if trial_match is not None and checkpoint_index >= 3:
        run_id = _slug(parts[checkpoint_index - 2]) or 'run'
        parsed_runs_dir = Path(*parts[: checkpoint_index - 2])
        run_dir = parsed_runs_dir / parts[checkpoint_index - 2] / parts[checkpoint_index - 1]
        return _TrainingTarget(
            run_id=run_id,
            trial_id=int(trial_match.group(1)),
            run_dir=run_dir,
            runs_dir=parsed_runs_dir,
        )

    run_id = _slug(parts[checkpoint_index - 1]) or 'run'
    parsed_runs_dir = Path(*parts[: checkpoint_index - 1])
    run_dir = parsed_runs_dir / parts[checkpoint_index - 1]
    return _TrainingTarget(run_id=run_id, trial_id=1, run_dir=run_dir, runs_dir=parsed_runs_dir)


def _adopt_checkpoint_root(
    cfg: Any,
    target: _TrainingTarget,
    registry_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Use the checkpoint's output tree while preserving explicit child-path overrides."""
    output_dir = target.runs_dir.parent
    evaluations_dir = _configured_path(cfg, 'run.evaluations_dir', output_dir / 'evaluations')
    config_dir = _configured_path(cfg, 'run.config_dir', output_dir / 'run_configs')
    configured_registry = cfg_get(cfg, 'run.config_registry', None)
    if configured_registry is None or str(configured_registry).lower() in {'none', 'null'}:
        registry_path = output_dir / 'run_registry.jsonl'
    return output_dir, target.runs_dir, evaluations_dir, config_dir, registry_path


def _existing_training_trial(runs_dir: Path, registry_path: Path, run_id: str) -> _TrainingTarget:
    base_dir = runs_dir / run_id
    trials = _filesystem_trial_dirs(base_dir)
    if _is_legacy_run_dir(base_dir):
        trials.setdefault(1, base_dir)
    registry_trials = sorted(_registry_trial_ids(registry_path, run_id, is_evaluation=False), reverse=True)
    candidates = [trial_id for trial_id in registry_trials if trial_id in trials]
    candidates.extend(sorted((trial_id for trial_id in trials if trial_id not in candidates), reverse=True))
    for trial_id in candidates:
        run_dir = trials[trial_id]
        if _has_checkpoint_files(run_dir / 'checkpoints'):
            return _TrainingTarget(
                run_id=run_id,
                trial_id=trial_id,
                run_dir=run_dir,
                runs_dir=runs_dir,
            )
    raise FileNotFoundError(f'No checkpoint-bearing training trial found for run_id={run_id}')


def _resolve_checkpoint_path(resume: str, training_run_dir: Path) -> Path:
    if _is_explicit_checkpoint_path(resume):
        return Path(resume).expanduser()
    checkpoint_name = _selector_checkpoint_name(resume)
    if checkpoint_name is None:
        raise ValueError(f'Unsupported checkpoint selector: {resume}')
    return training_run_dir / 'checkpoints' / checkpoint_name


def _selector_checkpoint_name(selector: str) -> str | None:
    normalized = selector.strip().lower()
    if normalized in {'latest', 'last', 'last.pt'}:
        return 'last.pt'
    if normalized in {'best', 'best.pt'}:
        return 'best.pt'
    match = re.fullmatch(r'(?:epoch[_-]?)?(\d+)(?:\.pt)?', normalized)
    if match is not None:
        return f'epoch_{int(match.group(1)):04d}.pt'
    return None


def _checkpoint_label(path: Path) -> str:
    return _slug(path.stem) or 'checkpoint'


def _tracking_id(run_id: str, trial_id: int, mode: str, checkpoint_label: str | None) -> str:
    base = f'{run_id}-trial-{trial_id}'
    if mode in {'eval', 'test', 'predict'}:
        return f'{base}-{mode}-{checkpoint_label or "uninitialized"}'
    return base


def _select_fresh_trial(
    artifact_root: Path,
    run_id: str,
    registry_path: Path,
    mode: str,
) -> tuple[int, Path]:
    if is_rank0():
        result = _allocate_trial(
            artifact_root,
            run_id,
            registry_path,
            is_evaluation=mode in {'eval', 'test', 'predict'},
        )
    else:
        result = (0, artifact_root / run_id)
    payload = broadcast_object((result[0], str(result[1])))
    return int(payload[0]), Path(str(payload[1]))


def _allocate_trial(
    artifact_root: Path,
    run_id: str,
    registry_path: Path,
    *,
    is_evaluation: bool,
) -> tuple[int, Path]:
    base_dir = artifact_root / run_id
    trial_id = _last_trial_id(base_dir, registry_path, run_id, is_evaluation=is_evaluation) + 1
    while True:
        run_dir = base_dir / _trial_dir_name(trial_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return trial_id, run_dir
        except FileExistsError:
            trial_id += 1


def _replace_evaluation_dir(run_dir: Path) -> bool:
    if is_rank0():
        existed = run_dir.exists() or run_dir.is_symlink()
        if run_dir.is_symlink():
            run_dir.unlink()
        elif run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        existed = False
    return bool(broadcast_object(existed))


def _evaluation_overwrite_warning(
    run_id: str,
    trial_id: int,
    mode: str,
    checkpoint_label: str,
    run_dir: Path,
) -> str:
    return (
        'OVERWRITING EXISTING EVALUATION OUTPUT: '
        f'run_id={run_id} trial_id={trial_id} mode={mode} checkpoint={checkpoint_label} directory={run_dir}'
    )


def _last_trial_id(base_dir: Path, registry_path: Path, run_id: str, *, is_evaluation: bool) -> int:
    trial_ids = set(_filesystem_trial_dirs(base_dir))
    trial_ids.update(_registry_trial_ids(registry_path, run_id, is_evaluation=is_evaluation))
    if _is_legacy_run_dir(base_dir):
        trial_ids.add(1)
    return max(trial_ids, default=0)


def _filesystem_trial_dirs(base_dir: Path) -> dict[int, Path]:
    if not base_dir.exists():
        return {}
    trials: dict[int, Path] = {}
    for path in base_dir.iterdir():
        match = re.fullmatch(r'trial_(\d+)', path.name)
        if path.is_dir() and match is not None:
            trials[int(match.group(1))] = path
    return trials


def _registry_trial_ids(registry_path: Path, run_id: str, *, is_evaluation: bool) -> set[int]:
    trial_ids: set[int] = set()
    if not registry_path.exists():
        return trial_ids
    for line in registry_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get('run_id', '')) != run_id:
            continue
        record_is_evaluation = str(cfg_get(record.get('config', {}), 'run.mode', 'train')).lower() in {
            'eval',
            'test',
            'predict',
        }
        if record_is_evaluation != is_evaluation:
            continue
        trial_id = record.get('trial_id', cfg_get(record.get('config', {}), 'run.trial_id', None))
        try:
            trial_ids.add(int(trial_id))
        except (TypeError, ValueError):
            trial_ids.add(1)
    return trial_ids


def _trial_dir_name(trial_id: int) -> str:
    return f'trial_{trial_id}'


def _is_legacy_run_dir(base_dir: Path) -> bool:
    return any((base_dir / name).exists() for name in ('checkpoints', 'logs', 'predictions', 'profiles'))


def _has_checkpoint_files(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.exists() and any(path.is_file() and path.suffix == '.pt' for path in checkpoint_dir.iterdir())


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
    record = {
        **asdict(info),
        'command': _invocation_command(),
        'command_cwd': str(Path.cwd()),
        'config': config,
    }
    with registry_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + '\n')


def _invocation_command() -> str:
    argv = list(getattr(sys, 'orig_argv', ()) or ())
    if not argv:
        argv = [sys.executable, *sys.argv]
    return shlex.join(_redact_sensitive_arguments(argv))


def _redact_sensitive_arguments(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append('<redacted>')
            redact_next = False
            continue
        if '=' in argument:
            key, _ = argument.split('=', maxsplit=1)
            redacted.append(f'{key}=<redacted>' if _SENSITIVE_ARGUMENT_PATTERN.search(key) else argument)
            continue
        redacted.append(argument)
        redact_next = bool(_SENSITIVE_ARGUMENT_PATTERN.search(argument))
    return redacted


def _write_config_yaml(path: Path, config: dict[str, Any]) -> None:
    if OmegaConf is not None:
        OmegaConf.save(config=OmegaConf.create(config), f=str(path))
    else:
        path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding='utf-8')


def _slug(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', value.strip().lower())
    slug = re.sub(r'-+', '-', slug).strip('-_.')
    return slug[:96]
