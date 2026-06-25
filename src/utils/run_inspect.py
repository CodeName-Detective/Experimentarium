"""Helpers for inspecting saved run artifacts and registry records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import cfg_get

DEFAULT_REGISTRY_PATH = Path('outputs/run_registry.jsonl')


@dataclass(frozen=True)
class MetricPoint:
    """One scalar metric value from a JSONL metrics log."""

    run_id: str
    metric: str
    step: int | None
    value: float


def read_registry_records(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Read run registry records from a JSONL file."""
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    return [json.loads(line) for line in registry_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest registry record for each run id, preserving latest order."""
    by_run: dict[str, dict[str, Any]] = {}
    for record in records:
        run_id = str(record.get('run_id', ''))
        if run_id:
            by_run[run_id] = record
    return list(by_run.values())


def latest_record_for_run(records: list[dict[str, Any]], run_id: str, mode: str | None = None) -> dict[str, Any]:
    """Return the newest registry record for one run id."""
    matches = [
        record
        for record in records
        if str(record.get('run_id')) == run_id
        and (mode is None or str(cfg_get(record.get('config', {}), 'run.mode', 'train')).lower() == mode.lower())
    ]
    if not matches:
        raise KeyError(f'run_id not found in registry: {run_id}' + (f' for mode={mode}' if mode else ''))
    return matches[-1]


def config_path_for_run(
    run_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH, mode: str | None = None
) -> Path:
    """Resolve the saved config path for a run id."""
    records = read_registry_records(registry_path)
    if records:
        try:
            record = latest_record_for_run(records, run_id, mode=mode)
            config_path = record.get('config_path')
            if config_path:
                path = _record_path(record, str(config_path))
                if path.exists():
                    return path
        except KeyError:
            pass
    config_root = Path('outputs/run_configs') / run_id
    trial_configs = sorted(
        config_root.glob('trial_*.yaml'),
        key=lambda path: _trial_id_from_name(path.stem),
    )
    if trial_configs:
        return trial_configs[-1]
    legacy_fallback = Path('outputs/run_configs') / f'{run_id}.yaml'
    if legacy_fallback.exists():
        return legacy_fallback
    raise FileNotFoundError(f'No saved config found for run_id={run_id}')


def config_path_for_record(record: dict[str, Any]) -> Path | None:
    """Return a registry config path resolved against its original working directory."""
    value = record.get('config_path')
    return _record_path(record, str(value)) if value else None


def training_run_dir_for_record(record: dict[str, Any]) -> Path:
    """Return the exact training trial directory associated with a registry record."""
    run_dir = record.get('run_dir')
    if run_dir:
        return _record_path(record, str(run_dir))
    config = record.get('config', {})
    configured = cfg_get(config, 'run.run_dir', None)
    if configured:
        return _record_path(record, str(configured))
    run_id = str(record.get('run_id', ''))
    runs_dir = _record_path(record, str(cfg_get(config, 'run.runs_dir', 'outputs/runs')))
    trial_id = int(record.get('trial_id', cfg_get(config, 'run.trial_id', 1)))
    return runs_dir / run_id / f'trial_{trial_id}'


def run_dir_for_record(record: dict[str, Any]) -> Path:
    """Return the artifact directory recorded for a registry record."""
    run_dir = record.get('run_dir')
    if run_dir:
        return _record_path(record, str(run_dir))
    config = record.get('config', {})
    configured = cfg_get(config, 'run.run_dir', None)
    return _record_path(record, str(configured)) if configured else training_run_dir_for_record(record)


def checkpoint_path_for_run(
    run_id: str,
    selector: str = 'best',
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Resolve a checkpoint selector or explicit path for a run id."""
    value = str(selector)
    explicit = Path(value).expanduser()
    if explicit.exists() or explicit.suffix == '.pt':
        return explicit
    records = read_registry_records(registry_path)
    record = latest_record_for_run(records, run_id, mode='train')
    checkpoint_dir = training_run_dir_for_record(record) / 'checkpoints'
    normalized = value.lower()
    if normalized == 'latest':
        normalized = 'last'
    if normalized in {'best', 'last'}:
        return checkpoint_dir / f'{normalized}.pt'
    if normalized.startswith('epoch_') and normalized.endswith('.pt'):
        return checkpoint_dir / normalized
    if normalized.startswith('epoch_'):
        normalized = normalized.removeprefix('epoch_')
    if normalized.isdigit():
        return checkpoint_dir / f'epoch_{int(normalized):04d}.pt'
    return checkpoint_dir / value


def metrics_path_for_record(record: dict[str, Any]) -> Path:
    """Return the JSONL metrics path associated with a registry record."""
    config = record.get('config', {})
    configured = cfg_get(config, 'logging.jsonl.path', None)
    if configured:
        return _record_path(record, str(configured))
    return run_dir_for_record(record) / 'logs' / 'metrics.jsonl'


def read_metric_points(record: dict[str, Any]) -> list[MetricPoint]:
    """Read scalar metric points for a registry record."""
    path = metrics_path_for_record(record)
    if not path.exists():
        return []
    run_id = str(record.get('run_id', ''))
    points: list[MetricPoint] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        metrics = payload.get('metrics')
        if not isinstance(metrics, dict):
            continue
        step = payload.get('step')
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            points.append(MetricPoint(run_id=run_id, metric=str(name), step=step, value=float(value)))
    return points


def final_metric_values(points: list[MetricPoint]) -> dict[str, float]:
    """Return the last logged value for each metric."""
    values: dict[str, float] = {}
    for point in points:
        values[point.metric] = point.value
    return values


def best_metric_values(points: list[MetricPoint]) -> dict[str, float]:
    """Return the best value for each metric using name-based min/max heuristics."""
    values: dict[str, float] = {}
    for point in points:
        previous = values.get(point.metric)
        if previous is None:
            values[point.metric] = point.value
        elif _metric_should_minimize(point.metric):
            values[point.metric] = min(previous, point.value)
        else:
            values[point.metric] = max(previous, point.value)
    return values


def flatten(value: Any, prefix: str = '') -> dict[str, Any]:
    """Flatten a nested dictionary into dotted keys."""
    if not isinstance(value, dict):
        return {prefix: value}
    items: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f'{prefix}.{key}' if prefix else str(key)
        items.update(flatten(child, child_prefix))
    return items


def _has_metric_records(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get('metrics'), dict) and payload['metrics']:
            return True
    return False


def run_status(record: dict[str, Any]) -> str:
    """Classify a run as success, failed, incomplete, or missing."""
    run_dir = run_dir_for_record(record)
    if not run_dir.exists():
        return 'missing'
    mode = str(cfg_get(record.get('config', {}), 'run.mode', 'train')).lower()
    log_path = run_dir / 'logs' / 'train.log'
    checkpoint_dir = run_dir / 'checkpoints'
    exception_paths = list(checkpoint_dir.glob('*_exception.pt'))
    normal_paths = [path for path in checkpoint_dir.glob('epoch_*.pt') if '_exception' not in path.stem]
    if (checkpoint_dir / 'last.pt').exists():
        normal_paths.append(checkpoint_dir / 'last.pt')
    if exception_paths and (
        not normal_paths
        or max(path.stat().st_mtime for path in exception_paths) >= max(path.stat().st_mtime for path in normal_paths)
    ):
        return 'failed'
    if log_path.exists():
        text = log_path.read_text(encoding='utf-8', errors='replace')
        session_start = text.rfind('run_id=')
        current_session = text[session_start:] if session_start >= 0 else text
        if 'Traceback (most recent call last)' in current_session or 'ERROR' in current_session:
            return 'failed'
    metrics_path = metrics_path_for_record(record)
    if mode == 'predict':
        return 'success' if (run_dir / 'predictions' / 'test_predictions.json').exists() else 'incomplete'
    if mode in {'eval', 'test'}:
        return 'success' if _has_metric_records(metrics_path) else 'incomplete'
    if mode == 'profile':
        profiles_dir = run_dir / 'profiles'
        return 'success' if profiles_dir.exists() and any(profiles_dir.iterdir()) else 'incomplete'
    manifest_path = checkpoint_dir / 'manifest.json'
    if not manifest_path.exists() or not _has_metric_records(metrics_path):
        return 'incomplete'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return 'failed'
    entries = manifest.get('checkpoints', [])
    has_checkpoint = isinstance(entries, list) and any(
        entry.get('path') and (checkpoint_dir / str(entry['path'])).exists() for entry in entries
    )
    return 'success' if has_checkpoint else 'incomplete'


def _record_path(record: dict[str, Any], value: str) -> Path:
    """Resolve a registry path, using command_cwd for relative saved paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    command_cwd = record.get('command_cwd')
    if command_cwd:
        return Path(str(command_cwd)).expanduser() / path
    return path


def _trial_id_from_name(name: str) -> int:
    """Return a numeric trial suffix for sorting trial_<n> paths."""
    try:
        return int(name.removeprefix('trial_'))
    except ValueError:
        return -1


def _metric_should_minimize(metric: str) -> bool:
    lowered = metric.lower()
    return any(token in lowered for token in ('loss', 'mse', 'mae', 'rmse', 'perplexity', 'error'))
