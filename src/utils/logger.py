"""Logging backends for console, JSONL files, TensorBoard, and W&B.

The training engine logs through ``LoggerCollection`` so logging backends can be
swapped or extended without changing trainer logic. Backends are rank-zero only
by default, which prevents duplicate logs in DDP. JSONL metrics provide a simple
local audit trail even when W&B and TensorBoard are disabled.

Typical usage:
    from src.utils.logger import build_loggers
    loggers = build_loggers(cfg)
    loggers.log_metrics({'train/loss': 0.5}, step=10)
    loggers.finish()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.runtime.distributed import is_rank0
from src.utils.config import cfg_get, config_to_dict

if TYPE_CHECKING:
    from src.utils.types import ConfigType


class LoggerBackend(Protocol):
    """Protocol implemented by every logging backend."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics at an optional step."""

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a file artifact and optional metadata."""

    def finish(self) -> None:
        """Flush and close backend resources."""


class ConsoleLogger:
    """Human-readable scalar logging through Python's logging module."""

    def __init__(self) -> None:
        self.logger = logging.getLogger('ml_template')

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics to the process logger."""
        if not is_rank0():
            return
        prefix = f'step={step} ' if step is not None else ''
        values = ' | '.join(f'{key}={value:.5f}' for key, value in metrics.items())
        self.logger.info('%s%s', prefix, values)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact path to the console logger."""
        if is_rank0():
            self.logger.info('artifact %s (%s): %s', name, artifact_type, path)

    def finish(self) -> None:
        """Finish console logging; no resources require closing."""


class JsonlLogger:
    """Append-only local metrics log for offline audit and plotting."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a', encoding='utf-8') if is_rank0() else None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append scalar metrics as one JSON record."""
        if self.handle is None:
            return
        self.handle.write(json.dumps({'step': step, 'metrics': metrics}, sort_keys=True) + '\n')
        self.handle.flush()

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append artifact metadata as one JSON record."""
        if self.handle is None:
            return
        record = {
            'artifact': {
                'path': str(path),
                'name': name,
                'type': artifact_type,
                'metadata': metadata or {},
            }
        }
        self.handle.write(json.dumps(record, sort_keys=True) + '\n')
        self.handle.flush()

    def finish(self) -> None:
        """Flush and close the JSONL file handle."""
        if self.handle is not None:
            self.handle.close()


class TensorBoardLogger:
    """TensorBoard scalar logger enabled by ``logging.tensorboard.enabled``."""

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir)) if is_rank0() else None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Write scalar metrics to TensorBoard."""
        if self.writer is None or step is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Accept artifact events; TensorBoard has no file-artifact backend."""

    def finish(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class WandBLogger:
    """W&B logger that fails closed when W&B is unavailable or disabled."""

    def __init__(self, cfg: ConfigType) -> None:
        self.run = None
        self.enabled = bool(cfg_get(cfg, 'logging.wandb.enabled', cfg_get(cfg, 'wandb.use_wandb', False)))
        if not self.enabled or not is_rank0():
            return
        try:
            import wandb

            wandb_cfg = cfg_get(cfg, 'logging.wandb', cfg_get(cfg, 'wandb', {}))
            resolved_config = config_to_dict(cfg)
            self.run = wandb.init(
                project=cfg_get(wandb_cfg, 'project', 'ml-template'),
                entity=cfg_get(wandb_cfg, 'entity', None),
                id=str(cfg_get(cfg, 'run.id', None)) if cfg_get(cfg, 'run.id', None) is not None else None,
                name=cfg_get(wandb_cfg, 'run_name', cfg_get(cfg, 'run.id', None)),
                tags=list(cfg_get(wandb_cfg, 'tags', []) or []),
                notes=str(cfg_get(wandb_cfg, 'notes', '') or ''),
                mode=str(cfg_get(wandb_cfg, 'mode', 'online')),
                config=resolved_config,
                resume='allow',
            )
            config_path = cfg_get(cfg, 'run.config_path', None)
            if config_path is not None and Path(str(config_path)).exists():
                artifact = wandb.Artifact(
                    name=f'{cfg_get(cfg, "run.id", "run")}-config',
                    type='config',
                    metadata={
                        'run_id': cfg_get(cfg, 'run.id', None),
                        'config_id': cfg_get(cfg, 'run.config_id', None),
                    },
                )
                artifact.add_file(str(config_path))
                self.run.log_artifact(artifact)
        except Exception as exc:
            logging.getLogger('ml_template').warning('W&B disabled after init failure: %s', exc)
            self.enabled = False
            self.run = None

    @property
    def run_id(self) -> str | None:
        """Return the active W&B run identifier."""
        return self.run.id if self.run is not None else None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics to W&B."""
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload a file artifact to W&B."""
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the active W&B run."""
        if self.run is not None:
            self.run.finish()


class LoggerCollection:
    """Fan-out logger that forwards calls to all enabled backends."""

    def __init__(self, backends: list[LoggerBackend]) -> None:
        self.backends = backends

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Forward scalar metrics to every backend."""
        for backend in self.backends:
            backend.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = 'artifact',
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Forward an artifact event to every backend."""
        for backend in self.backends:
            backend.log_artifact(path, name, artifact_type, metadata)

    def finish(self) -> None:
        """Finish every configured backend."""
        for backend in self.backends:
            backend.finish()


def setup_python_logging(cfg: ConfigType) -> None:
    """Configure console and file logging for the current process."""
    log_dir = Path(cfg_get(cfg, 'run.log_dir', cfg_get(cfg, 'log_dir', 'outputs/logs')))
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if bool(cfg_get(cfg, 'run.debug', False)) else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(log_dir / 'train.log')]
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=handlers,
        force=True,
    )
    logging.root.disabled = False
    logger = logging.getLogger('ml_template')
    logger.disabled = False
    logger.setLevel(level)


def build_loggers(cfg: ConfigType) -> LoggerCollection:
    """Build enabled logging backends from config."""
    setup_python_logging(cfg)
    backends: list[LoggerBackend] = [ConsoleLogger()]
    if bool(cfg_get(cfg, 'logging.jsonl.enabled', True)):
        backends.append(JsonlLogger(cfg_get(cfg, 'logging.jsonl.path', 'outputs/logs/metrics.jsonl')))
    if bool(cfg_get(cfg, 'logging.tensorboard.enabled', False)):
        try:
            backends.append(TensorBoardLogger(cfg_get(cfg, 'logging.tensorboard.log_dir', 'outputs/logs/tensorboard')))
        except Exception as exc:
            logging.getLogger('ml_template').warning('TensorBoard logger disabled: %s', exc)
    if bool(cfg_get(cfg, 'logging.wandb.enabled', cfg_get(cfg, 'wandb.use_wandb', False))):
        backends.append(WandBLogger(cfg))
    return LoggerCollection(backends)


_GLOBAL_LOGGERS: LoggerCollection | None = None


def init_logger(cfg: ConfigType | None = None) -> None:
    """Configure Python logging for older code paths."""
    setup_python_logging(cfg or {})


def init_wandb(cfg: ConfigType) -> None:
    """Build and retain all configured loggers for older code paths."""
    global _GLOBAL_LOGGERS
    _GLOBAL_LOGGERS = build_loggers(cfg)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics through the compatibility logger interface."""
    if _GLOBAL_LOGGERS is not None:
        _GLOBAL_LOGGERS.log_metrics(metrics, step=step)
    else:
        ConsoleLogger().log_metrics(metrics, step=step)


def log_info(message: str) -> None:
    """Log an informational message through the framework logger."""
    logging.getLogger('ml_template').info(message)


def finish() -> None:
    """Finish compatibility loggers when initialized."""
    if _GLOBAL_LOGGERS is not None:
        _GLOBAL_LOGGERS.finish()
