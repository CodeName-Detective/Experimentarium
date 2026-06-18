"""Fault-tolerant checkpoint manager for research training runs.

This module saves and restores full training state: model, optimizer, scheduler,
AMP scaler, epoch, global step, best metric, config, and RNG state. Checkpoints
are written atomically, recorded in a JSON manifest with SHA256 checksums, and
loaded with fallback so a partially corrupted newest checkpoint does not block
resume.

Typical usage:
    manager = CheckpointManager('outputs/checkpoints', keep_last_k=5, save_top_k=3)
    path = manager.save(state, epoch=epoch, metric=val_loss, is_best=is_best)
    state = manager.load_latest(model, optimizer, scheduler, scaler)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.runtime.distributed import is_rank0, unwrap_model


def get_rng_state(seed_cuda: bool | None = None) -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and optional CUDA RNG state."""
    state: dict[str, Any] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    should_capture_cuda = torch.cuda.is_available() if seed_cuda is None else seed_cuda and torch.cuda.is_available()
    if should_capture_cuda:
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state if present in a checkpoint."""
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


class CheckpointManager:
    """Checkpoint manager with atomic save, manifest, best/last/top-k, and fallback resume."""

    def __init__(
        self,
        directory: str | Path,
        save_every: int = 1,
        keep_last_k: int = 5,
        monitor: str = 'val/loss',
        mode: str = 'min',
        save_last: bool = True,
        save_top_k: int = 1,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every)
        self.keep_last_k = int(keep_last_k)
        self.monitor = monitor
        self.mode = mode
        self.save_last = bool(save_last)
        self.save_top_k = int(save_top_k)
        self.best_path = self.directory / 'best.pt'
        self.last_path = self.directory / 'last.pt'
        self.manifest_path = self.directory / 'manifest.json'

    def is_better(self, metric: float, best: float | None) -> bool:
        """Return whether a metric improves on the current best value."""
        if best is None:
            return True
        return metric < best if self.mode == 'min' else metric > best

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metric: float | None = None,
        is_best: bool = False,
        tag: str | None = None,
    ) -> Path | None:
        """Save a checkpoint if policy allows it and update manifest links."""
        if not is_rank0():
            return None
        if epoch % self.save_every != 0 and not is_best and tag is None:
            return None
        suffix = f'_{tag}' if tag else ''
        path = self.directory / f'epoch_{epoch:04d}{suffix}.pt'
        state = dict(state)
        state.setdefault('checkpoint_meta', self._checkpoint_meta(epoch=epoch, metric=metric, tag=tag))
        self._atomic_save(state, path)
        if self.save_last and tag is None:
            self._atomic_copy(path, self.last_path)
        if is_best:
            self._atomic_copy(path, self.best_path)
        self._record_manifest(path, epoch=epoch, metric=metric, is_best=is_best, tag=tag)
        self.rotate()
        return path

    def save_exception(self, state: dict[str, Any], epoch: int) -> Path | None:
        """Save an exception checkpoint that is not part of normal rotation."""
        return self.save(state, epoch=epoch, metric=None, is_best=False, tag='exception')

    def _checkpoint_meta(self, epoch: int, metric: float | None, tag: str | None) -> dict[str, Any]:
        return {
            'epoch': epoch,
            'metric': metric,
            'monitor': self.monitor,
            'mode': self.mode,
            'tag': tag,
            'created_at': time.time(),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
        }

    def _atomic_save(self, state: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, tmp_path)
        with tmp_path.open('rb') as handle:
            os.fsync(handle.fileno())
        tmp_path.replace(path)

    def _atomic_copy(self, source: Path, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        shutil.copyfile(source, tmp_path)
        with tmp_path.open('rb') as handle:
            os.fsync(handle.fileno())
        tmp_path.replace(path)

    def _record_manifest(self, path: Path, epoch: int, metric: float | None, is_best: bool, tag: str | None) -> None:
        manifest = self._read_manifest()
        entries = manifest.setdefault('checkpoints', [])
        entries.append({
            'path': path.name,
            'epoch': epoch,
            'metric': metric,
            'monitor': self.monitor,
            'mode': self.mode,
            'is_best': is_best,
            'tag': tag,
            'sha256': self._sha256(path),
            'created_at': time.time(),
        })
        manifest['latest'] = path.name
        if is_best:
            manifest['best'] = path.name
        manifest['updated_at'] = time.time()
        self._atomic_json(manifest, self.manifest_path)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {'version': 1, 'checkpoints': []}
        try:
            return json.loads(self.manifest_path.read_text(encoding='utf-8'))
        except Exception:
            return {'version': 1, 'checkpoints': []}

    def _atomic_json(self, payload: dict[str, Any], path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        tmp_path.replace(path)

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_checksum(self, path: Path) -> None:
        manifest = self._read_manifest()
        entries = [
            entry for entry in manifest.get('checkpoints', []) if entry.get('path') == path.name and entry.get('sha256')
        ]
        if not entries:
            return
        expected = entries[-1]['sha256']
        actual = self._sha256(path)
        if actual != expected:
            raise RuntimeError(f'Checkpoint checksum mismatch for {path}')

    def _validate_checkpoint_path(self, path: Path) -> None:
        if path.name == 'best.pt':
            self._validate_selector_checksum(path, 'best')
            return
        if path.name == 'last.pt':
            self._validate_selector_checksum(path, 'latest')
            return
        self._validate_checksum(path)

    def _validate_selector_checksum(self, path: Path, selector: str) -> None:
        manifest = self._read_manifest()
        target_name = manifest.get(selector)
        if not target_name:
            return
        entries = [
            entry
            for entry in manifest.get('checkpoints', [])
            if entry.get('path') == target_name and entry.get('sha256')
        ]
        if not entries:
            return
        expected = entries[-1]['sha256']
        actual = self._sha256(path)
        if actual != expected:
            raise RuntimeError(f'Checkpoint checksum mismatch for {path} ({selector} -> {target_name})')

    def verify(self) -> list[str]:
        """Validate manifest entries and best/last checkpoint selector files."""
        issues: list[str] = []
        manifest = self._read_manifest()
        entries = manifest.get('checkpoints', [])
        if not isinstance(entries, list):
            return ['manifest checkpoints field is not a list']
        for entry in entries:
            name = entry.get('path')
            expected = entry.get('sha256')
            if not name:
                issues.append('manifest entry missing path')
                continue
            path = self.directory / str(name)
            if not path.exists():
                issues.append(f'missing checkpoint: {path}')
                continue
            if expected and self._sha256(path) != expected:
                issues.append(f'checksum mismatch: {path}')
        for selector, selector_path in (('latest', self.last_path), ('best', self.best_path)):
            target_name = manifest.get(selector)
            if not target_name:
                continue
            if not selector_path.exists():
                issues.append(f'missing selector file: {selector_path} ({selector} -> {target_name})')
                continue
            try:
                self._validate_selector_checksum(selector_path, selector)
            except RuntimeError as exc:
                issues.append(str(exc))
        return issues

    def candidate_paths(self) -> list[Path]:
        """Return checkpoint candidates newest-first, including manifest links."""
        candidates: list[Path] = []
        if self.last_path.exists():
            candidates.append(self.last_path)
        ckpts = sorted(self.directory.glob('epoch_*.pt'), key=self._extract_epoch, reverse=True)
        candidates.extend(ckpts)
        if self.best_path.exists():
            candidates.append(self.best_path)
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
        return unique

    def latest_path(self) -> Path | None:
        """Return the newest available checkpoint path."""
        candidates = self.candidate_paths()
        return candidates[0] if candidates else None

    def resolve_resume_path(self, resume: str | Path) -> Path:
        """Resolve named checkpoint selectors relative to this checkpoint directory."""
        value = str(resume).strip()
        selector = value.lower()
        if selector == 'best':
            return self.best_path
        if selector == 'last':
            return self.last_path
        match = re.fullmatch(r'(?:epoch[_-]?)?(\d+)(?:\.pt)?', selector)
        if match is not None:
            return self.directory / f'epoch_{int(match.group(1)):04d}.pt'
        return Path(value)

    def load(
        self,
        path: str | Path,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        strict_model: bool = True,
        validate_checksum: bool = True,
    ) -> dict[str, Any]:
        """Load one checkpoint and restore any provided training objects."""
        path = Path(path)
        if validate_checksum:
            self._validate_checkpoint_path(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        unwrap_model(model).load_state_dict(state['model_state'], strict=strict_model)
        if optimizer is not None and state.get('optimizer_state') is not None:
            optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None and state.get('scheduler_state') is not None:
            scheduler.load_state_dict(state['scheduler_state'])
        if scaler is not None and state.get('scaler_state') is not None:
            scaler.load_state_dict(state['scaler_state'])
        if state.get('rng_state') is not None:
            set_rng_state(state['rng_state'])
        return state

    def load_latest(
        self,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        strict_model: bool = True,
    ) -> dict[str, Any] | None:
        """Load the newest valid checkpoint; skip corrupt candidates."""
        errors: list[str] = []
        for path in self.candidate_paths():
            try:
                return self.load(path, model, optimizer, scheduler, scaler, strict_model=strict_model)
            except Exception as exc:  # noqa: PERF203 - fallback must test each checkpoint independently.
                errors.append(f'{path}: {exc}')
                continue
        if errors:
            raise RuntimeError('No valid checkpoint could be loaded. Tried: ' + '; '.join(errors))
        return None

    def rotate(self) -> None:
        """Keep recent checkpoints and top-k best metric checkpoints."""
        if self.keep_last_k <= 0:
            return
        ckpts = [path for path in self.directory.glob('epoch_*.pt') if '_exception' not in path.stem]
        recent = set(sorted(ckpts, key=self._extract_epoch)[-self.keep_last_k :])
        topk = set(self._top_k_paths())
        keep = recent | topk
        for path in ckpts:
            if path not in keep:
                path.unlink(missing_ok=True)

    def _top_k_paths(self) -> list[Path]:
        if self.save_top_k <= 0:
            return []
        entries = [
            entry
            for entry in self._read_manifest().get('checkpoints', [])
            if entry.get('metric') is not None and entry.get('tag') is None
        ]
        reverse = self.mode == 'max'
        entries = sorted(entries, key=lambda entry: float(entry['metric']), reverse=reverse)
        return [
            self.directory / entry['path']
            for entry in entries[: self.save_top_k]
            if (self.directory / entry['path']).exists()
        ]

    def _extract_epoch(self, path: Path) -> int:
        match = re.search(r'epoch_(\d+)', path.name)
        return int(match.group(1)) if match else -1


# Compatibility helpers for older imports.
def save_checkpoint(state: dict[str, Any], path: str | Path, is_best: bool = False) -> None:
    """Save a checkpoint through the compatibility interface."""
    path = Path(path)
    manager = CheckpointManager(path.parent)
    manager._atomic_save(state, path)
    if is_best:
        manager._atomic_save(state, path.parent / 'best.pt')


def load_checkpoint(
    path: str | Path, model: Any, optimizer: Any | None = None, scheduler: Any | None = None, scaler: Any | None = None
) -> dict[str, Any]:
    """Load a checkpoint through the compatibility interface."""
    manager = CheckpointManager(Path(path).parent)
    return manager.load(path, model, optimizer, scheduler, scaler, validate_checksum=False)
