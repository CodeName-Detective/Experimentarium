"""Canonical pre-flight sanity checks for new machines and clusters.

Use this module before running real experiments on a fresh laptop, workstation,
or cluster node. It validates the Python environment, package requirements from
``pyproject.toml``, Hydra configuration ownership, registry entries, data
construction, model forward and backward passes, checkpoint/log directory
writability, disk space, and optional experiment config composition.

Typical usage:
    from src.utils.sanity import run_sanity_checks
    report = run_sanity_checks(cfg, strict=True)

CLI usage:
    uv run python scripts/run_sanity.py +experiment=sanity_cpu
"""

from __future__ import annotations

import importlib
import netrc as netrc_module
import os
import re
import shutil
import socket
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Callable

from src.utils.config import cfg_get
from src.utils.sanity.cuda import collect_cuda_diagnostics, format_torch_install_recommendation, recommend_torch_install

_IMPORT_NAME_OVERRIDES = {
    'hydra-core': 'hydra',
    'opencv-python': 'cv2',
    'pillow': 'PIL',
    'pyyaml': 'yaml',
    'scikit-learn': 'sklearn',
}


@dataclass
class CheckResult:
    """Single sanity-check result."""

    name: str
    passed: bool
    message: str = ''
    warning: bool = False
    always_show: bool = False

    @property
    def status(self) -> str:
        """Return PASS, WARN, or FAIL for display."""
        return 'PASS' if self.passed else 'WARN' if self.warning else 'FAIL'


@dataclass
class SanityReport:
    """Container for sanity-check results with summary helpers."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        """Return non-warning failed checks."""
        return [result for result in self.results if not result.passed and not result.warning]

    @property
    def warnings(self) -> list[CheckResult]:
        """Return warning-level failed checks."""
        return [result for result in self.results if result.warning and not result.passed]

    @property
    def passed(self) -> bool:
        """Return whether the report has no hard failures."""
        return not self.failures

    def add(self, name: str, passed: bool, message: str = '', warning: bool = False, always_show: bool = False) -> None:
        """Append one check result to the report."""
        self.results.append(
            CheckResult(
                name=name,
                passed=bool(passed),
                message=message,
                warning=warning,
                always_show=always_show,
            )
        )

    def print_summary(self) -> None:
        """Print a human-readable report summary."""
        print('\nSANITY CHECK REPORT')
        print('=' * 72)
        for result in self.results:
            show_detail = result.always_show or not result.passed or result.warning
            detail = f' - {result.message}' if result.message and show_detail else ''
            print(f'{result.status:4} {result.name}{detail}')
        print('=' * 72)
        print(
            f'passed={sum(r.passed for r in self.results)} warnings={len(self.warnings)} failures={len(self.failures)}'
        )


@dataclass(frozen=True)
class RequirementSpec:
    """Parsed project dependency requirement."""

    raw: str
    name: str
    specifier: str = ''
    marker: str = ''


@dataclass(frozen=True)
class ProjectRequirements:
    """Python and package requirements loaded from pyproject.toml."""

    requires_python: str = ''
    dependencies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    source: Path | None = None
    warning: str = ''


def bootstrap_registries() -> None:
    """Import packages that register models, datasets, tasks, losses, metrics, and optimizers."""
    import src.data  # noqa: F401
    import src.losses  # noqa: F401
    import src.metrics  # noqa: F401
    import src.models  # noqa: F401
    import src.optim  # noqa: F401
    import src.tasks  # noqa: F401


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return _parse_pyproject_fallback(path)
    with path.open('rb') as handle:
        return tomllib.load(handle)


def _parse_pyproject_fallback(path: Path) -> dict[str, Any]:
    """Small pyproject parser used only when tomllib/tomli are unavailable."""
    project: dict[str, Any] = {}
    optional: dict[str, list[str]] = {}
    section = ''
    lines = path.read_text(encoding='utf-8').splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line.strip('[]')
            continue
        if section == 'project' and line.startswith('requires-python'):
            project['requires-python'] = _strip_toml_string(line.split('=', 1)[1].strip())
        elif section == 'project' and line.startswith('dependencies'):
            values, idx = _collect_toml_array(line, lines, idx)
            project['dependencies'] = values
        elif section == 'project.optional-dependencies' and '=' in line:
            key, value = line.split('=', 1)
            values, idx = _collect_toml_array(f'={value}', lines, idx)
            optional[key.strip()] = values
    if optional:
        project['optional-dependencies'] = optional
    return {'project': project}


def _collect_toml_array(first_line: str, lines: list[str], idx: int) -> tuple[list[str], int]:
    raw = first_line.split('=', 1)[1].strip()
    while ']' not in raw and idx < len(lines):
        raw += ' ' + lines[idx].strip()
        idx += 1
    values = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    return [left or right for left, right in values], idx


def _strip_toml_string(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _load_project_requirements() -> ProjectRequirements:
    pyproject = _repo_root() / 'pyproject.toml'
    if not pyproject.exists():
        return ProjectRequirements(warning='pyproject.toml not found; package version checks skipped')
    try:
        data = _load_toml(pyproject)
        project = data.get('project', {})
        optional_requirements = _flatten_optional_requirements(project.get('optional-dependencies', {}))
        return ProjectRequirements(
            requires_python=str(project.get('requires-python') or ''),
            dependencies=tuple(str(dep) for dep in project.get('dependencies', ()) or ()),
            optional_dependencies=optional_requirements,
            source=pyproject,
        )
    except Exception as exc:
        return ProjectRequirements(warning=f'could not read pyproject.toml ({exc}); package version checks skipped')


def _flatten_optional_requirements(optional_dependencies: Any) -> tuple[str, ...]:
    if not isinstance(optional_dependencies, dict):
        return ()
    requirements: list[str] = []
    for dependencies in optional_dependencies.values():
        if isinstance(dependencies, (list, tuple)):
            requirements.extend(str(dep) for dep in dependencies)
    return tuple(requirements)


def _parse_requirement(raw: str) -> RequirementSpec:
    requirement, marker = [*raw.split(';', 1), ''][:2]
    match = re.match(r'\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)', requirement)
    if not match:
        return RequirementSpec(raw=raw, name=raw.strip(), marker=marker.strip())
    return RequirementSpec(raw=raw, name=match.group(1), specifier=match.group(2).strip(), marker=marker.strip())


def _marker_applies(marker: str) -> bool:
    if not marker:
        return True
    try:
        from packaging.markers import Marker

        return bool(Marker(marker).evaluate())
    except Exception:
        match = re.search(r'python_version\s*(>=|<=|==|>|<)\s*[\'"]([^\'"]+)[\'"]', marker)
        if not match:
            return True
        return _compare_versions(_python_version(), match.group(2), match.group(1))


def _python_version() -> str:
    return f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


def _version_satisfies(installed: str, specifier: str) -> bool:
    if not specifier:
        return True
    comparisons = re.findall(r'(>=|<=|==|>|<|~=)\s*([^,\s]+)', specifier)
    return all(
        _compare_versions(installed, version, '>=' if operator == '~=' else operator)
        for operator, version in comparisons
    )


def _compare_versions(installed: str, expected: str, operator: str) -> bool:
    try:
        from packaging.version import Version

        left: Any = Version(installed)
        right: Any = Version(expected)
    except Exception:
        left = _numeric_version(installed)
        right = _numeric_version(expected)
    if operator == '>=':
        return left >= right
    if operator == '<=':
        return left <= right
    if operator == '==':
        return left == right or installed.split('+', 1)[0] == expected.split('+', 1)[0]
    if operator == '>':
        return left > right
    if operator == '<':
        return left < right
    return True


def _numeric_version(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r'\d+', value)]
    return tuple(parts or [0])


def _import_candidates(distribution_name: str) -> list[str]:
    candidates: list[str] = []
    override = _IMPORT_NAME_OVERRIDES.get(distribution_name.lower())
    if override:
        candidates.append(override)
    top_level = ''
    try:
        distribution = importlib_metadata.distribution(distribution_name)
        top_level = distribution.read_text('top_level.txt') or ''
    except Exception:
        top_level = ''
    candidates.extend(line.strip() for line in top_level.splitlines() if line.strip())
    candidates.append(distribution_name.replace('-', '_'))
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _check_python(report: SanityReport, cfg: Any) -> None:
    requirements = _load_project_requirements()
    if requirements.warning:
        report.add('requirements.pyproject', False, requirements.warning, warning=True)
    requires_python = requirements.requires_python
    if not requires_python:
        report.add(
            'python.version',
            False,
            'requires-python is not declared in pyproject.toml; version check skipped',
            warning=True,
        )
        return
    report.add(
        'python.version',
        _version_satisfies(_python_version(), requires_python),
        f'pyproject requires Python {requires_python}; found {sys.version.split()[0]}',
    )


def _check_packages(report: SanityReport, cfg: Any) -> None:
    requirements = _load_project_requirements()
    recommend_mode = _torch_install_recommend_mode(cfg)
    for raw in requirements.dependencies:
        requirement = _parse_requirement(raw)
        _check_requirement(
            report, raw, warning=recommend_mode and requirement.name in {'torch', 'torchvision', 'torchaudio'}
        )
    for raw in sorted(set(requirements.optional_dependencies)):
        _check_requirement(report, raw, warning=True)


def _check_requirement(report: SanityReport, raw: str, warning: bool) -> None:
    requirement = _parse_requirement(raw)
    if not requirement.name or not _marker_applies(requirement.marker):
        return
    check_name = f'package.{requirement.name}'
    try:
        installed = importlib_metadata.version(requirement.name)
    except importlib_metadata.PackageNotFoundError:
        report.add(check_name, False, f'not installed; requires {requirement.raw}', warning=warning)
        return
    version_ok = _version_satisfies(installed, requirement.specifier)
    if not version_ok:
        report.add(check_name, False, f'installed {installed}; requires {requirement.raw}', warning=warning)
        return
    import_ok, import_name, import_error = _requirement_importable(requirement.name)
    if import_ok:
        report.add(check_name, True)
        return
    report.add(check_name, False, f'import {import_name!r} failed: {import_error}', warning=warning)


def _requirement_importable(distribution_name: str) -> tuple[bool, str, str]:
    last_name = distribution_name.replace('-', '_')
    last_error = ''
    for import_name in _import_candidates(distribution_name):
        last_name = import_name
        try:
            importlib.import_module(import_name)
            return True, import_name, ''
        except Exception as exc:
            last_error = str(exc)
    return False, last_name, last_error


def _check_config_keys(report: SanityReport, cfg: Any) -> None:
    required = list(cfg_get(cfg, 'sanity.required_config_keys', []) or [])
    if not required:
        required = [
            'run.seed',
            'run.device',
            'run.precision',
            'data.name',
            'model.name',
            'task.name',
            'optimizer.name',
            'scheduler.name',
            'trainer.max_epochs',
            'checkpoint.dir',
            'checkpoint.monitor',
        ]
    for key in required:
        report.add(f'config.{key}', cfg_get(cfg, key, None) is not None, f'missing {key}')


def _torch_install_recommend_mode(cfg: Any) -> bool:
    return bool(cfg_get(cfg, 'sanity.torch_install.recommend', False))


def _torch_missing_in_recommend_mode(cfg: Any, exc: BaseException) -> bool:
    return _torch_install_recommend_mode(cfg) and 'torch' in str(exc).lower()


def _flag_enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on', 'always'}


def _flag_disabled(value: Any) -> bool:
    return str(value).strip().lower() in {'0', 'false', 'no', 'off', 'none', 'disabled'}


def _wandb_logging_enabled(cfg: Any) -> bool:
    return _flag_enabled(cfg_get(cfg, 'logging.wandb.enabled', cfg_get(cfg, 'wandb.use_wandb', False)))


def _wandb_check_requested(cfg: Any) -> bool:
    mode = cfg_get(cfg, 'sanity.wandb.check', 'auto')
    if _flag_disabled(mode):
        return False
    if _flag_enabled(mode):
        return True
    return _wandb_logging_enabled(cfg)


def _wandb_failures_are_warnings(cfg: Any) -> bool:
    return not _wandb_logging_enabled(cfg) and not _flag_enabled(cfg_get(cfg, 'sanity.wandb.check', 'auto'))


def _wandb_mode(cfg: Any) -> str:
    return str(cfg_get(cfg, 'logging.wandb.mode', cfg_get(cfg, 'wandb.mode', 'online'))).strip().lower()


def _wandb_offline_mode(mode: str) -> bool:
    return mode in {'offline', 'disabled', 'dryrun'}


def _wandb_importable() -> tuple[bool, str]:
    try:
        importlib.import_module('wandb')
    except Exception as exc:
        return False, str(exc)
    return True, ''


def _wandb_host() -> str:
    base_url = os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai')
    parsed = urlparse(base_url if '://' in base_url else f'https://{base_url}')
    return parsed.hostname or 'api.wandb.ai'


def _wandb_api_key_source(host: str | None = None) -> str | None:
    if os.environ.get('WANDB_API_KEY'):
        return 'WANDB_API_KEY'
    host = host or _wandb_host()
    candidates = [host, 'api.wandb.ai']
    netrc_paths: list[Path] = []
    if os.environ.get('NETRC'):
        netrc_paths.append(Path(str(os.environ['NETRC'])).expanduser())
    netrc_paths.append(Path.home() / '.netrc')
    for netrc_path in netrc_paths:
        try:
            credentials = netrc_module.netrc(str(netrc_path))
        except (FileNotFoundError, netrc_module.NetrcParseError, OSError):
            continue
        for candidate in candidates:
            auth = credentials.authenticators(candidate)
            if auth and auth[2]:
                return str(netrc_path)
    return None


def _wandb_connectivity(host: str, timeout_seconds: float) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, 443), timeout=timeout_seconds):
            return True, f'reachable {host}:443'
    except OSError as exc:
        return False, f'cannot reach {host}:443 ({exc})'


def _check_wandb_runtime(report: SanityReport, cfg: Any) -> None:
    if not _wandb_check_requested(cfg):
        return
    warning = _wandb_failures_are_warnings(cfg)
    enabled = _wandb_logging_enabled(cfg)
    mode = _wandb_mode(cfg)
    report.add('runtime.wandb.mode', True, f'enabled={enabled}; mode={mode}', always_show=True)

    import_ok, import_error = _wandb_importable()
    report.add('runtime.wandb.import', import_ok, import_error or 'wandb importable', warning=warning, always_show=True)
    if not import_ok:
        return

    if _wandb_offline_mode(mode):
        report.add('runtime.wandb.api_key', True, f'not required for mode={mode}', always_show=True)
        report.add('runtime.wandb.network', True, f'not required for mode={mode}', always_show=True)
        return

    host = _wandb_host()
    key_source = _wandb_api_key_source(host)
    report.add(
        'runtime.wandb.api_key',
        key_source is not None,
        f'found credentials in {key_source}'
        if key_source
        else 'not found; run `uv run wandb login` or set WANDB_API_KEY',
        warning=warning,
        always_show=True,
    )

    connectivity_mode = cfg_get(cfg, 'sanity.wandb.check_connectivity', 'auto')
    if _flag_disabled(connectivity_mode):
        report.add('runtime.wandb.network', True, 'disabled by sanity.wandb.check_connectivity=false', always_show=True)
        return
    timeout_seconds = float(cfg_get(cfg, 'sanity.wandb.timeout_seconds', 5.0))
    network_ok, message = _wandb_connectivity(host, timeout_seconds)
    report.add('runtime.wandb.network', network_ok, message, warning=warning, always_show=True)


def _check_runtime(report: SanityReport, cfg: Any) -> None:
    device = str(cfg_get(cfg, 'run.device', 'cpu'))
    cuda_requested = device.startswith('cuda')
    diagnostics = collect_cuda_diagnostics()
    torch_version = diagnostics.torch_version or 'not installed'
    torch_installed = diagnostics.torch_version is not None
    report.add(
        'runtime.torch_version',
        torch_installed,
        f'torch={torch_version}',
        warning=not torch_installed and not cuda_requested,
        always_show=True,
    )
    if _should_check_cuda_driver(cfg, cuda_requested, diagnostics):
        _check_cuda_runtime(report, cfg, cuda_requested, diagnostics)
    else:
        report.add('runtime.cpu_mode', True)

    try:
        import torch.distributed as dist

        report.add('runtime.distributed_import', dist.is_available())
    except ModuleNotFoundError as exc:
        report.add('runtime.distributed_import', False, str(exc), warning=not cuda_requested)
    except Exception as exc:
        report.add('runtime.distributed_import', False, str(exc), warning=True)


def _should_check_cuda_driver(cfg: Any, cuda_requested: bool, diagnostics: Any) -> bool:
    mode = str(cfg_get(cfg, 'sanity.cuda.check_driver', 'auto')).lower()
    if mode in {'0', 'false', 'no', 'off', 'none'}:
        return bool(cfg_get(cfg, 'sanity.torch_install.recommend', False))
    if mode in {'1', 'true', 'yes', 'on', 'always'}:
        return True
    return (
        cuda_requested
        or diagnostics.torch_cuda is not None
        or bool(cfg_get(cfg, 'sanity.torch_install.recommend', False))
    )


def _check_cuda_runtime(report: SanityReport, cfg: Any, cuda_requested: bool, diagnostics: Any) -> None:
    warn_if_unusable = not cuda_requested and not bool(cfg_get(cfg, 'sanity.cuda.fail_on_cpu_mismatch', False))
    torch_version = diagnostics.torch_version or 'not installed'
    torch_cuda = 'unavailable' if diagnostics.torch_version is None else diagnostics.torch_cuda or 'cpu-only'
    torch_build_ok = diagnostics.torch_cuda is not None or (
        not cuda_requested and diagnostics.torch_version is not None
    )
    report.add(
        'runtime.cuda.torch_build',
        torch_build_ok,
        f'torch={torch_version}; torch.version.cuda={torch_cuda}',
        warning=not torch_build_ok and not cuda_requested,
        always_show=True,
    )
    if diagnostics.torch_cuda is None:
        if cuda_requested:
            report.add('runtime.cuda_available', False, diagnostics.compatibility_message, always_show=True)
            _maybe_add_torch_install_recommendation(report, cfg, diagnostics, force=True)
        else:
            report.add('runtime.cpu_mode', True)
            _maybe_add_torch_install_recommendation(report, cfg, diagnostics)
        return

    driver_message = _cuda_driver_message(diagnostics)
    driver_known = diagnostics.nvidia_smi_available or diagnostics.cuda_available
    report.add(
        'runtime.cuda.nvidia_driver',
        driver_known,
        driver_message,
        warning=warn_if_unusable and not driver_known,
        always_show=True,
    )
    compatibility_known = diagnostics.driver_compatible is not None
    compatibility_passed = bool(diagnostics.driver_compatible) if compatibility_known else diagnostics.cuda_available
    report.add(
        'runtime.cuda.driver_compatibility',
        compatibility_passed,
        diagnostics.compatibility_message,
        warning=warn_if_unusable and not compatibility_passed,
        always_show=True,
    )
    report.add(
        'runtime.cuda_available',
        diagnostics.cuda_available,
        diagnostics.compatibility_message,
        warning=warn_if_unusable and not diagnostics.cuda_available,
        always_show=True,
    )
    if diagnostics.cuda_available:
        device_names = ', '.join(diagnostics.device_names) or 'unknown device name'
        report.add(
            'runtime.cuda_device',
            diagnostics.device_count > 0,
            f'{diagnostics.device_count} CUDA device(s): {device_names}',
            always_show=True,
        )
    _maybe_add_torch_install_recommendation(report, cfg, diagnostics, force=not compatibility_passed)


def _maybe_add_torch_install_recommendation(
    report: SanityReport,
    cfg: Any,
    diagnostics: Any,
    force: bool = False,
) -> None:
    if not force and not bool(cfg_get(cfg, 'sanity.torch_install.recommend', False)):
        return
    requirements = _load_project_requirements()
    recommendation = recommend_torch_install(diagnostics, python_requirement=requirements.requires_python)
    report.add(
        'runtime.torch_install.recommendation',
        True,
        format_torch_install_recommendation(recommendation),
        always_show=True,
    )


def _cuda_driver_message(diagnostics: Any) -> str:
    if diagnostics.nvidia_driver_version is not None:
        gpu_names = ', '.join(diagnostics.nvidia_gpu_names) or 'GPU name unavailable'
        return f'NVIDIA driver={diagnostics.nvidia_driver_version}; GPUs={gpu_names}'
    return f'NVIDIA driver unavailable: {diagnostics.nvidia_smi_error or "unknown nvidia-smi error"}'


def _check_output_dirs(report: SanityReport, cfg: Any) -> None:
    paths = {
        'run.output_dir': cfg_get(cfg, 'run.output_dir', 'outputs'),
        'run.log_dir': cfg_get(cfg, 'run.log_dir', 'outputs/logs'),
        'run.prediction_dir': cfg_get(cfg, 'run.prediction_dir', 'outputs/predictions'),
        'checkpoint.dir': cfg_get(cfg, 'checkpoint.dir', 'outputs/checkpoints'),
    }
    for key, value in paths.items():
        path = Path(str(value))
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / '.sanity_write_test'
            probe.write_text('ok', encoding='utf-8')
            probe.unlink(missing_ok=True)
            report.add(f'writable.{key}', True)
        except Exception as exc:
            report.add(f'writable.{key}', False, str(exc))


def _check_disk(report: SanityReport, cfg: Any) -> None:
    min_gb = float(cfg_get(cfg, 'sanity.min_disk_gb', 1.0))
    free_gb = shutil.disk_usage('.').free / (1024**3)
    report.add(
        'disk.free_space', free_gb >= min_gb, f'{free_gb:.1f}GB free; need {min_gb:.1f}GB', warning=free_gb < min_gb
    )


def _check_registries(report: SanityReport, cfg: Any) -> None:
    try:
        bootstrap_registries()
        from src.utils.registry import (
            DATASET_REGISTRY,
            MODEL_REGISTRY,
            OPTIMIZER_REGISTRY,
            SCHEDULER_REGISTRY,
            TASK_REGISTRY,
        )
    except Exception as exc:
        report.add('registry.bootstrap', False, str(exc), warning=_torch_missing_in_recommend_mode(cfg, exc))
        return

    checks = {
        'model': (MODEL_REGISTRY, str(cfg_get(cfg, 'model.name', ''))),
        'dataset': (DATASET_REGISTRY, str(cfg_get(cfg, 'data.name', ''))),
        'task': (TASK_REGISTRY, str(cfg_get(cfg, 'task.name', ''))),
        'optimizer': (OPTIMIZER_REGISTRY, str(cfg_get(cfg, 'optimizer.name', ''))),
        'scheduler': (SCHEDULER_REGISTRY, str(cfg_get(cfg, 'scheduler.name', ''))),
    }
    for kind, (registry, name) in checks.items():
        report.add(f'registry.{kind}.{name}', name in registry, f'available: {sorted(registry.keys())}')


def _check_tensor_file_paths(report: SanityReport, cfg: Any) -> None:
    if str(cfg_get(cfg, 'data.name', '')) != 'tensor_file':
        report.add('data.paths', True)
        return
    for split in ('train', 'val', 'test'):
        path = cfg_get(cfg, f'data.splits.{split}.path', None)
        report.add(f'data.path.{split}', path is not None and Path(str(path)).exists(), f'missing {path}')


def _check_data_model_smoke(report: SanityReport, cfg: Any) -> None:
    if not bool(cfg_get(cfg, 'sanity.run_model_smoke', True)):
        report.add('smoke.data_model', True, 'disabled by config', warning=True)
        return
    try:
        import torch

        from src.data import build_dataloaders
        from src.engine.evaluator import move_to_device
        from src.optim import build_optimizer, build_scheduler
        from src.tasks import build_task
        from src.utils.registry import MODEL_REGISTRY

        device = torch.device(str(cfg_get(cfg, 'run.device', 'cpu')))
        if device.type == 'cuda' and not torch.cuda.is_available():
            report.add('smoke.device', False, f'{device} requested but torch.cuda.is_available() is False')
            return
        report.add('smoke.device', True, f'device={device}', always_show=True)
        loaders = build_dataloaders(cfg)
        batch = move_to_device(next(iter(loaders['train'])), device)
        model = MODEL_REGISTRY.build(str(cfg_get(cfg, 'model.name', 'mlp')), cfg_get(cfg, 'model')).to(device)
        task = build_task(cfg)
        result = task.step(model, batch, stage='sanity')
        finite_loss = result.loss is not None and torch.isfinite(result.loss.detach()).item()
        report.add('smoke.forward_loss', finite_loss, 'loss missing or non-finite')
        if result.loss is not None:
            result.loss.backward()
        grad_ok = any(
            p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad
        )
        report.add('smoke.backward_gradients', grad_ok, 'no finite gradients found')
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=max(1, len(loaders['train'])))
        report.add('smoke.optimizer_scheduler', optimizer is not None and scheduler is not None)
    except Exception as exc:
        report.add('smoke.data_model', False, str(exc), warning=_torch_missing_in_recommend_mode(cfg, exc))


def _check_experiment_composition(report: SanityReport, cfg: Any) -> None:
    if not bool(cfg_get(cfg, 'sanity.check_all_experiments', False)):
        return
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from src.utils.paths import CONFIG_DIR

        experiment_dir = Path(CONFIG_DIR / 'experiment')
        experiments = sorted(path.stem for path in experiment_dir.glob('*.yaml'))
        hydra_state = GlobalHydra.instance()
        if hydra_state.is_initialized():
            hydra_state.clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base='1.3'):
            for experiment in experiments:
                try:
                    compose(config_name='config', overrides=[f'+experiment={experiment}'])
                    report.add(f'experiment.{experiment}', True)
                except Exception as exc:  # noqa: PERF203 - each experiment must fail independently.
                    report.add(f'experiment.{experiment}', False, str(exc))
    except Exception as exc:
        report.add('experiment.compose_all', False, str(exc), warning=True)


def run_sanity_checks(
    cfg: Any, strict: bool = True, extra_checks: list[Callable[[SanityReport, Any], None]] | None = None
) -> SanityReport:
    """Run the canonical pre-flight sanity checks.

    Args:
        cfg: Hydra DictConfig or plain dict with the composed experiment config.
        strict: If true, raise RuntimeError on non-warning failures.
        extra_checks: Optional callables for project-specific checks.

    Returns:
        SanityReport with every check result.

    Raises:
        RuntimeError: If strict mode is enabled and a required check fails.
    """
    report = SanityReport()
    checks: list[Callable[[SanityReport, Any], None]] = [
        _check_python,
        _check_packages,
        _check_config_keys,
        _check_runtime,
        _check_wandb_runtime,
        _check_output_dirs,
        _check_disk,
        _check_registries,
        _check_tensor_file_paths,
        _check_data_model_smoke,
        _check_experiment_composition,
    ]
    if extra_checks:
        checks.extend(extra_checks)
    for check in checks:
        check(report, cfg)
    report.print_summary()
    if strict and report.failures:
        raise RuntimeError(f'Sanity checks failed: {[result.name for result in report.failures]}')
    return report
