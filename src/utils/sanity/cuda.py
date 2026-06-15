"""CUDA sanity helpers used by the canonical sanity runner.

Use this module only for low-level CUDA checks. Most users should call
``src.utils.sanity.run_sanity_checks`` instead.
"""

from __future__ import annotations

import re
import subprocess  # noqa: S404 - required to query nvidia-smi diagnostics.
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class CudaDiagnostics:
    """PyTorch CUDA and NVIDIA driver compatibility diagnostics."""

    torch_version: str | None
    torch_cuda: str | None
    cuda_available: bool
    cuda_warning: str
    device_count: int
    device_names: tuple[str, ...]
    nvidia_smi_available: bool
    nvidia_driver_version: str | None
    nvidia_gpu_names: tuple[str, ...]
    nvidia_smi_error: str
    min_driver_version: str | None
    driver_compatible: bool | None
    compatibility_message: str


@dataclass(frozen=True)
class TorchWheelOption:
    """Known PyTorch wheel option from the official PyTorch install matrix."""

    torch: str
    torchvision: str
    torchaudio: str
    cuda_tag: str

    @property
    def index_url(self) -> str:
        """Return the official wheel index URL."""
        return f'https://download.pytorch.org/whl/{self.cuda_tag}'

    @property
    def cuda_version(self) -> str | None:
        """Return the CUDA version encoded by the wheel tag."""
        return _cuda_tag_to_version(self.cuda_tag)

    @property
    def min_driver_version(self) -> str | None:
        """Return the minimum NVIDIA driver for this wheel."""
        return minimum_nvidia_driver_for_cuda(self.cuda_version)


@dataclass(frozen=True)
class TorchInstallRecommendation:
    """Install recommendation and commands for the detected machine."""

    python_requirement: str
    python_for_uv: str
    numpy_spec: str
    selected: TorchWheelOption
    reason: str
    compatible_options: tuple[TorchWheelOption, ...]
    uv_commands: tuple[str, ...]
    pip_commands: tuple[str, ...]
    pyproject_snippet: str


_PYTORCH_WHEEL_OPTIONS = (
    TorchWheelOption('2.11.0', '0.26.0', '2.11.0', 'cu130'),
    TorchWheelOption('2.11.0', '0.26.0', '2.11.0', 'cu128'),
    TorchWheelOption('2.11.0', '0.26.0', '2.11.0', 'cu126'),
    TorchWheelOption('2.10.0', '0.25.0', '2.10.0', 'cu130'),
    TorchWheelOption('2.10.0', '0.25.0', '2.10.0', 'cu128'),
    TorchWheelOption('2.10.0', '0.25.0', '2.10.0', 'cu126'),
    TorchWheelOption('2.9.1', '0.24.1', '2.9.1', 'cu130'),
    TorchWheelOption('2.9.1', '0.24.1', '2.9.1', 'cu128'),
    TorchWheelOption('2.9.1', '0.24.1', '2.9.1', 'cu126'),
    TorchWheelOption('2.9.0', '0.24.0', '2.9.0', 'cu130'),
    TorchWheelOption('2.9.0', '0.24.0', '2.9.0', 'cu128'),
    TorchWheelOption('2.9.0', '0.24.0', '2.9.0', 'cu126'),
    TorchWheelOption('2.8.0', '0.23.0', '2.8.0', 'cu129'),
    TorchWheelOption('2.8.0', '0.23.0', '2.8.0', 'cu128'),
    TorchWheelOption('2.8.0', '0.23.0', '2.8.0', 'cu126'),
    TorchWheelOption('2.7.1', '0.22.1', '2.7.1', 'cu128'),
    TorchWheelOption('2.7.1', '0.22.1', '2.7.1', 'cu126'),
    TorchWheelOption('2.7.1', '0.22.1', '2.7.1', 'cu118'),
    TorchWheelOption('2.7.0', '0.22.0', '2.7.0', 'cu128'),
    TorchWheelOption('2.7.0', '0.22.0', '2.7.0', 'cu126'),
    TorchWheelOption('2.7.0', '0.22.0', '2.7.0', 'cu118'),
    TorchWheelOption('2.6.0', '0.21.0', '2.6.0', 'cu126'),
    TorchWheelOption('2.6.0', '0.21.0', '2.6.0', 'cu124'),
    TorchWheelOption('2.6.0', '0.21.0', '2.6.0', 'cu118'),
    TorchWheelOption('2.5.1', '0.20.1', '2.5.1', 'cu124'),
    TorchWheelOption('2.5.1', '0.20.1', '2.5.1', 'cu121'),
    TorchWheelOption('2.5.1', '0.20.1', '2.5.1', 'cu118'),
)
_CPU_WHEEL_OPTION = TorchWheelOption('2.11.0', '0.26.0', '2.11.0', 'cpu')


_MIN_NVIDIA_DRIVER_BY_CUDA = {
    '11.0': '450.36.06',
    '11.1': '455.23.00',
    '11.2': '460.27.03',
    '11.3': '465.19.01',
    '11.4': '470.42.01',
    '11.5': '495.29.05',
    '11.6': '510.39.01',
    '11.7': '515.43.04',
    '11.8': '520.61.05',
    '12.0': '525.60.13',
    '12.1': '530.30.02',
    '12.2': '535.54.03',
    '12.3': '545.23.08',
    '12.4': '550.54.14',
    '12.5': '555.42.02',
    '12.6': '560.28.03',
    '12.8': '570.26.00',
    '12.9': '575.51.03',
    '13.0': '580.65.06',
}


def cuda_status() -> tuple[bool, str]:
    """Return whether CUDA is usable and a short diagnostic message."""
    diagnostics = collect_cuda_diagnostics()
    if not diagnostics.cuda_available:
        return False, diagnostics.compatibility_message or 'CUDA is not available to PyTorch'
    device = diagnostics.device_names[0] if diagnostics.device_names else 'CUDA device'
    return True, device


def collect_cuda_diagnostics(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> CudaDiagnostics:
    """Collect PyTorch CUDA, NVIDIA driver, and compatibility diagnostics."""
    torch_module, torch_error = _import_torch()
    smi_driver, smi_names, smi_error = _query_nvidia_smi(runner=runner)
    if torch_module is None:
        cuda_warning = f'torch is not installed: {torch_error}'
        message = _compatibility_message(
            torch_cuda=None,
            cuda_available=False,
            cuda_warning=cuda_warning,
            smi_driver=smi_driver,
            smi_error=smi_error,
            min_driver=None,
            compatible=None,
        )
        return CudaDiagnostics(
            torch_version=None,
            torch_cuda=None,
            cuda_available=False,
            cuda_warning=cuda_warning,
            device_count=0,
            device_names=(),
            nvidia_smi_available=smi_driver is not None,
            nvidia_driver_version=smi_driver,
            nvidia_gpu_names=smi_names,
            nvidia_smi_error=smi_error,
            min_driver_version=None,
            driver_compatible=None,
            compatibility_message=message,
        )

    cuda_available, cuda_warning = _torch_cuda_available(torch_module)
    device_count, device_names = _device_details(torch_module, cuda_available)
    torch_cuda = torch_module.version.cuda
    min_driver = minimum_nvidia_driver_for_cuda(torch_cuda)
    compatible = _driver_compatible(torch_cuda, smi_driver, min_driver, cuda_available)
    message = _compatibility_message(
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        cuda_warning=cuda_warning,
        smi_driver=smi_driver,
        smi_error=smi_error,
        min_driver=min_driver,
        compatible=compatible,
    )
    return CudaDiagnostics(
        torch_version=str(torch_module.__version__),
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        cuda_warning=cuda_warning,
        device_count=device_count,
        device_names=device_names,
        nvidia_smi_available=smi_driver is not None,
        nvidia_driver_version=smi_driver,
        nvidia_gpu_names=smi_names,
        nvidia_smi_error=smi_error,
        min_driver_version=min_driver,
        driver_compatible=compatible,
        compatibility_message=message,
    )


def recommend_torch_install(
    diagnostics: CudaDiagnostics,
    python_requirement: str = '>=3.10',
    numpy_spec: str = 'numpy>=1.24,<2.0',
) -> TorchInstallRecommendation:
    """Recommend a PyTorch wheel and UV/pip commands for this machine."""
    compatible = _compatible_wheel_options(diagnostics.nvidia_driver_version)
    if compatible:
        selected = compatible[0]
        reason = (
            f'selected highest supported CUDA wheel tag {selected.cuda_tag} for '
            f'NVIDIA driver {diagnostics.nvidia_driver_version}; '
            f'CUDA {selected.cuda_version} requires driver >= {selected.min_driver_version}'
        )
    else:
        selected = _CPU_WHEEL_OPTION
        reason = diagnostics.nvidia_smi_error or 'no compatible NVIDIA driver was detected; recommending CPU wheels'
    python_for_uv = _python_for_uv(python_requirement)
    uv_commands = _uv_commands(selected, python_for_uv, numpy_spec)
    pip_commands = _pip_commands(selected, numpy_spec)
    return TorchInstallRecommendation(
        python_requirement=python_requirement,
        python_for_uv=python_for_uv,
        numpy_spec=numpy_spec,
        selected=selected,
        reason=reason,
        compatible_options=tuple(compatible),
        uv_commands=uv_commands,
        pip_commands=pip_commands,
        pyproject_snippet=_pyproject_snippet(selected, numpy_spec),
    )


def format_torch_install_recommendation(recommendation: TorchInstallRecommendation) -> str:
    """Format a compact multi-line recommendation for sanity output."""
    option = recommendation.selected
    packages = _uv_package_specs(option)
    return '\n'.join([
        f'Python requirement from pyproject.toml: {recommendation.python_requirement}',
        f'Recommended wheel index: {option.index_url}',
        f'Recommended packages: {", ".join(packages)}',
        f'Reason: {recommendation.reason}',
        'UV commands:',
        *[f'  {command}' for command in recommendation.uv_commands],
        'pip commands:',
        *[f'  {command}' for command in recommendation.pip_commands],
    ])


def minimum_nvidia_driver_for_cuda(cuda_version: str | None) -> str | None:
    """Return the known minimum NVIDIA Linux driver for a CUDA toolkit version."""
    if not cuda_version:
        return None
    parts = cuda_version.split('.')
    if len(parts) < 2:
        return None
    return _MIN_NVIDIA_DRIVER_BY_CUDA.get(f'{parts[0]}.{parts[1]}')


def _import_torch() -> tuple[Any | None, str]:
    try:
        import torch
    except Exception as exc:
        return None, str(exc)
    return torch, ''


def _torch_cuda_available(torch_module: Any) -> tuple[bool, str]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            available = bool(torch_module.cuda.is_available())
        except Exception as exc:
            return False, str(exc)
    messages = '; '.join(str(warning.message) for warning in captured)
    return available, messages


def _device_details(torch_module: Any, cuda_available: bool) -> tuple[int, tuple[str, ...]]:
    if not cuda_available:
        return 0, ()
    try:
        count = int(torch_module.cuda.device_count())
        names = tuple(torch_module.cuda.get_device_name(index) for index in range(count))
        return count, names
    except Exception:
        return 0, ()


def _query_nvidia_smi(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[str | None, tuple[str, ...], str]:
    run = runner or subprocess.run
    try:
        result = run(
            ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return None, (), 'nvidia-smi not found'
    except Exception as exc:
        return None, (), str(exc)
    if result.returncode != 0:
        error = (result.stderr or result.stdout or '').strip()
        return None, (), error or f'nvidia-smi exited with code {result.returncode}'
    drivers: list[str] = []
    names: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        driver, _, name = line.partition(',')
        drivers.append(driver.strip())
        if name.strip():
            names.append(name.strip())
    return (drivers[0] if drivers else None), tuple(names), ''


def _driver_compatible(
    torch_cuda: str | None,
    smi_driver: str | None,
    min_driver: str | None,
    cuda_available: bool,
) -> bool | None:
    if torch_cuda is None:
        return None
    if smi_driver is not None and min_driver is not None:
        return _version_at_least(smi_driver, min_driver)
    if cuda_available:
        return True
    return None


def _compatibility_message(
    torch_cuda: str | None,
    cuda_available: bool,
    cuda_warning: str,
    smi_driver: str | None,
    smi_error: str,
    min_driver: str | None,
    compatible: bool | None,
) -> str:
    if torch_cuda is None:
        if 'not installed' in cuda_warning:
            return f'PyTorch is not installed; {cuda_warning}; install a PyTorch wheel compatible with this driver'
        return 'PyTorch is a CPU-only build; install a CUDA-enabled PyTorch build for GPU training'
    parts = [f'PyTorch CUDA build={torch_cuda}']
    if smi_driver is not None:
        parts.append(f'NVIDIA driver={smi_driver}')
    elif smi_error:
        parts.append(f'NVIDIA driver unavailable: {smi_error}')
    if min_driver is not None:
        parts.append(f'minimum driver for CUDA {torch_cuda} is {min_driver}')
    else:
        parts.append(f'no built-in minimum-driver mapping for CUDA {torch_cuda}')
    parts.append(f'torch.cuda.is_available()={cuda_available}')
    if cuda_warning:
        parts.append(f'PyTorch warning: {cuda_warning}')
    if compatible is False:
        parts.append('driver is too old for this PyTorch CUDA build')
    return '; '.join(parts)


def _version_at_least(installed: str, required: str) -> bool:
    return _version_tuple(installed) >= _version_tuple(required)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.replace('-', '.').split('.') if part.isdigit()]
    padded = [*parts, 0, 0, 0][:3]
    return padded[0], padded[1], padded[2]


def _compatible_wheel_options(driver_version: str | None) -> list[TorchWheelOption]:
    if driver_version is None:
        return []
    options = [
        option
        for option in _PYTORCH_WHEEL_OPTIONS
        if option.min_driver_version is not None and _version_at_least(driver_version, option.min_driver_version)
    ]
    return sorted(
        options, key=lambda option: (_cuda_sort_key(option.cuda_tag), _version_tuple(option.torch)), reverse=True
    )


def _uv_commands(option: TorchWheelOption, python_for_uv: str, numpy_spec: str) -> tuple[str, ...]:
    specs = _uv_package_specs(option)
    return (
        f'uv init --python {python_for_uv}',
        f'uv add "{numpy_spec}"',
        f'uv add "{specs[0]}" --index {option.index_url}',
        'uv sync --index-strategy unsafe-best-match',
        f'uv add "{specs[1]}" --index {option.index_url}',
        f'uv add "{specs[2]}" --index {option.index_url}',
    )


def _pip_commands(option: TorchWheelOption, numpy_spec: str) -> tuple[str, ...]:
    specs = _pip_package_specs(option)
    return (
        f'python -m pip install "{numpy_spec}"',
        f'python -m pip install "{specs[0]}" "{specs[1]}" "{specs[2]}" --index-url {option.index_url}',
    )


def _uv_package_specs(option: TorchWheelOption) -> tuple[str, str, str]:
    suffix = '' if option.cuda_tag == 'cpu' else f'+{option.cuda_tag}'
    return (
        f'torch=={option.torch}{suffix}',
        f'torchvision=={option.torchvision}{suffix}',
        f'torchaudio=={option.torchaudio}{suffix}',
    )


def _pip_package_specs(option: TorchWheelOption) -> tuple[str, str, str]:
    return (
        f'torch=={option.torch}',
        f'torchvision=={option.torchvision}',
        f'torchaudio=={option.torchaudio}',
    )


def _pyproject_snippet(option: TorchWheelOption, numpy_spec: str) -> str:
    specs = _uv_package_specs(option)
    dependencies = '\n'.join(f'    "{spec}",' for spec in (numpy_spec, *specs))
    return '\n'.join([
        'dependencies = [',
        dependencies,
        ']',
        '',
        '[[tool.uv.index]]',
        f'url = "{option.index_url}"',
    ])


def _python_for_uv(python_requirement: str) -> str:
    match = re.search(r'>=\s*(\d+\.\d+)', python_requirement)
    return match.group(1) if match else '3.10'


def _cuda_tag_to_version(cuda_tag: str) -> str | None:
    if cuda_tag == 'cpu' or not cuda_tag.startswith('cu'):
        return None
    digits = cuda_tag[2:]
    if len(digits) < 3:
        return None
    return f'{int(digits[:-1])}.{int(digits[-1])}'


def _cuda_sort_key(cuda_tag: str) -> tuple[int, int]:
    version = _cuda_tag_to_version(cuda_tag)
    if version is None:
        return 0, 0
    major, minor = version.split('.', 1)
    return int(major), int(minor)
