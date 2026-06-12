"""Sanity-check package for validating a machine before training.

Use this package when you clone or move the framework to a new machine and want
to confirm that Python, dependencies, configs, registries, data loading, model
forward/backward, checkpoints, and logging directories are usable.

Typical usage:
    from src.utils.sanity import run_sanity_checks
    report = run_sanity_checks(cfg, strict=True)

CLI usage:
    uv run python scripts/run_sanity.py +experiment=sanity_cpu
"""

from src.utils.sanity.core import CheckResult, SanityReport, bootstrap_registries, run_sanity_checks
from src.utils.sanity.cuda import (
    CudaDiagnostics,
    TorchInstallRecommendation,
    collect_cuda_diagnostics,
    cuda_status,
    format_torch_install_recommendation,
    recommend_torch_install,
)

__all__ = [
    'CheckResult',
    'CudaDiagnostics',
    'SanityReport',
    'TorchInstallRecommendation',
    'bootstrap_registries',
    'collect_cuda_diagnostics',
    'cuda_status',
    'format_torch_install_recommendation',
    'recommend_torch_install',
    'run_sanity_checks',
]
