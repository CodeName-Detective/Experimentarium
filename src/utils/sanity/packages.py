"""Package import sanity helpers.

Use ``missing_packages`` to check a list of imports without running the full
sanity suite. Most users should call ``src.utils.sanity.run_sanity_checks``.
"""

from __future__ import annotations

import importlib


def missing_packages(packages: list[str]) -> list[str]:
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except Exception:
            missing.append(package)
    return missing
