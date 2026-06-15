"""Package import sanity helpers.

Use ``missing_packages`` to check a list of imports without running the full
sanity suite. Most users should call ``src.utils.sanity.run_sanity_checks``.
"""

from __future__ import annotations

import importlib


def _package_available(package: str) -> bool:
    """Return whether a package can be imported."""
    try:
        importlib.import_module(package.replace('-', '_'))
    except Exception:
        return False
    return True


def missing_packages(packages: list[str]) -> list[str]:
    """Return package names that cannot be imported."""
    return [package for package in packages if not _package_available(package)]
