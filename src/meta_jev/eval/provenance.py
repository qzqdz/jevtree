"""Collect eval-afa provenance: git, dirty flag, runtime dependency versions.

Never present a clean scaffold commit as rebuilding a dirty working tree.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


# Paths whose uncommitted changes make git_dirty=true for eval provenance.
DIRTY_PATHSPECS = ("src", "configs", "tests", "pyproject.toml", "requirements.lock")


def _repo_root_from(path: Path | None = None) -> Path:
    here = (path or Path(__file__)).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").is_file() and (p / "src").is_dir():
            return p
    return Path.cwd()


def git_commit(repo: Path | None = None) -> str | None:
    root = repo or _repo_root_from()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_dirty(repo: Path | None = None) -> bool:
    """True if uncommitted changes exist under src/configs/tests/pyproject.toml/requirements.lock."""
    root = repo or _repo_root_from()
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                *DIRTY_PATHSPECS,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Unknown VCS state — treat as dirty so we never claim a clean rebuild.
        return True


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def dependency_versions() -> dict[str, str | None]:
    """Key runtime deps used by eval-afa (numpy GD logistic, stdlib otherwise)."""
    return {
        "numpy": _pkg_version("numpy"),
        "pytest": _pkg_version("pytest"),
    }


def collect_provenance(
    *,
    repo: Path | None = None,
    predictor_name: str | None = None,
    policy_name: str | None = None,
    dataset_hash: str | None = None,
    split_seed: int | None = None,
    budgets: list[int] | None = None,
    config_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full provenance block for summary.json."""
    root = repo or _repo_root_from()
    deps = dependency_versions()
    out: dict[str, Any] = {
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": deps.get("numpy"),
        "dependency_versions": {k: v for k, v in deps.items() if v is not None},
        "predictor_name": predictor_name,
        "policy_name": policy_name,
        "dataset_hash": dataset_hash,
        "split_seed": split_seed,
        "budgets": list(budgets) if budgets is not None else None,
        "config_path": config_path,
    }
    if extra:
        out.update(extra)
    return out


__all__ = [
    "DIRTY_PATHSPECS",
    "collect_provenance",
    "dependency_versions",
    "git_commit",
    "git_dirty",
]
