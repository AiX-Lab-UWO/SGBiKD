from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_root() -> Path:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return REPO_ROOT


def import_root_module(module_name: str):
    ensure_repo_root()
    return importlib.import_module(module_name)

