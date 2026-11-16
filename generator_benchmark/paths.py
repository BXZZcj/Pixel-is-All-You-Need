"""Locations of third-party weights and source trees.

UMM_MODEL_DIR  directory holding the downloaded weights, one sub-directory per model (default ./models)
UMM_SRC_DIR    directory holding the upstream source repositories that are not pip packages (default ./third_party)
The expected sub-directory names are listed in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("UMM_MODEL_DIR", "models"))
SRC_DIR = Path(os.environ.get("UMM_SRC_DIR", "third_party"))


def model(name: str) -> str:
    return str(MODEL_DIR / name)


def src(name: str) -> str:
    return str(SRC_DIR / name)
