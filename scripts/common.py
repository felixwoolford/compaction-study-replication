"""Shared helpers for the replication scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CEVAL_IGNORE_WINDOW", "1")  # no nightly window outside the paper's shared server
os.chdir(ROOT)


def rcfg() -> dict:
    return json.loads((ROOT / "replication.config.json").read_text())


def require_setup() -> None:
    if not (ROOT / "config/eval.config.json").exists():
        raise SystemExit("run `uv run python scripts/setup.py` first")
