"""Config loading, placeholder detection and repo paths."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLACEHOLDER_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")

# Files that must contain no placeholders before a run. Docs are excluded on purpose:
# they mention placeholder names when explaining how to fill them in.
PLACEHOLDER_SCAN = ["config/eval.config.json", "harness/bridge/pi-bridge.mjs"]

# Config keys whose values must be numbers once filled in.
NUMERIC_KEYS = {
    ("pi", "keep_recent_tokens"),
    ("pi", "reserve_tokens"),
    ("pi", "session_version"),
    ("model", "context_window"),
    ("tokens", "chars_per_token"),
}


def repo_root() -> Path:
    """The compaction-eval repo root (two levels above harness/src/ceval)."""
    return Path(__file__).resolve().parents[3]


def find_placeholders(root: Path | None = None) -> dict[str, list[str]]:
    """Return {relative file: [placeholder, ...]} for every scanned file with placeholders left."""
    root = root or repo_root()
    found: dict[str, list[str]] = {}
    for rel in PLACEHOLDER_SCAN:
        p = root / rel
        if not p.exists():
            found[rel] = ["<file missing>"]
            continue
        hits = sorted(set(PLACEHOLDER_RE.findall(p.read_text(encoding="utf-8"))))
        if hits:
            found[rel] = hits
    return found


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def keep_recent_tokens(self) -> int:
        return int(self.raw["pi"]["keep_recent_tokens"])

    @property
    def chars_per_token(self) -> float:
        return float(self.raw["tokens"]["chars_per_token"])

    @property
    def context_window(self) -> int:
        return int(self.raw["model"]["context_window"])

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def arm_env(self, arm: str) -> dict[str, str]:
        """Environment for an arm, resolving '@key' references into the run section."""
        env: dict[str, str] = {}
        for k, v in self.raw["arms"][arm]["env"].items():
            if isinstance(v, str) and v.startswith("@"):
                ref = self.raw["run"].get(v[1:])
                if ref is None:
                    raise ValueError(
                        f"arm {arm!r}: {k} refers to run.{v[1:]}, which is not set. "
                        "Run the pilot (SETUP.md Stage 4) and fill it in first."
                    )
                v = str(ref)
            env[k] = str(v)
        return env


def load_config(path: Path | None = None, *, allow_placeholders: bool = False) -> Config:
    root = repo_root()
    path = path or root / "config" / "eval.config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not allow_placeholders:
        left = PLACEHOLDER_RE.findall(json.dumps(raw))
        if left:
            raise ValueError(f"config still has placeholders: {sorted(set(left))} (see SETUP.md Stage 0)")
        for section, key in NUMERIC_KEYS:
            val = raw[section][key]
            try:
                float(val)
            except (TypeError, ValueError):
                raise ValueError(f"config {section}.{key} must be numeric, got {val!r}") from None
    return Config(raw=raw, root=root)
