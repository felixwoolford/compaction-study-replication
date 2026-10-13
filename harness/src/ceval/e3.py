"""E3 (exploratory): task continuation across a forced compaction.

Each task directory under materials/tasks/<id>/ has:
  task.json   {"id", "prompt", "test_cmd": [..], "timeout_s"}
  repo/       the starting repository (copied fresh for every arm)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import driver as D
from . import pisession as S
from .config import Config

E3_ARMS = ["pi-default", "designed", "mech-nolookup", "mech"]


def _sig(name: str, args: dict) -> str:
    return name + ":" + json.dumps(args, sort_keys=True)


def repeated_failures(entries: list[dict]) -> dict:
    """Calls after the first compaction that repeat a call which failed before it."""
    calls: dict[str, str] = {}
    failed_before: set[str] = set()
    after = repeats = 0
    seen_compaction = False
    for e in entries:
        if e.get("type") == "compaction":
            seen_compaction = True
            continue
        m = e.get("message") if e.get("type") == "message" else None
        if not m:
            continue
        if m.get("role") == "assistant":
            for b in m.get("content", []):
                if b.get("type") == "toolCall":
                    sig = _sig(b.get("name", ""), b.get("arguments", {}))
                    calls[b.get("id", "")] = sig
                    if seen_compaction:
                        after += 1
                        repeats += sig in failed_before
        elif m.get("role") == "toolResult" and m.get("isError") and not seen_compaction:
            sig = calls.get(m.get("toolCallId", ""))
            if sig:
                failed_before.add(sig)
    return {"compacted": seen_compaction, "calls_after": after, "repeated_failures": repeats,
            "failed_before": len(failed_before)}


def run_task(cfg: Config, task_dir: Path, arm: str, results: Path) -> dict:
    task = json.loads((task_dir / "task.json").read_text())
    wd = D.setup_workdir(cfg, "E3", arm, task["id"])
    repo = wd / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(task_dir / "repo", repo)
    # The agent works inside repo/, so the extension and skill must be visible from there too.
    shutil.copytree(wd / ".pi", repo / ".pi", dirs_exist_ok=True)
    # The session file and the compaction dumps stay outside the repo (see
    # driver.sessions_dir / arm_env): only the lookup arms are meant to see
    # pre-compaction content, and only via context_lookup.
    session = D.sessions_dir(cfg, "E3", arm, task["id"]) / "session.jsonl"
    S.SessionWriter(session, cwd=str(repo), version=int(cfg["pi"]["session_version"]),
                    provider=cfg["model"]["provider"], model=cfg["model"]["id"], seed=0,
                    thinking_level=cfg["model"]["thinking_level"])
    env = D.arm_env(cfg, arm, "E3", task["id"])
    tr = D.turn(cfg, session, repo, env, task["prompt"])
    test = subprocess.run(task["test_cmd"], cwd=repo, capture_output=True, text=True,
                          timeout=task.get("timeout_s", 600))
    entries = S.read_entries(session)
    tin, tout = S.token_usage(entries)
    row = {"experiment": "E3", "task": task["id"], "arm": arm, "turn_ok": tr.ok, "turn_error": tr.error,
           "success": test.returncode == 0, "tokens_in": tin, "tokens_out": tout, "duration_ms": tr.duration_ms,
           **repeated_failures(entries), "ts": datetime.now(timezone.utc).isoformat()}
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row
