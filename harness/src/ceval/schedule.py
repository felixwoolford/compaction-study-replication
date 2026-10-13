"""When the harness may use the model server.

The server is shared: the study may use it only inside a nightly window
(config `schedule`, e.g. 23:00-10:00 Asia/Tokyo), and never while the pause
file exists (the user wants the machine). Runs check before each (session, arm)
line, so a started line finishes and no trial fails because the window closed.
`call_bridge` checks too, as a backstop for anything else that calls pi.

CEVAL_IGNORE_WINDOW=1 lifts the window (not the pause file) for one command;
use it only when the user asks for a daytime run. The pause file always wins:
the user can stop any run, including a daytime one, by creating it (changed
2026-10-06; before, the flag lifted the pause file too).
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .config import Config


class WindowClosed(RuntimeError):
    """Raised when something tries to call the model outside the allowed window."""


def _hm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _settings(cfg: Config) -> dict | None:
    return cfg.raw.get("schedule")


def ignored() -> bool:
    return os.environ.get("CEVAL_IGNORE_WINDOW") == "1"


def pause_requested(cfg: Config) -> bool:
    s = _settings(cfg)
    return bool(s and s.get("pause_file") and cfg.path(s["pause_file"]).exists())


def in_window(cfg: Config, now: datetime | None = None) -> bool:
    """True inside [start, end) local time. No `schedule` section: always open."""
    s = _settings(cfg)
    if not s:
        return True
    tz = ZoneInfo(s["tz"])
    t = (now or datetime.now(tz)).astimezone(tz).time()
    start, end = _hm(s["start"]), _hm(s["end"])
    return start <= t < end if start < end else (t >= start or t < end)


def llm_allowed(cfg: Config, now: datetime | None = None) -> tuple[bool, str]:
    if pause_requested(cfg):
        return False, f"paused: {_settings(cfg)['pause_file']} exists"
    if ignored():
        return True, "window ignored (CEVAL_IGNORE_WINDOW=1)"
    if not in_window(cfg, now):
        s = _settings(cfg)
        return False, f"outside the LLM window {s['start']}-{s['end']} {s['tz']}"
    return True, "inside the LLM window"


def next_open(cfg: Config, now: datetime | None = None) -> datetime:
    """The next time the window opens (now, if it is open)."""
    s = _settings(cfg)
    tz = ZoneInfo(s["tz"]) if s else None
    now = (now or datetime.now(tz)).astimezone(tz)
    if in_window(cfg, now):
        return now
    start = _hm(s["start"])
    cand = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    return cand if cand > now else cand + timedelta(days=1)


def require_llm(cfg: Config) -> None:
    ok, why = llm_allowed(cfg)
    if not ok:
        raise WindowClosed(f"model calls not allowed now: {why} (set CEVAL_IGNORE_WINDOW=1 only if the user asked)")
