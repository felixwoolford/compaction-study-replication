"""Command line: `uv run ceval <command>`. See SETUP.md for when to use each."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import find_placeholders, load_config, repo_root


def _scenario(exp: str) -> dict:
    return json.loads((repo_root() / "materials" / "scenarios" / f"{exp.lower()}.json").read_text())


def _indices(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


BRIDGE = "harness/bridge/pi-bridge.mjs"


def _require_no_placeholders() -> bool:
    """Commands that drive pi refuse to start while any placeholder is left."""
    left = find_placeholders()
    for f, names in left.items():
        print(f"placeholders left in {f}: {', '.join(names)}", file=sys.stderr)
    return not left


def cmd_check(_: argparse.Namespace) -> int:
    left = find_placeholders()
    bridge = left.pop(BRIDGE, None)
    for f, names in left.items():
        print(f"{f}: {', '.join(names)}")
    if left:
        print("placeholders remain (SETUP.md Stage 0)")
        return 1
    if bridge:
        print(f"{BRIDGE}: {', '.join(bridge)}  (filled in Stage 2)")
    cfg = load_config()
    from .driver import check_pi_settings
    warnings = check_pi_settings(cfg)
    for w in warnings:
        print("WARN", w)
    for key in ("mech_compact_ts", "skill_dir"):
        if not cfg.path(cfg["extension"][key]).exists():
            print(f"missing: extension.{key} -> {cfg['extension'][key]}")
            return 1
    for name, e in cfg["extension"]["registry"].items():
        # In-place packages (pi-lcm, pi-fold) need vendor/node_modules (npm install in vendor/).
        for key in ("source", "package"):
            if key in e and not str(e[key]).startswith("@") and not cfg.path(e[key]).exists():
                print(f"missing: extension.registry.{name}.{key} -> {e[key]}")
                return 1
    print("config OK" + (" (with warnings)" if warnings else ""))
    from . import schedule
    print("model calls now:", "allowed" if schedule.llm_allowed(cfg)[0] else "not allowed",
          f"({schedule.llm_allowed(cfg)[1]})")
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    """Write stimulus plans, and optionally one full synthetic session file for inspection."""
    from . import pisession as S
    from .stimuli import build_plan

    cfg = load_config(allow_placeholders=a.allow_placeholders)
    sc = _scenario(a.exp)
    keep = cfg.keep_recent_tokens if not a.allow_placeholders else a.keep_recent
    cpt = cfg.chars_per_token if not a.allow_placeholders else a.chars_per_token
    from .stimuli import scenario_indices
    for i in _indices(a.sessions) or scenario_indices(sc):
        if sc.get("generator") == "stories":
            from .stories import build_story_plan
            p = build_story_plan(sc, i, keep_recent_tokens=keep, chars_per_token=cpt)
        elif sc.get("generator") == "challenge":
            from .challenge import build_ec_plan
            p = build_ec_plan(sc, i, keep_recent_tokens=keep, chars_per_token=cpt)
        else:
            p = build_plan(sc, i, keep_recent_tokens=keep, chars_per_token=cpt)
        out = Path(a.out) / f"{p.session_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(p.to_dict(), indent=1))
        if a.session_file:
            sf = Path(a.out) / f"{p.session_id}.jsonl"
            if sf.exists():
                sf.unlink()
            for idx, ph in enumerate(p.phases):
                w = S.SessionWriter(sf, cwd=str(Path(a.out).resolve()),
                                    version=int(a.version if a.allow_placeholders else cfg["pi"]["session_version"]),
                                    provider=a.provider if a.allow_placeholders else cfg["model"]["provider"],
                                    model=a.model if a.allow_placeholders else cfg["model"]["id"],
                                    seed=p.seed * 100 + idx,
                                    thinking_level=("xhigh" if a.allow_placeholders
                                                    else cfg["model"]["thinking_level"]))
                for st in ph["steps"]:
                    w.step(st)
            print(f"{sf}  ({S.line_count(sf)} entries)")
        print(out)
    return 0


def cmd_schema_diff(a: argparse.Namespace) -> int:
    from .pisession import schema_diff
    print(schema_diff(Path(a.real), Path(a.synthetic)))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    if not _require_no_placeholders():
        return 1
    from .runner import Runner
    cfg = load_config()
    from .driver import check_pi_settings
    for w in check_pi_settings(cfg):
        print("WARN", w)
        if not a.ignore_settings:
            print("refusing to run with mismatched pi settings (use --ignore-settings for pilots only)")
            return 1
    if a.exp in S2_LIKE:
        from .runner_s2 import S2Runner as Runner  # noqa: F811 (Study 2: continuation runner)
    if a.exp in SWE_LIKE:
        from .swe import SWERunner as Runner  # noqa: F811 (SWE1: real coding tasks, docs/build/swe1.md)
    complete = Runner(cfg, _scenario(a.exp)).run(arms=a.arms.split(",") if a.arms else None,
                                                  indices=_indices(a.sessions))
    return 0 if complete else 3


def cmd_nightly(a: argparse.Namespace) -> int:
    """`run`, but only inside the LLM window: wait for it to open, run, stop
    starting new lines when it closes or the pause file appears, resume the next
    night. Meant to live in tmux; exits when every line is done."""
    import time
    from datetime import datetime
    from . import schedule
    cfg = load_config()
    while True:
        ok, why = schedule.llm_allowed(cfg)
        if ok:
            print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}] {why}: running {a.exp}", flush=True)
            if cmd_run(a) == 0:
                print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}] {a.exp}: all lines done", flush=True)
                return 0
            ok, why = schedule.llm_allowed(cfg)
            if ok:  # cmd_run refused to start (settings or placeholders)
                return 1
        wake = schedule.next_open(cfg)
        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}] waiting: {why}; "
              f"window opens {wake:%Y-%m-%d %H:%M %Z}", flush=True)
        # Poll: the pause file can appear or disappear at any time.
        while not schedule.llm_allowed(cfg)[0]:
            time.sleep(300)


def cmd_score(a: argparse.Namespace) -> int:
    from .scoring import score_file
    cfg = load_config(allow_placeholders=True)
    from . import driver as D
    d = D.results_dir(cfg, a.exp.upper())
    counts = score_file(d / "trials.jsonl", d / "scored.jsonl", d / "review.csv", d / "review_coded.csv")
    print(json.dumps(counts, indent=1))
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .analysis import report
    cfg = load_config(allow_placeholders=True)
    from . import driver as D
    if a.exp.upper() == "S1B":
        # Study 1b: S1 rounds 1-3 merged with S1B rounds 4-8 (ceval.study1b).
        from .study1b import report as report_1b
        print(report_1b(cfg.path(cfg["run"]["results_dir"])))
        return 0
    d = D.results_dir(cfg, a.exp.upper())
    print(report(a.exp.upper(), d, fit_models=not a.no_models))
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    """One turn (and optionally one compaction) on a copy of a session file, in one arm."""
    if not _require_no_placeholders():
        return 1
    from . import driver as D
    from . import pisession as S
    cfg = load_config()
    wd = D.setup_workdir(cfg, "SMOKE", a.arm, "smoke")
    sess = S.fork(Path(a.session), D.sessions_dir(cfg, "SMOKE", a.arm, "smoke") / "smoke.jsonl")
    env = D.arm_env(cfg, a.arm, "SMOKE", "smoke")
    if a.compact:
        c = D.compact(cfg, sess, wd, env)
        print("compact:", c)
        if not c.ok or c.first_kept_id is None:
            print("no compaction entry was written")
            return 1
    t = D.turn(cfg, sess, wd, env, a.message)
    print("turn:", t)
    return 0 if t.ok else 1


def cmd_e3(a: argparse.Namespace) -> int:
    if not _require_no_placeholders():
        return 1
    from .e3 import E3_ARMS, run_task
    cfg = load_config()
    tasks_root = repo_root() / "materials" / "tasks"
    tasks = [d for d in sorted(tasks_root.iterdir()) if (d / "task.json").exists()]
    if a.tasks:
        tasks = [d for d in tasks if d.name in a.tasks.split(",")]
    out = cfg.path(cfg["run"]["results_dir"]) / "E3" / "runs.jsonl"
    for t in tasks:
        for arm in (a.arms.split(",") if a.arms else E3_ARMS):
            print(run_task(cfg, t, arm, out))
    return 0


EXPERIMENTS = ["E1", "E2", "EC", "S1", "S1B", "S2", "P1", "P1B", "P1C", "P2", "SWE1", "e1", "e2", "ec", "s1", "s1b", "s2",
               "p1", "p1b", "p1c", "p2", "swe1"]
S2_LIKE = ("S2", "P2")  # continuation experiments (runner_s2)
SWE_LIKE = ("SWE1",)  # real coding tasks (swe.py)


def cmd_swe_prepare(a: argparse.Namespace) -> int:
    """Clone, build and verify the SWE tasks (no model calls; idempotent)."""
    from . import swe
    cfg = load_config()
    sc = _scenario(a.exp)
    if a.select:
        chosen = swe.select(cfg, sc)
        print(json.dumps(chosen, indent=1))
        return 0
    recs = swe.prepare(cfg, sc, a.tasks.split(",") if a.tasks else None, force=a.force)
    for r in recs:
        disk = sum((r.get("disk_bytes") or {}).values()) / 1e6
        print(f"{r['instance_id']:32s} {'ok' if r['ok'] else 'FAILED':6s} python {r.get('python_exact', r['python'])}"
              f"  build {r.get('env_build_s')}s  verify {r.get('verify_s')}s  disk {disk:.0f} MB"
              + (f"  error: {r['error']}" if r.get("error") else ""))
    return 0 if all(r["ok"] for r in recs) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ceval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="placeholders, paths and pi settings").set_defaults(fn=cmd_check)

    b = sub.add_parser("build", help="write stimulus plans (and optionally session files)")
    b.add_argument("exp", choices=EXPERIMENTS)
    b.add_argument("--sessions", help="e.g. 0-2,5")
    b.add_argument("--out", default="runs/preview")
    b.add_argument("--session-file", action="store_true", help="also write the full synthetic .jsonl")
    b.add_argument("--allow-placeholders", action="store_true", help="preview before Stage 0 is done")
    b.add_argument("--keep-recent", type=int, default=20000)
    b.add_argument("--chars-per-token", type=float, default=3.5)
    b.add_argument("--version", type=int, default=3)
    b.add_argument("--provider", default="llama.cpp")
    b.add_argument("--model", default="preview-model")
    b.set_defaults(fn=cmd_build)

    d = sub.add_parser("schema-diff", help="compare a synthetic session's keys with a real pi session")
    d.add_argument("real")
    d.add_argument("synthetic")
    d.set_defaults(fn=cmd_schema_diff)

    r = sub.add_parser("run", help="run an experiment (resumable)")
    r.add_argument("exp", choices=EXPERIMENTS)
    r.add_argument("--arms", help="comma-separated subset")
    r.add_argument("--sessions", help="e.g. 0-1")
    r.add_argument("--ignore-settings", action="store_true")
    r.set_defaults(fn=cmd_run)

    ni = sub.add_parser("nightly", help="run, but only inside the LLM window; waits and resumes each night")
    ni.add_argument("exp", choices=EXPERIMENTS)
    ni.add_argument("--arms", help="comma-separated subset")
    ni.add_argument("--sessions", help="e.g. 0-1")
    ni.add_argument("--ignore-settings", action="store_true")
    ni.set_defaults(fn=cmd_nightly)

    s = sub.add_parser("score", help="score trials -> scored.jsonl + blind review.csv")
    s.add_argument("exp")
    s.set_defaults(fn=cmd_score)

    an = sub.add_parser("analyze", help="tables, models, figures -> report.md")
    an.add_argument("exp")
    an.add_argument("--no-models", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    sm = sub.add_parser("smoke", help="one turn (optionally after a compaction) through the bridge")
    sm.add_argument("session")
    sm.add_argument("--arm", default="mech")
    sm.add_argument("--message", default="Reply with the single word OK.")
    sm.add_argument("--compact", action="store_true")
    sm.set_defaults(fn=cmd_smoke)

    e = sub.add_parser("e3", help="run E3 task-continuation runs (exploratory)")
    e.add_argument("--tasks", help="comma-separated task ids")
    e.add_argument("--arms", help="comma-separated subset")
    e.set_defaults(fn=cmd_e3)

    sp = sub.add_parser("swe-prepare", help="clone, build and verify SWE task environments (no model calls)")
    sp.add_argument("exp", choices=[e for e in EXPERIMENTS if e.upper() in SWE_LIKE])
    sp.add_argument("--tasks", help="comma-separated instance ids (default: the scenario's tasks)")
    sp.add_argument("--force", action="store_true", help="rebuild even if prepared")
    sp.add_argument("--select", action="store_true",
                    help="walk the seeded candidate order and prepare until each difficulty quota is met")
    sp.set_defaults(fn=cmd_swe_prepare)

    s3 = sub.add_parser("s3", help="Study 3 (PROTOCOL §17): select, screen, freeze, run, score, analyze")
    s3.add_argument("rest", nargs=argparse.REMAINDER)
    s3.set_defaults(fn=lambda a: __import__("ceval.study3", fromlist=["main"]).main(a.rest))

    a = ap.parse_args(argv)
    if hasattr(a, "exp"):
        a.exp = a.exp.upper()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
