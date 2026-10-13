"""Implicit-context probes on real pi sessions (docs/build/realpi-implicit-probes.md),
run with OpenAI's gpt-6.1-sol (medium thinking, the user's own login) on two
compaction arms: mech and pi-vcc. Exploratory; outside the pre-registration.

    uv run python -m ceval.realpi_implicit replay   # model-free 48k replays (dead-port proxy)
    uv run python -m ceval.realpi_implicit check    # treatment check of the replays
    uv run python -m ceval.realpi_implicit probe [--workers 4]
    uv run python -m ceval.realpi_implicit summarize

1. replay: the two probe sessions are replayed through each arm's real
   compaction at the 48k window exactly as `realpi_replay run` does (same
   code, settings, private HOME/config); every bridge call goes through the
   usage proxy with a dead upstream, so a compaction model request cannot reach
   any server and is counted (must be 0).
2. probe: per probe x arm x kind (probe / control), a full copy of the
   compacted session (as Study 1 probes) gets the instruction as one user turn,
   no built-in tools, the arm's recovery tool available, 15-min timeout. The
   model is openai-codex/gpt-6.1-sol; the models file given to pi declares NO
   providers, so the llama.cpp provider does not exist in the probe runtime
   (the bridge loads pi's llama.cpp extension only when the session model is
   llama.cpp). mech's context_lookup subagent uses MECH_COMPACT_LOOKUP_MODEL =
   openai-codex/gpt-6.1-sol. Stops when the session files record > 4M input or
   > 200k output tokens.
3. The probe's fixed pass rule is applied mechanically to the reply text;
   rules with residual judgement are flagged for a read (recorded separately).

Session text, replies and per-run details stay under runs/realpi/implicit/
(gitignored). results/REALPI-IMPLICIT/runs.jsonl holds aggregate fields only.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import realpi_replay as R

MODEL = {"provider": "openai-codex", "id": "gpt-6.1-sol", "thinking_level": "medium"}
ARMS = ("mech", "pi-vcc")
RECOVERY_TOOL = {"mech": "context_lookup", "pi-vcc": "vcc_recall",
                 "mech-direct": ("context_list_entries", "context_grep", "context_show_entry"),
                 "mech-hybrid": ("context_lookup", "context_list_entries", "context_grep", "context_show_entry"),
                 "verbatim": ("context_lookup", "context_list_entries", "context_grep", "context_show_entry"),
                 "pi-default": ()}


def recovery_tools(arm: str) -> tuple:
    t = RECOVERY_TOOL[arm]
    return t if isinstance(t, tuple) else (t,)
EXP = "REALPI-48k"
BASE = Path("runs/realpi/implicit")
PROBES = Path("runs/realpi/probes/probes.json")
RESULTS = Path("results/REALPI-IMPLICIT")
CAP_IN, CAP_OUT = 4_000_000, 200_000
TURN_TIMEOUT_S = 900
# Expected last 48k compaction per session (docs/build/realpi-implicit-probes.md, mech replay).
EXPECTED = {"6e52f5c76d": {"k": 7, "cut_entry_index": 406}, "a5ce2d8ed7": {"k": 5, "cut_entry_index": 363}}


# ----------------------------------------------------------------- scoring
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
INPLACE = re.compile(r"(perl\s+(-[A-Za-z]*\s+)*-[A-Za-z]*i|sed\s+(-[A-Za-z]*i|--in-place))")


def _blocks(text: str) -> list[str]:
    """Commit groups for P1: blocks separated by blank lines, headings or
    top-level numbered items."""
    out, cur = [], []
    for ln in text.splitlines():
        top = re.match(r"^(#{1,6}\s|\*\*|\d+[.)]\s|commit\s*\d|提交)", ln.strip(), re.I) and not ln.startswith((" ", "\t"))
        if not ln.strip() or (top and cur):
            if cur:
                out.append("\n".join(cur))
            cur = [ln] if ln.strip() else []
        else:
            cur.append(ln)
    if cur:
        out.append("\n".join(cur))
    return out


P1_REFACTOR = ("live-session", "server.ts", "ws-rpc-adapter", "terminal-log-view", "lifecycle", "exit-input",
               "pi-live-session")


def score(probe: dict, text: str) -> dict:
    """The probe's fixed pass rule applied to the reply. Returns pass (bool)
    and review (True where the rule names residual judgement and the
    mechanical result could be wrong)."""
    pid, rule = probe["id"], probe["pass_rule"]
    if not text.strip():
        return {"pass": False, "review": False, "why": "empty"}
    if pid == "P1":
        groups = [g for g in _blocks(text) if "vite.config.ts" in g]
        if not groups:
            return {"pass": False, "review": True, "why": "vite.config.ts not mentioned"}
        if re.search(rule["regex_label"], text, re.I) and any(re.search(rule["regex_label"], g, re.I) for g in groups):
            return {"pass": True, "review": True, "why": "labelled"}
        bundled = any(any(f in g for f in P1_REFACTOR) for g in groups)
        return {"pass": not bundled, "review": True, "why": "bundled" if bundled else "separate group"}
    if pid == "P2":
        bad = re.search(rule["fail_regex"], text)
        ok = bool(INPLACE.search(text)) and not bad
        return {"pass": ok, "review": False, "why": "bsd form" if bad else ("ok" if ok else "no in-place command")}
    if pid == "P3":
        bad = re.search(rule["fail_regex"], text)
        good = re.search(rule["pass_regex"], text)
        return {"pass": bool(good) and not bad, "review": False,
                "why": "absolute/user path" if bad else ("ok" if good else "no ~/...extensions/ path")}
    if pid == "P4":
        ok = all(re.search(r, text) for r in rule["pass_regex_all"])
        return {"pass": ok, "review": ok, "why": "both named" if ok else "one file only"}
    if pid == "P5":
        items = [ln for ln in text.splitlines() if LIST_ITEM.match(ln)] or text.splitlines()
        hits = [ln for ln in items if re.search(rule["fail_regex"], ln, re.I)]
        return {"pass": not hits, "review": bool(hits), "why": "feature re-proposed?" if hits else "ok"}
    raise ValueError(pid)


# ----------------------------------------------------------------- config
def replay_cfg(root: Path):
    return R._cfg_for("48k", (root / BASE / "replay").resolve())


# Appended to every probe and control message when S3_MEMORY_NOTE=1 (2026-10-08,
# after Qwen looped on lookups trying to "verify the working tree" with no shell):
# arm-neutral; it names no tool and no compaction method.
MEMORY_NOTE = ("(In this reply you have no shell, file or repository access: you cannot run commands or "
               "read files. Answer from what you know of this conversation; if you have a tool for "
               "searching the conversation's history, you may use it. Do not try to verify the working tree.)")


def memory_note() -> bool:
    import os
    return os.environ.get("S3_MEMORY_NOTE") == "1"


def local_model() -> bool:
    """S3_MODEL=qwen (shared with Study 3): probe turns on the local Qwen with
    Study 1's settings instead of gpt-6.1-sol (2026-10-08, the user's request)."""
    import os
    return os.environ.get("S3_MODEL") == "qwen"


def probe_cfg(root: Path):
    import os
    from .config import Config, load_config
    base = replay_cfg(root)
    if local_model():
        main = load_config()
        raw = copy.deepcopy(base.raw)
        raw["model"] = copy.deepcopy(main.raw["model"])
        if os.environ.get("S3_THINKING"):  # e.g. medium (pilot use, 2026-10-08)
            raw["model"]["thinking_level"] = os.environ["S3_THINKING"]
        raw["pi"]["models_file"] = str(main.path(main.raw["pi"]["models_file"]))
        raw["usage_proxy"] = copy.deepcopy(main.raw["usage_proxy"])
        raw["bridge"]["turn_timeout_s"] = TURN_TIMEOUT_S
        return Config(raw=raw, root=base.root)
    raw = copy.deepcopy(base.raw)
    raw["model"].update(MODEL)
    for k in ("sampling_params", "context_window"):
        raw["model"].pop(k, None)
    empty = (root / BASE / "pi-models-none.json").resolve()
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text(json.dumps({"providers": {}}) + "\n")
    raw["pi"]["models_file"] = str(empty)
    raw["bridge"]["turn_timeout_s"] = TURN_TIMEOUT_S
    raw.setdefault("usage_proxy", {})["enabled"] = False
    return Config(raw=raw, root=base.root)


# ----------------------------------------------------------------- replay
def cmd_setup(root: Path, src: Path) -> None:
    """Copy the probe sessions (by sid) and their selection rows from an
    existing realpi replay area `src` (read only) into runs/realpi/."""
    import shutil
    sids = {p["sid"] for p in json.loads((root / PROBES).read_text())}
    sel = [x for x in json.loads((src / "replay" / "selection.json").read_text())["selected"] if x["sid"] in sids]
    (root / "runs/realpi/hf").mkdir(parents=True, exist_ok=True)
    for x in sel:
        shutil.copyfile(src / "hf" / x["file"], root / "runs/realpi/hf" / x["file"])
    (root / BASE / "replay").mkdir(parents=True, exist_ok=True)
    (root / BASE / "replay" / "selection.json").write_text(json.dumps({"selected": sel}, indent=1))
    print("sessions:", [x["sid"] for x in sel])


def cmd_replay(root: Path) -> None:
    for m in ARMS:
        R.main(["run", "--method", m, "--window", "48k", "--sessions", "0-1",
                "--data", "runs/realpi/hf", "--out", str(BASE / "replay")])


def check(root: Path) -> dict:
    from . import driver as D
    from . import pisession as S
    cfg = replay_cfg(root)
    base = root / BASE / "replay"
    out = {}
    for m in ARMS:
        reps = [json.loads(x) for x in (base / f"replays-{m}-48k.jsonl").read_text().splitlines()]
        rows = [json.loads(x) for x in (base / f"rows-{m}-48k.jsonl").read_text().splitlines()]
        for rep in reps:
            sid = rep["sid"]
            rs = [r for r in rows if r["sid"] == sid]
            main = D.sessions_dir(cfg, EXP, m, sid) / "main.jsonl"
            ents = S.read_entries(main)
            last = S.last_compaction(ents)
            ok = (not rep["failed"] and rep["model_requests"] == 0 and rep["entries_replayed"] == rep["entries"]
                  and all(r["ok"] and r["compactor"] == D.expected_compactor(cfg, m) for r in rs)
                  and S.compactor_of(last) == D.expected_compactor(cfg, m))
            if m == "mech":
                ok = ok and rs[-1]["k"] == EXPECTED[sid]["k"] and rs[-1]["cut_entry_index"] == EXPECTED[sid]["cut_entry_index"]
            out[f"{m}/{sid}"] = {"ok": ok, "compactions": len(rs), "last_k": rs[-1]["k"],
                                 "last_cut_entry_index": rs[-1].get("cut_entry_index"),
                                 "compactor": sorted({r["compactor"] for r in rs}),
                                 "model_requests": rep["model_requests"],
                                 "final_est_tokens": rep["final_est_tokens"],
                                 "entries_after_last_compaction": len(ents) - 1 - ents.index(last)}
    return out


# ----------------------------------------------------------------- probes
def _delta_usage(delta: list[dict]) -> dict:
    from . import pisession as S
    u = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "cost": 0.0, "models": set(), "assistant_msgs": 0}
    for m in S.messages(delta):
        if m.get("role") != "assistant":
            continue
        us = m.get("usage") or {}
        u["input"] += int(us.get("input", 0) or 0)
        u["cache_read"] += int(us.get("cacheRead", 0) or 0)
        u["cache_write"] += int(us.get("cacheWrite", 0) or 0)
        u["output"] += int(us.get("output", 0) or 0)
        u["cost"] += float((us.get("cost") or {}).get("total", 0) or 0)
        u["models"].add(f"{m.get('provider')}/{m.get('model')}")
        u["assistant_msgs"] += 1
    u["models"] = sorted(u["models"])
    return u


def run_one(cfg, root: Path, probe: dict, arm: str, kind: str) -> tuple[dict, str]:
    from . import driver as D
    from . import pisession as S
    sid = probe["sid"]
    main = D.sessions_dir(cfg, EXP, arm, sid) / "main.jsonl"
    name = f"implicit-{probe['id']}-{kind}"
    fork = D.prepare_fork(cfg, main, main.parent / "probes" / f"{name}.jsonl", prune=False)
    wd = D.agent_cwd(cfg, EXP, arm, sid)
    env = D.arm_env(cfg, arm, EXP, sid, {"MECH_COMPACT_MAX_SUMMARY_CHARS": "40000"} if arm.startswith("mech") else {})
    env.update(D.copy_probe_state(cfg, EXP, arm, sid, name))
    want = f"{cfg['model']['provider']}/{cfg['model']['id']}" if local_model() else f"{MODEL['provider']}/{MODEL['id']}"
    assert env["MECH_COMPACT_LOOKUP_MODEL"] == want
    n0 = S.line_count(fork)
    msg = probe["instruction" if kind == "probe" else "control_instruction"]
    if memory_note():
        msg = msg.rstrip() + "\n\n" + MEMORY_NOTE
    t0 = time.monotonic()
    tr = D.turn(cfg, fork, wd, env, msg, builtin_tools=False)
    wall = round(time.monotonic() - t0, 1)
    delta = S.entries_after(fork, n0)
    u = _delta_usage(delta)
    tool = RECOVERY_TOOL[arm]
    sc = score(probe, tr.text)
    row = {"probe": probe["id"], "sid": sid, "arm": arm, "kind": kind, "ok": tr.ok, "error": tr.error,
           "pass_auto": sc["pass"], "needs_review": sc["review"], "rule_note": sc["why"],
           "recovery_calls": sum(c.get("name") == tool for c in tr.calls),
           "other_tool_calls": sorted({c.get("name") for c in tr.calls if c.get("name") != tool}),
           "input_tokens": u["input"], "cache_read_tokens": u["cache_read"], "cache_write_tokens": u["cache_write"],
           "output_tokens": u["output"], "cost_usd": round(u["cost"], 6), "assistant_messages": u["assistant_msgs"],
           "models": u["models"], "compactions_during_turn": sum(e.get("type") == "compaction" for e in delta),
           "wall_s": wall, "reply_chars": len(tr.text)}
    return row, tr.text


def cmd_probe(root: Path, workers: int, only: str | None = None) -> None:
    from . import driver as D
    cfg = probe_cfg(root)
    D.set_usage_proxy(cfg, None)
    probes = json.loads((root / PROBES).read_text())
    out = root / BASE / "runs-full.jsonl"
    replies = root / BASE / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    done = {(r["probe"], r["arm"], r["kind"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    tot = {"in": 0, "out": 0}
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            tot["in"] += r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"]
            tot["out"] += r["output_tokens"]
    lock, stop = threading.Lock(), threading.Event()
    jobs = [(p, a, k) for p in probes for a in ARMS for k in ("probe", "control") if (p["id"], a, k) not in done
            and (not only or f"{p['id']}:{a}:{k}" in only.split(","))]

    def job(p, a, k):
        if stop.is_set():
            return
        row, text = run_one(cfg, root, p, a, k)
        with lock:
            (replies / f"{a}-{p['id']}-{k}.txt").write_text(text)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            tot["in"] += row["input_tokens"] + row["cache_read_tokens"] + row["cache_write_tokens"]
            tot["out"] += row["output_tokens"]
            print("RUN", json.dumps({x: row[x] for x in ("probe", "arm", "kind", "ok", "pass_auto", "recovery_calls",
                                                          "input_tokens", "cache_read_tokens", "output_tokens",
                                                          "cost_usd", "wall_s", "models")}),
                  "TOTAL", tot, flush=True)
            if not local_model() and (tot["in"] > CAP_IN or tot["out"] > CAP_OUT):  # a cost cap: GPT only
                print("TOKEN CAP EXCEEDED: stopping", flush=True)
                stop.set()

    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(job, *j) for j in jobs]:
            f.result()


AGG = ("probe", "arm", "kind", "ok", "error", "pass_auto", "needs_review", "pass", "review_note", "recovery_calls",
       "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "cost_usd", "assistant_messages",
       "models", "compactions_during_turn", "wall_s")


def cmd_summarize(root: Path) -> None:
    """runs-full.jsonl + reviews.json (manual decisions on flagged rows,
    abstract notes only) -> results/REALPI-IMPLICIT/runs.jsonl."""
    rows = [json.loads(x) for x in (root / BASE / "runs-full.jsonl").read_text().splitlines()]
    rv_f = root / BASE / "reviews.json"
    reviews = json.loads(rv_f.read_text()) if rv_f.exists() else {}
    res = root / RESULTS
    res.mkdir(parents=True, exist_ok=True)
    with (res / "runs.jsonl").open("w") as f:
        for r in sorted(rows, key=lambda r: (r["probe"], r["arm"], r["kind"])):
            rv = reviews.get(f"{r['arm']}-{r['probe']}-{r['kind']}")
            r["pass"] = rv["pass"] if rv else r["pass_auto"]
            r["review_note"] = rv["note"] if rv else None
            f.write(json.dumps({k: r.get(k) for k in AGG}) + "\n")
    print((res / "runs.jsonl").read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("setup", "replay", "check", "probe", "summarize"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", help="probe: comma list of PID:arm:kind")
    ap.add_argument("--src", help="setup: an existing runs/realpi directory (read only)")
    a = ap.parse_args(argv)
    from .config import repo_root
    root = repo_root()
    if a.cmd == "setup":
        cmd_setup(root, Path(a.src))
    elif a.cmd == "replay":
        cmd_replay(root)
    elif a.cmd == "check":
        print(json.dumps(check(root), indent=1))
    elif a.cmd == "probe":
        cmd_probe(root, a.workers, a.only)
    else:
        cmd_summarize(root)


if __name__ == "__main__":
    main()
