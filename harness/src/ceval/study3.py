"""Study 3: implicit-context probes on real sessions, mech vs pi-vcc (PROTOCOL.md §17;
docs/build/study3.md).

    uv run ceval s3 select                          # §17.2: pick, replay at 48k (dead-port proxy), qualify
    uv run ceval s3 writer-input                    # per-session input for the probe writers
    uv run ceval s3 dump-session <sid> [--from i] [--until j] [--full i] [--grep RE]
    uv run ceval s3 screen                          # §17.3 screens -> runs/S3/probes.json
    uv run ceval s3 freeze                          # sha256 -> results/S3/probes.sha256
    uv run ceval s3 run [--workers 4]               # §17.4 (model calls: ask first)
    uv run ceval s3 blind-pack | blind-merge        # §17.5
    uv run ceval s3 analyze                         # §17.6 -> results/S3/report.md

Session text, probes and replies stay under runs/S3/ (gitignored). results/S3/
holds hashes, counts and aggregate per-turn fields only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import realpi_replay as R
from .realpi_implicit import MODEL, TURN_TIMEOUT_S, _delta_usage, recovery_tools

SEED = 20261008
CAP_SESSIONS, CAP_PER_SOURCE, MIN_COMPACTIONS = 25, 8, 3
ARMS = ("mech", "pi-vcc")
KINDS = ("probe", "control")
EXP = "REALPI-48k"  # realpi_replay's experiment name at the 48k window
BASE = Path("runs/S3")
RES = Path("results/S3")
CAP_IN, CAP_OUT = 40_000_000, 1_000_000
MAX_PROBES_WRITER, MAX_PROBES_KEPT = 6, 4
LOCATIONS = ("user", "assistant", "tool_output", "thinking")
TYPES = ("constraint", "decision", "tool_output_fact", "failed_attempt")
TRUNC, HEAD = 2000, 800
BLIND_FRACTION = 0.25


def _root() -> Path:
    from .config import repo_root
    return repo_root()


def _jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# ----------------------------------------------------------------- select (§17.2)
def candidates(data: Path) -> list[dict]:
    """Licence-clear (MIT / CC-BY-4.0), uncompacted sessions with >= 128k
    estimated tokens on the active branch."""
    out = []
    for r in _jl(data / "manifest.jsonl"):
        p = data / r["file"]
        lic = R.licence(r["source_dataset"])
        if not lic or not p.exists():
            continue
        _, chain = R.active_path(p)
        if any(e.get("type") == "compaction" for e in chain):
            continue
        tok = sum(R.entry_chars(e) for e in chain if e.get("type") in R.KEEP_TYPES) / R.CPT
        if tok >= R.MIN_TOKENS:
            out.append({"file": r["file"], "src": r["source_dataset"], "licence": lic, "tokens": round(tok),
                        "sid": R.sid_of(r["file"])})
    return out


def pick(cands: list[dict], cap: int = CAP_SESSIONS, per_source: int = CAP_PER_SOURCE, seed: int = SEED) -> list[dict]:
    """Seeded round-robin across sources (sources and files in sorted order,
    each source's list shuffled), at most `per_source` from any one source."""
    rng = random.Random(seed)
    by_src: dict[str, list] = {}
    for c in sorted(cands, key=lambda c: c["file"]):
        by_src.setdefault(c["src"], []).append(c)
    for v in by_src.values():
        rng.shuffle(v)
        del v[per_source:]
    picked = []
    while len(picked) < cap and any(by_src.values()):
        for s in sorted(by_src):
            if by_src[s] and len(picked) < cap:
                picked.append(by_src[s].pop())
    return sorted(picked, key=lambda c: c["sid"])


def cmd_select(root: Path, data: str) -> dict:
    cands = candidates((root / data).resolve())
    picked = pick(cands)
    rep_dir = root / BASE / "replay"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "selection.json").write_text(json.dumps({"selected": picked}, indent=1))
    for m in ARMS:  # resumable; dead-port proxy, 0 requests checked below
        R.main(["run", "--method", m, "--window", "48k", "--sessions", f"0-{len(picked) - 1}",
                "--data", data, "--out", str(BASE / "replay")])
    reps = {m: {r["sid"]: r for r in _jl(rep_dir / f"replays-{m}-48k.jsonl")} for m in ARMS}
    rows = {m: _jl(rep_dir / f"rows-{m}-48k.jsonl") for m in ARMS}
    sessions = []
    for c in picked:
        s = {"sid": c["sid"], "src": c["src"], "licence": c["licence"], "tokens": c["tokens"]}
        why = []
        for m in ARMS:
            rp = reps[m].get(c["sid"])
            if rp is None:
                why.append(f"{m}: not replayed")
                continue
            last = [r for r in rows[m] if r["sid"] == c["sid"]][-1:] or [{}]
            s[m] = {"compactions": rp["compactions"], "model_requests": rp["model_requests"],
                    "failed": rp["failed"], "entries": rp["entries"],
                    "last_cut_entry_index": last[0].get("cut_entry_index")}
            if rp["failed"] or rp["model_requests"]:
                why.append(f"{m}: replay failed or made model requests")
            elif rp["compactions"] < MIN_COMPACTIONS:
                why.append(f"{m}: {rp['compactions']} compactions < {MIN_COMPACTIONS}")
        s["qualifies"], s["dropped_because"] = not why, why
        sessions.append(s)
    out = {"seed": SEED, "candidates": len(cands), "candidate_sources": R._count(c["src"] for c in cands),
           "picked": len(picked), "qualifying": sum(s["qualifies"] for s in sessions),
           "model_requests_total": sum(r["model_requests"] for m in ARMS for r in reps[m].values()),
           "sessions": sessions}
    (root / RES).mkdir(parents=True, exist_ok=True)
    (root / RES / "sessions.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def qualifying(root: Path) -> list[dict]:
    return [s for s in json.loads((root / RES / "sessions.json").read_text())["sessions"] if s["qualifies"]]


def raw_chain(root: Path, sid: str) -> list[dict]:
    """The raw session's active-branch context entries (the replay's entry indices)."""
    local = root / BASE / "replay" / "selection.json"
    if local.exists():
        sel = {x["sid"]: x for x in json.loads(local.read_text())["selected"]}
    else:  # e.g. a cloud checkout without the replay tree: map hashes via the public manifest
        man = root / "runs/realpi/hf/manifest.jsonl"
        sel = {R.sid_of(m["file"]): m for m in (json.loads(l) for l in man.read_text().splitlines() if l.strip())}
    _, chain = R.active_path(root / "runs/realpi/hf" / sel[sid]["file"])
    return [e for e in chain if e.get("type") in R.KEEP_TYPES]


def arm_main(root: Path, arm: str, sid: str) -> Path:
    from . import driver as D
    return D.sessions_dir(R._cfg_for("48k", (root / BASE / "replay").resolve()), EXP, arm, sid) / "main.jsonl"


# ----------------------------------------------------------------- rendering
def _cut(s: str, full: bool) -> str:
    if full or len(s) <= TRUNC:
        return s
    return f"{s[:HEAD]}\n…[{len(s) - 2 * HEAD} chars cut; --full to show]…\n{s[-HEAD:]}"


def entry_parts(e: dict) -> list[tuple[str, str]]:
    """(label, text) pieces of one context entry, untruncated."""
    m = R.entry_message(e) or {}
    role = m.get("role")
    if role == "assistant":
        out = []
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                out.append(("assistant", b.get("text") or ""))
            elif t == "thinking":
                out.append(("thinking", b.get("thinking") or ""))
            elif t == "toolCall":
                out.append(("tool_call", f"{b.get('name')}({R._args_json(b.get('arguments'))})"))
        return out
    if role == "toolResult":
        return [(f"tool_output {m.get('toolName', '')}".strip(), R._text(m.get("content")))]
    if role == "bashExecution":
        return [("bash", f"$ {m.get('command', '')}\n{m.get('output', '')}")]
    if role in ("compactionSummary", "branchSummary"):
        return [(role, m.get("summary", "") or "")]
    return [(role or "?", R._text(m.get("content")))]


def entry_text(e: dict) -> str:
    return "\n".join(t for _, t in entry_parts(e))


def render(chain: list[dict], lo: int = 0, hi: int | None = None, full: int | None = None,
           grep: str | None = None) -> str:
    hi = len(chain) if hi is None else min(hi, len(chain))
    out = []
    rx = re.compile(grep) if grep else None
    for i in range(lo, hi):
        parts = entry_parts(chain[i])
        if rx and not any(rx.search(t) for _, t in parts):
            continue
        for lab, t in parts:
            out.append(f"[{i}] {lab}: {_cut(t, full == i)}")
    return "\n".join(out)


def writer_inputs(root: Path) -> list[dict]:
    """One input per qualifying session: its hash, entry count (the cut is the
    end) and the early range (entries before the earliest last-compaction cut
    of the two arms; neutral: it says nothing about any method)."""
    out = []
    for s in qualifying(root):
        early = min(s[m]["last_cut_entry_index"] for m in ARMS)
        out.append({"sid": s["sid"], "entries": s["mech"]["entries"], "early_until": early,
                    "output": str(BASE / "probes_raw" / f"{s['sid']}.json")})
    return out


# ----------------------------------------------------------------- screen (§17.3)
def keys_of(p: dict) -> list[str]:
    k = p.get("dependency_text")
    return [x for x in ([k] if isinstance(k, str) else k or []) if isinstance(x, str) and x.strip()]


def rule_kind(rule) -> str | None:
    """'mechanical', 'judged' or None (invalid)."""
    if not isinstance(rule, dict):
        return None
    if isinstance(rule.get("judged"), str) and rule["judged"].strip() and not rule.get("must") and not rule.get("must_not"):
        return "judged"
    must, must_not = rule.get("must") or [], rule.get("must_not") or []
    if "judged" in rule or not isinstance(must, list) or not isinstance(must_not, list) or not (must or must_not):
        return None
    try:
        for r in must + must_not:
            re.compile(r)
    except (re.error, TypeError):
        return None
    return "mechanical"


def restated(keys: list[str], texts: list[str]) -> dict:
    """Each key string (whitespace-normalised) must occur exactly once in the
    raw session up to the cut: absent -> 'not_found', more than once ->
    'restated'."""
    hay = R.norm("\n".join(texts))
    res = {}
    for k in keys:
        n = hay.count(R.norm(k))
        res[k] = "ok" if n == 1 else ("not_found" if n == 0 else "restated")
    return res


REQUIRED = ("id", "dependency_text", "dependency_entry_index", "location", "type", "instruction", "pass_rule",
            "control_instruction")


def screen_session(sid: str, raw: list[dict], chain: list[dict]) -> tuple[list[dict], list[dict]]:
    texts = [entry_text(e) for e in chain]
    kept, log = [], []
    for n, p in enumerate(raw[:MAX_PROBES_WRITER] if isinstance(raw, list) else []):
        why = None
        if not isinstance(p, dict) or any(f not in p for f in REQUIRED):
            why = "missing_fields"
        elif not keys_of(p):
            why = "no_key_strings"
        else:
            r = restated(keys_of(p), texts)
            kind = rule_kind(p["pass_rule"])
            if "not_found" in r.values():
                why = "key_not_found"
            elif "restated" in r.values():
                why = "restated"
            elif kind is None:
                why = "invalid_rule"
            elif len(kept) >= MAX_PROBES_KEPT:
                why = "over_cap"
        entry = {"sid": sid, "writer_order": n, "id": p.get("id") if isinstance(p, dict) else None,
                 "kept": why is None, "why": why}
        if isinstance(p, dict) and why is None:
            q = dict(p, sid=sid, pid=f"{sid}-{p['id']}", rule_kind=rule_kind(p["pass_rule"]), dependency_text=keys_of(p))
            kept.append(q)
            entry.update(location=p["location"], type=p["type"])
        log.append(entry)
    if isinstance(raw, list) and len(raw) > MAX_PROBES_WRITER:
        log.append({"sid": sid, "writer_order": MAX_PROBES_WRITER, "kept": False, "why": f"writer_over_{MAX_PROBES_WRITER}",
                    "count": len(raw) - MAX_PROBES_WRITER})
    return kept, log


def cmd_screen(root: Path, chain_of=None) -> dict:
    chain_of = chain_of or (lambda sid: raw_chain(root, sid))
    probes, log, missing = [], [], []
    early = {x["sid"]: x["early_until"] for x in writer_inputs(root)}
    for s in qualifying(root):
        f = root / BASE / "probes_raw" / f"{s['sid']}.json"
        if not f.exists():
            missing.append(s["sid"])
            continue
        try:
            raw = json.loads(f.read_text())
            raw = raw.get("probes", raw) if isinstance(raw, dict) else raw
        except ValueError:
            log.append({"sid": s["sid"], "kept": False, "why": "unparseable_json"})
            continue
        k, lg = screen_session(s["sid"], raw, chain_of(s["sid"]))
        probes += k
        log += lg
    (root / BASE).mkdir(parents=True, exist_ok=True)
    (root / BASE / "probes.json").write_text(json.dumps(probes, indent=1, ensure_ascii=False) + "\n")
    (root / BASE / "screen_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    counts = {"sessions_with_writer_output": len({x["sid"] for x in log}), "sessions_missing_output": len(missing),
              "proposed": sum(1 for x in log if "writer_order" in x and not str(x.get("why", "")).startswith("writer_over"))
              + sum(x.get("count", 0) for x in log if str(x.get("why", "")).startswith("writer_over")),
              "kept": len(probes), "dropped_by_reason": R._count(x["why"] for x in log if not x["kept"]),
              "kept_judged": sum(p["rule_kind"] == "judged" for p in probes),
              "kept_by_location": R._count(p["location"] for p in probes),
              "kept_by_type": R._count(p["type"] for p in probes),
              "kept_dependency_at_or_after_early_until": sum(
                  isinstance(p["dependency_entry_index"], int) and p["dependency_entry_index"] >= early.get(p["sid"], 0)
                  for p in probes)}
    (root / RES).mkdir(parents=True, exist_ok=True)
    (root / RES / "screening.json").write_text(json.dumps(counts, indent=1) + "\n")
    return counts


# ----------------------------------------------------------------- freeze (§17.3)
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_freeze(root: Path) -> str:
    h, f = sha(root / BASE / "probes.json"), root / RES / "probes.sha256"
    if f.exists() and f.read_text().split()[0] != h:
        raise SystemExit(f"refusing: {f} already holds a different hash (the probe set is frozen)")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"{h}  runs/S3/probes.json\n")
    return h


def frozen_probes(root: Path) -> list[dict]:
    f = root / RES / "probes.sha256"
    if not f.exists():
        raise SystemExit("refusing: probes are not frozen (run `ceval s3 freeze`)")
    if f.read_text().split()[0] != sha(root / BASE / "probes.json"):
        raise SystemExit("refusing: runs/S3/probes.json does not match results/S3/probes.sha256")
    return json.loads((root / BASE / "probes.json").read_text())


# ----------------------------------------------------------------- run (§17.4)
def apply_rule(rule: dict, text: str) -> bool | None:
    """Mechanical pass (all `must` match, no `must_not` matches); None if judged."""
    if rule_kind(rule) != "mechanical":
        return None
    if not text.strip():
        return False
    return all(re.search(r, text) for r in rule.get("must") or []) and \
        not any(re.search(r, text) for r in rule.get("must_not") or [])


def context_text_at_cut(main: Path) -> str:
    """The arm's context at the cut: last compaction's summary + kept entries."""
    from . import pisession as S
    ents = S.read_entries(main)
    last = S.last_compaction(ents)
    if last is None:
        return "\n".join(entry_text(e) for e in ents if e.get("type") in R.KEEP_TYPES)
    idx = {e.get("id"): i for i, e in enumerate(ents)}
    start = idx.get(last.get("firstKeptEntryId") or "", idx[last["id"]])
    kept = [entry_text(e) for i, e in enumerate(ents) if i >= start and e is not last and e.get("type") in R.KEEP_TYPES]
    return "\n".join([last.get("summary", "") or ""] + kept)


def verbatim(keys: list[str], ctx: str) -> bool:
    hay = R.norm(ctx)
    return all(R.norm(k) in hay for k in keys)


def local_model() -> bool:
    """S3_MODEL=qwen: run the probe turns on the study's local model (Qwen via
    llama.cpp, the Study 1 model settings) instead of gpt-6.1-sol. Pilot use
    (2026-10-08); the pre-registered Study 3 model is gpt-6.1-sol (§17.4)."""
    return os.environ.get("S3_MODEL") == "qwen"


def probe_cfg(root: Path):
    from .config import Config, load_config
    from .realpi_implicit import probe_cfg as _pc
    import copy
    base = R._cfg_for("48k", (root / BASE / "replay").resolve())
    if local_model():
        # Study 1's model, sampling and 128k declaration for the probe turns (the
        # compacted sessions come from the 48k replay; a probe turn must not
        # trigger a further compaction), the usage proxy to the real server.
        main = load_config()
        raw = copy.deepcopy(base.raw)
        raw["model"] = copy.deepcopy(main.raw["model"])
        raw["pi"]["models_file"] = str(main.path(main.raw["pi"]["models_file"]))
        raw["usage_proxy"] = copy.deepcopy(main.raw["usage_proxy"])
        raw["bridge"]["turn_timeout_s"] = TURN_TIMEOUT_S
        return Config(raw=raw, root=base.root)
    raw = copy.deepcopy(_pc(root).raw)  # model, empty models file, timeout, proxy off
    raw["run"] = copy.deepcopy(base.raw["run"])
    return Config(raw=raw, root=base.root)


def real_turn(root: Path, cfg):
    """turn_fn(probe, arm, kind, message) -> (TurnResult, delta entries)."""
    from . import driver as D
    from . import pisession as S

    def fn(p, arm, kind, msg):
        main = D.sessions_dir(cfg, EXP, arm, p["sid"]) / "main.jsonl"
        name = f"s3-{p['pid']}-{kind}"
        fork = D.prepare_fork(cfg, main, main.parent / "probes" / f"{name}.jsonl", prune=False)
        env = D.arm_env(cfg, arm, EXP, p["sid"], {"MECH_COMPACT_MAX_SUMMARY_CHARS": "40000"} if arm.startswith("mech") else {})
        env.update(D.copy_probe_state(cfg, EXP, arm, p["sid"], name))
        want = f"{cfg['model']['provider']}/{cfg['model']['id']}" if local_model() else f"{MODEL['provider']}/{MODEL['id']}"
        assert env["MECH_COMPACT_LOOKUP_MODEL"] == want
        n0 = S.line_count(fork)
        tr = D.turn(cfg, fork, D.agent_cwd(cfg, EXP, arm, p["sid"]), env, msg, builtin_tools=False)
        return tr, S.entries_after(fork, n0)
    return fn


def run_one(p: dict, arm: str, kind: str, turn_fn, dep_verbatim: bool | None) -> tuple[dict, str]:
    msg = p["instruction" if kind == "probe" else "control_instruction"]
    from .realpi_implicit import MEMORY_NOTE, memory_note
    if memory_note():  # S3_MEMORY_NOTE=1: the arm-neutral "no shell or file access" note
        msg = msg.rstrip() + "\n\n" + MEMORY_NOTE
    t0 = time.monotonic()
    row = {"probe": p["pid"], "sid": p["sid"], "arm": arm, "kind": kind, "location": p["location"],
           "type": p["type"], "rule_kind": p["rule_kind"], "dep_verbatim_at_cut": dep_verbatim}
    try:
        tr, delta = turn_fn(p, arm, kind, msg)
    except Exception as ex:  # recorded, never retried
        row.update(ok=False, error=f"{type(ex).__name__}: {ex}", pass_mech=None, recovery_calls=None,
                   input_tokens=0, cache_read_tokens=0, cache_write_tokens=0, output_tokens=0, cost_usd=0.0,
                   wall_s=round(time.monotonic() - t0, 1), reply_chars=0)
        return row, ""
    u = _delta_usage(delta)
    tools = recovery_tools(arm)
    row.update(ok=tr.ok, error=tr.error, pass_mech=apply_rule(p["pass_rule"], tr.text) if tr.ok else
               (None if p["rule_kind"] == "judged" else False),
               recovery_calls=sum(c.get("name") in tools for c in tr.calls),
               other_tool_calls=sorted({str(c.get("name")) for c in tr.calls if c.get("name") not in tools}),
               input_tokens=u["input"], cache_read_tokens=u["cache_read"], cache_write_tokens=u["cache_write"],
               output_tokens=u["output"], cost_usd=round(u["cost"], 6), models=u["models"],
               compactions_during_turn=sum(e.get("type") == "compaction" for e in delta),
               wall_s=round(time.monotonic() - t0, 1), reply_chars=len(tr.text))
    return row, tr.text


def _tin(r: dict) -> int:
    return r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"]


def cmd_run(root: Path, workers: int = 4, turn_fn=None, ctx_of=None, proxy=None) -> dict:
    probes = frozen_probes(root)  # refuses on a hash mismatch, before any model call
    out, replies = root / RES / "runs.jsonl", root / BASE / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    turn_is_real = turn_fn is None
    if turn_fn is None:
        from . import driver as D
        from . import usageproxy
        cfg = probe_cfg(root)
        D.set_usage_proxy(cfg, None)
        if local_model() and workers == 1 and cfg.raw.get("usage_proxy", {}).get("enabled"):
            # Every model call of a turn, including mech's lookup subagent (which the
            # session file does not record), goes through the proxy; one worker so
            # each turn's calls are its own.
            proxy = usageproxy.UsageProxy(cfg["usage_proxy"]["upstream"]).start()
            D.set_usage_proxy(cfg, proxy, root / BASE / ".usage-proxy-models.json")
        turn_fn = real_turn(root, cfg)
    ctx_of = ctx_of or (lambda arm, sid: context_text_at_cut(arm_main(root, arm, sid)))
    prev = _jl(out)
    done = {(r["probe"], r["arm"], r["kind"]) for r in prev}
    tot = {"in": sum(_tin(r) for r in prev), "out": sum(r["output_tokens"] for r in prev)}
    stop = threading.Event()
    capped = lambda: not local_model() and (tot["in"] > CAP_IN or tot["out"] > CAP_OUT)  # §17.4 GPT cap; none for Qwen (§17.10)
    if capped():
        stop.set()
    ctx: dict = {}
    lock = threading.Lock()
    jobs = [(p, a, k) for p in probes for a in ARMS for k in KINDS if (p["pid"], a, k) not in done]

    def job(p, a, k):
        if stop.is_set():
            return
        if turn_is_real:
            # Stop cleanly (nothing recorded) when the model may not be used now:
            # runs/PAUSE or, without CEVAL_IGNORE_WINDOW, outside the nightly window.
            # Rerunning resumes; recorded failures are never retried.
            from . import schedule
            ok, why = schedule.llm_allowed(cfg)
            if not ok:
                print(f"stopping: {why}", flush=True)
                stop.set()
                return
        with lock:
            if (a, p["sid"]) not in ctx:
                ctx[(a, p["sid"])] = ctx_of(a, p["sid"])
            dv = verbatim(p["dependency_text"], ctx[(a, p["sid"])])
        mark = proxy.mark() if proxy is not None else None
        row, text = run_one(p, a, k, turn_fn, dv)
        if proxy is not None:
            from . import usageproxy
            row["usage"] = usageproxy.summarize(proxy.since(mark))  # all calls of this turn
        with lock:
            (replies / f"{p['pid']}-{a}-{k}.txt").write_text(text)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            tot["in"] += _tin(row)
            tot["out"] += row["output_tokens"]
            print("RUN", json.dumps({x: row.get(x) for x in ("probe", "arm", "kind", "ok", "pass_mech", "recovery_calls",
                                                              "output_tokens", "cost_usd", "wall_s")}), "TOTAL", tot,
                  flush=True)
            if capped():
                print("TOKEN CAP EXCEEDED: stopping", flush=True)
                stop.set()

    try:
        with ThreadPoolExecutor(workers) as ex:
            for f in [ex.submit(job, *j) for j in jobs]:
                f.result()
    finally:
        if proxy is not None and turn_is_real:
            from . import driver as D
            D.set_usage_proxy(cfg, None)
            proxy.stop()
    return {"rows": len(_jl(out)), "total": tot, "capped": stop.is_set()}


# ----------------------------------------------------------------- blind scoring (§17.5)
def _criterion(p: dict) -> str:
    r = p["pass_rule"]
    if p["rule_kind"] == "judged":
        return r["judged"]
    parts = [f"must match /{x}/" for x in r.get("must") or []] + [f"must not match /{x}/" for x in r.get("must_not") or []]
    return "Pass if the reply " + "; ".join(parts)


def cmd_blind_pack(root: Path, seed: int = SEED) -> dict:
    """primary: every reply of a judged probe. second: a seeded 25% of all
    replies (re-scored by a second blind scorer for Cohen's kappa). Items carry
    no arm, session or method; ids are shuffled; the key file is separate."""
    probes = {p["pid"]: p for p in frozen_probes(root)}
    rows = sorted(_jl(root / RES / "runs.jsonl"), key=lambda r: (r["probe"], r["arm"], r["kind"]))
    rng = random.Random(seed)
    second = set(rng.sample(range(len(rows)), round(BLIND_FRACTION * len(rows))))
    d = root / BASE / "blind"
    d.mkdir(parents=True, exist_ok=True)
    key, counts = {}, {}
    for pack, sel in (("primary", [i for i, r in enumerate(rows) if probes[r["probe"]]["rule_kind"] == "judged"]),
                      ("second", sorted(second))):
        rng.shuffle(sel)
        items = []
        for n, i in enumerate(sel):
            r, p = rows[i], probes[rows[i]["probe"]]
            iid = f"{pack[0]}{n:04d}"
            key[iid] = {"probe": r["probe"], "arm": r["arm"], "kind": r["kind"]}
            items.append({"item": iid, "dependency": p.get("dependency") or "; ".join(p["dependency_text"]),
                          "dependency_key_strings": p["dependency_text"],
                          "instruction": p["instruction" if r["kind"] == "probe" else "control_instruction"],
                          "criterion": _criterion(p),
                          "reply": (root / BASE / "replies" / f"{r['probe']}-{r['arm']}-{r['kind']}.txt").read_text()})
        (d / f"{pack}_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
        counts[pack] = len(items)
    (d / "key.json").write_text(json.dumps(key, indent=1))
    return counts


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def cmd_blind_merge(root: Path) -> dict:
    """Final pass = the frozen mechanical rule, or the primary blind score for
    judged probes. Kappa: second scorer vs the final value on its items."""
    d = root / BASE / "blind"
    key = json.loads((d / "key.json").read_text())

    def scores(pack):
        f = d / f"{pack}_scores.json"
        raw = json.loads(f.read_text()) if f.exists() else {}
        return {(key[i]["probe"], key[i]["arm"], key[i]["kind"]): bool(v["pass"] if isinstance(v, dict) else v)
                for i, v in raw.items() if i in key}
    prim, sec = scores("primary"), scores("second")
    rows = _jl(root / RES / "runs.jsonl")
    for r in rows:
        k = (r["probe"], r["arm"], r["kind"])
        r["pass"] = (prim.get(k) if r["rule_kind"] == "judged" else r["pass_mech"]) if r["ok"] else False
        if r["rule_kind"] == "judged" and r["ok"] and k not in prim:
            r["pass"] = None  # not yet scored
    final = {(r["probe"], r["arm"], r["kind"]): r["pass"] for r in rows}
    pairs = [(final[k], v) for k, v in sec.items() if final.get(k) is not None]
    agr = {"second_items_scored": len(sec), "pairs": len(pairs),
           "kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
           "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
           "judged_unscored": sum(r["pass"] is None for r in rows)}
    (root / RES / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / RES / "agreement.json").write_text(json.dumps(agr, indent=1) + "\n")
    return agr


# ----------------------------------------------------------------- analysis (§17.6)
def sign_test(b: int, c: int) -> float:
    """Exact two-sided binomial sign test on discordant pairs (b vs c, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pairs_of(rows: list[dict]) -> tuple[dict, dict]:
    """(per probe: {arm: probe-turn pass}, exclusion counts). Excluded: a
    control that failed (or errored, or is unscored) in either arm; a probe
    turn that errored or is unscored."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["probe"], {})[(r["arm"], r["kind"])] = r
    keep, ex = {}, {"control_failed": 0, "probe_turn_error_or_unscored": 0, "incomplete": 0}
    for pid, d in by.items():
        if any((a, k) not in d for a in ARMS for k in KINDS):
            ex["incomplete"] += 1
        elif any(d[(a, "control")]["pass"] is not True for a in ARMS):
            ex["control_failed"] += 1
        elif any(not d[(a, "probe")]["ok"] or d[(a, "probe")]["pass"] is None for a in ARMS):
            ex["probe_turn_error_or_unscored"] += 1
        else:
            keep[pid] = {a: d[(a, "probe")] for a in ARMS}
    return keep, ex


def h11(keep: dict) -> dict:
    b = sum(v["mech"]["pass"] and not v["pi-vcc"]["pass"] for v in keep.values())
    c = sum(v["pi-vcc"]["pass"] and not v["mech"]["pass"] for v in keep.values())
    return {"probes": len(keep), "mech_pass": sum(v["mech"]["pass"] for v in keep.values()),
            "vcc_pass": sum(v["pi-vcc"]["pass"] for v in keep.values()),
            "discordant_mech_only": b, "discordant_vcc_only": c, "p_sign_two_sided": round(sign_test(b, c), 6)}


def analyze(rows: list[dict]) -> dict:
    from . import stats
    keep, ex = pairs_of(rows)
    out = {"exclusions": ex, "h11": h11(keep)}
    by_s: dict = {}
    for v in keep.values():
        by_s.setdefault(v["mech"]["sid"], []).append(int(v["mech"]["pass"]) - int(v["pi-vcc"]["pass"]))
    diffs = [sum(x) / len(x) for x in by_s.values()]
    out["sensitivity_session_sign_flip"] = (
        {k: stats.sign_flip_test(diffs)[k] for k in ("statistic", "p", "n_sessions")} if diffs else None)
    for f in ("location", "type"):
        out[f"by_{f}"] = {}
        for val in sorted({v["mech"][f] for v in keep.values()}):
            sub = {k: v for k, v in keep.items() if v["mech"][f] == val}
            out[f"by_{f}"][val] = h11(sub)
    rec = {}
    for a in ARMS:
        pr = [v[a] for v in keep.values()]
        other = [x for x in ARMS if x != a][0]
        lost = [v[a] for v in keep.values() if not v[a]["pass"] and v[other]["pass"]]
        rec[a] = {"probe_turns": len(pr), "turns_with_recovery_call": sum((r["recovery_calls"] or 0) > 0 for r in pr),
                  "lost_discordant": len(lost), "lost_with_recovery_call": sum((r["recovery_calls"] or 0) > 0 for r in lost),
                  "dep_verbatim_at_cut": sum(bool(r["dep_verbatim_at_cut"]) for r in pr),
                  "pass_if_verbatim": _rate([r["pass"] for r in pr if r["dep_verbatim_at_cut"]]),
                  "pass_if_not_verbatim": _rate([r["pass"] for r in pr if not r["dep_verbatim_at_cut"]])}
    out["recovery_and_verbatim"] = rec
    out["cost"] = {f"{a}/{k}": {"turns": len(rs), "input_tokens_median": _med([_tin(r) for r in rs]),
                                "output_tokens_median": _med([r["output_tokens"] for r in rs]),
                                "wall_s_median": _med([r["wall_s"] for r in rs]),
                                "cost_usd_total": round(sum(r["cost_usd"] for r in rs), 4),
                                "errors": sum(not r["ok"] for r in rs)}
                   for a in ARMS for k in KINDS for rs in [[r for r in rows if r["arm"] == a and r["kind"] == k]]}
    return out


def _rate(xs: list) -> list | None:
    return [sum(bool(x) for x in xs), len(xs)] if xs else None


def _med(xs: list):
    return statistics.median(xs) if xs else None


def report(a: dict, agr: dict | None) -> str:
    h = a["h11"]
    L = ["# Study 3: implicit-context probes (PROTOCOL.md §17)", "",
         "Generated by `ceval s3 analyze` from `results/S3/scored.jsonl`.", "",
         f"Excluded probes: {a['exclusions']}.", "",
         "## H11 (confirmatory): mech passes more probes than pi-vcc", "",
         f"Probes analysed: {h['probes']}. mech passes {h['mech_pass']}, pi-vcc passes {h['vcc_pass']}. "
         f"Discordant pairs: {h['discordant_mech_only']} mech only, {h['discordant_vcc_only']} pi-vcc only. "
         f"Exact two-sided sign test p = {h['p_sign_two_sided']} (alpha 0.05).", "",
         f"Sensitivity (session-level sign-flip on per-session pass-rate differences): {a['sensitivity_session_sign_flip']}.",
         ""]
    for f in ("location", "type"):
        L += [f"## By dependency {f}", "", "| " + f + " | probes | mech | pi-vcc | mech only | pi-vcc only | p |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['probes']} | {v['mech_pass']} | {v['vcc_pass']} | {v['discordant_mech_only']} | "
              f"{v['discordant_vcc_only']} | {v['p_sign_two_sided']} |" for k, v in a[f"by_{f}"].items()] + [""]
    L += ["## Recovery tool and verbatim retention", "", "| arm | probe turns | with recovery call | lost discordant | "
          "of which called recovery | dependency verbatim at cut | pass if verbatim | pass if not |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['probe_turns']} | {v['turns_with_recovery_call']} | {v['lost_discordant']} | "
          f"{v['lost_with_recovery_call']} | {v['dep_verbatim_at_cut']} | {v['pass_if_verbatim']} | {v['pass_if_not_verbatim']} |"
          for k, v in a["recovery_and_verbatim"].items()] + [""]
    L += ["## Tokens and time per turn (medians)", "", "| arm/kind | turns | input tokens | output tokens | wall s | cost USD | errors |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['turns']} | {v['input_tokens_median']} | {v['output_tokens_median']} | {v['wall_s_median']} | "
          f"{v['cost_usd_total']} | {v['errors']} |" for k, v in a["cost"].items()] + [""]
    if agr:
        L += ["## Blind scoring agreement", "", f"{agr}", ""]
    return "\n".join(L)


def cmd_analyze(root: Path) -> dict:
    frozen_probes(root)
    rows = _jl(root / RES / "scored.jsonl")
    a = analyze(rows)
    agf = root / RES / "agreement.json"
    (root / RES / "analysis.json").write_text(json.dumps(a, indent=1) + "\n")
    (root / RES / "report.md").write_text(report(a, json.loads(agf.read_text()) if agf.exists() else None) + "\n")
    return a


# ----------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ceval s3", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").add_argument("--data", default="runs/realpi/hf")
    sub.add_parser("writer-input")
    d = sub.add_parser("dump-session")
    d.add_argument("sid")
    d.add_argument("--from", dest="lo", type=int, default=0)
    d.add_argument("--until", type=int, help="exclusive; default: the cut (end of session)")
    d.add_argument("--full", type=int, help="show this entry untruncated")
    d.add_argument("--grep", help="only entries matching this regex")
    sub.add_parser("screen")
    sub.add_parser("freeze")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    for n in ("blind-pack", "blind-merge", "analyze"):
        sub.add_parser(n)
    a = ap.parse_args(argv)
    root = _root()
    if a.cmd == "select":
        res = cmd_select(root, a.data)
        print(json.dumps({k: v for k, v in res.items() if k != "sessions"}, indent=1))
    elif a.cmd == "writer-input":
        ins = writer_inputs(root)
        (root / BASE).mkdir(parents=True, exist_ok=True)
        (root / BASE / "writer_input.json").write_text(json.dumps(ins, indent=1))
        print(json.dumps(ins, indent=1))
    elif a.cmd == "dump-session":
        print(render(raw_chain(root, a.sid), a.lo, a.until, a.full, a.grep))
    elif a.cmd == "screen":
        print(json.dumps(cmd_screen(root), indent=1))
    elif a.cmd == "freeze":
        print(cmd_freeze(root))
    elif a.cmd == "run":
        print(json.dumps(cmd_run(root, a.workers)))
    elif a.cmd == "blind-pack":
        print(json.dumps(cmd_blind_pack(root)))
    elif a.cmd == "blind-merge":
        print(json.dumps(cmd_blind_merge(root)))
    else:
        print(json.dumps(cmd_analyze(root)["h11"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
