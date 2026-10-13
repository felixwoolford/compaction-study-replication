"""Model-free replay of long real pi sessions through mech-compact and pi-vcc
(docs/build/realpi-replay.md).

    uv run python -m ceval.realpi_replay select [--data runs/realpi/hf]
    uv run python -m ceval.realpi_replay run --method mech --window 48k [--sessions 0-39]
    uv run python -m ceval.realpi_replay summarize

Each selected session's active-branch entries are written, in order, into a
fresh session file in the arm's normal work area (driver.setup_workdir /
arm_env: exact extension set, private HOME, pi-vcc's private config). At every
turn boundary the context pi would build is estimated in tokens (chars / 3.4);
above window - reserve the real bridge `compact` runs. Every bridge call goes
through the usage proxy with a DEAD upstream, so a model request cannot reach
any server; requests are counted (must be 0). The shared server is never
contacted (no /tokenize: tokens are estimated from characters).

Data and per-replay rows stay under runs/realpi/ (gitignored). The committed
results are aggregates; sessions are named by a hash prefix of the file name.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path

CPT = 3.4  # chars per token (estimate; the shared server's tokenizer is not used)
DEAD = "http://127.0.0.1:9"
WINDOWS = {  # declared window -> settings (128k as Study 1, 48k as SWE1/LOCA1)
    "128k": {"window": 128_000, "reserve": 32_768, "keep": 20_000, "mech_cap": None,
             "models_file": "config/pi-models.json"},
    "48k": {"window": 48_000, "reserve": 8_192, "keep": 8_000, "mech_cap": 40_000,
            "models_file": "config/pi-models-swe1.json"},
}
METHODS = {"mech": "mech", "pi-vcc": "pi-vcc", "mech-direct": "mech-direct", "verbatim": "verbatim"}  # method -> arm in config/eval.config.json
MIN_TOKENS = 128_000
CAP_SESSIONS = 40
SEED = 20261007
MAX_COMPACTIONS = 400  # per replay; a guard against a compaction loop
CHUNK = 32  # retention granularity (chars)
# Upstream licences per source (docs/build/realpi-scoping.md, from the HF dataset cards).
LICENCE_CLEAR = {"woxQAQ/pi-web": "MIT", "julien-c/pi-sessions": "CC-BY-4.0"}
LICENCE_CLEAR_PREFIX = {"thomasmustier/": "MIT"}
KEEP_TYPES = ("message", "custom_message", "branch_summary")


def licence(src: str) -> str | None:
    if src in LICENCE_CLEAR:
        return LICENCE_CLEAR[src]
    for p, lic in LICENCE_CLEAR_PREFIX.items():
        if src.startswith(p):
            return lic
    return None


def sid_of(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:10]


# ----------------------------------------------------------------- loading
def active_path(path: Path) -> tuple[dict, list[dict]]:
    """(header, entries on the active branch = parent chain of the last entry)."""
    entries, header = [], None
    for line in path.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "session" and header is None:
            header = d
        elif isinstance(d, dict) and "id" in d:
            entries.append(d)
    by_id = {e["id"]: e for e in entries}
    chain, cur, seen = [], entries[-1] if entries else None, set()
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        cur = by_id.get(cur.get("parentId"))
    return header or {}, chain[::-1]


def _text(c) -> str:
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") or "" for b in c or [] if isinstance(b, dict) and b.get("type") == "text")


def _args_json(a) -> str:
    return a if isinstance(a, str) else json.dumps(a if a is not None else {}, ensure_ascii=False)


def msg_chars(m: dict) -> int:
    """Characters of one context message (text, thinking, tool-call arguments,
    tool output, bash command and output, summaries; images ignored)."""
    role = m.get("role")
    if role == "assistant":
        n = 0
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            n += len(b.get("text", "") or "") if t == "text" else len(b.get("thinking", "") or "") if t == "thinking" \
                else len(b.get("name", "")) + len(_args_json(b.get("arguments"))) if t == "toolCall" else 0
        return n
    if role == "bashExecution":
        return len(str(m.get("command", ""))) + len(str(m.get("output", "")))
    if role in ("compactionSummary", "branchSummary"):
        return len(m.get("summary", "") or "")
    return len(_text(m.get("content")))


def entry_message(e: dict) -> dict | None:
    """The context message an entry contributes (None: not in context)."""
    t = e.get("type")
    if t == "message":
        return e.get("message")
    if t == "custom_message":
        return {"role": "custom", "content": e.get("content")}
    if t == "branch_summary":
        return {"role": "branchSummary", "summary": e.get("summary", "")}
    return None


def entry_chars(e: dict) -> int:
    m = entry_message(e)
    return msg_chars(m) if m else 0


# ----------------------------------------------------------------- selection
def select(data: Path) -> dict:
    rows = [json.loads(x) for x in (data / "manifest.jsonl").read_text().splitlines() if x.strip()]
    cands = []
    for r in rows:
        p = data / r["file"]
        if not p.exists():
            continue
        _, chain = active_path(p)
        if any(e.get("type") == "compaction" for e in chain):
            continue
        tok = sum(entry_chars(e) for e in chain if e.get("type") in KEEP_TYPES) / CPT
        if tok >= MIN_TOKENS:
            cands.append({"file": r["file"], "src": r["source_dataset"], "tokens": round(tok),
                          "licence": licence(r["source_dataset"])})
    clear = [c for c in cands if c["licence"]]
    pool = clear if len(clear) >= CAP_SESSIONS else clear + [c for c in cands if not c["licence"]]
    # Seeded round-robin across sources, so no source dominates the cap.
    rng = random.Random(SEED)
    by_src: dict[str, list] = {}
    for c in sorted(pool, key=lambda c: c["file"]):
        by_src.setdefault(c["src"], []).append(c)
    for v in by_src.values():
        rng.shuffle(v)
    picked, srcs = [], sorted(by_src)
    while len(picked) < CAP_SESSIONS and any(by_src.values()):
        for s in srcs:
            if by_src[s] and len(picked) < CAP_SESSIONS:
                picked.append(by_src[s].pop())
    picked.sort(key=lambda c: sid_of(c["file"]))
    return {"candidates_uncompacted_ge_128k": len(cands), "licence_clear": len(clear),
            "licence_unclear": len(cands) - len(clear),
            "selected": [dict(c, sid=sid_of(c["file"])) for c in picked]}


# ----------------------------------------------------------------- retention
def norm(s: str) -> str:
    return " ".join(s.split())


def coverage(text: str, hay: str) -> int:
    """Characters of `text` retained verbatim in `hay` (both whitespace-normalised):
    `text` is cut into consecutive CHUNK-char pieces (the last may be shorter);
    a piece counts if it occurs anywhere in `hay`."""
    t = norm(text)
    return sum(len(t[i:i + CHUNK]) for i in range(0, len(t), CHUNK) if t[i:i + CHUNK] in hay)


def mech_signature(name: str, args, snippet: int = 160) -> str:
    """mech-compact's toolCallSignature (mech-compact.ts), in Python."""
    parts = []
    for k, v in (args.items() if isinstance(args, dict) else []):
        s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        if len(s) > snippet:
            s = f"{s[:snippet]}…[+{len(s) - snippet} chars]"
        parts.append(f"{k}={s}")
    return f"{name}({', '.join(parts)})"


def originals(entries: list[dict]) -> dict[str, list]:
    """User texts, assistant text blocks and tool calls in `entries`."""
    out = {"user": [], "assistant": [], "calls": []}
    for e in entries:
        if e.get("type") != "message":
            continue
        m = e["message"]
        if m.get("role") == "user":
            t = _text(m.get("content"))
            if t.strip():
                out["user"].append(t)
        elif m.get("role") == "assistant":
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    out["assistant"].append(b["text"])
                elif b.get("type") == "toolCall":
                    out["calls"].append(mech_signature(str(b.get("name")), b.get("arguments")))
    return out


def retention(pre: dict[str, list], summary: str) -> dict:
    hay = norm(summary)
    res = {}
    for k in ("user", "assistant"):
        tot = sum(len(norm(t)) for t in pre[k])
        kept = [coverage(t, hay) for t in pre[k]]
        res[f"{k}_chars"] = tot
        res[f"{k}_retained"] = round(sum(kept) / tot, 4) if tot else None
    calls = pre["calls"]
    hs = norm(summary)
    res["tool_calls"] = len(calls)
    res["tool_calls_retained"] = round(sum(norm(c) in hs for c in calls) / len(calls), 4) if calls else None
    return res


def mech_trim(entries: list[dict], comp: dict) -> dict:
    """mech-compact's size guard at checkpoint `comp`: the un-capped span lines
    of every span it renders (details.span.lines of the mech entries since the
    base), how many occur verbatim in the summary, user lines stubbed, spans
    marked trimmed. Like pisession.mech_checkpoint_stats, but a span line is
    matched as a substring, not as one '\\n'-separated summary line: real
    sessions have multi-line user and assistant texts, which a line split would
    count as dropped."""
    from . import pisession as S
    spans: list[list[str]] = []
    for e in entries:
        if e.get("type") == "compaction":
            lines = S._mech_span_lines(e)
            spans = [] if lines is None else spans + [lines]
        if e is comp:
            break
    summary = comp.get("summary", "") or ""
    flat = [ln for sp in spans for ln in sp]
    kept = sum(ln in summary for ln in flat)
    stubbed, tspans = summary.count(S.MECH_USER_STUB), summary.count(S.MECH_SPAN_TRIMMED_ATTR)
    return {"spans": len(spans), "span_lines": len(flat), "lines_kept": kept, "lines_dropped": len(flat) - kept,
            "user_lines_stubbed": stubbed, "spans_trimmed": tspans,
            "trimmed": len(flat) > kept or stubbed > 0 or tspans > 0}


# ----------------------------------------------------------------- context model
def context_after(entries: list[dict], comp: dict) -> int:
    """Chars of the context pi builds after compaction `comp` (the last
    compaction in `entries`, a linear file): its summary plus every context
    entry from firstKeptEntryId on (pi's buildSessionContext)."""
    idx = {e.get("id"): i for i, e in enumerate(entries)}
    ci = idx[comp["id"]]
    start = idx.get(comp.get("firstKeptEntryId") or "", ci)
    return len(comp.get("summary", "") or "") + sum(
        entry_chars(e) for i, e in enumerate(entries) if i >= start and i != ci and e.get("type") in KEEP_TYPES)


def boundary(e: dict) -> bool:
    """A turn boundary falls before a user or assistant message (after the
    previous turn's tool results)."""
    return e.get("type") == "message" and (e.get("message") or {}).get("role") in ("user", "assistant")


def should_compact(nxt: dict | None, since: int, ctx_chars: int, threshold_tokens: int) -> bool:
    """Compact before entry `nxt` (None: the end of the session) if it is a
    turn boundary, something was appended since the last compaction, and the
    estimated context exceeds window - reserve."""
    return (nxt is None or boundary(nxt)) and since > 0 and ctx_chars / CPT > threshold_tokens


# ----------------------------------------------------------------- replay
def replay(cfg, data: Path, sel: dict, method: str, win: str, out_rows: Path, proxy) -> dict:
    from . import driver as D
    from . import pisession as S

    w = WINDOWS[win]
    arm = METHODS[method]
    exp = f"REALPI-{win}"
    sid = sel["sid"]
    _, chain = active_path(data / sel["file"])
    chain = [e for e in chain if e.get("type") in KEEP_TYPES]
    overrides = {"MECH_COMPACT_MAX_SUMMARY_CHARS": str(w["mech_cap"])} if (method.startswith("mech") and w["mech_cap"]) else {}
    env = D.arm_env(cfg, arm, exp, sid, overrides)
    wd = D.setup_workdir(cfg, exp, arm, sid)
    sess = D.sessions_dir(cfg, exp, arm, sid) / "main.jsonl"
    hdr = S.schema_header(f"realpi-{sid}", int(cfg["pi"]["session_version"]),
                          chain[0].get("timestamp", "2026-01-01T00:00:00.000Z"), str(wd))
    fh = sess.open("w", encoding="utf-8")
    fh.write(json.dumps(hdr) + "\n")
    threshold = w["window"] - w["reserve"]
    written: list[dict] = []
    last_id, ctx, since, k = None, 0, 0, 0
    rows, failed, requests = [], False, 0
    t_start = time.monotonic()

    for i, e in enumerate(chain + [None]):
        if should_compact(e, since, ctx, threshold) and k < MAX_COMPACTIONS:
            fh.close()
            t0 = time.monotonic()
            comp = D.compact(cfg, sess, wd, env, timeout_s=D.compact_timeout_s(cfg, arm))
            wall = int((time.monotonic() - t0) * 1000)
            req = (comp.usage or {}).get("requests", 0) or 0
            requests += req
            ents = S.read_entries(sess)
            last = S.last_compaction(ents)
            new = last is not None and (not rows or last.get("id") != rows[-1]["_id"])
            row = {"sid": sid, "method": method, "window": win, "k": k, "entry_index": i,
                   "ok": bool(comp.ok and new), "error": comp.error if not (comp.ok and new) else None,
                   "compactor": S.compactor_of(last) if new else None, "model_requests": req,
                   "est_tokens_before": round(ctx / CPT), "wall_ms": wall, "bridge_ms": comp.duration_ms,
                   "_id": last.get("id") if new else None}
            if new:
                summary = last.get("summary", "") or ""
                ids = [x.get("id") for x in written]
                fk = last.get("firstKeptEntryId") or ""
                cut = ids.index(fk) if fk in ids else len(written)
                ctx = context_after(ents, last)
                row.update(summary_chars=len(summary), est_tokens_after=round(ctx / CPT),
                           pi_tokens_before=last.get("tokensBefore"), cut_entry_index=cut,
                           kept_entries=len(written) - cut,
                           **retention(originals(written[:cut]), summary))
                if method.startswith("mech") or method == "verbatim":
                    row["mech"] = mech_trim(ents, last)
                last_id = last["id"]
            rows.append(row)
            print(json.dumps({x: row.get(x) for x in ("sid", "method", "window", "k", "ok", "est_tokens_before",
                                                      "est_tokens_after", "summary_chars", "model_requests",
                                                      "wall_ms")}), flush=True)
            since, k = 0, k + 1
            if not row["ok"] or req:
                failed = True
                break
            fh = sess.open("a", encoding="utf-8")
        if e is None:
            break
        ne = dict(e, parentId=last_id)
        fh.write(json.dumps(ne, ensure_ascii=False) + "\n")
        written.append(ne)
        last_id = ne["id"]
        ctx += entry_chars(ne)
        since += 1
    if not fh.closed:
        fh.close()
    # Check the context model against pi's own buildSessionContext (no model call).
    br = D.call_bridge(cfg, "context", {"sessionFile": str(sess.resolve()), "timeoutMs": 120_000, "includeText": True},
                       dict(env), sess.parent, 180)
    pi_ctx = sum(msg_chars(m) for m in json.loads(br.raw["text"])) if br.ok else None
    with out_rows.open("a", encoding="utf-8") as f:
        for r in rows:
            r.pop("_id", None)
            f.write(json.dumps(r) + "\n")
    return {"sid": sid, "method": method, "window": win, "compactions": len(rows),
            "failed": failed, "model_requests": requests, "entries": len(chain),
            "entries_replayed": len(written), "hit_max_compactions": k >= MAX_COMPACTIONS,
            "final_est_tokens": round(ctx / CPT),
            "final_pi_context_est_tokens": round(pi_ctx / CPT) if pi_ctx is not None else None,
            "context_model_error": br.error if not br.ok else None,
            "wall_s": round(time.monotonic() - t_start, 1)}


def _cfg_for(win: str, out: Path):
    from .config import Config, load_config
    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)
    w = WINDOWS[win]
    raw["pi"]["keep_recent_tokens"], raw["pi"]["reserve_tokens"] = w["keep"], w["reserve"]
    raw["pi"]["models_file"] = w["models_file"]
    raw["run"]["runs_dir"] = str(out / "runs")
    raw["run"]["results_dir"] = str(out / "results")
    raw.setdefault("model", {})["tokenize_url"] = DEAD + "/tokenize"  # never the shared server
    return Config(raw=raw, root=cfg.root)


def cmd_run(a) -> None:
    from . import driver as D
    from . import schedule, usageproxy
    from .config import repo_root

    root = repo_root()
    base = (root / a.out).resolve()
    assert root in base.parents
    sel = json.loads((base / "selection.json").read_text())["selected"]
    lo, hi = (int(x) for x in a.sessions.split("-"))
    tag = f"{a.method}-{a.window}"
    done_f, rows_f = base / f"replays-{tag}.jsonl", base / f"rows-{tag}.jsonl"
    done = {json.loads(x)["sid"] for x in done_f.read_text().splitlines()} if done_f.exists() else set()
    cfg = _cfg_for(a.window, base)
    # The upstream is dead: no model call can reach any server, so the
    # nightly-window gate (which protects the shared server) does not apply.
    schedule.require_llm = lambda _c: None
    proxy = usageproxy.UsageProxy(DEAD).start()
    D.set_usage_proxy(cfg, proxy, base / f".proxy-models-{tag}.json")
    try:
        for s in sel[lo:hi + 1]:
            if s["sid"] in done:
                continue
            rep = replay(cfg, (root / a.data).resolve(), s, a.method, a.window, rows_f, proxy)
            rep["proxy_requests_total_so_far"] = len(proxy.since(0))
            with done_f.open("a") as f:
                f.write(json.dumps(rep) + "\n")
            print("REPLAY", json.dumps(rep), flush=True)
    finally:
        D.set_usage_proxy(cfg, None)
        proxy.stop()


# ----------------------------------------------------------------- summary
def q(xs: list, p: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[0] if len(xs) == 1 else statistics.quantiles(xs, n=4, method="inclusive")[{.25: 0, .5: 1, .75: 2}[p]]


def mqi(xs: list) -> list | None:
    xs = [x for x in xs if x is not None]
    return [round(q(xs, .5), 3), round(q(xs, .25), 3), round(q(xs, .75), 3), len(xs)] if xs else None


def summarize(base: Path) -> dict:
    sel = json.loads((base / "selection.json").read_text())
    out = {"chars_per_token": CPT, "selection": {k: v for k, v in sel.items() if k != "selected"},
           "selected_sessions": len(sel["selected"]),
           "selected_by_licence": {}, "selected_sources": len({s["src"] for s in sel["selected"]}),
           "selected_tokens_mqi": mqi([s["tokens"] for s in sel["selected"]]), "configs": {}}
    for s in sel["selected"]:
        out["selected_by_licence"][s["licence"] or "unclear"] = out["selected_by_licence"].get(s["licence"] or "unclear", 0) + 1
    rows_all, reps_all = [], []
    for m in METHODS:
        for w in WINDOWS:
            tag = f"{m}-{w}"
            rf, df = base / f"rows-{tag}.jsonl", base / f"replays-{tag}.jsonl"
            rows = [json.loads(x) for x in rf.read_text().splitlines()] if rf.exists() else []
            reps = [json.loads(x) for x in df.read_text().splitlines()] if df.exists() else []
            rows_all += rows
            reps_all += reps
            if not reps:
                continue
            by_k: dict[int, list] = {}
            for r in rows:
                by_k.setdefault(r["k"], []).append(r)
            n = [r["compactions"] for r in reps]
            trim_first = [min((r["k"] for r in rows if r["sid"] == p["sid"] and (r.get("mech") or {}).get("trimmed")),
                              default=None) for p in reps] if (m.startswith("mech") or m == "verbatim") else []
            out["configs"][tag] = {
                "replays": len(reps), "compactions_total": sum(n),
                "compactions_per_session": {"min": min(n), "median": statistics.median(n), "max": max(n),
                                            "q25": q(n, .25), "q75": q(n, .75), "zero": sum(x == 0 for x in n)},
                "failed_replays": sum(r["failed"] for r in reps), "model_requests": sum(r["model_requests"] for r in reps),
                "compactions_not_ok": sum(not r["ok"] for r in rows),
                "compactor": sorted({str(r.get("compactor")) for r in rows}),
                "hit_max_compactions": sum(r["hit_max_compactions"] for r in reps),
                "final_est_tokens_mqi": mqi([r["final_est_tokens"] for r in reps]),
                "context_model_max_abs_diff_tokens": max((abs(r["final_est_tokens"] - r["final_pi_context_est_tokens"])
                                                          for r in reps if r["final_pi_context_est_tokens"] is not None),
                                                         default=None),
                "wall_ms_mqi": mqi([r["wall_ms"] for r in rows]),
                "first_trim_k": {"sessions_trimmed": sum(x is not None for x in trim_first),
                                 "k_mqi": mqi(trim_first)} if (m.startswith("mech") or m == "verbatim") else None,
                "by_k": {str(kk): {
                    "n": len(v),
                    "summary_chars": mqi([r.get("summary_chars") for r in v]),
                    "est_tokens_before": mqi([r.get("est_tokens_before") for r in v]),
                    "est_tokens_after": mqi([r.get("est_tokens_after") for r in v]),
                    "user_retained": mqi([r.get("user_retained") for r in v]),
                    "assistant_retained": mqi([r.get("assistant_retained") for r in v]),
                    "tool_calls_retained": mqi([r.get("tool_calls_retained") for r in v]) if m == "mech" else None,
                    "mech_trimmed": sum(bool((r.get("mech") or {}).get("trimmed")) for r in v) if m == "mech" else None,
                } for kk, v in sorted(by_k.items())},
                "overall": {x: mqi([r.get(x) for r in rows]) for x in
                            ("summary_chars", "est_tokens_after", "user_retained", "assistant_retained",
                             "tool_calls_retained")},
            }
    out["model_requests_total"] = sum(r["model_requests"] for r in reps_all)
    out["compactions_total"] = len(rows_all)
    return out


KS = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30)


def _f(v, pct=False) -> str:
    if v is None:
        return "-"
    med, lo, hi, n = v
    if pct:
        return f"{100 * med:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"
    return f"{med:,.0f} [{lo:,.0f}-{hi:,.0f}]"


def tables(summ: dict) -> str:
    """Markdown tables for docs/build/realpi-replay.md, from summary.json."""
    c, out = summ["configs"], []
    out.append("| Config | Replays | Compactions | Per session min / q25 / median / q75 / max | Final context (est. tokens) | Compaction ms | Failed | Model requests |")
    out.append("|---|---|---|---|---|---|---|---|")
    for tag, v in c.items():
        d = v["compactions_per_session"]
        out.append(f"| {tag} | {v['replays']} | {v['compactions_total']} | {d['min']} / {d['q25']} / {d['median']} / "
                   f"{d['q75']} / {d['max']} | {_f(v['final_est_tokens_mqi'])} | {_f(v['wall_ms_mqi'])} | "
                   f"{v['failed_replays']} | {v['model_requests']} |")
    for w in WINDOWS:
        out.append(f"\n**{w}** (median [IQR], n = compactions at that index)\n")
        out.append("| k | n mech / vcc | Checkpoint chars mech | Summary chars vcc | Context after mech | Context after vcc "
                   "| User text mech | User text vcc | Asst text mech | Asst text vcc | Tool calls mech | mech trimmed |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        m, p = c.get(f"mech-{w}", {}).get("by_k", {}), c.get(f"pi-vcc-{w}", {}).get("by_k", {})
        for k in KS:
            a, b = m.get(str(k)), p.get(str(k))
            if not a and not b:
                continue
            a, b = a or {}, b or {}
            out.append(f"| {k} | {a.get('n', 0)} / {b.get('n', 0)} | {_f(a.get('summary_chars'))} | {_f(b.get('summary_chars'))} | "
                       f"{_f(a.get('est_tokens_after'))} | {_f(b.get('est_tokens_after'))} | "
                       f"{_f(a.get('user_retained'), True)} | {_f(b.get('user_retained'), True)} | "
                       f"{_f(a.get('assistant_retained'), True)} | {_f(b.get('assistant_retained'), True)} | "
                       f"{_f(a.get('tool_calls_retained'), True)} | {a.get('mech_trimmed', '-')}/{a.get('n', 0)} |")
    out.append("\n| Config | Checkpoint/summary chars | Context after | User text retained | Assistant text retained | Tool calls retained |")
    out.append("|---|---|---|---|---|---|")
    for tag, v in c.items():
        o = v["overall"]
        out.append(f"| {tag} | {_f(o['summary_chars'])} | {_f(o['est_tokens_after'])} | {_f(o['user_retained'], True)} | "
                   f"{_f(o['assistant_retained'], True)} | {_f(o['tool_calls_retained'], True) if tag.startswith('mech') else '-'} |")
    out.append("\n| mech config | Sessions where mech trims | First trimming compaction k (median [IQR]) |")
    out.append("|---|---|---|")
    for tag, v in c.items():
        if v.get("first_trim_k"):
            out.append(f"| {tag} | {v['first_trim_k']['sessions_trimmed']} / {v['replays']} | {_f(v['first_trim_k']['k_mqi'])} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "run", "summarize", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="runs/realpi/hf")
        p.add_argument("--out", default="runs/realpi/replay")
        if name == "run":
            p.add_argument("--method", choices=list(METHODS), required=True)
            p.add_argument("--window", choices=list(WINDOWS), required=True)
            p.add_argument("--sessions", default=f"0-{CAP_SESSIONS - 1}")
        if name in ("summarize", "tables"):
            p.add_argument("--results", default="results/REALPI")
    a = ap.parse_args(argv)
    from .config import repo_root
    root = repo_root()
    base = (root / a.out).resolve()
    base.mkdir(parents=True, exist_ok=True)
    if a.cmd == "select":
        sel = select((root / a.data).resolve())
        (base / "selection.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps({k: v for k, v in sel.items() if k != "selected"}),
              {"selected": len(sel["selected"]), "by_src": _count(s["src"] for s in sel["selected"])})
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "tables":
        print(tables(json.loads((root / a.results / "summary.json").read_text())))
    else:
        res = root / a.results
        res.mkdir(parents=True, exist_ok=True)
        rows = []
        for m in METHODS:
            for w in WINDOWS:
                f = base / f"rows-{m}-{w}.jsonl"
                if f.exists():
                    rows += f.read_text().splitlines()
        (res / "replay.jsonl").write_text("".join(x + "\n" for x in rows))
        summ = summarize(base)
        (res / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(json.dumps({k: v for k, v in summ.items() if k != "configs"}, indent=1))


def _count(xs) -> dict:
    d: dict = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
