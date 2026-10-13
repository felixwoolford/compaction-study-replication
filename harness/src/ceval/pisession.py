"""Read, write and fork pi session JSONL files.

Everything that depends on pi's exact on-disk schema lives in the SCHEMA
section below. It was verified in SETUP.md Stage 3 with `ceval schema-diff`
against real pi 1.0.0 sessions of the study model (llama.cpp, openai-completions
API); keep all schema knowledge in this section.
"""

from __future__ import annotations

import json
import random
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# =============================================================================
# SCHEMA (verified against real pi 1.0.0 sessions, SETUP.md Stage 3)
# =============================================================================

# Wire API per provider, as recorded on assistant messages (verified: the
# llama.cpp provider of pi 1.0.0 records "openai-completions").
API_BY_PROVIDER = {"llama.cpp": "openai-completions"}

# Provider stop status per pi stopReason (verified in real sessions: tool calls
# are "tool_calls", a final answer is "stop").
RAW_STOP_BY_REASON = {"toolUse": "tool_calls", "stop": "stop"}


def schema_header(session_id: str, version: int, ts: str, cwd: str) -> dict:
    return {"type": "session", "version": version, "id": session_id, "timestamp": ts, "cwd": cwd}


def schema_entry(entry_type: str, entry_id: str, parent_id: str | None, ts: str, **fields: Any) -> dict:
    return {"type": entry_type, "id": entry_id, "parentId": parent_id, "timestamp": ts, **fields}


def schema_user_message(text: str, ts_ms: int) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": ts_ms}


def schema_assistant_message(blocks: list[dict], provider: str, model: str, stop_reason: str, ts_ms: int,
                             *, response_id: str | None = None, thinking_level: str = "xhigh",
                             error_message: str | None = None) -> dict:
    # Verified against real llama.cpp sessions of pi 1.0.0. api is the wire
    # API, rawStopReason the provider's raw stop status, responseId/responseModel
    # the provider's response identity, thinkingLevel the level pi was run with,
    # errorMessage null unless the request failed. Usage is all zeros: pi treats
    # an all-zero usage as absent and falls back to char-based token estimates
    # (dist/core/compaction/compaction.js, getAssistantUsage/estimateTokens),
    # which is what places the compaction cut.
    return {
        "role": "assistant",
        "content": blocks,
        "api": API_BY_PROVIDER.get(provider, "openai-completions"),
        "provider": provider,
        "model": model,
        "stopReason": stop_reason,
        "rawStopReason": RAW_STOP_BY_REASON.get(stop_reason, stop_reason),
        "responseId": response_id,
        "responseModel": model,
        "thinkingLevel": thinking_level,
        "errorMessage": error_message,
        "usage": {"input": 0, "output": 0},
        "timestamp": ts_ms,
    }


def schema_thinking_block(text: str, signature: str = "reasoning_content") -> dict:
    # Verified: pi records where the reasoning came from. Openai-completions
    # providers (llama.cpp) store the raw field name that carried it,
    # "reasoning_content" (pi 1.0.0 openai-completions chunk). Codex Responses
    # sessions instead carry a JSON reasoning item; not used by this study.
    return {"type": "thinking", "thinking": text, "thinkingSignature": signature}


def schema_text_block(text: str) -> dict:
    # Verified: text blocks carry only text/type for openai-completions providers
    # (the codex provider adds a textSignature; not used by this study).
    return {"type": "text", "text": text}


def schema_tool_call_block(call_id: str, name: str, args: dict) -> dict:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": args}


def schema_tool_result_message(call_id: str, name: str, text: str, is_error: bool, ts_ms: int) -> dict:
    # Verified envelope. The optional `details` field is metadata written by the
    # editing tools only (e.g. {"diff", "patch", "firstChangedLine"} for edit/write;
    # pi 1.0.0 chunk-ZSBPJAJ2.js) and is absent on read/bash results, which is all
    # the synthetic sessions use.
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": name,
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
        "timestamp": ts_ms,
    }


# Which compactor wrote a compaction entry, from its `details` (treatment
# check). pi's own summary writes {readFiles, modifiedFiles}; mech-compact
# marks its entries details.kind == "mech-compact" (mech-compact.ts MECH_KIND);
# pi-lcm returns details {lcm: true, stats} from session_before_compact
# (pi-lcm 0.1.3 index.ts); pi-vcc returns details {compactor: "pi-vcc",
# version, sections, sourceMessageCount, previousSummaryUsed, messageRange,
# tokensBefore, keptCount, keptTokensEst, ...} (pi-vcc 0.8.12
# src/hooks/before-compact.ts, src/details.ts). Anything else is pi's default.
MECH_KIND = "mech-compact"
VCC_COMPACTOR = "pi-vcc"
# The parts of pi-vcc's details recorded in the compaction row (timestamps and
# entry ids are left out).
VCC_DETAIL_KEYS = ("compactor", "version", "sections", "sourceMessageCount", "previousSummaryUsed",
                   "keptCount", "keptTokensEst")


def compactor_of(entry: dict) -> str:
    """'mech', 'lcm', 'vcc' or 'pi' for a compaction entry."""
    d = entry.get("details") or {}
    if d.get("kind") == MECH_KIND:
        return "mech"
    if d.get("lcm") is True:
        return "lcm"
    if d.get("compactor") == VCC_COMPACTOR:
        return "vcc"
    return "pi"


# Session-state entries pi appends when it opens a session whose recorded
# state differs from the run's (seen in S1 main lines: a thinking_level_change
# just before the first compaction). Not messages; they carry no content.
SESSION_STATE_TYPES = frozenset({"thinking_level_change", "model_change"})


# mech-compact's size guard (mech-compact.ts capSpans / renderCheckpoint): each
# mech entry stores its own un-capped span in details.span.lines; the rendered
# checkpoint holds every span after the base (the last non-span compaction)
# until the conversation section exceeds MECH_COMPACT_MAX_SUMMARY_CHARS. Then
# it drops the oldest non-user lines without a marker, stubs long user lines
# ("…[truncated +N chars — use context_lookup]") and marks emptied spans
# trimmed="true" with SPAN_TRIMMED_NOTE.
MECH_USER_STUB = "…[truncated +"
MECH_SPAN_TRIMMED_ATTR = ' trimmed="true"'
MECH_SPAN_TRIMMED_NOTE = "(trimmed for size — use context_lookup)"


def _mech_span_lines(entry: dict) -> list[str] | None:
    d = entry.get("details") or {}
    if entry.get("type") != "compaction" or d.get("kind") != MECH_KIND:
        return None
    lines = (d.get("span") or {}).get("lines")
    return lines if isinstance(lines, list) else None


def mech_checkpoint_stats(entries: list[dict], comp: dict) -> dict | None:
    """How much of its spans a mech-compact checkpoint (`comp`, one of
    `entries`) still carries: the un-capped span lines of every span it renders,
    how many appear verbatim in the summary, user lines stubbed, spans marked
    trimmed. None for any other compactor, or a mech entry without spans."""
    if compactor_of(comp) != "mech" or _mech_span_lines(comp) is None:
        return None
    upto = entries[:next(i for i, e in enumerate(entries) if e is comp or e.get("id") == comp.get("id")) + 1]
    spans: list[list[str]] = []
    for e in upto:
        if e.get("type") != "compaction":
            continue
        lines = _mech_span_lines(e)
        spans = [] if lines is None else spans + [lines]  # a non-span compaction is the new base
    summary = comp.get("summary", "") or ""
    have: dict[str, int] = {}
    for ln in summary.split("\n"):
        have[ln] = have.get(ln, 0) + 1
    kept = 0
    flat = [ln for sp in spans for ln in sp]
    for ln in flat:
        if have.get(ln, 0) > 0:
            have[ln] -= 1
            kept += 1
    stubbed = summary.count(MECH_USER_STUB)
    trimmed_spans = summary.count(MECH_SPAN_TRIMMED_ATTR)
    return {"spans": len(spans), "span_lines": len(flat), "span_chars": sum(len(x) + 1 for x in flat),
            "lines_kept": kept, "lines_dropped": len(flat) - kept, "user_lines_stubbed": stubbed,
            "spans_trimmed": trimmed_spans,
            "trimmed": len(flat) > kept or stubbed > 0 or trimmed_spans > 0}
def vcc_details(details: dict) -> dict:
    """The identifying part of a pi-vcc compaction's details (empty otherwise)."""
    if (details or {}).get("compactor") != VCC_COMPACTOR:
        return {}
    return {k: details[k] for k in VCC_DETAIL_KEYS if k in details}


# pi-fold 4.2.6 writes its state through pi.appendEntry, i.e. pi custom
# entries {"type": "custom", "customType", "data"} (session-manager.js
# appendCustomEntry). One fold-record per new fold (data.fold: kind, parentId,
# sourceChars, placeholderChars, ...; lib/persistence.ts), the fold state
# (checkpoint/delta), a provider measurement receipt per measured reply, and
# instrumentation events (data.kind "context.commit" with deferred false for a
# real commit: applied_marks, freed_bytes, occupancy_tokens_before,
# budget_tokens, clipped_results; active-context.ts). Event entries are written
# asynchronously and may be missing; fold records are authoritative.
FOLD_RECORD = "pi-fold-active-context-fold-record"
FOLD_STATE = "pi-fold-active-context-state"
FOLD_EVENT = "pi-fold-context-event"
FOLD_MEASUREMENT = "pi-fold-provider-context-measurement"


def custom_entries(entries: list[dict], custom_type: str) -> list[dict]:
    return [e for e in entries if e.get("type") == "custom" and e.get("customType") == custom_type]


def fold_summary(entries: list[dict]) -> dict:
    """What pi-fold did in these entries (a commit turn's delta)."""
    folds = [(e.get("data") or {}).get("fold") or {} for e in custom_entries(entries, FOLD_RECORD)]
    kinds: dict[str, int] = {}
    for f in folds:
        kinds[str(f.get("kind"))] = kinds.get(str(f.get("kind")), 0) + 1
    events = [e.get("data") or {} for e in custom_entries(entries, FOLD_EVENT)]
    keys = ("trigger", "applied_marks", "freed_bytes", "freed_tokens", "occupancy_tokens_before",
            "budget_tokens", "target_freed_share", "clipped_results", "consolidation_marks")
    commits = [{k: ev.get(k) for k in keys} for ev in events
               if ev.get("kind") == "context.commit" and ev.get("deferred") is False]
    deferred = [ev.get("reason") for ev in events if ev.get("kind") == "context.commit" and ev.get("deferred")]
    return {"folds": len(folds), "fold_kinds": kinds,
            "source_chars": sum(int(f.get("sourceChars") or 0) for f in folds),
            "placeholder_chars": sum(int(f.get("placeholderChars") or 0) for f in folds),
            "commits": commits, "deferred_commits": deferred,
            "measurements": len(custom_entries(entries, FOLD_MEASUREMENT)),
            "state_entries": len(custom_entries(entries, FOLD_STATE))}


def context_tokens(message: dict) -> int | None:
    """Provider-reported prompt size of an assistant message (input plus cache
    read/write; the reply's output excluded). None if no usage was recorded."""
    u = message.get("usage") or {}
    n = sum(int(u.get(k, 0) or 0) for k in ("input", "cacheRead", "cacheWrite"))
    return n or None


def usage_tokens(message: dict) -> tuple[int, int]:
    """(input, output) tokens recorded on an assistant message. SCHEMA-ASSUMPTION."""
    u = message.get("usage") or {}
    return int(u.get("input", 0) or 0), int(u.get("output", 0) or 0)


# =============================================================================
# Writer
# =============================================================================


class SessionWriter:
    """Append-only writer producing pi session entries with parent chaining."""

    def __init__(self, path: Path, *, cwd: str, version: int, provider: str, model: str,
                 seed: int, start: datetime | None = None, step_s: int = 17,
                 thinking_level: str = "xhigh"):
        self.path = Path(path)
        self.provider = provider
        self.model = model
        self.thinking_level = thinking_level
        self.rng = random.Random(seed)
        self._dt = timedelta(seconds=step_s)
        self.last_id: str | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            entries = read_entries(self.path)
            last = entries[-1]
            self.last_id = last.get("id") if last.get("type") != "session" else None
            self.clock = _parse_ts(last["timestamp"])
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.clock = start or datetime(2026, 1, 5, 9, 0, 0, tzinfo=timezone.utc)
            self._write(schema_header(str(uuid.UUID(int=self.rng.getrandbits(128))), version,
                                      _iso(self.clock), cwd))

    # -- low level ------------------------------------------------------------
    def _write(self, obj: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _tick(self) -> tuple[str, int]:
        self.clock += self._dt
        return _iso(self.clock), int(self.clock.timestamp() * 1000)

    def _new_id(self) -> str:
        return f"{self.rng.getrandbits(32):08x}"

    def _entry(self, entry_type: str, ts: str | None = None, **fields: Any) -> str:
        eid = self._new_id()
        if ts is None:
            ts, _ = self._tick()
        self._write(schema_entry(entry_type, eid, self.last_id, ts, **fields))
        self.last_id = eid
        return eid

    # -- messages -------------------------------------------------------------
    def user(self, text: str) -> str:
        ts, ms = self._tick()
        return self._entry("message", ts, message=schema_user_message(text, ms))

    def assistant(self, *, thinking: str | None = None, text: str | None = None,
                  calls: list[dict] | None = None) -> list[str]:
        """Write an assistant message (+ toolResult messages for its calls). Returns all entry ids."""
        calls = calls or []
        blocks: list[dict] = []
        if thinking:
            blocks.append(schema_thinking_block(thinking))
        if text:
            blocks.append(schema_text_block(text))
        call_ids = []
        for c in calls:
            cid = f"call_{self.rng.getrandbits(48):012x}"
            call_ids.append(cid)
            blocks.append(schema_tool_call_block(cid, c["name"], c["args"]))
        ts, ms = self._tick()
        response_id = "chatcmpl-" + f"{self.rng.getrandbits(96):024x}"
        ids = [self._entry("message", ts, message=schema_assistant_message(
            blocks, self.provider, self.model, "toolUse" if calls else "stop", ms,
            response_id=response_id, thinking_level=self.thinking_level))]
        for c, cid in zip(calls, call_ids):
            ts, ms = self._tick()
            ids.append(self._entry("message", ts, message=schema_tool_result_message(
                cid, c["name"], c["result"], bool(c.get("is_error")), ms)))
        return ids

    def step(self, step: dict) -> list[str]:
        if step["kind"] == "user":
            return [self.user(step["text"])]
        return self.assistant(thinking=step.get("thinking"), text=step.get("text"), calls=step.get("calls"))


# =============================================================================
# Reading, forking, deltas
# =============================================================================


def read_entries(path: Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def line_count(path: Path) -> int:
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def entries_after(path: Path, n_before: int) -> list[dict]:
    return read_entries(path)[n_before:]


def fork(src: Path, dst: Path) -> Path:
    """Copy a session file and give the copy a fresh session id in its header."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    entries = read_entries(dst)
    if entries and entries[0].get("type") == "session":
        entries[0]["id"] = str(uuid.uuid4())
        with dst.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return dst


def copy_session(src: Path, dst: Path) -> Path:
    """A byte-identical copy that keeps the session id: the same conversation,
    continued elsewhere (Study 1's full probe copies). pi-lcm finds its stored
    history by pi session id, so a copy with a fresh id would start an empty
    store; mech and pi-fold read the session in-process and do not use the id,
    and pi sends no session id to the llama.cpp provider (no prompt-cache key
    or affinity header for this server)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def fork_pruned(src: Path, dst: Path) -> Path:
    """A fork holding only what pi's context is built from after the latest
    compaction: the header, the kept entries from firstKeptEntryId on, the
    compaction entry and everything after it. Entries the compaction summarised
    are not in the file, so an agent that reads its own session file finds
    nothing compaction removed (PROTOCOL.md v2 §6.1). Without a compaction the
    fork is a plain copy. The first kept entry becomes the root (parentId None).
    Assumes a linear session file, as the harness writes them."""
    entries = read_entries(src)
    header, body = entries[0], entries[1:]
    comps = [i for i, e in enumerate(body) if e.get("type") == "compaction"]
    if not comps:
        return fork(src, dst)
    comp = body[comps[-1]]
    first = next((i for i, e in enumerate(body) if e.get("id") == comp.get("firstKeptEntryId")), None)
    if first is None or first > comps[-1]:
        raise ValueError(f"{src}: firstKeptEntryId {comp.get('firstKeptEntryId')!r} not found before the compaction")
    kept = [dict(e) for e in body[first:]]
    kept[0]["parentId"] = None
    # Older compaction entries between the first kept entry and the latest one
    # are not part of pi's context either.
    kept = [e for e in kept if e.get("type") != "compaction" or e.get("id") == comp["id"]]
    for i in range(1, len(kept)):
        if kept[i].get("parentId") not in {k.get("id") for k in kept[:i]}:
            kept[i]["parentId"] = kept[i - 1]["id"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = dict(header, id=str(uuid.uuid4()))
    with dst.open("w", encoding="utf-8") as f:
        for e in [header, *kept]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return dst


def messages(entries: list[dict]) -> list[dict]:
    return [e["message"] for e in entries if e.get("type") == "message" and "message" in e]


def final_assistant_text(entries: list[dict]) -> str:
    for m in reversed(messages(entries)):
        if m.get("role") == "assistant":
            texts = [b.get("text", "") for b in m.get("content", []) if b.get("type") == "text"]
            if any(t.strip() for t in texts):
                return "\n".join(texts)
    return ""


def tool_calls(entries: list[dict]) -> list[dict]:
    calls = []
    for m in messages(entries):
        if m.get("role") == "assistant":
            for b in m.get("content", []):
                if b.get("type") == "toolCall":
                    calls.append({"name": b.get("name"), "args": b.get("arguments", {})})
    return calls


def token_usage(entries: list[dict]) -> tuple[int, int]:
    tin = tout = 0
    for m in messages(entries):
        if m.get("role") == "assistant":
            i, o = usage_tokens(m)
            tin += i
            tout += o
    return tin, tout


def last_compaction(entries: list[dict]) -> dict | None:
    for e in reversed(entries):
        if e.get("type") == "compaction":
            return e
    return None


def entry_index(entries: list[dict]) -> dict[str, int]:
    return {e["id"]: i for i, e in enumerate(entries) if "id" in e and e.get("type") != "session"}


# =============================================================================
# Schema diff (SETUP.md Stage 3)
# =============================================================================


def _shape(entries: list[dict]) -> dict[str, set[str]]:
    shapes: dict[str, set[str]] = {}

    def add(k: str, keys) -> None:
        shapes.setdefault(k, set()).update(keys)

    for e in entries:
        t = e.get("type", "?")
        if t == "message":
            m = e.get("message", {})
            role = m.get("role", "?")
            add(f"entry:message[{role}]", e.keys())
            if role == "toolResult" and m.get("toolName"):
                # Tool-specific fields (e.g. `details` on edit/write results)
                # are compared per tool, not pooled across all tools.
                add(f"message[toolResult:{m['toolName']}]", m.keys())
            else:
                add(f"message[{role}]", m.keys())
            content = m.get("content")
            if isinstance(content, list):
                for b in content:
                    add(f"block[{role}:{b.get('type', '?')}]", b.keys())
        else:
            add(f"entry:{t}", e.keys())
    return shapes


def schema_diff(real: Path, synthetic: Path) -> str:
    """Report key differences between a real pi session and a synthetic one."""
    r, s = _shape(read_entries(real)), _shape(read_entries(synthetic))
    lines = []
    for k in sorted(set(r) | set(s)):
        if k not in s:
            continue  # synthetic files only need the shapes they use
        if k not in r:
            lines.append(f"{k}: not seen in the real file (cannot verify)")
            continue
        missing, extra = r[k] - s[k], s[k] - r[k]
        if missing or extra:
            lines.append(f"{k}: missing {sorted(missing)} extra {sorted(extra)}")
        else:
            lines.append(f"{k}: OK")
    return "\n".join(lines)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
