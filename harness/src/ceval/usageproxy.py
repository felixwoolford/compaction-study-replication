"""Token and time accounting for every model call, whoever makes it.

pi's session file records only the main agent's replies. Model calls made
inside extensions (mech-compact's context_lookup subagent, pi-lcm's
summariser) and pi's own compaction summary are not in it, and the llama.cpp
server's /metrics endpoint is disabled (the server is shared: never
reconfigure it). So the runner puts this proxy between pi and the server: pi's
models file is copied with its baseUrl pointing here, every request is
forwarded unchanged, and the usage the server reports for each request is
recorded. Calls are sequential (one bridge process at a time), so the records
taken during one bridge call are that operation's cost.

The server is llama.cpp's OpenAI-compatible API. A reply carries `usage`
(prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens) and
llama.cpp's `timings` (prompt_n: prompt tokens evaluated, i.e. not served from
cache; cache_n; predicted_n; prompt_ms; predicted_ms); a streamed reply carries
them in its last data chunks. Whatever is present is recorded; nothing is
estimated.
"""

from __future__ import annotations

import gzip
import http.client
import json
import os
from collections import deque
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length"}

USAGE_KEYS = ("requests", "prompt_tokens", "cached_tokens", "completion_tokens",
              "prompt_eval_tokens", "predicted_tokens", "prompt_ms", "predicted_ms", "wall_ms",
              "errors", "unreported")


def _usage_of(obj: dict, rec: dict) -> None:
    u = obj.get("usage")
    if isinstance(u, dict):
        rec["prompt_tokens"] = u.get("prompt_tokens")
        rec["completion_tokens"] = u.get("completion_tokens")
        d = u.get("prompt_tokens_details") or {}
        if isinstance(d, dict) and d.get("cached_tokens") is not None:
            rec["cached_tokens"] = d.get("cached_tokens")
    t = obj.get("timings")
    if isinstance(t, dict):
        for src, dst in (("prompt_n", "prompt_eval_tokens"), ("cache_n", "cache_n"),
                         ("predicted_n", "predicted_tokens"), ("prompt_ms", "prompt_ms"),
                         ("predicted_ms", "predicted_ms")):
            if t.get(src) is not None:
                rec[dst] = t[src]


def _dump(d: Path, i: int, rec: dict, body: bytes | None, tail: list[bytes]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    try:
        req = json.loads(body) if body else {}
    except json.JSONDecodeError:
        req = {}
    msgs = req.pop("messages", None) or []
    line = {"i": i, **rec, "params": req, "n_messages": len(msgs), "messages_chars": len(json.dumps(msgs)),
            "last_roles": [m.get("role") for m in msgs[-3:] if isinstance(m, dict)],
            "response_tail": [t.decode("utf-8", "replace")[:2000] for t in tail]}
    if body:
        (d / f"req-{i:05d}.json.gz").write_bytes(gzip.compress(body))
    with (d / "requests.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")


class UsageProxy:
    """A local forwarding proxy that records the usage of every request."""

    def __init__(self, upstream: str):
        u = urlsplit(upstream)
        self.host, self.port_up = u.hostname or "127.0.0.1", u.port or 80
        self.records: list[dict] = []
        self.lock = threading.Lock()
        # Diagnostics only: CEVAL_PROXY_DUMP=<dir> writes every request (gzipped) and a summary line of its
        # parameters and response tail to <dir>/requests.jsonl. Off by default; the forwarding is unchanged.
        self.dump = Path(os.environ["CEVAL_PROXY_DUMP"]) if os.environ.get("CEVAL_PROXY_DUMP") else None
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # one request per connection; the body ends at close

            def log_message(self, *_):
                pass

            def _forward(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                rec = {"t0": time.time(), "path": self.path, "status": None}
                t0 = time.monotonic()
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
                conn = http.client.HTTPConnection(proxy.host, proxy.port_up, timeout=7200)
                sse, full = False, []
                tail = deque(maxlen=6)  # the last SSE data lines (finish_reason, usage, timings), for the dump
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                    rec["status"] = resp.status
                    self.send_response(resp.status, resp.reason)
                    for k, v in resp.getheaders():
                        if k.lower() not in HOP:
                            self.send_header(k, v)
                    self.end_headers()
                    sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
                    buf = b""
                    while True:
                        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if sse:
                            buf += chunk
                            *lines, buf = buf.split(b"\n")
                            for line in lines:
                                line = line.strip()
                                if line.startswith(b"data:"):
                                    tail.append(line[5:].strip())
                                if line.startswith(b"data:") and (b'"usage"' in line or b'"timings"' in line):
                                    try:
                                        _usage_of(json.loads(line[5:].strip()), rec)
                                    except json.JSONDecodeError:
                                        pass
                        else:
                            full.append(chunk)
                    if not sse and full:
                        try:
                            obj = json.loads(b"".join(full))
                            if isinstance(obj, dict):
                                _usage_of(obj, rec)
                        except json.JSONDecodeError:
                            pass
                except Exception as e:  # recorded; the client sees a broken response
                    rec["error"] = repr(e)
                finally:
                    conn.close()
                    rec["wall_ms"] = int((time.monotonic() - t0) * 1000)
                    with proxy.lock:
                        proxy.records.append(rec)
                        if proxy.dump:
                            _dump(proxy.dump, len(proxy.records), rec, body, list(tail) if sse else full)

            do_POST = do_GET = _forward

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "UsageProxy":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def mark(self) -> int:
        with self.lock:
            return len(self.records)

    def since(self, mark: int) -> list[dict]:
        with self.lock:
            return list(self.records[mark:])


def summarize(records: list[dict]) -> dict:
    """Totals over the model requests of one operation (requests to other paths,
    e.g. /tokenize, are not model calls and are excluded)."""
    calls = [r for r in records if "/chat/completions" in r.get("path", "") or "/completions" in r.get("path", "")]
    s = {k: 0 for k in USAGE_KEYS}
    s["requests"] = len(calls)
    for r in calls:
        if r.get("error") or (r.get("status") or 0) >= 400:
            s["errors"] += 1
        if r.get("prompt_tokens") is None and r.get("prompt_eval_tokens") is None:
            s["unreported"] += 1
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "prompt_eval_tokens",
                  "predicted_tokens", "prompt_ms", "predicted_ms", "wall_ms"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                s[k] += v
    for k in ("prompt_ms", "predicted_ms"):
        s[k] = int(s[k])
    return s


def merge(a: dict | None, b: dict | None) -> dict | None:
    """Sum two operation summaries (e.g. a probe and its follow-up)."""
    if a is None or b is None:
        return a or b
    return {k: (a.get(k) or 0) + (b.get(k) or 0) for k in USAGE_KEYS}
