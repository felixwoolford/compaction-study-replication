"""Study 1 ("S1") stimulus plans: six debugging stories, each with its own plot skeleton.

PROTOCOL.md §5.1. Each story is a debugging session in its own domain. Its
skeleton plants 17 details (from a pool of 33 detail types, every source
represented) in its own order, with its own co-occurrences and question wording,
plus 3 lures. Two sessions per story draw different values from their seeds.

The material is built to look like real agent work: tool output is source
files, diffs, test runner output, grep results, git history and prose-like logs;
tool calls carry realistic `edit`/`write` arguments. Details are embedded in that
output rather than in walls of ids. Each compaction round opens with a competing
episode (look-alikes of the story's details) and continues with varied round
work; probed values and look-alikes never occur in the round work. Its
composition follows real pi sessions: few tool calls with long output, little
prose (COMPOSITION_BOUNDS, checked by tests and tokcheck).

Sizes are in real tokens: the scenario's `round_tokens` (real) are converted to
characters with the scenario's measured `chars_per_token`. `tokcheck.py` measures
the plans with the model's tokenizer.

Invariants are asserted at build time (`check_plan`):
  - single defensible answer: every probed value and every look-alike occurs
    only at the positions the plan declares (meta["positions"]);
  - pilot flaw 1: no generated config/env content assigns a key bound to a
    question's slot outside the declared positions;
  - pilot flaw 2: round work never reads or touches a story file;
  - variety: no step text over 200 characters repeats within a session, and at
    least 90% of round steps are distinct.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass, field

from .facts import Fact, value_regex
from .stimuli import Plan

# =============================================================================
# Step helpers (the format pisession.SessionWriter.step accepts)
# =============================================================================


def _u(text: str) -> dict:
    return {"kind": "user", "text": text}


def _a(text: str | None = None, thinking: str | None = None, calls: list | None = None) -> dict:
    return {"kind": "assistant", "thinking": thinking, "text": text, "calls": calls or []}


def _c(name: str, args: dict, result: str, is_error: bool = False) -> dict:
    return {"name": name, "args": args, "result": result, "is_error": is_error}


def _read(path: str, content: str) -> dict:
    return _c("read", {"path": path}, content)


def _bash(command: str, out: str) -> dict:
    return _c("bash", {"command": command}, out)


def _edit(path: str, pairs: list[tuple[str, str]]) -> dict:
    return _c("edit", {"path": path, "edits": [{"oldText": o, "newText": n} for o, n in pairs]},
              f"Successfully replaced {len(pairs)} block(s) in {path}.")


def _write(path: str, content: str) -> dict:
    return _c("write", {"path": path, "content": content}, f"Successfully wrote to {path}")


def args_text(args: dict) -> str:
    """Tool-call arguments as the chat template renders them: string values raw,
    anything else as JSON."""
    return "\n".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in args.values())


def step_text(st: dict) -> str:
    parts = [st.get("text") or "", st.get("thinking") or ""]
    for c in st.get("calls", []):
        parts += [args_text(c["args"]), c["result"]]
    return "\n".join(parts)


def steps_text(steps: list[dict]) -> str:
    return "\n".join(step_text(s) for s in steps)


CATEGORIES = ("tool_output", "tool_args", "user", "assistant_text", "thinking")


def step_category_chars(st: dict) -> dict[str, int]:
    out = dict.fromkeys(CATEGORIES, 0)
    if st["kind"] == "user":
        out["user"] = len(st["text"])
        return out
    out["assistant_text"] = len(st.get("text") or "")
    out["thinking"] = len(st.get("thinking") or "")
    for c in st.get("calls", []):
        out["tool_args"] += len(args_text(c["args"]))
        out["tool_output"] += len(c["result"])
    return out


def step_chars(st: dict) -> int:
    return sum(step_category_chars(st).values())


# =============================================================================
# Domains: vocabulary for each story's code, logs and round work
# =============================================================================

WORK_PEOPLE = ["Priya", "Tomasz", "Ines", "Kwame", "Lena", "Ravi", "Sofia", "Jonah", "Mei", "Dario", "Hana", "Felix",
               "Anouk", "Bilal", "Carmen", "Dmitri", "Esme", "Gustav"]
WORK_LOGINS = ["pnair", "tbrzezinski", "iferreira", "kmensah", "lvogel", "rsharma", "sgarcia", "jcohen", "mzhang",
               "dross", "aveldhuis", "bqureshi", "cortiz", "dvolkov"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["Oct", "Nov"]
HEX = "0123456789abcdef"

DOMAINS: dict[str, dict] = {
    "export": {
        "lang": "py", "pkg": "ledgerd", "service": "ledgerd", "repo": "ledgerd",
        "entities": ["invoice", "credit_note", "ledger_entry", "payout", "statement", "tax_line", "export_batch",
                     "remittance", "dunning_notice", "fx_rate"],
        "attrs": ["amount_minor", "currency", "tenant_id", "issued_at", "period", "status", "checksum", "row_count",
                  "attempt", "external_ref", "due_date", "vat_rate"],
        "areas": ["export", "upload", "render", "ledger", "tax", "dunning", "reconcile", "notify", "fx", "archive"],
        "verbs": ["load", "flush", "render", "upload", "reconcile", "validate", "split", "merge", "schedule", "sign",
                  "archive", "normalise", "collect", "emit"],
        "exceptions": ["UploadError", "LedgerLockTimeout", "RenderError", "ExportAborted", "ChecksumMismatch",
                       "TenantNotFound"],
        "tenants": ["acme-retail", "northwind", "globex-eu", "initech", "umbrella-pharma", "hooli-de", "stark-ind"],
        "log_dir": "/var/log/ledgerd", "test_dir": "tests",
        "commits": ["fix VAT rounding for credit notes", "add export manifest checksum", "refactor uploader session handling",
                    "tune ledger writer flush interval", "add dunning email template", "log request ids on retry",
                    "parallelise PDF rendering", "add idempotency keys to uploads", "move S3 client to shared pool",
                    "raise ledger snapshot retention", "fix CSV quoting for notes", "handle FX rate gaps on weekends",
                    "split reconcile job by tenant", "drop legacy remittance format", "add statement period guard"],
        "hosts": ["ledger-prod-1", "ledger-prod-2", "ledger-prod-3", "ledger-batch-1", "ledger-batch-2", "ledger-db-1"],
        "cfg_keys": {"flush_interval_ms": [250, 500, 1000], "render_timeout_s": [20, 40, 60], "max_retries": [3, 5, 8],
                     "pdf_dpi": [150, 200, 300], "archive_after_days": [30, 90, 365], "log_level": ["info", "debug"],
                     "checksum_algo": ["sha256", "blake2b"], "manifest_version": [2, 3]},
        "env_prefix": "LEDGERD",
    },
    "ci": {
        "lang": "ts", "pkg": "@atlas/ci", "service": "ci-agent", "repo": "atlas",
        "entities": ["job", "shard", "artifact", "cacheEntry", "pipeline", "runner", "step", "report", "matrixCell",
                     "timing"],
        "attrs": ["id", "status", "startedAt", "durationMs", "attempt", "shardIndex", "cacheKey", "sha", "queue",
                  "exitCode", "labels", "os"],
        "areas": ["scheduler", "cache", "artifacts", "reporter", "runners", "matrix", "notify", "timing", "lint", "e2e"],
        "verbs": ["resolve", "restore", "upload", "schedule", "expand", "collect", "summarise", "retry", "prune",
                  "assign", "hash", "annotate"],
        "exceptions": ["CacheMissError", "ShardTimeoutError", "ArtifactUploadError", "RunnerLostError",
                       "MatrixExpansionError"],
        "tenants": ["web", "api", "mobile", "design-system", "docs", "payments-ui", "admin"],
        "log_dir": "/var/log/ci", "test_dir": "packages/ci/test",
        "commits": ["cache pnpm store per lockfile", "split e2e job", "pin chromium", "add flaky-test quarantine list",
                    "parallelise lint", "drop node 16", "retry artifact upload", "move to arm runners for linux",
                    "add build timing report", "annotate PRs with shard timings", "dedupe matrix cells",
                    "prune stale caches nightly", "upload junit reports", "label runners by os version"],
        "hosts": ["linux-ci-01", "linux-ci-02", "arm-ci-01", "arm-ci-02", "win-ci-01", "cache-proxy-1"],
        "cfg_keys": {"parallelism": [2, 4, 8], "agents_queue": ["linux", "arm"], "artifact_paths": ["dist/**", "coverage/**"],
                     "priority": [1, 5, 10], "soft_fail": ["false", "true"], "cancel_on_new_push": ["true"],
                     "skip_checkout": ["false"]},
        "env_prefix": "CI",
    },
    "migration": {
        "lang": "py", "pkg": "orders", "service": "orders-migrate", "repo": "orders",
        "entities": ["order", "shipment", "refund", "customer", "coupon", "address", "line_item", "audit_event",
                     "invoice_link", "warehouse"],
        "attrs": ["created_at", "status", "amount_cents", "currency", "customer_id", "shipped_at", "version",
                  "warehouse_id", "deleted_at", "external_id", "channel", "tax_cents"],
        "areas": ["migrations", "backfill", "models", "repo", "api", "audit", "reporting", "cleanup", "search", "sync"],
        "verbs": ["backfill", "copy", "validate", "lock", "index", "archive", "sync", "partition", "purge", "load",
                  "reindex", "dedupe"],
        "exceptions": ["LockNotAvailable", "BackfillAborted", "MigrationConflict", "ReplicaLagTooHigh",
                       "ConstraintViolation"],
        "tenants": ["web", "pos", "marketplace", "b2b", "wholesale", "app-ios", "app-android"],
        "log_dir": "/var/log/migrations", "test_dir": "db/tests",
        "commits": ["add index on orders.created_at", "split customers.address", "drop legacy coupons table",
                    "add orders.currency default", "rename shipments.eta", "add partial index on refunds",
                    "batch tax backfill", "add check constraint on amount", "move audit log to own schema",
                    "vacuum tuning for shipments", "partition audit_events by month", "add warehouse_id to line_items",
                    "dedupe customers by email", "sync search index on refund"],
        "hosts": ["pg-orders-p1", "pg-orders-p2", "pg-billing-p1", "pg-audit-p1", "pg-reporting-1"],
        "cfg_keys": {"sleep_ms": [50, 100, 200], "concurrently": ["true", "false"], "max_retries": [3, 5],
                     "parallel_workers": [2, 4], "analyze_after": ["true"], "dry_run": ["false"],
                     "progress_every": [1000, 5000]},
        "env_prefix": "ORDERS",
    },
    "leak": {
        "lang": "ts", "pkg": "@relay/gateway", "service": "relay", "repo": "relay",
        "entities": ["connection", "frame", "channel", "subscriber", "heartbeat", "presence", "backlog", "session",
                     "fanout", "deflater"],
        "attrs": ["id", "bufferedBytes", "lastSeen", "channelId", "userId", "compressed", "seq", "state", "region",
                  "protocol", "openedAt", "drops"],
        "areas": ["gateway", "fanout", "presence", "codec", "auth", "metrics", "heartbeat", "sessions", "backlog",
                  "admin"],
        "verbs": ["accept", "broadcast", "encode", "decode", "subscribe", "evict", "drain", "flush", "compress",
                  "authorize", "track", "resume"],
        "exceptions": ["BackpressureError", "SocketClosedError", "FrameTooLargeError", "AuthExpiredError",
                       "ChannelNotFoundError"],
        "tenants": ["chat", "presence", "notifications", "live-scores", "collab-docs", "typing", "cursors"],
        "log_dir": "/var/log/relay", "test_dir": "test",
        "commits": ["batch presence updates", "drop idle sockets after 2 missed pings", "move auth to upgrade handler",
                    "add per-channel fanout metrics", "cap backlog per subscriber", "reuse deflate dictionaries",
                    "log close codes", "add resume tokens", "shard fanout by channel hash", "trim frame headers",
                    "expose admin drain endpoint", "count dropped frames", "lazy-load presence roster"],
        "hosts": ["relay-prod-01", "relay-prod-02", "relay-prod-03", "relay-edge-01", "relay-edge-02"],
        "cfg_keys": {"ping_timeout_ms": [10000, 20000], "max_channels_per_conn": [64, 128, 256],
                     "backlog_frames": [100, 500], "compression_level": [1, 3, 6], "drain_grace_s": [15, 30, 60],
                     "auth_cache_ttl_s": [60, 300]},
        "env_prefix": "RELAY",
    },
    "rollback": {
        "lang": "py", "pkg": "checkout", "service": "checkout", "repo": "checkout",
        "entities": ["cart", "quote", "payment", "tax_quote", "merchant", "coupon", "receipt", "shipping_quote",
                     "refund_request", "price_rule"],
        "attrs": ["merchant_id", "total_minor", "currency", "idempotency_key", "country", "status", "attempt",
                  "expires_at", "line_count", "channel", "risk_score", "locale"],
        "areas": ["cart", "tax", "payments", "quotes", "receipts", "rollout", "coupons", "fraud", "shipping", "pricing"],
        "verbs": ["quote", "price", "authorize", "capture", "apply", "expire", "recalculate", "round", "reserve",
                  "notify", "score", "settle"],
        "exceptions": ["TaxServiceTimeout", "PaymentDeclined", "QuoteExpired", "RateLimited", "CouponNotApplicable"],
        "tenants": ["shop-de", "shop-fr", "shop-jp", "shop-uk", "shop-us", "marketplace", "b2b-portal"],
        "log_dir": "/var/log/checkout", "test_dir": "tests",
        "commits": ["cache tax quotes per cart", "add idempotency to capture", "round JPY totals at the end",
                    "expire quotes after 15 minutes", "move coupon stacking rules to config", "log merchant ids on 429",
                    "add receipt locale fallback", "score risk before authorize", "split shipping quotes by carrier",
                    "retry tax service once", "add price rule audit trail", "batch receipt emails"],
        "hosts": ["checkout-prod-1", "checkout-prod-2", "checkout-prod-3", "deploy-us-01", "deploy-us-02"],
        "cfg_keys": {"quote_ttl_s": [600, 900], "tax_timeout_ms": [800, 1200, 2000], "max_coupons": [2, 3],
                     "risk_threshold": [0.7, 0.8], "receipt_batch": [50, 100], "replicas": [6, 8, 12]},
        "env_prefix": "CHECKOUT",
    },
    "thumbs": {
        "lang": "ts", "pkg": "@img/thumbs", "service": "thumbs", "repo": "thumbs",
        "entities": ["thumbnail", "variant", "originFetch", "crop", "image", "cacheEntry", "srcset", "purge",
                     "transform", "focalPoint"],
        "attrs": ["width", "height", "format", "etag", "quality", "sourceUrl", "bytes", "ttl", "gravity", "dpr",
                  "lastModified", "hits"],
        "areas": ["resize", "origin", "cache", "crop", "purge", "formats", "cdn", "srcset", "metadata", "admin"],
        "verbs": ["resize", "fetch", "encode", "crop", "purge", "negotiate", "sign", "probe", "strip", "cache",
                  "warm", "stream"],
        "exceptions": ["ResizeError", "OriginTimeoutError", "UnsupportedFormatError", "SignatureMismatchError",
                       "TooLargeError"],
        "tenants": ["catalog", "avatars", "editorial", "ads", "marketplace", "user-uploads", "banners"],
        "log_dir": "/var/log/thumbs", "test_dir": "test",
        "commits": ["add AVIF negotiation", "strip EXIF by default", "keep ICC profiles", "add focal point crop",
                    "sign transform URLs", "warm cache for top SKUs", "stream large originals", "add purge-on-edit hook",
                    "tune JPEG quality", "dedupe srcset widths", "log origin status codes", "cap source megapixels"],
        "hosts": ["img-prod-01", "img-prod-02", "img-prod-03", "img-edge-01", "img-edge-02"],
        "cfg_keys": {"jpeg_quality": [78, 82, 85], "max_megapixels": [40, 60], "avif_effort": [4, 6],
                     "strip_metadata": ["true"], "purge_batch": [100, 500], "warm_top_n": [500, 2000]},
        "env_prefix": "THUMBS",
    },
}


def _camel(s: str) -> str:
    parts = re.split(r"[_\-]", s)
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _pascal(s: str) -> str:
    c = _camel(s)
    return c[:1].upper() + c[1:]


def _snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _ing(v: str) -> str:
    if v.endswith("e") and not v.endswith(("ee", "ye")):
        return v[:-1] + "ing"
    return v + "ing"


def _past(v: str) -> str:
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def _plural(s: str) -> str:
    if s.endswith("y") and not s.endswith("ey"):
        return s[:-1] + "ies"
    return s + ("es" if s.endswith(("s", "x", "ch")) else "s")


# =============================================================================
# The artifact engine: realistic source, diffs, test output, logs, git history
# =============================================================================


class Work:
    """Seeded generator of realistic tool output for one domain.

    Every number it draws goes through `i()`, which never returns a value in
    `forbid` (probed values and look-alikes), so round work cannot contain them
    by accident. `check_plan` still asserts it.
    """

    def __init__(self, rng: random.Random, story: str, forbid: set[str] | None = None,
                 forbid_keys: set[str] | None = None):
        self.rng, self.story, self.d = rng, story, DOMAINS[story]
        self.forbid = set(forbid or ())
        self.forbid_keys = set(forbid_keys or ())
        self.used_paths: set[str] = set()
        self.commit_pool = list(self.d["commits"])
        self.krx = _keys_regex(self.forbid_keys)
        # Size hint (chars) for the unit being drawn: units that can scale their
        # output (log lines, test counts, file length) stay under it. _fill sets it.
        self.cap = 10 ** 9
        self.deck: list = []  # round units still to deal (round_unit)

    def fit(self, n: int, lo: int, per: int, overhead: int = 0) -> int:
        """n items of about `per` chars, reduced to fit under the size hint (at least lo)."""
        return max(lo, min(n, (self.cap - overhead) // max(1, per)))

    # ------------------------------------------------------------ primitives
    def i(self, lo: int, hi: int) -> int:
        while True:
            v = self.rng.randint(lo, hi)
            if str(v) not in self.forbid:
                return v

    def f(self, lo: float, hi: float, nd: int = 2) -> str:
        while True:
            s = f"{self.rng.uniform(lo, hi):.{nd}f}"
            if s not in self.forbid and not any(p in self.forbid for p in s.split(".")):
                return s

    def ch(self, xs):
        """A choice that avoids forbidden values when it can."""
        ok = [x for x in xs if str(x) not in self.forbid]
        return self.rng.choice(ok or xs)

    def clock(self, t: int, ms: bool = False) -> str:
        """HH:MM:SS for second-of-day t, moved on past forbidden HH:MM values."""
        t %= 86400
        while (f"{t // 3600:02d}:{t // 60 % 60:02d}" in self.forbid
               or f"{t // 60 % 60:02d}:{t % 60:02d}" in self.forbid):
            t = (t + 61) % 86400
        s = f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"
        return s + (f".{self.i(100, 999)}" if ms else "")

    def hexs(self, n: int) -> str:
        return "".join(self.rng.choice(HEX) for _ in range(n))

    def sample(self, xs, k):
        return self.rng.sample(xs, k)

    def ts(self, day: int | None = None) -> str:
        return (f"2026-09-{day or self.i(10, 27):02d} {self.i(0, 23):02d}:{self.i(0, 59):02d}:"
                f"{self.i(0, 59):02d},{self.i(100, 999)}")

    def hm(self) -> str:
        while True:
            s = f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}"
            if s not in self.forbid:
                return s

    def ent(self):
        return self.ch(self.d["entities"])

    def attr(self):
        return self.ch(self.d["attrs"])

    def area(self):
        return self.ch(self.d["areas"])

    def verb(self):
        return self.ch(self.d["verbs"])

    def exc(self):
        return self.ch(self.d["exceptions"])

    def person(self):
        return self.ch(WORK_PEOPLE)

    def login(self):
        return self.ch(WORK_LOGINS)

    def host(self):
        return self.ch(self.d["hosts"])

    def tenant(self):
        return self.ch(self.d["tenants"])

    def commit_msg(self):
        return self.ch(self.d["commits"])

    def ident(self) -> str:
        return f"{self.verb()}_{self.ent()}" if self.d["lang"] == "py" else _camel(f"{self.verb()}_{self.ent()}")

    def new_path(self, kind: str = "src") -> str:
        """A source path not used before in this session."""
        d, py = self.d, self.d["lang"] == "py"
        for _ in range(200):
            a, e = self.area(), self.ent()
            suffix = self.ch(["", "_service", "_store", "_client", "_utils", "_jobs", "_handlers", "_policy",
                              "_models", "_queue", "_cache"])
            if kind == "test":
                p = (f"{d['test_dir']}/{a}/test_{_snake(e)}{suffix}.py" if py
                     else f"{d['test_dir']}/{a}/{_camel(e + suffix)}.test.ts")
            else:
                p = (f"{d['pkg']}/{a}/{_snake(e)}{suffix}.py" if py
                     else f"src/{a}/{_camel(e + suffix)}.ts")
            if p not in self.used_paths:
                self.used_paths.add(p)
                return p
        raise RuntimeError("path space exhausted")

    # ------------------------------------------------------------ prose
    def sentence(self) -> str:
        e, e2, a = self.ent().replace("_", " "), self.ent().replace("_", " "), self.attr()
        t = self.ch([
            "Helpers for {vi} {es} before they reach the {ar} layer.",
            "{E} handling for the {ar} pipeline: validation, retries and metrics.",
            "Keeps {es} and {e2s} in sync; the {a} field is the source of truth.",
            "Thin wrapper around the {ar} backend so callers never see transport errors directly.",
            "Background jobs that {v} {es} in small batches and report progress.",
            "Policy objects deciding when a {e} may be {vp} again.",
            "Read-through cache for {es}, keyed by {a}.",
            "Serialisation of {es} for the {ar} API (v2 wire format).",
        ])
        return t.format(v=self.verb(), vi=_ing(self.verb()), vp=_past(self.verb()), es=_plural(e), e2s=_plural(e2), E=e.capitalize(), e=e, a=a, ar=self.area())

    def prose(self, n: int = 2) -> str:
        bits = [
            "This keeps the hot path free of extra round trips.",
            "Callers that relied on the old ordering need to sort explicitly now.",
            "The previous behaviour hid partial failures behind a generic error.",
            "No functional change for tenants that never hit the limit.",
            "Covered by the new unit tests; the integration suite is unchanged.",
            "We saw this in staging under replayed production traffic.",
            "The metric names stay the same so existing dashboards keep working.",
            "Rolled out behind the usual canary; no alerts during the bake.",
            "Reverting is safe: the migration is additive.",
            "Follow-up ticket covers the remaining call sites.",
            "This also removes a lock that was held across a network call.",
            "Logging is at debug level to avoid flooding the aggregator.",
        ]
        return " ".join(self.sample(bits, n))

    # ------------------------------------------------------------ source code
    def py_function(self, cls: bool = False) -> list[str]:
        for _ in range(20):
            t = self._py_function(cls)
            if not (self.krx and self.krx.search("\n".join(t))):
                return t
        return t

    def _py_function(self, cls: bool = False) -> list[str]:
        e, a, a2 = self.ent(), self.attr(), self.attr()
        E, es = _pascal(e), _plural(e)
        fn, other = f"{self.verb()}_{e}", f"{self.verb()}_{self.ent()}"
        n1, n2 = self.i(2, 64), self.i(100, 5000)
        exc = self.exc()
        ind = "    " if cls else ""
        self_ = "self, " if cls else ""
        t = self.ch([
            [f"def {fn}({self_}{e}: {E}, *, timeout: float = {self.f(1, 30, 1)}) -> {E}:",
             f'    """{self.sentence()}"""',
             f"    started = time.monotonic()",
             f"    try:",
             f"        result = {'self._client' if cls else '_client()'}.{other}({e}.{a}, timeout=timeout)",
             f"    except {exc} as exc:",
             f'        log.warning("{fn} failed for %s after %.2fs: %s", {e}.{a}, time.monotonic() - started, exc)',
             f"        raise",
             f'    metrics.observe("{self.d["service"].replace("-", "_")}_{fn}_seconds", time.monotonic() - started)',
             f"    return result"],
            [f"def {fn}({self_}{es}: Iterable[{E}], batch_size: int = {n2}) -> Iterator[list[{E}]]:",
             f'    """Yield {es} in batches of at most ``batch_size``."""',
             f"    batch: list[{E}] = []",
             f"    for item in {es}:",
             f"        if item.{a} is None:",
             f"            continue",
             f"        batch.append(item)",
             f"        if len(batch) >= batch_size:",
             f"            yield batch",
             f"            batch = []",
             f"    if batch:",
             f"        yield batch"],
            [f"def {fn}({self_}{e}: {E}) -> bool:",
             f"    if {e}.{a} in _SKIP_{a.upper()}:",
             f"        return False",
             f"    if {e}.{a2} and {e}.attempt >= MAX_ATTEMPTS:",
             f'        log.info("giving up on %s after %d attempts", {e}.{a2}, {e}.attempt)',
             f"        return False",
             f"    return True"],
            [f"def {fn}({self_}rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:",
             f"    counts: dict[str, int] = defaultdict(int)",
             f"    for row in rows:",
             f'        key = row.get("{a}") or "unknown"',
             f"        counts[key] += 1",
             f'    log.debug("{fn}: %d rows, %d distinct {a}", len(rows), len(counts))',
             f"    return dict(counts)"],
            [f"async def {fn}({self_}{e}_id: str) -> {E} | None:",
             f"    cached = await _cache.get(f\"{e}:{{{e}_id}}\")",
             f"    if cached is not None:",
             f"        return {E}.from_json(cached)",
             f"    {e} = await _repo.get_{e}({e}_id)",
             f"    if {e} is None:",
             f"        return None",
             f"    await _cache.set(f\"{e}:{{{e}_id}}\", {e}.to_json(), ttl={self.i(30, 900)})",
             f"    return {e}"],
            [f"def {fn}({self_}{e}: {E}, now: datetime | None = None) -> None:",
             f"    now = now or datetime.now(timezone.utc)",
             f"    if {e}.status == Status.DONE:",
             f"        raise {exc}(f\"{e} {{{e}.{a}}} is already done\")",
             f"    {e}.{a2} = now",
             f"    {e}.status = Status.PENDING",
             f"    _events.publish(\"{e}.{_past(self.verb())}\", {{\"id\": {e}.{a}, \"at\": now.isoformat()}})"],
            [f"def {fn}({self_}config: Mapping[str, Any]) -> Settings:",
             f'    retries = int(config.get("max_retries", {self.i(2, 8)}))',
             f'    backoff = float(config.get("backoff_s", {self.f(0.1, 4, 1)}))',
             f"    if retries < 0 or backoff <= 0:",
             f'        raise ValueError(f"invalid retry settings: {{retries}}, {{backoff}}")',
             f"    return Settings(retries=retries, backoff=backoff, jitter={self.ch(['True', 'False'])})"],
            [f"def {fn}({self_}{es}: list[{E}]) -> list[{E}]:",
             f"    seen: set[str] = set()",
             f"    out = []",
             f"    for item in sorted({es}, key=lambda x: (x.{a}, x.{a2})):",
             f"        if item.{a} in seen:",
             f"            continue",
             f"        seen.add(item.{a})",
             f"        out.append(item)",
             f"    return out"],
            [f"def {fn}({self_}{e}: {E}, limit: int = {n1}) -> list[str]:",
             f"    problems = []",
             f"    if not {e}.{a}:",
             f'        problems.append("missing {a}")',
             f"    if len({e}.{a2} or ()) > limit:",
             f'        problems.append(f"{a2} has more than {{limit}} entries")',
             f"    return problems"],
            [f"def {fn}({self_}rows: Iterable[{E}], *, dry_run: bool = False) -> Summary:",
             f'    """{self.sentence()}',
             f"",
             f"    Rows are processed in chunks of {n1 * 10}; a failing chunk is logged and",
             f"    counted, the rest continue.",
             f'    """',
             f"    summary = Summary()",
             f"    for chunk in chunked(rows, {n1 * 10}):",
             f"        try:",
             f"            with transaction() as tx:",
             f"                for row in chunk:",
             f"                    if row.{a} is None:",
             f"                        summary.skipped += 1",
             f"                        continue",
             f'                    tx.execute(UPDATE_{e.upper()}, {{"id": row.id, "{a2}": row.{a2}}})',
             f"                    summary.updated += 1",
             f"                if dry_run:",
             f"                    tx.rollback()",
             f"        except {exc}:",
             f'            log.exception("{fn}: chunk starting at %s failed", chunk[0].id)',
             f"            summary.failed += len(chunk)",
             f'    log.info("{fn}: %s", summary)',
             f"    return summary"],
            [f"def {fn}_cmd({self_}tenant: str, limit: int = {n2}, verbose: bool = False) -> int:",
             f'    """CLI entry point: {self.verb()} {es} for one tenant and print a table."""',
             f"    items = list(_repo.list_{es}(tenant=tenant, limit=limit))",
             f"    if not items:",
             f'        print(f"no {es} for {{tenant}}", file=sys.stderr)',
             f"        return 1",
             f'    width = max(len(str(i.{a})) for i in items)',
             f"    for item in items:",
             f'        line = f"{{str(item.{a}).ljust(width)}}  {{item.{a2}}}"',
             f"        if verbose:",
             f'            line += f"  attempt={{item.attempt}}"',
             f"        print(line)",
             f'    log.debug("{fn}_cmd printed %d rows for %s", len(items), tenant)',
             f"    return 0"],
            [f"def {fn}({self_}{e}: {E}, *, clock: Callable[[], float] = time.monotonic) -> {E}:",
             f"    attempt = 0",
             f"    delay = {self.f(0.05, 1.5, 2)}",
             f"    while True:",
             f"        attempt += 1",
             f"        try:",
             f"            return {'self._client' if cls else '_client()'}.{other}({e})",
             f"        except {exc} as exc:",
             f"            if attempt >= MAX_ATTEMPTS:",
             f'                log.error("{fn}: giving up on %s after %d attempts", {e}.{a}, attempt)',
             f"                raise",
             f'            log.warning("{fn}: attempt %d failed (%s), retrying in %.2fs", attempt, exc, delay)',
             f"            time.sleep(delay)",
             f"            delay = min(delay * 2, {self.i(5, 60)})"],
        ])
        return [ind + line if line else "" for line in t]

    def py_file(self, path: str, extra: list[str] | None = None, n: int | None = None) -> str:
        d = self.d
        e = self.ent()
        E = _pascal(e)
        imports = self.sample(["import logging", "import time", "from collections import defaultdict",
                               "from dataclasses import dataclass, field",
                               "from datetime import datetime, timezone",
                               "from typing import Any, Iterable, Iterator, Mapping, Sequence"], 4)
        imports = sorted(set(imports) | {"import logging", "import time"})
        local = [f"from {d['pkg']}.{self.area()} import {_pascal(self.ent())}, {self.exc()}",
                 f"from {d['pkg']}.metrics import metrics"]
        consts = []
        for _ in range(self.i(1, 3)):
            k = self.ch(["MAX_ATTEMPTS", "DEFAULT_BATCH", "CACHE_TTL_S", "SLOW_LOG_MS", "PAGE_SIZE", "LOCK_WAIT_S"])
            lim = {"MAX_ATTEMPTS": (3, 9), "DEFAULT_BATCH": (100, 2000), "CACHE_TTL_S": (30, 900),
                   "SLOW_LOG_MS": (200, 2000), "PAGE_SIZE": (50, 500), "LOCK_WAIT_S": (2, 30)}[k]
            if not any(c.startswith(k) for c in consts):
                consts.append(f"{k} = {self.i(*lim)}")
        fields = [f"    {a}: {self.ch(['str', 'int', 'str | None', 'datetime | None', 'float'])}"
                  for a in dict.fromkeys(self.sample(d["attrs"], self.i(3, 6)))]
        lines = [f'"""{self.sentence()}"""', "", "from __future__ import annotations", ""] + imports + [""] + local + [
            "", "log = logging.getLogger(__name__)", ""] + consts + ["", "", "@dataclass", f"class {E}Record:"] + fields
        if extra:
            lines += [""] + extra
        n = n or self.i(3, 6)
        n_methods = n // 4 if n >= 8 else 0
        for k in range(n - n_methods):
            lines += ["", ""] + self.py_function()
        if n_methods:
            cname = f"{_pascal(self.ent())}{self.ch(['Service', 'Store', 'Client', 'Worker', 'Manager'])}"
            lines += ["", "", f"class {cname}:", f'    """{self.sentence()}"""', "",
                      f"    def __init__(self, client, *, max_attempts: int = {self.i(2, 9)}) -> None:",
                      "        self._client = client", "        self._max_attempts = max_attempts"]
            for _ in range(n_methods):
                lines += [""] + self.py_function(cls=True)
        return "\n".join(lines) + "\n"

    def ts_function(self) -> list[str]:
        for _ in range(20):
            t = self._ts_function()
            if not (self.krx and self.krx.search("\n".join(t))):
                return t
        return t

    def _ts_function(self) -> list[str]:
        e, a, a2 = self.ent(), self.attr(), self.attr()
        E, es = _pascal(e), _plural(e)
        fn, other = _camel(f"{self.verb()}_{e}"), _camel(f"{self.verb()}_{self.ent()}")
        exc = self.exc()
        t = self.ch([
            [f"export async function {fn}({e}: {E}, opts: {{ timeoutMs?: number }} = {{}}): Promise<{E}> {{",
             f"  const started = performance.now();",
             f"  try {{",
             f"    return await client.{other}({e}.{a}, {{ timeoutMs: opts.timeoutMs ?? {self.i(500, 30000)} }});",
             f"  }} catch (err) {{",
             f"    if (err instanceof {exc}) {{",
             f"      log.warn({{ {a}: {e}.{a}, err }}, '{fn} failed');",
             f"    }}",
             f"    throw err;",
             f"  }} finally {{",
             f"    metrics.histogram('{self.d['service']}_{_snake(fn)}_ms').observe(performance.now() - started);",
             f"  }}",
             f"}}"],
            [f"export function {fn}({es}: readonly {E}[], size = {self.i(10, 500)}): {E}[][] {{",
             f"  const out: {E}[][] = [];",
             f"  for (let i = 0; i < {es}.length; i += size) {{",
             f"    out.push({es}.slice(i, i + size));",
             f"  }}",
             f"  return out;",
             f"}}"],
            [f"export function {fn}({e}: {E}): boolean {{",
             f"  if (!{e}.{a}) return false;",
             f"  if ({e}.{a2} === undefined || {e}.attempt >= MAX_ATTEMPTS) {{",
             f"    log.info({{ id: {e}.{a} }}, 'skipping after max attempts');",
             f"    return false;",
             f"  }}",
             f"  return true;",
             f"}}"],
            [f"export class {E}{self.ch(['Tracker', 'Registry', 'Pool', 'Queue'])} {{",
             f"  private readonly items = new Map<string, {E}>();",
             f"",
             f"  add(item: {E}): void {{",
             f"    this.items.set(String(item.{a}), item);",
             f"    gauge.set(this.items.size);",
             f"  }}",
             f"",
             f"  remove(key: string): {E} | undefined {{",
             f"    const item = this.items.get(key);",
             f"    this.items.delete(key);",
             f"    gauge.set(this.items.size);",
             f"    return item;",
             f"  }}",
             f"}}"],
            [f"export function {fn}(input: Record<string, unknown>): {E}Options {{",
             f"  const retries = Number(input.retries ?? {self.i(1, 9)});",
             f"  const backoffMs = Number(input.backoffMs ?? {self.i(50, 4000)});",
             f"  if (!Number.isFinite(retries) || retries < 0) {{",
             f"    throw new TypeError(`invalid retries: ${{input.retries}}`);",
             f"  }}",
             f"  return {{ retries, backoffMs, jitter: {self.ch(['true', 'false'])} }};",
             f"}}"],
            [f"export async function* {fn}(source: AsyncIterable<{E}>): AsyncGenerator<{E}> {{",
             f"  const seen = new Set<string>();",
             f"  for await (const item of source) {{",
             f"    const key = `${{item.{a}}}:${{item.{a2}}}`;",
             f"    if (seen.has(key)) continue;",
             f"    seen.add(key);",
             f"    yield item;",
             f"  }}",
             f"}}"],
            [f"export function {fn}(a: {E}, b: {E}): number {{",
             f"  if (a.{a} !== b.{a}) return a.{a} < b.{a} ? -1 : 1;",
             f"  return (a.{a2} ?? 0) > (b.{a2} ?? 0) ? 1 : -1;",
             f"}}"],
            [f"/**",
             f" * {self.sentence()}",
             f" * Failures are retried up to MAX_ATTEMPTS times with exponential backoff.",
             f" */",
             f"export async function {fn}WithRetry({e}: {E}, signal?: AbortSignal): Promise<{E}> {{",
             f"  let delayMs = {self.i(20, 900)};",
             f"  for (let attempt = 1; ; attempt++) {{",
             f"    try {{",
             f"      return await client.{other}({e}, {{ signal }});",
             f"    }} catch (err) {{",
             f"      if (attempt >= MAX_ATTEMPTS || signal?.aborted) {{",
             f"        log.error({{ {a}: {e}.{a}, attempt, err }}, '{fn}: giving up');",
             f"        throw err;",
             f"      }}",
             f"      log.warn({{ {a}: {e}.{a}, attempt, delayMs }}, '{fn}: retrying');",
             f"      await sleep(delayMs);",
             f"      delayMs = Math.min(delayMs * 2, {self.i(2000, 60000)});",
             f"    }}",
             f"  }}",
             f"}}"],
            [f"export function {fn}Handler(deps: {{ repo: {E}Repo; now: () => number }}) {{",
             f"  return async (req: Request, res: Response): Promise<void> => {{",
             f"    const id = String(req.params.id ?? '');",
             f"    if (!id) {{",
             f"      res.status(400).json({{ error: 'missing id' }});",
             f"      return;",
             f"    }}",
             f"    const started = deps.now();",
             f"    const item = await deps.repo.find(id);",
             f"    if (!item) {{",
             f"      res.status(404).json({{ error: '{e} not found' }});",
             f"      return;",
             f"    }}",
             f"    metrics.histogram('{self.d['service']}_{_snake(fn)}_handler_ms').observe(deps.now() - started);",
             f"    res.json({{ {a}: item.{a}, {a2}: item.{a2} }});",
             f"  }};",
             f"}}"],
            [f"export interface {E}{self.ch(['Options', 'Settings', 'Config'])} {{",
             f"  /** {self.sentence()} */",
             f"  {a}: string;",
             f"  {a2}?: number;",
             f"  retries: number;",
             f"  backoffMs: number;",
             f"  jitter: boolean;",
             f"}}"],
        ])
        return t

    def ts_file(self, path: str, extra: list[str] | None = None, n: int | None = None) -> str:
        e = self.ent()
        imports = [f"import {{ {self.exc()}, {self.exc()} }} from '../errors';",
                   f"import {{ log }} from '../log';",
                   f"import {{ metrics{', gauge' if self.rng.random() < 0.5 else ''} }} from '../metrics';",
                   f"import type {{ {_pascal(e)}, {_pascal(self.ent())} }} from '../types';"]
        imports = list(dict.fromkeys(imports))
        consts = [f"const MAX_ATTEMPTS = {self.i(2, 9)};"] + (
            [f"const {self.ch(['FLUSH_EVERY_MS', 'CACHE_TTL_MS', 'SLOW_MS', 'PAGE_SIZE'])} = {self.i(50, 60000)};"]
            if self.rng.random() < 0.6 else [])
        lines = [f"// {self.sentence()}", ""] + imports + [""] + consts
        if extra:
            lines += [""] + extra
        for _ in range(n or self.i(3, 6)):
            lines += [""] + self.ts_function()
        return "\n".join(lines) + "\n"

    def source(self, path: str, extra: list[str] | None = None, n: int | None = None) -> str:
        return self.py_file(path, extra, n) if path.endswith(".py") else self.ts_file(path, extra, n)

    def test_file(self, path: str, names: list[str]) -> str:
        e = self.ent()
        if path.endswith(".py"):
            lines = ["import pytest", "", f"from {self.d['pkg']}.{self.area()} import {_pascal(e)}, {self.exc()}", ""]
            for n in names:
                a = self.attr()
                lines += ["", f"def {n}({self.ch(['make_' + e, 'clock', 'tmp_path', 'monkeypatch'])}):",
                          f"    item = make_{e}({a}={self.ch(['None', repr(self.tenant()), str(self.i(1, 99))])})",
                          f"    result = {self.verb()}_{e}(item)",
                          f"    assert result.{a} == item.{a}",
                          f"    assert result.status == {self.ch(['Status.DONE', 'Status.PENDING', 'Status.FAILED'])}"]
            return "\n".join(lines) + "\n"
        lines = [f"import {{ describe, expect, it }} from 'vitest';",
                 f"import {{ {_camel(self.verb() + '_' + e)} }} from '../../src/{self.area()}/{_camel(e)}';", "",
                 f"describe('{_camel(e)}', () => {{"]
        for n in names:
            lines += [f"  it('{n}', async () => {{",
                      f"    const input = make{_pascal(e)}({{ {self.attr()}: {self.i(1, 99)} }});",
                      f"    const out = await {_camel(self.verb() + '_' + e)}(input);",
                      f"    expect(out.{self.attr()}).toBe(input.{self.attr()});",
                      f"  }});", ""]
        return "\n".join(lines + ["});"]) + "\n"

    def test_names(self, k: int) -> list[str]:
        out = []
        while len(out) < k:
            e, v = self.ent(), self.verb()
            cond = self.ch(["when_empty", "after_retry", "with_unicode", "on_timeout", "is_idempotent", "keeps_order",
                            "rejects_negative", "handles_none", "across_midnight", "with_large_batch",
                            "under_contention", "preserves_metadata"])
            n = f"test_{v}_{_snake(e)}_{cond}"
            if self.d["lang"] == "ts":
                n = n[5:].replace("_", " ")
            if n not in out:
                out.append(n)
        return out

    # ------------------------------------------------------------ edits & diffs
    def mutate(self, content: str) -> tuple[str, str, str]:
        """An edit of a few lines inside one function body: (oldText, newText, new content).
        oldText is unique in the file, as pi's edit tool requires."""
        py = content.startswith('"""')
        lines = content.split("\n")
        skip = ('"""', "//", "#", "else", "elif", "except", "finally", "}", ")", "@")
        cands = [i for i, l in enumerate(lines) if l.startswith("  ") and l.strip() and not l.strip().startswith(skip)]
        for _ in range(50):
            start = self.ch(cands)
            end = start
            n = self.i(3, 12)
            while end + 1 < len(lines) and lines[end + 1].strip() and end - start < n - 1:
                end += 1
            old = lines[start:end + 1]
            if content.count("\n".join(old)) == 1:
                break
        ind = re.match(r"\s*", old[0]).group()
        kind = self.rng.random()
        if kind < 0.3:
            new = [ind + ("# " if py else "// ") + self.ch([
                "Guard against partial results; see the incident notes.", "Keep this ordering: callers depend on it.",
                "Retry once; transient errors are common here.", "Measured: this is not on the hot path.",
                "Do not log the payload here, it can contain customer data."])] + old
        elif kind < 0.6:
            new = [re.sub(r"\b\d{2,5}\b", lambda m: str(self.i(2, 9000)), l, count=1) for l in old]
            if new == old:
                new = [ind + ('log.debug("%s: start", __name__)' if py else "log.debug('start');")] + old
        else:
            a, b = self.attr(), self.attr()
            new = [l.replace(a, b) for l in old]
            if new == old:
                name = f"{self.verb()}_{_snake(self.ent())}"
                new = [ind + (f'metrics.incr("{name}")' if py else f"metrics.counter('{name}').inc();")] + old
        old_t, new_t = "\n".join(old), "\n".join(new)
        return old_t, new_t, content.replace(old_t, new_t, 1)

    def diff(self, path: str, before: str, after: str) -> str:
        return "\n".join(difflib.unified_diff(before.split("\n"), after.split("\n"), f"a/{path}", f"b/{path}",
                                              lineterm="", n=3))

    # ------------------------------------------------------------ test runners
    def py_traceback(self, test: str, path: str) -> list[str]:
        e, a = self.ent(), self.attr()
        exc = self.exc()
        ln = self.i(20, 240)
        return [f"{'_' * 20} {test} {'_' * 20}", "",
                f"    def {test}(make_{e}):",
                f"        item = make_{e}({a}={self.i(1, 99)})",
                f">       result = {self.verb()}_{e}(item)", "",
                f"{path}:{ln}: ",
                f"{'_ ' * 30}",
                f"{self.d['pkg']}/{self.area()}/{_snake(e)}.py:{self.i(20, 300)}: in {self.verb()}_{e}",
                f"    raise {exc}(f\"{e} {{item.{a}}} could not be processed\")",
                f"E   {self.d['pkg']}.errors.{exc}: {e} {self.i(1000, 99999)} could not be processed", ""]

    def pytest_run(self, path: str, names: list[str], failing: list[str], extra_pass: int = 0) -> str:
        n_all = len(names) + extra_pass
        out = [f"============================= test session starts ==============================",
               f"platform linux -- Python 3.14.0, pytest-8.4.2, pluggy-1.6.0",
               f"rootdir: /work/{self.d['repo']}", f"configfile: pyproject.toml",
               f"plugins: {self.ch(['anyio-4.11.0, ', ''])}xdist-3.8.0, timeout-2.4.0",
               f"collected {n_all} items", ""]
        for k, n in enumerate(names):
            pct = int(100 * (k + 1) / max(1, len(names)))
            out.append(f"{path}::{n} {'FAILED' if n in failing else 'PASSED'} {'[' + str(pct).rjust(3) + '%]'}")
        if failing:
            out += ["", "=================================== FAILURES ==================================="]
            for n in failing:
                out += self.py_traceback(n, path)
            out += ["=========================== short test summary info ============================"]
            out += [f"FAILED {path}::{n} - {self.exc()}" for n in failing]
        dur = self.f(0.8, 40)
        tail = f"{len(failing)} failed, {n_all - len(failing)} passed" if failing else f"{n_all} passed"
        out.append(f"{'=' * 22} {tail} in {dur}s {'=' * 22}")
        return "\n".join(out)

    def vitest_run(self, path: str, names: list[str], failing: list[str]) -> str:
        out = [f" RUN  v3.2.4 /work/{self.d['repo']}", ""]
        for n in names:
            mark = "×" if n in failing else "✓"
            out.append(f"   {mark} {path} > {n} {self.i(1, 400)}ms")
        if failing:
            out += ["", f"⎯⎯⎯⎯⎯⎯⎯ Failed Tests {len(failing)} ⎯⎯⎯⎯⎯⎯⎯", ""]
            for n in failing:
                e = self.ent()
                out += [f" FAIL  {path} > {n}",
                        f"AssertionError: expected {self.i(1, 9999)} to be {self.i(1, 9999)} // Object.is equality",
                        f" ❯ {path}:{self.i(10, 200)}:{self.i(5, 40)}",
                        f"    {self.i(10, 200)}|     const out = await {_camel(self.verb() + '_' + e)}(input);",
                        f"       |                       ^", ""]
        p, fl = len(names) - len(failing), len(failing)
        out += ["", f" Test Files  {1 if fl else 0} failed | {0 if fl else 1} passed (1)" if fl else
                " Test Files  1 passed (1)",
                f"      Tests  {fl} failed | {p} passed ({len(names)})" if fl else f"      Tests  {p} passed ({p})",
                f"   Start at  {self.hm()}:{self.i(10, 59)}",
                f"   Duration  {self.f(0.5, 30)}s (transform {self.i(20, 400)}ms, setup 0ms, collect {self.i(50, 900)}ms)"]
        return "\n".join(out)

    def test_run(self, path: str, names: list[str], failing: list[str]) -> str:
        return self.pytest_run(path, names, failing) if path.endswith(".py") else self.vitest_run(path, names, failing)

    # ------------------------------------------------------------ logs
    def log_msg(self) -> tuple[str, str]:
        d = self.d
        e = self.ent().replace("_", " ")
        lvl = self.ch(["INFO"] * 7 + ["DEBUG"] * 2 + ["WARNING"])
        msg = self.ch([
            "{vp} {n} {es} for tenant {t} in {s}s",
            "starting {ar} cycle: {n} {es} queued, {m} in flight",
            "{e} batch {k}/{n} done ({m} rows, {s}s)",
            "slow {ar} call: {ms} ms for {t} (threshold {th} ms)",
            "retrying {e} for tenant {t} (attempt {k} of {th2}): {x}",
            "cache {hm} for {e} {t}; size now {n}",
            "connection to {h} re-established after {s}s",
            "scheduled next {ar} run at {hhmm}",
            "skipped {n} {es} for {t}: nothing changed since last run",
            "{ar} worker {k} idle for {s}s, parking",
            "config reloaded from disk ({n} keys, {m} changed)",
            "drained {n} pending {es} from {h} before shutdown",
        ]).format(vp=_past(self.verb()), n=self.i(2, 4000), es=_plural(e), e=e, t=self.tenant(), s=self.f(0.01, 30),
                  ar=self.area(), m=self.i(1, 300), k=self.i(1, 4), ms=self.i(800, 9000), th=self.ch([500, 750, 1000]),
                  th2=self.ch([5, 8]), x=self.exc(), hm=self.ch(["hit", "miss"]), h=self.host(), hhmm=self.hm())
        if lvl == "WARNING" and "retrying" not in msg:
            msg = f"{self.exc()} while processing {e}: {msg}"
        return lvl, msg

    def app_log(self, n: int, special: dict[int, str] | None = None, day: int | None = None) -> str:
        """`n` lines in time order; `special` maps line numbers to fixed lines. Lines
        carry the time only: the file is per day."""
        special = special or {}
        out = []
        times = sorted(self.rng.randint(0, 86399) for _ in range(n))
        for k, t in enumerate(times):
            if k in special:
                out.append(special[k])
                continue
            lvl, msg = self.log_msg()
            logger = f"{self.d['service']}.{self.area()}"
            out.append(f"{self.clock(t)} {lvl:<7} {logger}: {msg}")
        return "\n".join(out)

    def log_time(self, k: int, n: int) -> str:
        """A time for line k of n that fits between the other lines of app_log (roughly)."""
        t = int(86399 * (k + 0.5) / n)
        return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"

    # ------------------------------------------------------------ git
    def author_line(self) -> str:
        p = self.person()
        return f"Author: {p} {self.ch(['Okafor', 'Lindqvist', 'Moreau', 'Silva', 'Becker', 'Tanaka', 'Novak', 'Haddad'])} <{p.lower()}@{self.d['repo']}.dev>"

    def git_date(self) -> str:
        return (f"Date:   {self.ch(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])} Sep {self.i(10, 27)} "
                f"{self.hm()}:{self.i(10, 59)} 2026 +0200")

    def git_log(self, n: int, special: dict[int, list[str]] | None = None, body: bool = True,
                stat: bool = False) -> str:
        special = special or {}
        out = []
        for k in range(n):
            if k in special:
                out += special[k] + [""]
                continue
            out += [f"commit {self.hexs(40)}", self.author_line(), self.git_date(), "", f"    {self.commit_msg()}"]
            if body:
                out += ["", f"    {self.prose(self.i(1, 3))}"]
            if stat:
                out.append("")
                k_files = self.i(1, 6)
                for _ in range(k_files):
                    path = self.new_path(self.ch(["src", "src", "test"]))
                    a, b = self.i(1, 60), self.i(0, 40)
                    out.append(f" {path:<52} | {a + b:>3} {'+' * min(a, 30)}{'-' * min(b, 20)}")
                out.append(f" {k_files} file{'s' if k_files > 1 else ''} changed, {self.i(2, 200)} insertions(+), "
                           f"{self.i(0, 90)} deletions(-)")
            out.append("")
        return "\n".join(out).rstrip()

    def oneline(self, n: int, special: dict[int, str] | None = None) -> str:
        special = special or {}
        msgs = self.sample(self.d["commits"], min(n, len(self.d["commits"])))
        msgs += [self.commit_msg() for _ in range(n - len(msgs))]
        return "\n".join(special.get(k) or f"{self.hexs(7)} {msgs[k]}" for k in range(n))

    # ------------------------------------------------------------ search
    def grep(self, pattern: str, n: int) -> str:
        py = self.d["lang"] == "py"
        out = []
        for _ in range(n):
            p = (f"{self.d['pkg']}/{self.area()}/{_snake(self.ent())}.py" if py
                 else f"src/{self.area()}/{_camel(self.ent())}.ts")
            e = self.ent()
            line = self.ch([
                f"    {pattern}({_snake(e) if py else _camel(e)}, timeout={self.i(1, 60)})",
                f"    result = {pattern}(batch)" if py else f"  const result = await {pattern}(batch);",
                f"from {self.d['pkg']}.{self.area()} import {pattern}" if py else f"import {{ {pattern} }} from '../{self.area()}';",
                f"        return {pattern}(item, retries=retries)" if py else f"    return {pattern}(item, {{ retries }});",
                f"    # TODO: {pattern} should not block here" if py else f"  // TODO: {pattern} should not block here",
            ])
            out.append(f"{p}:{self.i(3, 400)}:{line}")
        return "\n".join(sorted(out))

    # ------------------------------------------------------------ configs
    def yaml_cfg(self, root: str, special: list[str] | None = None, n: int | None = None) -> str:
        keys = [k for k in self.d["cfg_keys"] if k not in self.forbid_keys]
        chosen = self.sample(keys, min(len(keys), n or self.i(4, len(keys))))
        items = [f"  {k}: {self.ch(self.d['cfg_keys'][k])}" for k in chosen]
        for s in special or []:
            items.insert(self.i(0, len(items)), s)
        return "\n".join([f"{root}:"] + items)

    def k8s_manifest(self, name: str, special: list[str] | None = None) -> str:
        img = f"registry.internal/{self.d['repo']}/{name}:{self.i(2024, 2026)}.{self.i(1, 12)}.{self.i(1, 40)}"
        lines = ["apiVersion: apps/v1", "kind: Deployment", "metadata:", f"  name: {name}",
                 f"  namespace: {self.d['repo']}", "  labels:", f"    app: {name}", f"    team: {self.area()}",
                 "spec:", f"  replicas: {self.i(2, 12)}", "  strategy:", "    type: RollingUpdate",
                 "    rollingUpdate:", f"      maxSurge: {self.i(1, 3)}", "      maxUnavailable: 0",
                 "  template:", "    spec:", "      containers:", f"        - name: {name}", f"          image: {img}",
                 "          resources:", "            requests:", f"              cpu: {self.i(100, 900)}m",
                 f"              memory: {self.i(128, 2048)}Mi", "            limits:",
                 f"              memory: {self.i(1024, 4096)}Mi", "          readinessProbe:", "            httpGet:",
                 "              path: /healthz", f"              port: {self.i(8000, 9100)}",
                 f"            periodSeconds: {self.i(5, 30)}", "          env:"]
        envs = [f"            - name: {self.d['env_prefix']}_{self.ch(['LOG_FORMAT', 'TRACE_SAMPLE', 'REGION_HINT', 'WORKERS', 'SHUTDOWN_GRACE_S'])}",
                f"              value: \"{self.i(1, 300)}\""]
        lines += envs
        for s in special or []:
            lines.append(s)
        return "\n".join(lines)

    def env_file(self, special: list[str] | None = None) -> str:
        pre = self.d["env_prefix"]
        keys = [f"{pre}_{k}" for k in self.sample(["LOG_LEVEL", "WORKERS", "SHUTDOWN_GRACE_S", "TRACE_SAMPLE_RATE",
                                                   "METRICS_PORT", "HTTP_TIMEOUT_MS", "CACHE_SIZE", "FEATURE_FLAGS_URL",
                                                   "MAX_BODY_KB", "STATSD_PREFIX"], 6)]
        keys = [k for k in keys if k not in self.forbid_keys]
        lines = [f"# {self.d['service']} environment, managed by deploy/render-env.sh", ""]
        for k in keys:
            v = {"LOG_LEVEL": self.ch(["info", "warn"]), "FEATURE_FLAGS_URL": f"https://flags.internal/{self.d['repo']}",
                 "STATSD_PREFIX": self.d["service"].replace("-", "_"), "TRACE_SAMPLE_RATE": self.f(0.01, 0.2)}.get(
                k[len(pre) + 1:], str(self.i(2, 9000)))
            lines.append(f"{k}={v}")
        for s in special or []:
            lines.insert(self.i(2, len(lines)), s)
        return "\n".join(lines)

    # ------------------------------------------------------------ misc command output
    def deps(self) -> str:
        if self.d["lang"] == "py":
            pk = ["aiohttp", "attrs", "boto3", "botocore", "certifi", "click", "httpx", "jinja2", "orjson",
                  "prometheus-client", "psycopg", "pydantic", "pytest", "python-dateutil", "pyyaml", "redis",
                  "sqlalchemy", "structlog", "tenacity", "urllib3"]
            rows = [f"{p:<22}{self.i(0, 9)}.{self.i(0, 40)}.{self.i(0, 20)}" for p in sorted(self.sample(pk, 14))]
            return "Package               Version\n--------------------- ---------\n" + "\n".join(rows)
        pk = ["@fastify/websocket", "pino", "prom-client", "undici", "zod", "vitest", "typescript", "tsx", "ioredis",
              "fastify", "sharp", "ws", "lru-cache", "p-limit", "yaml", "esbuild", "msw", "eslint"]
        rows = [f"├── {p}@{self.i(0, 9)}.{self.i(0, 40)}.{self.i(0, 20)}" for p in sorted(self.sample(pk, 12))]
        return f"{self.d['pkg']}@{self.i(1, 4)}.{self.i(0, 30)}.0 /work/{self.d['repo']}\n" + "\n".join(rows)

    def lint(self, paths: list[str]) -> str:
        out = []
        if self.d["lang"] == "py":
            for p in paths:
                for _ in range(self.i(1, 4)):
                    code, msg = self.ch([("F401", "`{x}` imported but unused"), ("E501", "Line too long ({n} > 120)"),
                                         ("B008", "Do not perform function call `{x}` in argument defaults"),
                                         ("SIM108", "Use ternary operator instead of `if`-`else`-block"),
                                         ("UP035", "Import from `collections.abc` instead: `{x}`")])
                    out.append(f"{p}:{self.i(3, 300)}:{self.i(1, 80)}: {code} {msg.format(x=self.ident(), n=self.i(121, 180))}")
            out.append(f"Found {len(out)} errors.\n[*] {self.i(0, len(out))} fixable with the `--fix` option.")
            return "\n".join(out)
        for p in paths:
            out.append(f"/work/{self.d['repo']}/{p}")
            for _ in range(self.i(1, 4)):
                rule, msg = self.ch([("@typescript-eslint/no-unused-vars", "'{x}' is defined but never used"),
                                     ("@typescript-eslint/no-floating-promises", "Promises must be awaited"),
                                     ("prefer-const", "'{x}' is never reassigned. Use 'const' instead"),
                                     ("no-console", "Unexpected console statement")])
                out.append(f"  {self.i(3, 300)}:{self.i(1, 60)}  {self.ch(['error', 'warning'])}  {msg.format(x=self.ident())}  {rule}")
            out.append("")
        out.append(f"✖ {self.i(2, 20)} problems")
        return "\n".join(out)

    def typecheck(self, paths: list[str]) -> str:
        out = []
        for p in paths:
            if self.d["lang"] == "py":
                out.append(f"{p}:{self.i(3, 300)}: error: Argument 1 to \"{self.ident()}\" has incompatible type "
                           f"\"{_pascal(self.ent())} | None\"; expected \"{_pascal(self.ent())}\"  [arg-type]")
            else:
                out.append(f"{p}({self.i(3, 300)},{self.i(1, 60)}): error TS2345: Argument of type "
                           f"'{_pascal(self.ent())} | undefined' is not assignable to parameter of type '{_pascal(self.ent())}'.")
        if self.d["lang"] == "py":
            out.append(f"Found {len(out)} errors in {len(paths)} files (checked {self.i(40, 300)} source files)")
        return "\n".join(out)

    def pods(self, n: int | None = None) -> str:
        rows = ["NAME                                  READY   STATUS    RESTARTS        AGE"]
        for _ in range(n or self.i(4, 9)):
            name = f"{self.d['service']}-{self.area()}-{self.hexs(9)}-{self.hexs(5)}"
            st = self.ch(["Running"] * 6 + ["CrashLoopBackOff", "Pending"])
            rows.append(f"{name:<38}{self.ch(['1/1', '1/1', '0/1']):<8}{st:<10}{self.i(0, 14)} ({self.i(2, 59)}m ago)   {self.i(1, 30)}d")
        return "\n".join(rows)

    def events(self, n: int | None = None) -> str:
        rows = ["LAST SEEN   TYPE      REASON      OBJECT                         MESSAGE"]
        for _ in range(n or self.i(4, 10)):
            obj = f"pod/{self.d['service']}-{self.hexs(5)}"
            reason, msg = self.ch([("Pulled", "Container image already present on machine"),
                                   ("Killing", "Stopping container (rolling update)"),
                                   ("Unhealthy", f"Readiness probe failed: HTTP probe failed with statuscode: {self.ch([500, 503])}"),
                                   ("Scheduled", f"Successfully assigned {self.d['repo']}/{obj[4:]} to {self.host()}"),
                                   ("BackOff", "Back-off restarting failed container")])
            rows.append(f"{self.i(1, 59)}m{'':<9}{self.ch(['Normal', 'Normal', 'Warning']):<10}{reason:<12}{obj:<31}{msg}")
        return "\n".join(rows)

    def health_json(self) -> str:
        comps = {a: {"status": self.ch(["ok", "ok", "ok", "degraded"]), "latency_ms": self.i(1, 900)}
                 for a in self.sample(self.d["areas"], 4)}
        body = {"service": self.d["service"], "version": f"{self.i(1, 9)}.{self.i(0, 40)}.{self.i(0, 9)}",
                "uptime_s": self.i(100, 900000), "components": comps,
                "queue_depth": self.i(0, 5000), "build": self.hexs(12)}
        return json.dumps(body, indent=2)

    def metrics_text(self, k: int = 4) -> str:
        out = []
        svc = self.d["service"].replace("-", "_")
        for a in self.sample(self.d["areas"], min(k, len(self.d["areas"]))):
            name = f"{svc}_{a}_duration_seconds"
            out += [f"# HELP {name} Time spent in {a} operations.", f"# TYPE {name} histogram"]
            tot = 0
            for b in ["0.05", "0.1", "0.25", "0.5", "1", "2.5", "+Inf"]:
                tot += self.i(0, 4000)
                out.append(f'{name}_bucket{{le="{b}"}} {tot}')
            out += [f"{name}_sum {self.f(10, 9000)}", f"{name}_count {tot}"]
        return "\n".join(out)

    def explain(self) -> str:
        t = _plural(self.ent())
        return "\n".join([
            "                                   QUERY PLAN",
            "--------------------------------------------------------------------------------",
            f" Limit  (cost={self.f(0, 9, 2)}..{self.f(10, 9000, 2)} rows={self.i(10, 500)} width={self.i(20, 200)}) (actual time={self.f(0, 5, 3)}..{self.f(5, 900, 3)} rows={self.i(10, 500)} loops=1)",
            f"   ->  Index Scan using {t}_{self.attr()}_idx on {t}  (cost=0.43..{self.f(100, 90000, 2)} rows={self.i(1000, 900000)} width={self.i(20, 200)})",
            f"         Index Cond: ({self.attr()} >= '2026-09-{self.i(10, 27)}'::date)",
            f"         Filter: ({self.attr()} IS NOT NULL)",
            f"         Rows Removed by Filter: {self.i(0, 9000)}",
            f" Planning Time: {self.f(0.1, 3, 3)} ms",
            f" Execution Time: {self.f(1, 900, 3)} ms"])

    def bench(self, label: str, values: list[int]) -> str:
        rows = [f"{label:<14}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'errors':>9}{'rps':>8}"]
        for v in values:
            rows.append(f"{v:<14}{self.i(20, 400):>9}{self.i(400, 2500):>9}{self.i(900, 6000):>9}{self.i(0, 40):>9}{self.i(200, 4000):>8}")
        return "\n".join(rows)

    # ------------------------------------------------------------ long output for round work
    def edits(self, content: str, k: int) -> tuple[list[tuple[str, str]], str]:
        """Up to k non-overlapping edits of `content`, each oldText unique in the
        original (as pi's multi-edit requires): (pairs, new content)."""
        pairs, spans = [], []
        for _ in range(k):
            old, new, _ = self.mutate(content)
            at = content.find(old)
            if any(at < b and a < at + len(old) for a, b in spans):
                continue
            spans.append((at, at + len(old)))
            pairs.append((old, new))
        after = content
        for old, new in pairs:
            after = after.replace(old, new, 1)
        return pairs, after

    def suite_run(self, files: list[tuple[str, list[str]]], failing: list[str]) -> str:
        """A verbose run over several test files, with failures, warnings and timings."""
        return self._pytest_suite(files, failing) if self.d["lang"] == "py" else self._vitest_suite(files, failing)

    def _pytest_suite(self, files, failing) -> str:
        n_all = sum(len(ns) for _, ns in files)
        out = ["============================= test session starts ==============================",
               "platform linux -- Python 3.14.0, pytest-8.4.2, pluggy-1.6.0 -- /work/.venv/bin/python",
               "cachedir: .pytest_cache", f"rootdir: /work/{self.d['repo']}", "configfile: pyproject.toml",
               f"plugins: {self.ch(['anyio-4.11.0, ', ''])}xdist-3.8.0, timeout-2.4.0, cov-7.0.0",
               f"timeout: {self.ch([60, 120, 300])}.0s", f"collecting ... collected {n_all} items", ""]
        k, where = 0, {}
        for path, names in files:
            for n in names:
                k += 1
                where[n] = path
                out.append(f"{path}::{n} {'FAILED' if n in failing else 'PASSED'} [{int(100 * k / n_all):>3}%]")
        if failing:
            out += ["", "=================================== FAILURES ==================================="]
            for n in failing:
                out += self.py_traceback(n, where[n])
        out += ["", "=============================== warnings summary ==============================="]
        for _ in range(self.i(1, 3)):
            path, names = self.ch(files)
            mod = self.ch(["botocore/auth.py", "dateutil/tz/tz.py", "sqlalchemy/sql/schema.py", "pydantic/_internal/_config.py"])
            out += [f"{path}::{self.ch(names)}",
                    f"  /work/.venv/lib/python3.14/site-packages/{mod}:{self.i(40, 900)}: DeprecationWarning: "
                    f"{self.ch(['datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version.', 'Support for class-based `config` is deprecated, use ConfigDict instead.', 'The default value of `timezone` will change in the next major release.'])}",
                    f"    {self.ch(['now = datetime.datetime.utcnow()', 'warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)', 'return cls(**kwargs)'])}", ""]
        out.append("-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html")
        out += ["", "============================= slowest 10 durations ============================="]
        allt = [(p, n) for p, ns in files for n in ns]
        for p, n in self.sample(allt, min(10, len(allt))):
            out.append(f"{self.f(0.05, 9)}s {self.ch(['call    ', 'setup   ', 'teardown'])} {p}::{n}")
        if failing:
            out += ["=========================== short test summary info ============================"]
            out += [f"FAILED {where[n]}::{n} - {self.d['pkg']}.errors.{self.exc()}" for n in failing]
        dur = self.f(4, 180)
        tail = (f"{len(failing)} failed, {n_all - len(failing)} passed" if failing else f"{n_all} passed")
        out.append(f"{'=' * 12} {tail}, {self.i(1, 9)} warnings in {dur}s {'=' * 12}")
        return "\n".join(out)

    def _vitest_suite(self, files, failing) -> str:
        out = [f" RUN  v3.2.4 /work/{self.d['repo']}", ""]
        n_fail_files = 0
        for path, names in files:
            bad = [n for n in names if n in failing]
            n_fail_files += bool(bad)
            out.append(f" {'❯' if bad else '✓'} {path} ({len(names)} tests{f' | {len(bad)} failed' if bad else ''}) "
                       f"{self.i(20, 4000)}ms")
            for n in names:
                out.append(f"   {'×' if n in failing else '✓'} {_camel(self.ent())} > {n} {self.i(1, 900)}ms")
        if failing:
            out += ["", f"⎯⎯⎯⎯⎯⎯⎯ Failed Tests {len(failing)} ⎯⎯⎯⎯⎯⎯⎯", ""]
            where = {n: p for p, ns in files for n in ns}
            for n in failing:
                e = self.ent()
                out += [f" FAIL  {where[n]} > {n}",
                        f"AssertionError: expected {self.i(1, 9999)} to be {self.i(1, 9999)} // Object.is equality", "",
                        f"- Expected", f"+ Received", "",
                        f"- {self.i(1, 9999)}", f"+ {self.i(1, 9999)}", "",
                        f" ❯ {where[n]}:{self.i(10, 200)}:{self.i(5, 40)}",
                        f"    {self.i(10, 200)}|     const out = await {_camel(self.verb() + '_' + e)}(input);",
                        f"    {self.i(10, 200)}|     expect(out.{self.attr()}).toBe(input.{self.attr()});",
                        f"       |                       ^", "",
                        f"⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯", ""]
        n_all = sum(len(ns) for _, ns in files)
        fl = len(failing)
        out += ["", f" Test Files  {n_fail_files} failed | {len(files) - n_fail_files} passed ({len(files)})" if fl
                else f" Test Files  {len(files)} passed ({len(files)})",
                f"      Tests  {fl} failed | {n_all - fl} passed ({n_all})" if fl else f"      Tests  {n_all} passed ({n_all})",
                f"   Start at  {self.hm()}:{self.i(10, 59)}",
                f"   Duration  {self.f(2, 90)}s (transform {self.i(200, 4000)}ms, setup {self.i(0, 900)}ms, "
                f"collect {self.i(500, 9000)}ms, tests {self.i(1000, 60000)}ms, environment {self.i(0, 900)}ms, "
                f"prepare {self.i(100, 2000)}ms)"]
        return "\n".join(out)

    def describe_pod(self) -> str:
        name = f"{self.d['service']}-{self.area()}-{self.hexs(9)}-{self.hexs(5)}"
        img = f"registry.internal/{self.d['repo']}/{self.d['service']}:{self.i(2024, 2026)}.{self.i(1, 12)}.{self.i(1, 40)}"
        started = f"{self.ch(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])}, {self.i(10, 27)} Sep 2026 {self.hm()}:{self.i(10, 59)} +0000"
        lines = [f"Name:             {name}", f"Namespace:        {self.d['repo']}", "Priority:         0",
                 f"Service Account:  {self.d['service']}", f"Node:             {self.host()}/10.{self.i(0, 9)}.{self.i(0, 255)}.{self.i(2, 254)}",
                 f"Start Time:       {started}", "Labels:           app=" + self.d["service"],
                 f"                  pod-template-hash={self.hexs(9)}", "Status:           Running",
                 f"IP:               10.{self.i(40, 90)}.{self.i(0, 255)}.{self.i(2, 254)}",
                 f"Controlled By:    ReplicaSet/{self.d['service']}-{self.area()}-{self.hexs(9)}", "Containers:",
                 f"  {self.d['service']}:", f"    Container ID:   containerd://{self.hexs(64)}", f"    Image:          {img}",
                 f"    Image ID:       {img.rsplit(':', 1)[0]}@sha256:{self.hexs(64)}",
                 f"    Port:           {self.i(8000, 9100)}/TCP", "    State:          Running",
                 f"      Started:      {started}", "    Last State:     Terminated",
                 f"      Reason:       {self.ch(['OOMKilled', 'Error', 'Completed'])}",
                 f"      Exit Code:    {self.ch([137, 1, 0])}", f"    Ready:          True",
                 f"    Restart Count:  {self.i(0, 14)}", "    Limits:", f"      memory:  {self.i(1024, 4096)}Mi",
                 "    Requests:", f"      cpu:     {self.i(100, 900)}m", f"      memory:  {self.i(128, 1024)}Mi",
                 f"    Liveness:   http-get http://:{self.i(8000, 9100)}/healthz delay=10s timeout=1s period={self.i(5, 30)}s #success=1 #failure=3",
                 f"    Readiness:  http-get http://:{self.i(8000, 9100)}/readyz delay=5s timeout=1s period={self.i(5, 30)}s #success=1 #failure=3",
                 "    Environment:"]
        for k in self.sample(["LOG_FORMAT", "TRACE_SAMPLE", "WORKERS", "SHUTDOWN_GRACE_S", "METRICS_PORT"], 4):
            lines.append(f"      {self.d['env_prefix']}_{k}:  {self.i(1, 300)}")
        lines += ["    Mounts:", "      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-"
                  f"{self.hexs(5)} (ro)", "Conditions:", "  Type              Status", "  Initialized       True",
                  "  Ready             True", "  ContainersReady   True", "  PodScheduled      True",
                  "QoS Class:                   Burstable", "Node-Selectors:              <none>",
                  "Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s",
                  "                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s",
                  "Events:", "  Type     Reason     Age                From     Message",
                  "  ----     ------     ----               ----     -------"]
        for _ in range(self.i(2, 6)):
            lines.append(f"  {self.ch(['Normal ', 'Warning'])}  {self.ch(['Pulled   ', 'Started  ', 'Unhealthy', 'BackOff  '])}  "
                         f"{self.i(2, 59)}m (x{self.i(1, 9)} over {self.i(1, 9)}h)  kubelet  "
                         f"{self.ch(['Container image already present on machine', 'Started container', 'Readiness probe failed: HTTP probe failed with statuscode: 503'])}")
        return "\n".join(lines)

    # ------------------------------------------------------------ the agent's own words in round work
    _THINK = [
        "If nothing here lines up with the problem I'll move on rather than dig further.",
        "The output will be long; what matters is the failures and the summary at the end.",
        "Keep the change small so it can be reviewed separately from the main fix.",
        "Check whether this is covered by tests before touching it.",
        "This is housekeeping, not the main issue, so keep it quick.",
        "Look for anything that changed recently or anything that retries in a loop.",
        "Don't widen the scope beyond what was asked.",
        "Note anything odd but don't chase it unless it matches what we've already seen.",
        "Problems here would show up as warnings first, so scan for those.",
        "Start from the public entry point and follow the calls down.",
        "A circular import would explain the odd initialisation order; check the imports too.",
        "The last change here went in without much review, so read it carefully.",
        "Compare with the neighbouring module; they should handle errors the same way.",
        "Prefer a keyword argument with a default over changing every caller.",
        "Run only the relevant tests first; the full suite takes a while.",
    ]

    def think(self, *first: str) -> str:
        """Short reasoning before a call: a specific sentence, often with a generic follow-up."""
        e, ar = _plural(self.ent().replace("_", " ")), self.area()
        lead = self.ch(first) if first else self.ch([
            f"The {ar} code is where {e} are handled; that's the part to read closely.",
            f"Worth seeing how {e} flow through {ar} before deciding anything.",
            f"Check how {ar} deals with {e} on failure."])
        r = self.rng.random()
        if r < 0.5:
            return f"{lead} {self.ch(self._THINK)}"
        if r < 0.62:
            return f"{lead} {' '.join(self.sample(self._THINK, 2))}"
        return lead

    def say(self, *opts: str) -> str:
        return self.ch(opts)


# =============================================================================
# Round work: varied, realistic units of agent work (no story details)
# =============================================================================


ASK_P = 0.75  # share of units opened by a short user request; the rest continue the agent's own work


def _ask(w: Work, *opts: str) -> list[dict]:
    """Some units start with a short user request; others continue without one."""
    return [_u(w.ch(opts))] if w.rng.random() < ASK_P else []


def _fname(p: str) -> str:
    return p.rsplit("/", 1)[-1]


def _test_cmd(tp: str, verbose: bool = False) -> str:
    if tp.endswith(".py"):
        return f"pytest {tp} {'-v' if verbose else '-q'}"
    return f"pnpm vitest run {tp}{' --reporter verbose' if verbose else ''}"


def _maybe(w: Work, p: float, *opts: str) -> str | None:
    """A short line of narration with probability p."""
    return w.ch(opts) if w.rng.random() < p else None


# Round units produce what real agent work produces: few calls, long output
# (whole files, full test runs, long log tails, many grep hits), and little
# prose (a short line between calls, thinking before most calls). Sizes scale
# down to the size hint `w.cap` when _fill needs a small unit.

def _unit_read_edit(w: Work) -> list[dict]:
    p = w.new_path()
    src = w.source(p, n=w.fit(w.i(8, 28) + 4 * (w.d["lang"] == "ts"), 4, 520, 1500))
    pairs, _after = w.edits(src, w.ch([1, 2, 2, 3, 3]))
    fn = w.ident()
    steps = _ask(w, f"Can you tidy up {p}? The {fn} path is hard to follow.",
                 f"Have a look at {p}, something in there looks off.",
                 f"While you're in {p.rsplit('/', 2)[-2]}, clean up {_fname(p)}.",
                 f"Next: {p}. Make the error handling consistent with the rest of {w.area()}.")
    steps.append(_a(thinking=w.think(f"Read {p} first.", f"Need to see all of {_fname(p)} before changing it.",
                                     f"Open {p} and check how errors propagate."),
                    text=_maybe(w, 0.3, f"Reading {_fname(p)}.", "Let me look at the module first."),
                    calls=[_read(p, src)]))
    first = pairs[0][0].strip().splitlines()[0].strip()[:60]
    steps.append(_a(thinking=w.think("A small targeted edit is enough.", "Keep the change minimal.",
                                     f"Only {len(pairs)} block{'s' if len(pairs) > 1 else ''} need to change.",
                                     f"The block around `{first}` is the culprit."),
                    text=_maybe(w, 0.4, f"Adjusting {_fname(p)}.", "Small fix, nothing else changes."),
                    calls=[_edit(p, pairs)]))
    if w.rng.random() < 0.6:
        tp = w.new_path("test")
        names = w.test_names(w.fit(w.i(10, 40), 4, 95))
        fail = w.sample(names, 1) if w.rng.random() < 0.25 else []
        steps.append(_a(thinking=w.think(f"Run {_fname(tp)} to make sure nothing else moved."),
                        calls=[_bash(_test_cmd(tp, True), w.test_run(tp, names, fail))]))
        steps.append(_a(text=(f"`{fail[0]}` fails, but it predates this change." if fail else
                              f"All {len(names)} pass.")))
    elif w.rng.random() < 0.5:
        steps.append(_a(text=w.ch([f"Done; behaviour is unchanged.", f"Edited {_fname(p)}."])))
    return steps


def _unit_write_test(w: Work) -> list[dict]:
    tp = w.new_path("test")
    names = w.test_names(w.fit(w.i(8, 22), 3, 330))
    content = w.test_file(tp, names)
    fail = w.sample(names, 1) if w.rng.random() < 0.3 else []
    steps = _ask(w, f"Add tests for the {w.ident()} edge cases we talked about.",
                 f"We have no coverage for {_plural(w.ent().replace('_', ' '))} in {w.area()}; add a few tests.",
                 f"Write a regression test file for the {w.ident()} behaviour.")
    steps.append(_a(thinking=w.think("Write a focused test module.", "Cover the empty, retry and ordering cases.",
                                     "One test per edge case, named after the behaviour."),
                    calls=[_write(tp, content)]))
    steps.append(_a(thinking=w.think(f"Run the new tests."),
                    calls=[_bash(_test_cmd(tp, True), w.test_run(tp, names, fail))]))
    steps.append(_a(text=(f"`{fail[0]}` fails: that's the bug we expected to catch." if fail else
                          f"{len(names)} new tests, all passing.")))
    return steps


def _unit_write_module(w: Work) -> list[dict]:
    p = w.new_path()
    content = w.source(p, n=w.fit(w.i(4, 12), 2, 520, 1200))
    steps = _ask(w, f"Pull the {w.area()} helpers out into their own module.",
                 f"Can you add a small {w.area()} module for the {w.ident()} logic?")
    steps.append(_a(thinking=w.think(f"Put the helpers in {_fname(p)} and keep the old call sites working.",
                                     "A new module is cleaner than growing the existing one."),
                    calls=[_write(p, content)]))
    py = w.d["lang"] == "py"
    if w.rng.random() < 0.5:
        cmd, out = ("mypy " + p if py else "pnpm tsc --noEmit -p ."), w.typecheck([p])
    else:
        cmd, out = ("ruff check " if py else "pnpm eslint ") + p, w.lint([p])
    steps.append(_a(thinking=w.think("Check the new file before wiring it in."), calls=[_bash(cmd, out)]))
    steps.append(_a(text=w.ch([f"Wrote {_fname(p)}; one checker complaint to follow up.",
                               f"New module in place; the findings are style nits."])))
    return steps


def _unit_grep_read(w: Work) -> list[dict]:
    fn = w.ident()
    root = w.d["pkg"] + "/" if w.d["lang"] == "py" else "src/"
    steps = _ask(w, f"Where else is {fn} called?", f"Find all callers of {fn} before we change its signature.",
                 f"Is {fn} still used anywhere?")
    steps.append(_a(thinking=w.think(f"Search for {fn} across the package."),
                    calls=[_bash(f"rg -n '{fn}' {root}", w.grep(fn, w.fit(w.i(15, 60), 5, 110)))]))
    for k in range(1 + (w.rng.random() < 0.35)):
        p = w.new_path()
        src = w.source(p, n=w.fit(w.i(6, 22) + 4 * (w.d["lang"] == "ts"), 3, 520, 600))
        steps.append(_a(thinking=w.think(f"Check the caller in {p}.", f"{_fname(p)} looks like the main caller."),
                        text=_maybe(w, 0.3, f"Opening {_fname(p)}.", "Now the main caller."),
                        calls=[_read(p, src)]))
    steps.append(_a(text=w.ch([f"Callers are contained; a keyword argument with a default is enough.",
                               f"{fn} is used in {w.i(3, 9)} modules; I'd add a defaulted argument."])))
    return steps


def _unit_history(w: Work) -> list[dict]:
    p = w.new_path()
    if w.rng.random() < 0.5:
        n = w.fit(w.i(8, 25), 3, 650)
        steps = _ask(w, f"Who has been changing {p.rsplit('/', 2)[-2]} lately?", f"What's the recent history of {p}?")
        steps.append(_a(thinking=w.think(f"Read the log for {p.rsplit('/', 2)[-2]}."),
                        calls=[_bash(f"git log --stat -n {n} -- {p.rsplit('/', 1)[0]}", w.git_log(n, stat=True))]))
        steps.append(_a(text=w.ch([f"Mostly small fixes; nothing that changes {w.area()} behaviour.",
                                   f"{w.person()} did most of the recent work there."])))
        return steps
    before = w.source(p, n=w.fit(w.i(8, 20), 4, 520, 1500))
    _, after = w.edits(before, w.i(2, 4))
    h = w.hexs(40)
    show = "\n".join([f"commit {h}", w.author_line(), w.git_date(), "", f"    {w.commit_msg()}", "",
                      f"    {w.prose(2)}", "", f"diff --git a/{p} b/{p}", f"index {w.hexs(7)}..{w.hexs(7)} 100644",
                      w.diff(p, before, after)])
    steps = _ask(w, f"What did commit {h[:7]} change?", f"Can you check what {h[:9]} did to {w.area()}?")
    steps.append(_a(thinking=w.think(f"Show the commit."), calls=[_bash(f"git show {h[:7]}", show)]))
    steps.append(_a(text=w.ch([f"{h[:7]} only touches {_fname(p)}; a behaviour-preserving cleanup.",
                               f"No functional change in {h[:7]}."])))
    return steps


def _unit_log(w: Work) -> list[dict]:
    while True:
        day, ar = w.i(10, 27), w.area()
        path = f"{w.d['log_dir']}/{ar}-2026-09-{day:02d}.log"
        if path not in w.used_paths:
            break
    w.used_paths.add(path)
    n = w.fit(w.i(80, 320), 15, 105)
    steps = _ask(w, f"Anything odd in {_fname(path)}?", f"Check the {ar} log from the {day}th.",
                 f"Can you skim the {ar} logs from Sep {day}?")
    steps.append(_a(thinking=w.think(f"Tail the {ar} log for the {day}th."),
                    calls=[_bash(f"tail -n {n} {path}", w.app_log(n, day=day))]))
    steps.append(_a(text=w.ch([f"Sep {day} looks routine: retries and slow calls, nothing related.",
                               f"Nothing new on the {day}th; the usual {w.exc()} noise.",
                               f"A burst of slow calls on the {day}th that recovers on its own."])))
    return steps


def _unit_suite(w: Work) -> list[dict]:
    budget = w.fit(w.i(60, 220) + 40 * (w.d["lang"] == "ts"), 8, 100, 2500)
    files = []
    while budget > 0:
        k = min(budget, w.i(6, 30))
        files.append((w.new_path("test"), w.test_names(k)))
        budget -= k
    allnames = [n for _, ns in files for n in ns]
    fail = w.sample(allnames, w.ch([0, 0, 1, 1, 2]))
    tdir = w.d["test_dir"] + "/" + files[0][0].split("/")[-2]
    cmd = f"pytest {tdir} -v" if w.d["lang"] == "py" else f"pnpm vitest run {tdir} --reporter verbose"
    steps = _ask(w, f"Run the {w.area()} tests.", "Can you run the suite?", f"Let's see if {tdir} is green.")
    steps.append(_a(thinking=w.think("Run the whole directory, verbose, to see every test."),
                    calls=[_bash(cmd, w.suite_run(files, fail))]))
    steps.append(_a(text=(f"{len(fail)} failures, both known and tracked separately." if len(fail) == 2 else
                          f"One failure, in code we haven't touched." if fail else
                          f"{len(allnames)} passed.")))
    return steps


def _unit_lint(w: Work) -> list[dict]:
    paths = [w.new_path() for _ in range(w.fit(w.i(6, 18), 2, 200))]
    py = w.d["lang"] == "py"
    if w.rng.random() < 0.5:
        cmd = ("ruff check " if py else "pnpm eslint ") + " ".join(paths)
        out = w.lint(paths)
    else:
        cmd = "mypy " + " ".join(paths) if py else "pnpm tsc --noEmit -p ."
        out = w.typecheck(paths)
    steps = _ask(w, "Is the linter happy?", "Run the type checker over what we touched.", "Check lint before we commit.",
                 f"Does {w.area()} type-check?")
    steps.append(_a(thinking=w.think("Run the checker over the files we touched and their neighbours."),
                    calls=[_bash(cmd, out)]))
    steps.append(_a(text=w.ch(["Pre-existing findings, none from our change.",
                               f"Only style nits in {_fname(paths[0])}; leaving them."])))
    return steps


def _unit_manifest(w: Work) -> list[dict]:
    while True:
        name = f"{w.d['service']}-{w.area()}-{w.ch(['worker', 'api', 'cron', 'consumer', 'sync'])}"
        path = f"deploy/k8s/{name}.yaml"
        if path not in w.used_paths:
            break
    w.used_paths.add(path)
    steps = _ask(w, f"What resources does {name} get?", f"Check the manifest for {name}.")
    steps.append(_a(thinking=w.think(f"Read the {name} manifest."), calls=[_read(path, w.k8s_manifest(name))]))
    if w.rng.random() < 0.85:
        steps.append(_a(thinking=w.think("Compare with what is actually running."),
                        calls=[_bash(f"kubectl -n {w.d['repo']} describe pod -l app={name}", w.describe_pod())]))
    steps.append(_a(text=w.ch([f"Requests and limits for {name} look sensible.",
                               f"The readiness probe on {name} is aggressive but unrelated."])))
    return steps


def _unit_cluster(w: Work) -> list[dict]:
    ns = w.d["repo"]
    steps = _ask(w, "How are the pods doing?", "Any restarts in the cluster?", f"Check the {ns} namespace.",
                 f"Is the {w.area()} deployment healthy?")
    pods = w.pods(w.i(8, 24))
    steps.append(_a(thinking=w.think("List the pods first."), calls=[_bash(f"kubectl -n {ns} get pods", pods)]))
    steps.append(_a(calls=[_bash(f"kubectl -n {ns} get events --sort-by=.lastTimestamp | tail -40",
                                 w.events(w.i(15, 40)))]))
    if w.rng.random() < 0.5:
        first = pods.splitlines()[1].split()[0]
        steps.append(_a(thinking=w.think(f"Describe {first}, it restarted most recently."),
                        calls=[_bash(f"kubectl -n {ns} describe pod {first}", w.describe_pod())]))
    steps.append(_a(text=w.ch(["Restarts are from the rolling update; pods are stable.",
                               "Healthy apart from routine rescheduling."])))
    return steps


def _unit_http(w: Work) -> list[dict]:
    host = w.host()
    steps = _ask(w, f"Is {host} healthy?", f"Check the metrics on {host}.")
    if w.rng.random() < 0.3:
        steps.append(_a(thinking=w.think(f"Hit the health endpoint on {host}."),
                        calls=[_bash(f"curl -s http://{host}:{w.i(8000, 9100)}/healthz | jq .", w.health_json())]))
    else:
        steps.append(_a(thinking=w.think(f"Pull the duration histograms from {host}."),
                        calls=[_bash(f"curl -s http://{host}:{w.i(9100, 9300)}/metrics | grep duration_seconds",
                                     w.metrics_text(w.fit(w.i(4, 10), 2, 650)))]))
    steps.append(_a(text=w.ch([f"{host} looks fine.", f"{host} is up; latencies within the usual range."])))
    return steps


def _unit_deps(w: Work) -> list[dict]:
    cmd = "uv pip list" if w.d["lang"] == "py" else "pnpm ls --depth 0"
    steps = _ask(w, "What versions are we actually running?", "List the installed dependencies.")
    steps.append(_a(thinking=w.think("List installed versions."), calls=[_bash(cmd, w.deps())]))
    steps.append(_a(text=w.ch(["Nothing surprising in the dependency list.", "Versions match the lockfile."])))
    return steps


def _unit_bench(w: Work) -> list[dict]:
    a = w.area()
    vals = sorted(w.sample([2, 4, 8, 12, 16, 24, 32, 48, 64], w.i(3, 5)))
    steps = _ask(w, f"Benchmark the {a} path with a few concurrency levels.", f"How does {a} scale with more workers?")
    steps.append(_a(thinking=w.think(f"Benchmark {a} at a few concurrency levels."),
                    calls=[_bash(f"./scripts/bench.sh {a} --levels {','.join(map(str, vals))}",
                                 w.bench("concurrency", vals))]))
    steps.append(_a(text=w.ch([f"{a} scales until {vals[-1]}, where p99 jumps.",
                               f"Differences for {a} are within noise."])))
    return steps


def _unit_explain(w: Work) -> list[dict]:
    t = _plural(w.ent())
    steps = _ask(w, f"Is the {t} query using the index?", f"Why is the {t} listing slow?")
    steps.append(_a(thinking=w.think(f"EXPLAIN ANALYZE the {t} listing."),
                    calls=[_bash(f"psql -X -c 'EXPLAIN ANALYZE SELECT * FROM {t} WHERE ... LIMIT 200'", w.explain())]))
    plan = steps[-1]["calls"][0]["result"]
    ms = re.search(r"Execution Time: ([\d.]+)", plan).group(1)
    steps.append(_a(text=f"Index scan, {ms} ms; the time goes into the filter."))
    return steps


def _unit_review_diff(w: Work) -> list[dict]:
    parts, files = [], []
    for _ in range(w.fit(w.i(2, 6), 1, 2000)):
        p = w.new_path()
        files.append(_fname(p))
        before = w.source(p, n=w.i(6, 16))
        _, after = w.edits(before, w.i(2, 4))
        parts.append(f"diff --git a/{p} b/{p}\nindex {w.hexs(7)}..{w.hexs(7)} 100644\n" + w.diff(p, before, after))
    steps = _ask(w, "Show me what we have uncommitted.", "What's in the working tree right now?",
                 f"Did we leave anything uncommitted in {w.area()}?", "Can you show me the diff before we commit?")
    steps.append(_a(thinking=w.think("Look at the full working-tree diff."),
                    calls=[_bash("git diff" + w.ch(["", " --stat -p", " HEAD"]), "\n".join(parts))]))
    steps.append(_a(text=w.ch([f"Only {', '.join(files)} changed; self-contained.",
                               "The diff is what we expect; ready to commit."])))
    return steps


UNITS = [(_unit_read_edit, 20), (_unit_write_test, 11), (_unit_write_module, 6), (_unit_grep_read, 10),
         (_unit_history, 6), (_unit_log, 7), (_unit_suite, 9), (_unit_lint, 4), (_unit_manifest, 4),
         (_unit_cluster, 4), (_unit_http, 3), (_unit_deps, 1), (_unit_bench, 2), (_unit_explain, 2),
         (_unit_review_diff, 6)]


def round_unit(w: Work) -> list[dict]:
    """The next unit of round work. Units are dealt from a shuffled deck holding
    each unit type `weight` times, so a session's mix (and with it the
    composition) follows the weights closely instead of drifting with i.i.d. draws."""
    if not w.deck:
        w.deck = [fn for fn, wt in UNITS for _ in range(wt)]
        w.rng.shuffle(w.deck)
    return w.deck.pop()(w)


# =============================================================================
# Skeleton builder: facts, lures, look-alikes and where each value may occur
# =============================================================================


class Skel:
    """Collects one session's planned steps and declares, per label, the values it
    owns and the steps they may occur in.

    Labels: a fact key ("ticket"), "decoy:<name>" (a look-alike inside the plant
    phase, e.g. the decommissioned host), "rival:<name>" (a competing-episode
    look-alike). `add(step, plants=..., at=...)` records the step for those labels.
    `gap(weight)` marks a place for realistic work in the plant phase, filled later
    so the phase reaches its size; positions survive the expansion because they
    are recorded by step identity.
    """

    def __init__(self, sid: str, story: str):
        self.sid, self.story = sid, story
        self.facts: dict[str, Fact] = {}
        self.labels: dict[str, dict] = {}
        self.phases: dict[int, list[dict]] = {0: []}
        self.phase = 0
        self.files: set[str] = set()
        self.w: Work | None = None

    # ---------------------------------------------------------------- declarations
    def fact(self, key: str, dtype: str, source: str, value, question: str, *, old=None, alts=(),
             slot_keys=()) -> str:
        value = str(value)
        old = None if old is None else str(old)
        acc = "|".join(value_regex(x) for x in [value, *alts])
        self.facts[key] = Fact(uid=f"{self.sid}:{key}", slot=key, key=key, kind=dtype, source=source,
                               position="early", value=value, value_old=old, question=question, accept=acc,
                               intrusion=value_regex(old) if old else None)
        self.labels[key] = {"values": [value, *alts] + ([old] if old else []), "steps": [],
                            "slot_keys": list(slot_keys)}
        return value

    def lure(self, key: str, dtype: str, question: str) -> None:
        self.facts[key] = Fact(uid=f"{self.sid}:{key}", slot=key, key=key, kind=dtype, source="lure",
                               position="none", value=None, question=question)

    def decoy(self, name: str, *values) -> None:
        self.labels[f"decoy:{name}"] = {"values": [str(v) for v in values], "steps": [], "slot_keys": []}

    def rival(self, name: str, *values) -> None:
        self.labels[f"rival:{name}"] = {"values": [str(v) for v in values], "steps": [], "slot_keys": []}

    def values(self) -> set[str]:
        return {v for lab in self.labels.values() for v in lab["values"]}

    # ---------------------------------------------------------------- steps
    def add(self, step: dict, *, plants=(), at=()) -> dict:
        for k in (*plants, *at):
            self.labels[k]["steps"].append(step)
        if plants:
            step["fact_uid"] = self.facts[plants[0]].uid
        if self.phase == 0:
            for c in step.get("calls", []):
                if c["name"] in ("read", "edit", "write"):
                    self.files.add(c["args"]["path"])
        self.phases[self.phase].append(step)
        return step

    def gap(self, weight: float = 1.0) -> None:
        self.phases[self.phase].append({"kind": "_gap", "weight": weight})

    def story_file(self, *paths: str) -> None:
        """Files the story reads via bash (cat, tail, crontab dumps...)."""
        self.files.update(paths)

    def begin(self, phase: int) -> None:
        self.phase = phase
        self.phases.setdefault(phase, [])


# =============================================================================
# Static checks (PROTOCOL.md §5.1): single defensible answer, pilot flaws, variety
# =============================================================================


def _values_regex(values) -> re.Pattern | None:
    vs = sorted({v for v in values if len(v) >= 3}, key=len, reverse=True)
    if not vs:
        return None
    return re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(v) for v in vs) + r")(?![\w-])")


def _keys_regex(keys) -> re.Pattern | None:
    ks = sorted(set(keys), key=len, reverse=True)
    if not ks:
        return None
    # A config/env/code assignment of the key: `key: v`, `key = v`, `KEY=v`, `"key": v`.
    return re.compile(r"(?<![\w-])[\"']?(" + "|".join(re.escape(k) for k in ks) + r")[\"']?\s*[:=]")


def check_plan(plan: Plan) -> None:
    """Assert the plan's material invariants; AssertionError on any violation.

    1. Single defensible answer. Every value of 3+ characters owned by a label
       (a fact's value, its superseded value and accepted variants; a decoy; a
       competing look-alike) occurs only in the steps the plan declares for it.
       So probed values never recur in round work, look-alikes stay in their
       episodes, and no generated noise repeats a planted value.
    2. No look-alike equals a fact value; every fact value is in a declared step,
       and the step carrying the fact's uid contains an accepted answer.
    3. Pilot flaw 1: no step outside a fact's declared steps assigns a config/env
       key bound to that fact's slot (`slot_keys`), e.g. `regions: us`.
    4. Pilot flaw 2: round work never reads, edits or mentions a story file, so it
       cannot re-show a story file without a planted key.
    5. Variety: no step text over 200 characters repeats within the session, and
       at least 90% of round steps are distinct (over the episode rounds; each
       plain round of an extended plan, Study 1b, within itself).
    """
    sid = plan.session_id
    pos = plan.meta["positions"]
    allowed: dict[str, set[tuple[int, int]]] = {}
    owners: dict[str, set[str]] = {}
    for label, p in pos.items():
        for v in p["values"]:
            allowed.setdefault(v, set()).update((a, b) for a, b in p["at"])
            owners.setdefault(v, set()).add(label)
    fact_keys = {f.key for f in plan.facts if not f.is_lure}
    fact_vals = {v for k in fact_keys for v in pos[k]["values"]}
    for label, p in pos.items():
        if label.startswith(("rival:", "decoy:")):
            clash = set(p["values"]) & fact_vals
            assert not clash, f"{sid}: look-alike {label} equals a fact value: {sorted(clash)}"
    rx = _values_regex(allowed)
    key_owner = {k: label for label, p in pos.items() for k in p.get("slot_keys", [])}
    krx = _keys_regex(key_owner)
    files = plan.meta.get("story_files", [])
    seen: dict[str, tuple[int, int]] = {}
    found: dict[str, set] = {}
    by_uid = {f.uid: f for f in plan.facts}
    for ph, phase in enumerate(plan.phases):
        for i, st in enumerate(phase["steps"]):
            t = step_text(st)
            if rx:
                for m in rx.finditer(t):
                    v = m.group(0)
                    assert (ph, i) in allowed[v], (
                        f"{sid}: value {v!r} ({', '.join(sorted(owners[v]))}) occurs in {phase['name']}[{i}], "
                        f"outside its planned steps")
                    found.setdefault(v, set()).add((ph, i))
            if krx:
                for m in krx.finditer(t):
                    label = key_owner[m.group(1)]
                    assert [ph, i] in pos[label]["at"], (
                        f"{sid}: {phase['name']}[{i}] assigns {m.group(1)!r}, a key bound to {label}'s question "
                        f"(pilot flaw 1)")
            if ph > 0:
                for c in st.get("calls", []):
                    a = args_text(c["args"])
                    for path in files:
                        assert path not in a, (f"{sid}: {phase['name']}[{i}] touches story file {path} "
                                               f"(pilot flaw 2)")
            if len(t) > 200:
                assert t not in seen, f"{sid}: step text repeats: {phase['name']}[{i}] and {seen[t]}"
                seen[t] = (ph, i)
            if st.get("fact_uid"):
                f = by_uid[st["fact_uid"]]
                assert re.search(f.accept, t), f"{sid}: step {phase['name']}[{i}] carries {f.key} without its value"
    for k in fact_keys:
        f = next(x for x in plan.facts if x.key == k)
        assert pos[k]["at"], f"{sid}: fact {k} has no planned step"
        if len(f.value) >= 3:
            accepted = [x for x in pos[k]["values"] if x != f.value_old]
            assert any(found.get(x) for x in accepted), f"{sid}: fact {k} value {f.value!r} never occurs"
    n_ep = plan.meta.get("episode_rounds", len(plan.phases) - 1)
    rounds = [json.dumps(s, sort_keys=True) for ph in plan.phases[1:n_ep + 1] for s in ph["steps"]]
    if rounds:
        assert len(set(rounds)) >= 0.9 * len(rounds), f"{sid}: round steps are only {len(set(rounds))}/{len(rounds)} distinct"
    # Plain rounds (Study 1b, rounds 4-8): each round's steps are 90% distinct
    # among themselves. Across eight rounds, short acknowledgements ("All 37
    # pass.") recur as they do in real sessions; long steps never repeat (above).
    for ph in plan.phases[n_ep + 1:]:
        steps = [json.dumps(s, sort_keys=True) for s in ph["steps"]]
        assert len(set(steps)) >= 0.9 * len(steps), (
            f"{sid}: {ph['name']} steps are only {len(set(steps))}/{len(steps)} distinct")


# =============================================================================
# Composition and sizing (pure: chars and per-category chars/token)
# =============================================================================


def phase_chars(plan: Plan, names=None) -> dict[str, int]:
    out = dict.fromkeys(CATEGORIES, 0)
    for ph in plan.phases:
        if names is None or ph["name"] in names:
            for st in ph["steps"]:
                for k, v in step_category_chars(st).items():
                    out[k] += v
    return out


def composition(plan: Plan, cpt, names=None) -> dict:
    """Estimated real tokens per category and their shares.

    `cpt` is one chars/token ratio or a dict per category (CATEGORIES). Shares
    are reported for tool output, tool-call arguments and the rest (user text,
    assistant text, thinking)."""
    chars = phase_chars(plan, names)
    per = cpt if isinstance(cpt, dict) else dict.fromkeys(CATEGORIES, cpt)
    toks = {k: chars[k] / per[k] for k in CATEGORIES}
    total = sum(toks.values()) or 1.0
    rest = toks["user"] + toks["assistant_text"] + toks["thinking"]
    return {"chars": chars, "tokens": {k: round(v) for k, v in toks.items()}, "total_tokens": round(total),
            "share": {"tool_output": toks["tool_output"] / total, "tool_args": toks["tool_args"] / total,
                      "rest": rest / total}}


# Composition of real pi coding sessions (character shares of all message
# content; tool calls per 100k characters), measured on the user's own sessions:
# tool output 81-90%, tool-call arguments 8-16%, assistant text 0.4-1.2%, user
# 0.2-0.9%, thinking 0.4-1.7%, 19-25 calls per 100k chars. The bounds below keep
# the material from drifting back to prose-heavy, call-heavy sessions (which
# would inflate mech-compact's verbatim checkpoint). Thinking is held at a
# floor, not the real median, so thinking-sourced details stay plausible.
COMPOSITION_BOUNDS = {
    "tool_output": (0.80, 1.0),
    "assistant_text": (0.0, 0.015),
    "user": (0.0, 0.010),
    "thinking": (0.012, 1.0),
    "calls_per_100k": (17.0, 27.0),
}


def char_composition(plan: Plan) -> dict:
    """Character shares per category over the whole session, and tool calls per
    100k characters (all message content, as step_category_chars counts it)."""
    chars = phase_chars(plan)
    total = sum(chars.values()) or 1
    calls = sum(len(s.get("calls", [])) for ph in plan.phases for s in ph["steps"])
    return {"chars": total, "calls": calls, "share": {k: chars[k] / total for k in CATEGORIES},
            "calls_per_100k": calls * 100_000 / total}


def composition_problems(plan: Plan) -> list[str]:
    """Violations of COMPOSITION_BOUNDS (empty when the session is in bounds)."""
    c = char_composition(plan)
    got = dict(c["share"], calls_per_100k=c["calls_per_100k"])
    return [f"{k} {got[k]:.4g} outside [{lo}, {hi}]" for k, (lo, hi) in COMPOSITION_BOUNDS.items()
            if not lo <= got[k] <= hi]


def phase_tokens(plan: Plan, cpt: float) -> dict[str, int]:
    """Estimated real tokens per phase with one chars/token ratio (as the generator sizes them)."""
    return {ph["name"]: round(sum(step_chars(s) for s in ph["steps"]) / cpt) for ph in plan.phases}


# =============================================================================
# Plans
# =============================================================================

SKELETONS: dict[str, dict] = {}


def _register(story: str, skeleton: str, values, plant, episodes) -> None:
    """values(sk, rng) declares facts, lures, decoys and rivals and returns the
    value dict; plant(sk, v) adds the plant steps; episodes(sk, v, r) adds round
    r's competing episode."""
    SKELETONS[story] = {"skeleton": skeleton, "values": values, "plant": plant, "episodes": episodes}


def _fill(w: Work, target_chars: int, rx: re.Pattern | None, krx: re.Pattern | None) -> list[dict]:
    """Round units until target_chars; a unit that would contain a planted value,
    a look-alike or a slot key assignment is drawn again. The last unit is drawn
    so that the total lands near the target rather than a whole unit past it."""
    steps, tot, tries = [], 0, 0
    while tot < target_chars:
        w.cap = target_chars - tot + int(0.03 * target_chars) + 1500
        u = round_unit(w)
        t = steps_text(u)
        if (rx and rx.search(t)) or (krx and krx.search(t)):
            continue
        n = sum(step_chars(s) for s in u)
        over = tot + n - target_chars
        if over > 0.03 * target_chars + 1500 and tries < 40:
            tries += 1
            if target_chars - tot < 1500:
                break
            continue
        steps += u
        tot += n
    w.cap = 10 ** 9
    return steps


def episode_rounds(scenario: dict) -> int:
    """Rounds opened by a competing episode: all of them, unless the scenario
    extends a study with plain rounds (Study 1b: `episode_rounds` 3, `rounds` 8)."""
    return int(scenario.get("episode_rounds", scenario["rounds"]))


def plan_session_id(scenario: dict, index: int) -> str:
    """The session id of plan `index`. A continuation study (`continue_from`)
    keeps the source study's ids (s1-s000, ...), so its facts (uids), plans and
    trial rows line up with the lines it continues."""
    exp = (scenario.get("continue_from") or {}).get("experiment", scenario["experiment"])
    return f"{exp.lower()}-s{index:03d}"


def _ext_rng(seed: int, r: int, ext_attempt: int) -> random.Random:
    """The noise stream of plain round r (r > episode_rounds): its own seed, so
    adding rounds never changes the phases before them."""
    return random.Random(seed * 7919 + 104729 * r + 15485863 * ext_attempt + 3)


def _build(scenario: dict, index: int, story: str, cpt: float, attempt: int, ext_attempt: int = 0) -> Plan:
    exp = scenario["experiment"]
    seed = scenario["seed"] * 1000 + index
    sid = plan_session_id(scenario, index)
    n_ep = episode_rounds(scenario)
    spec = SKELETONS[story]
    rng = random.Random(seed)
    sk = Skel(sid, story)
    v = spec["values"](sk, rng)
    vals = sk.values()
    keys = {k for lab in sk.labels.values() for k in lab["slot_keys"]}
    w = Work(random.Random(seed * 7 + 17 + 1000 * attempt), story, forbid=vals, forbid_keys=keys)
    sk.w = w
    spec["plant"](sk, v)
    for r in range(1, scenario["rounds"] + 1):
        sk.begin(r)
        if r <= n_ep:
            spec["episodes"](sk, v, r)
    rx, krx = _values_regex(vals), _keys_regex(keys)
    w.used_paths |= sk.files
    # Plant phase: expand gaps into realistic work so the phase reaches plant_tokens.
    plant = sk.phases[0]
    gaps = [s for s in plant if s["kind"] == "_gap"]
    have = sum(step_chars(s) for s in plant if s["kind"] != "_gap")
    deficit = max(0, int(scenario.get("plant_tokens", 0) * cpt) - have)
    tw = sum(g["weight"] for g in gaps) or 1.0
    out = []
    for s in plant:
        if s["kind"] == "_gap":
            out += _fill(w, int(deficit * s["weight"] / tw), rx, krx) if deficit else []
        else:
            out.append(s)
    for s in out:
        s["plant"] = True
    phases = [{"name": "plant", "steps": out}]
    for r in range(1, scenario["rounds"] + 1):
        if r > n_ep:
            # Plain round (Study 1b, rounds 4-8): round work only, from its own
            # seed stream; the Work state (paths, deck) carries on from round r-1.
            w.rng = _ext_rng(seed, r, ext_attempt)
        ep = sk.phases[r]
        target = int(scenario["round_tokens"][r - 1] * cpt) - sum(step_chars(s) for s in ep)
        phases.append({"name": f"round{r}", "steps": ep + _fill(w, target, rx, krx)})
    ids = {id(s): (ph, i) for ph, p in enumerate(phases) for i, s in enumerate(p["steps"])}
    positions = {label: {"values": lab["values"], "at": [list(ids[id(s)]) for s in lab["steps"]],
                         "slot_keys": lab["slot_keys"]} for label, lab in sk.labels.items()}
    facts = list(sk.facts.values())
    order = [f.uid for f in facts]
    random.Random(seed + 5).shuffle(order)
    meta = {"story": story, "skeleton": spec["skeleton"], "competing_rounds": True,
            "detail_types": {f.key: f.kind for f in facts if not f.is_lure},
            "chars_per_token": cpt, "positions": positions, "story_files": sorted(sk.files),
            "build_attempt": attempt}
    if scenario["rounds"] > n_ep:
        meta.update({"episode_rounds": n_ep, "extension_attempt": ext_attempt})
    plan = Plan(exp, sid, seed, facts, phases, order, meta)
    check_plan(plan)
    return plan


def build_story_plan(scenario: dict, index: int, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    """The S1/P1 plan for session `index` (story scenario["session_story"][index]).

    Sizes use scenario["chars_per_token"] (measured on this material with
    tokcheck.py) when present; it overrides the `chars_per_token` argument, which
    is pi's estimate ratio from the config. `keep_recent_tokens` is unused: rounds
    are sized in real tokens by scenario["round_tokens"].

    A build whose random noise happens to hit a planted value is redrawn with a
    new noise stream (`meta["build_attempt"]`); a systematic violation fails
    every attempt and raises.
    """
    cpt = float(scenario.get("chars_per_token", chars_per_token))
    story = scenario["session_story"][index]
    n_ep = episode_rounds(scenario)
    err = None
    if scenario["rounds"] > n_ep:
        # Plain rounds after the episode rounds (Study 1b). The episode rounds
        # are drawn exactly as a scenario with rounds = episode_rounds draws them
        # (same attempt), so they stay byte-identical to the source study's
        # plans; only the plain rounds' own stream is redrawn on a violation.
        base = _first_ok(lambda a: _build(dict(scenario, rounds=n_ep), index, story, cpt, a))
        for ext in range(8):
            try:
                return _build(scenario, index, story, cpt, base.meta["build_attempt"], ext)
            except AssertionError as e:
                err = e
        raise err
    return _first_ok(lambda a: _build(scenario, index, story, cpt, a))


def _first_ok(build) -> Plan:
    err = None
    for attempt in range(8):
        try:
            return build(attempt)
        except AssertionError as e:
            err = e
    raise err


# =============================================================================
# Shared helpers for skeletons
# =============================================================================


def _hexr(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(HEX) for _ in range(n))


def _weekday(day: int, month: int = 10) -> str:
    import datetime as _dt
    return _dt.date(2026, month, day).strftime("%A")


def _date_alts(day: int, month: str = "October") -> list[str]:
    m3 = month[:3]
    mn = {"September": 9, "October": 10, "November": 11}[month]
    return [f"{month} {day}", f"{m3} {day}", f"{day} {m3}", f"2026-{mn:02d}-{day:02d}", f"{day}th {month}",
            f"{month} {day}th"]


def _distinct(rng: random.Random, draw, avoid: set[str]):
    """Draw until the value (as a string) is not in `avoid`; add it to `avoid`."""
    while True:
        x = draw(rng)
        if str(x) not in avoid:
            avoid.add(str(x))
            return x


def _ls_logs(w: Work, names: list[str]) -> str:
    rows = [f"total {w.i(400, 9000)}"]
    for n in names:
        m = re.search(r"2026-09-(\d\d)", n)
        day = int(m.group(1)) + 1 if m else w.i(20, 27)
        rows.append(f"-rw-r----- 1 {w.d['service'].split('-')[0]} adm {w.i(20000, 900000):>7} Sep {day:>2} "
                    f"{w.hm()} {n}")
    return "\n".join(rows)


def _readme(w: Work, title: str, paras: list[str], code: list[str]) -> str:
    out = [f"# {title}", "", w.sentence(), ""]
    for i, p in enumerate(paras):
        out += [p, ""]
        if i == 0 and code:
            out += ["```", *code, "```", ""]
    out += ["## Development", "", f"Run the tests with `{'pytest' if w.d['lang'] == 'py' else 'pnpm test'}`. "
            f"{w.prose(2)}", ""]
    return "\n".join(out)


# =============================================================================
# Story 1: export (ledgerd nightly export; Python). Skeleton "ticket-first":
# the ticket and deadline open the session, the schedule and two failed nights
# come from tool output, a three-step tuning of the multipart part size sits in
# the middle, and people and test details close the plant phase, followed by a
# long stretch of work before round 1.
# =============================================================================

EXPORT_TESTS = ["test_rollover_at_month_end", "test_currency_rounding_eur", "test_invoice_number_gap",
                "test_vat_split_payment", "test_refund_rounding_chf", "test_ledger_close_period"]
EXPORT_PEOPLE = ["Marta", "Olu", "Brigid", "Teodor", "Amara", "Yusuf", "Greta", "Haruto"]
EXPORT_AUTHORS = ["rkowalczyk", "abernard", "nokafor", "ylindqvist", "cmoreau", "hpetrov"]


def _export_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["ticket"] = _distinct(rng, lambda r: f"FIN-{r.randint(2100, 4899)}", used)
    v["dday"] = rng.randint(6, 16)
    v["deadline"] = f"{v['dday']} October"
    stg = rng.randint(3, 9)
    v["host"], v["prev"] = f"ledger-stg-{stg:02d}", f"ledger-stg-{stg - 1:02d}"
    v["day1"] = rng.randint(14, 19)
    v["day2"] = v["day1"] + 1
    v["code1"] = _distinct(rng, lambda r: f"LGR-{r.randint(4100, 4999)}", used)
    v["code2"] = _distinct(rng, lambda r: f"LGR-{r.randint(5100, 5999)}", used)
    v["req"] = _distinct(rng, lambda r: f"req-{_hexr(r, 6)}", used)
    v["cron"] = f"{rng.choice([7, 13, 17, 23, 41, 47])} {rng.choice([1, 2, 3])} * * {rng.choice(['1-5', '1-6', '*'])}"
    v["hash"] = _hexr(rng, 7)
    v["author"] = rng.choice(EXPORT_AUTHORS)
    v["initial"] = rng.choice([5120, 6144, 7168])
    v["tried"] = rng.choice([24576, 32768])
    v["final"] = rng.choice([10240, 12288, 14336])
    v["pin"] = f"1.40.{rng.randint(3, 19)}"
    k = rng.randint(2, 4)
    v["hidden"], v["readme_prefix"] = f"exports/v{k}-legacy/", f"exports/v{k}/"
    v["flaky"], v["real"] = rng.sample(EXPORT_TESTS, 2)
    v["env"] = rng.choice([4500, 6500, 7500, 9500])
    v["contact"], v["on_leave"] = rng.sample(EXPORT_PEOPLE, 2)
    v["flag"] = f"ledger_upload_jitter_v{rng.randint(2, 5)}"
    v["n_tenants"] = rng.randint(31, 48)
    # rivals for the competing episodes
    v["ticket2"] = _distinct(rng, lambda r: f"FIN-{r.randint(2100, 4899)}", used)
    v["d3"] = v["day2"] + rng.randint(2, 4)
    v["d4"] = v["d3"] + 1
    v["code3"] = _distinct(rng, lambda r: f"LGR-{r.randint(4100, 5999)}", used)
    v["code4"] = _distinct(rng, lambda r: f"LGR-{r.randint(4100, 5999)}", used)
    v["req3"] = _distinct(rng, lambda r: f"req-{_hexr(r, 6)}", used)
    v["a"], v["b"], v["c"] = rng.choice([4096, 3072]), rng.choice([20480, 16384]), rng.choice([8192, 9216])
    v["pin2"] = f"2.5.{rng.randint(1, 9)}"
    v["cron2"] = f"{rng.choice([5, 11, 29, 37, 53])} {rng.choice([4, 5])} * * {rng.choice(['0', '6', '0,3'])}"
    stg2 = rng.choice([x for x in range(3, 12) if x not in (stg, stg - 1)])
    v["host2"] = f"ledger-stg-{stg2:02d}"
    v["contact2"], v["leave2"] = rng.sample([p for p in EXPORT_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["hidden2"] = f"statements/v{rng.randint(5, 7)}-legacy/"
    v["flag2"] = f"ledger_statement_render_v{rng.randint(2, 5)}"

    sk.fact("ticket", "ticket", "user", v["ticket"], "Which finance ticket is this work tracked under?")
    sk.fact("deadline", "deadline_date", "user", v["deadline"], "By which date does finance need the export working again?",
            alts=_date_alts(v["dday"]))
    sk.fact("staging_host", "host", "user", v["host"], "What is the staging box called now?",
            alts=[f"stg-{stg:02d}"])
    sk.fact("error_ref", "error_code", "tool_output", v["code1"],
            f"What error reference did the failed uploads log on September {v['day1']}?")
    sk.fact("request_both", "shared_id", "tool_output", v["req"],
            f"Which request id failed on both September {v['day1']} and September {v['day2']}?")
    sk.fact("cron", "cron", "tool_output", v["cron"], "What is the cron schedule of the nightly export job?")
    sk.fact("jitter_hash", "commit_hash", "tool_output", v["hash"],
            "What is the short hash of the commit that made the uploader retries jittered?")
    sk.fact("jitter_author", "author", "thinking", v["author"], "Who wrote the jittered-retry change?")
    sk.fact("part_initial", "count_initial", "assistant_text", v["initial"],
            "What multipart part size, in KB, was the uploader using when we started?",
            slot_keys=["part_size_kb", "LEDGERD_UPLOAD_PART_KB"])
    sk.fact("part_tried", "count_tried", "assistant_text", v["tried"],
            "Which part size did we try first that made things worse?")
    sk.fact("part_final", "count_final", "superseded", v["final"], "Which part size did we end up keeping?",
            old=v["tried"])
    sk.fact("boto_pin", "pinned_dep", "user", v["pin"], "Which boto3 version must we stay on?")
    sk.fact("s3_prefix", "hidden_location", "thinking", v["hidden"],
            "Under which S3 prefix does prod really write the exports?", alts=[v["hidden"].rstrip("/")],
            slot_keys=["export_prefix", "s3_prefix"])
    sk.fact("flaky_test", "flaky_test", "tool_output", v["flaky"], "Which export test is flaky?")
    sk.fact("upload_timeout", "env_value", "tool_output", v["env"],
            "What is LEDGERD_UPLOAD_TIMEOUT_MS set to in deploy/ledgerd.env?",
            slot_keys=["LEDGERD_UPLOAD_TIMEOUT_MS"])
    sk.fact("dash_contact", "contact", "user", v["contact"], "Who should we ask about the finance dashboards?")
    sk.fact("retry_flag", "feature_flag", "assistant_text", v["flag"],
            "What is the name of the feature flag for the new retry path?")
    sk.lure("pdf_engine", "lure", "Which PDF rendering engine did we switch the invoices to?")
    sk.lure("kafka_topic", "lure", "Which Kafka topic do the export events go to?")
    sk.lure("bucket_retention", "lure", "What lifecycle retention did we set on the export bucket?")
    sk.decoy("prev_host", v["prev"])
    sk.decoy("code2", v["code2"])
    sk.decoy("readme_prefix", v["readme_prefix"], v["readme_prefix"].rstrip("/"))
    sk.decoy("real_failure", v["real"])
    sk.decoy("on_leave", v["on_leave"])
    for name in ("ticket2", "code3", "code4", "req3", "pin2", "cron2", "host2", "contact2", "leave2", "hidden2",
                 "flag2"):
        sk.rival(name, v[name])
    sk.rival("parts2", v["a"], v["b"], v["c"])
    return v


def _export_upload_line(w: Work, k: int, n: int, code: str, req: str) -> str:
    t = w.tenant()
    return (f"{w.log_time(k, n)} ERROR   ledgerd.upload: upload of {t} export part {w.i(2, 40)}/{w.i(41, 90)} failed "
            f"after 3 attempts: SlowDown: Please reduce your request rate. (error_ref={code}, request={req})")


def _export_failed_night(w: Work, n: int, code: str, req: str, req_at: int, n_fail: int) -> str:
    special = {}
    at = sorted(w.sample(range(5, n - 3), n_fail - 1) + [req_at])
    for k in at:
        special[k] = _export_upload_line(w, k, n, code, req if k == req_at else f"req-{w.hexs(6)}")
    special[n - 1] = (f"{w.log_time(n - 1, n)} ERROR   ledgerd.export: nightly export finished with {n_fail} failed "
                      f"uploads; manifest not written")
    return w.app_log(n, special)


def _export_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    logdir = "/var/log/ledgerd"
    d1, d2 = v["day1"], v["day2"]
    sk.add(_u(f"Picking up {v['ticket']}: the nightly ledger export has failed every night since Monday and finance "
              f"needs it working again by {v['dday']} October ({_weekday(v['dday'])}). Logs are on ledger-batch-1 "
              f"under {logdir}/."), plants=("ticket", "deadline"))
    cron = "\n".join([
        "# m h  dom mon dow   command",
        "MAILTO=ops-ledger@ledgerd.dev",
        "*/15 * * * * /opt/ledgerd/bin/ledgerd fx-sync --quiet",
        f"{v['cron']} /opt/ledgerd/bin/ledgerd export --nightly --all-tenants >> {logdir}/export-$(date +\\%F).log 2>&1",
        f"{w.ch([3, 9, 33])} 6 * * 0 /opt/ledgerd/bin/ledgerd reconcile --weekly",
        f"{w.ch([12, 44])} 4 1 * * /opt/ledgerd/bin/ledgerd archive --older-than 400d"])
    sk.add(_a(thinking="Find out when the export runs and which logs exist.",
              calls=[_bash("ssh ledger-batch-1 crontab -l -u ledgerd", cron),
                     _bash(f"ssh ledger-batch-1 ls -la {logdir}/",
                           _ls_logs(w, [f"export-2026-09-{d:02d}.log" for d in range(d1 - 4, d2 + 3)]
                                    + ["fx-sync.log", "reconcile.log"]))]), plants=("cron",))
    sk.add(_a(text=f"The export runs from cron on ledger-batch-1. The logs from the {d1 - 4}th to the {d1 - 1}th are "
                   f"normal size; the {d1}th is the first big one. Reading the first two failed nights."))
    n = 64
    sk.add(_a(thinking=f"Read the log of September {d1}.",
              calls=[_read(f"{logdir}/export-2026-09-{d1:02d}.log",
                           _export_failed_night(w, n, v["code1"], v["req"], 23, 6))]),
           plants=("error_ref",), at=("request_both",))
    sk.add(_a(text=f"Every failed upload on the {d1}th has the same error reference, and the message is S3's SlowDown: "
                   f"we're being throttled. Let me compare with the {d2}th."))
    sk.add(_a(thinking=f"Read the log of September {d2}.",
              calls=[_read(f"{logdir}/export-2026-09-{d2:02d}.log",
                           _export_failed_night(w, n, v["code2"], v["req"], 41, 5))]),
           plants=("request_both",), at=("decoy:code2",))
    sk.add(_a(text=f"The {d2}th fails with a different reference, but one request failed on both nights. That may be "
                   f"a symptom of the throttling rather than a cause; I'll keep it in mind."))
    sk.gap(0.6)
    sk.add(_a(thinking="What changed in the uploader right before the failures started?",
              calls=[_bash("git log --oneline -15 -- ledgerd/upload/",
                           w.oneline(15, {w.i(2, 8): f"{v['hash']} uploader: switch retries to jittered exponential backoff"}))]),
           plants=("jitter_hash",))
    src = w.py_file("ledgerd/upload/multipart.py")
    sk.add(_a(thinking=f"{v['author']} wrote the jittered backoff change; the PR was theirs. If the retries are the "
                       f"problem they should review the fix. Now the multipart code.",
              calls=[_read("ledgerd/upload/multipart.py", src)]), plants=("jitter_author",))
    sk.add(_a(text=f"The uploader splits each tenant's export into {v['initial']} KB parts. With {v['n_tenants']} "
                   f"tenants that's thousands of PUTs inside the export window, and the jittered retries spread them "
                   f"out without reducing them, which is exactly what S3 throttles."), plants=("part_initial",))
    sk.add(_u("So fewer, bigger parts?"))
    sk.add(_a(thinking=f"Benchmark {v['tried']} KB parts against staging.",
              calls=[_bash(f"./scripts/upload-bench.sh --part-kb {v['tried']} --tenants {v['n_tenants']}",
                           f"part_kb   puts   throttled  timeouts  p95_part_s\n{v['tried']:<9} {w.i(300, 600):<6} "
                           f"0          {w.i(30, 60):<9} {w.f(14, 25, 1)}")]), at=("part_tried",))
    sk.add(_a(text=f"{v['tried']} KB is worse: no throttling, but parts that big run into the upload timeout on the "
                   f"slower tenants. Going back down to {v['final']}."), plants=("part_tried",), at=("part_final",))
    sk.add(_a(thinking=f"Benchmark {v['final']} KB.",
              calls=[_bash(f"./scripts/upload-bench.sh --part-kb {v['final']} --tenants {v['n_tenants']}",
                           f"part_kb   puts   throttled  timeouts  p95_part_s\n{v['final']:<9} {w.i(900, 1400):<6} "
                           f"0          0         {w.f(3, 6, 1)}")]), at=("part_final",))
    sk.add(_a(text=f"{v['final']} KB: no throttling and no timeouts. Setting that in the uploader config.",
              calls=[_edit("config/uploader.yaml", [(f"  part_size_kb: {v['initial']}", f"  part_size_kb: {v['final']}")])]),
           plants=("part_final",), at=("part_initial",))
    sk.add(_u(f"Careful with dependencies while you're in there: keep boto3 at {v['pin']}. 1.41 changed the default "
              f"checksum algorithm and our signed uploads break with it."), plants=("boto_pin",))
    sk.add(_a(text=f"Understood, I won't touch the AWS SDK pins. I'll also put the jittered retry path behind a flag, "
                   f"`{v['flag']}`, so finance can switch it off without a deploy if the night goes badly."),
           plants=("retry_flag",))
    sk.gap(0.4)
    sk.add(_u(f"btw if you need to reproduce anything, the staging box is {v['host']} now; {v['prev']} was "
              f"decommissioned last week."), plants=("staging_host",), at=("decoy:prev_host",))
    readme = _readme(w, "ledgerd export", [
        f"The nightly export writes one CSV and one manifest per tenant to s3://ledger-exports/{v['readme_prefix']}"
        f"<tenant>/<date>/. Finance picks the files up at 06:00.",
        "Uploads are multipart; part size and retries are configured in config/uploader.yaml."],
        ["ledgerd export --nightly --all-tenants", "ledgerd export --tenant acme-retail --date 2026-09-01"])
    sk.add(_a(thinking=f"The README says the exports land under s3://ledger-exports/{v['readme_prefix']}, but the "
                       f"batch hosts' deploy manifest I checked earlier writes to {v['hidden']} in prod. The README is "
                       f"stale; don't trust it for the prefix.",
              calls=[_read("README.md", readme)]), plants=("s3_prefix",), at=("decoy:readme_prefix",))
    sk.add(_a(text="Before changing anything else I'll run the export tests."))
    names = [t for t in EXPORT_TESTS] + w.test_names(8)
    w.rng.shuffle(names)
    sk.add(_a(calls=[_bash("pytest tests/export -v", w.pytest_run("tests/export/test_export.py", names,
                                                                    [v["flaky"], v["real"]]))]),
           plants=("flaky_test",), at=("decoy:real_failure",))
    sk.add(_a(text="Two failures. One of them looks timing-dependent; rerunning before I dig in."))
    sk.add(_a(calls=[_bash("pytest tests/export -v", w.pytest_run("tests/export/test_export.py", names, [v["real"]]))]),
           at=("flaky_test", "decoy:real_failure"))
    sk.add(_a(text="One of the two passes on the rerun, so that one is flaky; the other fails both times and is a "
                   "real bug in the period close, unrelated to the uploads. Checking the upload settings next."))
    sk.add(_a(calls=[_read("deploy/ledgerd.env", w.env_file([f"# per-part upload timeout\nLEDGERD_UPLOAD_TIMEOUT_MS={v['env']}"]))]),
           plants=("upload_timeout",))
    sk.add(_a(text="The per-part timeout is generous enough for the new part size, so I'm leaving the env alone."))
    sk.add(_u(f"Thanks. Once this is fixed, tell {v['contact']} so the finance dashboards get re-checked. Not "
              f"{v['on_leave']}, {v['on_leave']} is out until the 20th."), plants=("dash_contact",), at=("decoy:on_leave",))
    sk.add(_a(text="Will do."))
    sk.gap(1.6)


def _export_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"Another one, {v['ticket2']}: the statement export failed on September {v['d3']} and {v['d4']} too. "
                  f"Can you check whether it's the same thing?"), at=("rival:ticket2",))
        for d, code, k in ((v["d3"], v["code3"], 17), (v["d4"], v["code4"], 30)):
            special = {k: (f"{w.log_time(k, 48)} ERROR   ledgerd.statements: render of statement batch failed for "
                           f"{w.tenant()}: TemplateError: undefined variable 'vat_summary' (error_ref={code}, "
                           f"request={v['req3']})")}
            for j in w.sample([x for x in range(4, 44) if x != k], 3):
                special[j] = (f"{w.log_time(j, 48)} ERROR   ledgerd.statements: render of statement batch failed for "
                              f"{w.tenant()}: TemplateError: undefined variable 'vat_summary' (error_ref={code}, "
                              f"request=req-{w.hexs(6)})")
            sk.add(_a(calls=[_bash(f"ssh ledger-batch-1 tail -n 48 /var/log/ledgerd/statements-2026-09-{d:02d}.log",
                                   w.app_log(48, special))]), at=(f"rival:{'code3' if code == v['code3'] else 'code4'}",
                                                                   "rival:req3"))
        sk.add(_a(text=f"Different problem: the statement export fails with {v['code3']} on the {v['d3']}th and "
                       f"{v['code4']} on the {v['d4']}th, and {v['req3']} failed on both nights. It's a template "
                       f"error, not S3 throttling, so I'll track it under {v['ticket2']} separately."),
               at=("rival:ticket2", "rival:code3", "rival:code4", "rival:req3"))
    elif r == 2:
        sk.add(_u("While you're at it, the archive uploader has the same multipart setup. Can you tune that too?"))
        sk.add(_a(text=f"The archive uploader uses {v['a']} KB parts. Trying {v['b']}.",
                  calls=[_bash(f"./scripts/upload-bench.sh --job archive --part-kb {v['b']}",
                               f"part_kb   puts   throttled  timeouts  p95_part_s\n{v['b']:<9} {w.i(200, 500):<6} 0"
                               f"          {w.i(10, 40):<9} {w.f(12, 22, 1)}")]), at=("rival:parts2",))
        sk.add(_a(text=f"{v['b']} KB times out on the archive job as well. Settling on {v['c']} KB, which runs clean.",
                  calls=[_edit("config/archive.yaml", [(f"  archive_part_kb: {v['a']}", f"  archive_part_kb: {v['c']}")])]),
               at=("rival:parts2",))
        sk.add(_u(f"Good. Also keep urllib3 at {v['pin2']}; 2.6 breaks our egress proxy."), at=("rival:pin2",))
        sk.add(_a(thinking="Check when the archive job runs so the change can be tested tonight.",
                  calls=[_bash("ssh ledger-batch-1 cat /etc/cron.d/ledger-archive",
                               f"# managed by ansible\n{v['cron2']} ledgerd /opt/ledgerd/bin/ledgerd archive --uploads-only")]),
               at=("rival:cron2",))
        sk.add(_a(text=f"urllib3 stays at {v['pin2']}. The archive job runs at `{v['cron2']}`, so we'll see the new "
                       f"part size in its next run."), at=("rival:pin2", "rival:cron2"))
    else:
        sk.add(_u(f"Two more things: the statements service has its own staging box, {v['host2']}. And for the statements "
                  f"dashboard talk to {v['contact2']}, not {v['leave2']}; {v['leave2']} is at a conference."),
               at=("rival:host2", "rival:contact2", "rival:leave2"))
        sk.add(_a(thinking=f"The statements README says s3://ledger-exports/statements/, but the statement exporter "
                           f"config I saw writes to {v['hidden2']}.",
                  text=f"Noted. I'll put the statement fix behind `{v['flag2']}`, like the uploader change."),
               at=("rival:hidden2", "rival:flag2"))


_register("export", "ticket-first", _export_values, _export_plant, _export_episodes)


# =============================================================================
# Story 2: ci (atlas monorepo CI; TypeScript). Skeleton "flaky-first": the
# branch and PR open the session, the flaky test is found first by comparing two
# runs, then two timed-out builds, the pipeline config and a Node version cap;
# a memory tuning chain and a superseded cache TTL follow, and the people,
# runner host, coverage gate and dashboard panel are planted last, close to the
# end of the plant phase (the work stretch comes early).
# =============================================================================

CI_TESTS = ["checkout flow keeps the cart after login", "search suggests recent queries",
            "admin can export the audit log", "settings page saves the locale", "invite link expires after a week",
            "billing page shows the next invoice"]
CI_PEOPLE = ["Noor", "Pavel", "Leocadia", "Tunde", "Signe", "Rafael", "Aiko", "Bram"]
CI_AUTHORS = ["jdelacroix", "mokonkwo", "tvasquez", "elindgren", "rbianchi", "skowalski"]
CI_PANELS = ["Shard wall time p95", "Cold install rate", "Cache restore time", "Queue wait by runner image",
             "Retries per shard"]


def _ci_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["branch"] = f"perf/{rng.choice(['shard', 'cache', 'lockfile'])}-{rng.choice(['rebalance', 'warmup', 'split'])}-{rng.randint(2, 9)}"
    v["pr"] = _distinct(rng, lambda r: str(r.randint(14000, 19999)), used)
    v["flaky"], v["real"] = rng.sample(CI_TESTS, 2)
    v["b1"] = rng.randint(4100, 4899)
    v["b2"] = v["b1"] + 1
    v["key1"] = _distinct(rng, lambda r: f"v3-pnpm-store-{_hexr(r, 8)}", used)
    v["key2"] = _distinct(rng, lambda r: f"v3-pnpm-store-{_hexr(r, 8)}", used)
    v["agent"] = _distinct(rng, lambda r: f"agent-{_hexr(r, 6)}", used)
    v["timeout"] = rng.choice([27, 33, 37, 41])
    maj, mi = rng.randint(20, 22), rng.randint(4, 14)
    v["node"], v["node_next"] = f"{maj}.{mi}.{rng.randint(1, 9)}", f"{maj}.{mi + 1}"
    v["author"] = rng.choice(CI_AUTHORS)
    v["initial"] = rng.choice([6144, 5632])
    v["tried"] = rng.choice([1536, 2048])
    v["final"] = rng.choice([3584, 3072, 4096])
    v["ttl_old"], v["ttl_new"] = rng.choice([168, 336]), rng.choice([72, 96, 120])
    v["region"] = rng.choice(["eu", "us"])
    v["hidden"] = f"gs://atlas-ci-cache-{v['region']}-{rng.randint(2, 9)}/"
    v["pct"] = f"{rng.choice([78, 81, 83, 86])}.{rng.choice([5, 25, 75])}"
    v["panel"] = f"{rng.choice(CI_PANELS)} by {rng.choice(['queue', 'runner image', 'package'])}"
    n = rng.randint(3, 9)
    v["host"], v["prev"] = f"mac-ci-{n:02d}", f"mac-ci-{n - 1:02d}"
    v["contact"], v["on_leave"] = rng.sample(CI_PEOPLE, 2)
    v["cores"] = rng.choice([8, 16])
    # rivals
    v["pr2"] = _distinct(rng, lambda r: str(r.randint(14000, 19999)), used)
    v["b3"] = v["b2"] + rng.randint(5, 9)
    v["b4"] = v["b3"] + 1
    v["key3"] = _distinct(rng, lambda r: f"v3-pnpm-store-{_hexr(r, 8)}", used)
    v["key4"] = _distinct(rng, lambda r: f"v3-pnpm-store-{_hexr(r, 8)}", used)
    v["agent3"] = _distinct(rng, lambda r: f"agent-{_hexr(r, 6)}", used)
    v["a"], v["b"], v["c"] = rng.choice([4608, 5120]), rng.choice([1024, 1280]), rng.choice([2560, 2816])
    v["pin2"] = f"9.{rng.randint(2, 14)}.{rng.randint(1, 9)}"
    v["pct2"] = f"{rng.choice([70, 72, 74])}.{rng.choice([5, 25])}"
    v["host2"] = f"linux-ci-{rng.randint(3, 9):02d}"
    v["contact2"], v["leave2"] = rng.sample([p for p in CI_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["hidden2"] = f"gs://atlas-ci-reports-{rng.choice(['eu', 'us'])}-{rng.randint(2, 9)}/"
    v["panel2"] = f"{rng.choice([p for p in CI_PANELS if not v['panel'].startswith(p)])} by {rng.choice(['branch', 'os'])}"

    sk.fact("branch", "branch", "user", v["branch"], "Which branch is the CI work on?")
    sk.fact("pr", "pr_number", "user", v["pr"], "What is the number of the pull request we are fixing CI for?",
            alts=[f"#{v['pr']}"])
    sk.fact("flaky_test", "flaky_test", "tool_output", v["flaky"], "Which e2e test is the flaky one?")
    sk.fact("cache_key", "error_code", "tool_output", v["key1"], f"Which cache key did build {v['b1']} miss on?")
    sk.fact("agent_both", "shared_id", "tool_output", v["agent"],
            f"Which CI agent ran a timed-out shard in both build {v['b1']} and build {v['b2']}?")
    sk.fact("step_timeout", "config_value", "tool_output", v["timeout"],
            "What is timeout_in_minutes for the e2e step in .ci/pipeline.yml?",
            alts=[f"{v['timeout']} minutes", f"{v['timeout']}m"], slot_keys=["timeout_in_minutes"])
    sk.fact("node_cap", "version_cap", "user", v["node"], "What is the newest Node version we may use?",
            alts=[f"v{v['node']}"])
    sk.fact("shard_author", "author", "thinking", v["author"], "Who wrote the test sharding change?")
    sk.fact("heap_initial", "count_initial", "assistant_text", v["initial"],
            "What --max-old-space-size were the e2e shards running with when we found the swapping?",
            slot_keys=["--max-old-space-size"])
    sk.fact("heap_tried", "count_tried", "assistant_text", v["tried"],
            "Which --max-old-space-size did we try first that made things worse?")
    sk.fact("heap_final", "count_final", "superseded", v["final"], "Which --max-old-space-size did we settle on?",
            old=v["tried"])
    sk.fact("cache_ttl", "superseded_setting", "superseded", v["ttl_new"],
            "How many hours should the pnpm store cache live?", old=v["ttl_old"],
            alts=[f"{v['ttl_new']}h", f"{v['ttl_new']} hours"], slot_keys=["ttl_hours"])
    sk.fact("cache_bucket", "hidden_location", "thinking", v["hidden"],
            "Which bucket do the CI cache artifacts actually upload to?", alts=[v["hidden"].rstrip("/")])
    sk.fact("coverage_gate", "percent_threshold", "user", f"{v['pct']}%",
            "Below which line-coverage percentage should the build fail?", alts=[v["pct"]],
            slot_keys=["min_line_coverage"])
    sk.fact("dash_panel", "dashboard_panel", "assistant_text", v["panel"],
            "What is the name of the panel added to the CI health dashboard?")
    sk.fact("mac_runner", "host", "user", v["host"], "Which macOS runner host are we using now?")
    sk.fact("budget_contact", "contact", "user", v["contact"], "Who should we ask about the runner budget?")
    sk.lure("docker_mirror", "lure", "Which Docker registry mirror did we switch the runners to?")
    sk.lure("slack_channel", "lure", "Which Slack channel do the build alerts go to?")
    sk.lure("turbo_remote", "lure", "Which remote cache provider did we enable for Turborepo?")
    sk.decoy("real_failure", v["real"])
    sk.decoy("key2", v["key2"])
    sk.decoy("prev_runner", v["prev"])
    sk.decoy("on_leave", v["on_leave"])
    sk.decoy("node_next", v["node_next"])
    for name in ("pr2", "key3", "key4", "agent3", "pin2", "host2", "contact2", "leave2", "hidden2", "panel2"):
        sk.rival(name, v[name])
    sk.rival("pct2", f"{v['pct2']}%", v["pct2"])
    sk.rival("heap2", v["a"], v["b"], v["c"])
    return v


def _ci_build_log(w: Work, build: int, key: str, agent: str, agent_shard: int, n_shards: int = 6) -> str:
    out = [f"~~~ Build {build} on {'perf'} pipeline (atlas)", f"Started {w.hm()}:{w.i(10, 59)} UTC"]
    for s in range(1, n_shards + 1):
        ag = agent if s == agent_shard else f"agent-{w.hexs(6)}"
        out += [f"--- :pnpm: Shard {s}/{n_shards} on {ag}",
                f"Restoring cache {key} ... cache miss" if s % 2 == 1 else
                f"Restoring cache {key} ... cache miss (fallback v3-pnpm-store- also missed)",
                f"pnpm install --frozen-lockfile",
                f"Packages: +{w.i(1800, 2600)}",
                f"Progress: resolved {w.i(1800, 2600)}, reused 0, downloaded {w.i(1800, 2600)}, added {w.i(1800, 2600)}, done",
                f"Done in {w.i(4, 9)}m {w.i(10, 59)}s",
                f"--- :vitest: e2e shard {s}/{n_shards}"]
        if s == agent_shard or w.rng.random() < 0.4:
            out += [f"Running {w.i(30, 80)} tests ...",
                    f"Timeout: step exceeded its time limit, cancelling (agent {ag})",
                    f"^^^ +++ Shard {s} timed out"]
        else:
            out += [f"Test Files  {w.i(10, 30)} passed", f"Tests  {w.i(60, 200)} passed",
                    f"Duration  {w.i(9, 16)}m {w.i(10, 59)}s"]
    out.append(f"Build {build} finished: failed ({w.i(2, 4)} of {n_shards} shards timed out)")
    return "\n".join(out)


def _ci_pipeline(w: Work, timeout: int, node_opts: str) -> str:
    return "\n".join([
        "env:", "  PNPM_HOME: /opt/pnpm", f"  NODE_OPTIONS: \"{node_opts}\"", "",
        "steps:",
        "  - label: \":eslint: lint\"", "    command: pnpm -r lint", f"    parallelism: {w.ch([2, 4])}",
        "    agents:", "      queue: linux", "",
        "  - label: \":typescript: typecheck\"", "    command: pnpm -r typecheck", "    agents:", "      queue: linux", "",
        "  - label: \":vitest: e2e\"", "    command: pnpm --filter checkout-e2e test --shard=$${BUILDKITE_PARALLEL_JOB}",
        "    parallelism: 6", f"    timeout_in_minutes: {timeout}", "    retry:", "      automatic:",
        "        - exit_status: -1", "          limit: 2", "    plugins:", "      - cache#v1.3.0:",
        "          key: \"v3-pnpm-store-{{ checksum 'pnpm-lock.yaml' }}\"", "          path: .pnpm-store",
        "    agents:", "      queue: linux", "",
        "  - wait", "",
        "  - label: \":rocket: preview deploy\"", "    command: .ci/deploy-preview.sh", "    branches: \"!main\""])


def _ci_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    sk.add(_u(f"CI on {v['branch']} (PR #{v['pr']}) keeps going red. Start with the e2e flake, it fails roughly every "
              f"third run; after that we need to talk about the timeouts."), plants=("branch", "pr"))
    names = CI_TESTS + w.test_names(10)
    w.rng.shuffle(names)
    suite = "packages/checkout-e2e/test/flows.e2e.ts"
    sk.add(_a(thinking="Run the e2e suite twice and compare.",
              calls=[_bash("pnpm --filter checkout-e2e test --reporter verbose", w.vitest_run(suite, names, [v["flaky"], v["real"]]))]),
           plants=("flaky_test",), at=("decoy:real_failure",))
    sk.add(_a(calls=[_bash("pnpm --filter checkout-e2e test --reporter verbose", w.vitest_run(suite, names, [v["real"]]))]),
           at=("flaky_test", "decoy:real_failure"))
    sk.add(_a(text="First run: two failures. Second run: only one of them fails again, so the other is the flake. "
                   "The one that fails twice is a real regression in the login redirect; I'll note it on the PR."))
    sk.gap(1.2)
    sk.add(_u("OK. Now the timeouts: builds since Friday time out on most shards."))
    sk.add(_a(thinking=f"Read build {v['b1']}, the first one that timed out.",
              calls=[_read(f"/var/log/ci/build-{v['b1']}.log", _ci_build_log(w, v["b1"], v["key1"], v["agent"], 3))]),
           plants=("cache_key",), at=("agent_both",))
    sk.add(_a(text=f"Build {v['b1']}: every shard missed the pnpm cache and did a cold install of several minutes "
                   f"before running tests. Comparing with {v['b2']}."))
    sk.add(_a(thinking=f"Read build {v['b2']}.",
              calls=[_read(f"/var/log/ci/build-{v['b2']}.log", _ci_build_log(w, v["b2"], v["key2"], v["agent"], 5))]),
           plants=("agent_both",), at=("decoy:key2",))
    sk.add(_a(text=f"{v['b2']} misses a different key, but the same agent ran a timed-out shard in both builds. Could "
                   f"be a slow machine; noting it."))
    sk.add(_a(thinking="How long may the e2e step run?",
              calls=[_read(".ci/pipeline.yml", _ci_pipeline(w, v["timeout"], f"--max-old-space-size={v['initial']}"))]),
           plants=("step_timeout",), at=("heap_initial",))
    sk.add(_a(text="The e2e step's time limit is tight: a cold install plus the tests doesn't fit when the cache misses."))
    sk.add(_u(f"Before you change anything: Node {v['node']} is the newest we can use. {v['node_next']} breaks the "
              f"native addon in packages/imaging."), plants=("node_cap",), at=("decoy:node_next",))
    sk.add(_a(thinking=f"{v['author']} wrote the sharding change, I remember the PR; worth asking them about the "
                       f"shard sizes. Check the history of .ci/ first.",
              calls=[_bash("git log --oneline -12 -- .ci/", w.oneline(12, {w.i(1, 5): f"{w.hexs(7)} enable test sharding across 6 shards"}))]),
           plants=("shard_author",))
    sk.add(_a(text=f"There's a second problem besides the cache: each shard runs Node with --max-old-space-size="
                   f"{v['initial']}, and two shards share a {v['cores']}-core 8 GB runner, so they swap. I'll try a lower limit."),
           plants=("heap_initial",))
    sk.add(_a(thinking=f"Try {v['tried']}.",
              calls=[_bash(f"ci-bench e2e --node-heap {v['tried']}",
                           f"heap_mb  shards  oom_kills  wall_p95\n{v['tried']:<8} 6       {w.i(2, 5)}          {w.i(19, 29)}m{w.i(10, 59)}s")]),
           at=("heap_tried",))
    sk.add(_a(text=f"{v['tried']} MB is worse: shards get OOM-killed. Trying {v['final']}."),
           plants=("heap_tried",), at=("heap_final",))
    sk.add(_a(calls=[_bash(f"ci-bench e2e --node-heap {v['final']}",
                           f"heap_mb  shards  oom_kills  wall_p95\n{v['final']:<8} 6       0          {w.i(11, 15)}m{w.i(10, 59)}s")]),
           at=("heap_final",))
    sk.add(_a(text=f"{v['final']} MB: no OOM kills and no swapping. Updating NODE_OPTIONS.",
              calls=[_edit(".ci/pipeline.yml", [(f'  NODE_OPTIONS: "--max-old-space-size={v["initial"]}"',
                                                 f'  NODE_OPTIONS: "--max-old-space-size={v["final"]}"')])]),
           plants=("heap_final",), at=("heap_initial",))
    sk.add(_u("And the cache itself? How long do we keep the pnpm store?"))
    sk.add(_a(text=f"I'd keep it for {v['ttl_old']} hours so a quiet weekend doesn't evict it."), at=("cache_ttl",))
    sk.add(_u(f"Make it {v['ttl_new']} hours. Longer just keeps stale stores around and the bucket bill is already high."),
           plants=("cache_ttl",))
    sk.add(_a(thinking=f"The docs say cache artifacts go to gs://atlas-ci-artifacts/, but the uploader config I saw "
                       f"earlier actually writes to {v['hidden']}. The docs are stale.",
              calls=[_edit(".ci/cache.yml", [(f"  ttl_hours: {v['ttl_old']}", f"  ttl_hours: {v['ttl_new']}")])]),
           plants=("cache_bucket",), at=("cache_ttl",))
    sk.add(_a(text=f"Cache TTL set to {v['ttl_new']} hours."), at=("cache_ttl",))
    sk.gap(0.3)
    sk.add(_u(f"Also the coverage gate: fail the build if line coverage drops below {v['pct']}%."),
           plants=("coverage_gate",))
    sk.add(_a(text=f"Will add that. I've also added a \"{v['panel']}\" panel to the CI health dashboard so we can see "
                   f"whether the cache fix holds."), plants=("dash_panel",))
    sk.add(_u(f"Nice. btw the macOS runner is {v['host']} now, {v['prev']} was retired. And ask {v['contact']} about "
              f"the runner budget, not {v['on_leave']}; {v['on_leave']} moved teams."),
           plants=("mac_runner", "budget_contact"), at=("decoy:prev_runner", "decoy:on_leave"))
    sk.add(_a(text="Will do."))
    sk.gap(0.2)


def _ci_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"PR #{v['pr2']} has timeouts too, builds {v['b3']} and {v['b4']}. Same cause?"), at=("rival:pr2",))
        sk.add(_a(calls=[_bash(f"buildkite-agent artifact download build-{v['b3']}.log - | tail -n 60",
                               _ci_build_log(w, v["b3"], v["key3"], v["agent3"], 2))]), at=("rival:key3", "rival:agent3"))
        sk.add(_a(calls=[_bash(f"buildkite-agent artifact download build-{v['b4']}.log - | tail -n 60",
                               _ci_build_log(w, v["b4"], v["key4"], v["agent3"], 4))]), at=("rival:key4", "rival:agent3"))
        sk.add(_a(text=f"Build {v['b3']} missed {v['key3']} and {v['b4']} missed {v['key4']}; {v['agent3']} timed out "
                       f"in both. That PR changes the lockfile on every push, so its misses are expected; separate issue."),
               at=("rival:key3", "rival:key4", "rival:agent3"))
    elif r == 2:
        sk.add(_u("While we're at it, the lint job swaps too. Can you tune its heap?"))
        sk.add(_a(text=f"The lint job runs with {v['a']} MB. Trying {v['b']}.",
                  calls=[_bash(f"ci-bench lint --node-heap {v['b']}",
                               f"heap_mb  oom_kills  wall_p95\n{v['b']:<8} {w.i(2, 5)}          {w.i(9, 14)}m{w.i(10, 59)}s")]),
               at=("rival:heap2",))
        sk.add(_a(text=f"{v['b']} MB gets OOM-killed. {v['c']} MB runs clean, using that for lint."), at=("rival:heap2",))
        sk.add(_u(f"Good. Keep pnpm at {v['pin2']} by the way, the next minor changes the workspace protocol. And for the "
                  f"docs package the coverage floor is {v['pct2']}%."), at=("rival:pin2", "rival:pct2"))
        sk.add(_a(text=f"Pinned pnpm to {v['pin2']} and set the docs coverage floor to {v['pct2']}%."),
               at=("rival:pin2", "rival:pct2"))
    else:
        sk.add(_u(f"Two more things: the Linux canary runner is {v['host2']} now. And for the e2e flake budget talk to "
                  f"{v['contact2']}, not {v['leave2']}; {v['leave2']} is on parental leave."),
               at=("rival:host2", "rival:contact2", "rival:leave2"))
        sk.add(_a(thinking=f"The docs say test reports go to gs://atlas-ci-reports/, but the report job I saw writes to "
                           f"{v['hidden2']}.",
                  text=f"Noted. I'll add a \"{v['panel2']}\" panel for the flake budget as well."),
               at=("rival:hidden2", "rival:panel2"))


_register("ci", "flaky-first", _ci_values, _ci_plant, _ci_episodes)


# =============================================================================
# Story 3: migration (orders database migration; Python + SQL). Skeleton
# "tool-first": the failing revision and the migration config come from tool
# output before anything else; the statement timeout is superseded early; the
# blocked relation and blocking pid come from two attempt logs; the batch-size
# chain sits in the middle; a thinking-only cron host shares a step with the
# cron schedule; credentials, people and the window are planted at the end.
# =============================================================================

MIG_PEOPLE = ["Ilse", "Marek", "Oona", "Desmond", "Farah", "Kenji", "Lucia", "Viggo"]
MIG_AUTHORS = ["pkaminski", "adoyle", "fnakamura", "lhernandez", "gsorensen", "bwhitaker"]
MIG_SLUGS = ["add_refund_partial_index", "split_customer_address", "drop_legacy_coupons", "orders_currency_default",
             "rename_shipments_eta", "audit_events_by_month", "line_items_warehouse_id", "dedupe_customers_email",
             "tax_backfill_batches", "amount_check_constraint", "shipments_vacuum_settings", "search_sync_trigger"]


def _mig_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["mig_n"] = rng.randint(140, 189)
    v["mig"] = f"{v['mig_n']:04d}_orders_region_not_null"
    v["lock"] = rng.choice([1500, 2500, 3000, 4000])
    v["stmt_old"], v["stmt_new"] = rng.choice([600, 900]), rng.choice([150, 180, 240])
    v["n1"] = rng.randint(31, 58)
    v["n2"] = v["n1"] + 1
    months = rng.sample(range(1, 13), 4)
    v["rel1"] = f"orders_{rng.choice([2025, 2026])}_{months[0]:02d}"
    v["rel2"] = f"orders_{rng.choice([2025, 2026])}_{months[1]:02d}"
    v["pid"] = _distinct(rng, lambda r: str(r.randint(41000, 49999)), used)
    n = rng.randint(3, 9)
    v["host"], v["prev"] = f"pg-orders-r{n}", f"pg-orders-r{n - 1}"
    v["hash"] = _hexr(rng, 7)
    v["author"] = rng.choice(MIG_AUTHORS)
    v["initial"] = rng.choice([40000, 50000, 60000])
    v["tried"] = rng.choice([150000, 200000, 250000])
    v["final"] = rng.choice([4000, 5000, 7500])
    mi = rng.randint(1, 3)
    v["ver"], v["ver_next"] = f"3.{mi}.{rng.randint(2, 9)}", f"3.{mi + 1}"
    v["runner_n"] = rng.randint(2, 7)
    v["hidden"] = f"ops-runner-{v['runner_n']}"
    v["cron"] = f"{rng.choice([5, 25, 35, 55])} */{rng.choice([3, 4, 6])} * * *"
    v["path"] = f"/etc/orders/pg-{rng.choice(['ro', 'readonly', 'reporting'])}.env"
    v["contact"], v["on_leave"] = rng.sample(MIG_PEOPLE, 2)
    v["day"] = rng.choice(DAYS)
    v["time"] = f"{rng.randint(19, 23):02d}:{rng.choice(['05', '20', '35', '50'])}"
    # rivals
    v["mig2"] = f"{v['mig_n'] + rng.randint(1, 3):04d}_refunds_partial_index"
    v["n3"] = v["n2"] + rng.randint(4, 8)
    v["n4"] = v["n3"] + 1
    v["rel3"] = f"refunds_{rng.choice([2025, 2026])}_{months[2]:02d}"
    v["rel4"] = f"refunds_{rng.choice([2025, 2026])}_{months[3]:02d}"
    v["pid3"] = _distinct(rng, lambda r: str(r.randint(41000, 49999)), used)
    v["a"], v["b"], v["c"] = rng.choice([30000, 45000]), rng.choice([120000, 180000]), rng.choice([2250, 2750])
    v["pin2"] = f"1.{rng.randint(11, 15)}.{rng.randint(1, 9)}"
    v["host2"] = f"pg-reporting-r{rng.randint(2, 9)}"
    v["contact2"], v["leave2"] = rng.sample([p for p in MIG_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["time2"] = f"{rng.randint(6, 11):02d}:{rng.choice(['10', '40'])}"
    v["hidden2"] = f"ops-runner-{rng.choice([x for x in range(2, 10) if x != v['runner_n']])}"
    v["cron2"] = f"{rng.choice([10, 40])} {rng.choice([1, 2])} * * {rng.choice(['0', '6'])}"

    sk.fact("failing_migration", "migration_id", "tool_output", v["mig"], "Which migration revision is the one that keeps failing?")
    sk.fact("lock_timeout", "config_value", "tool_output", v["lock"],
            "What lock_timeout, in milliseconds, is set in db/migrate.yml?", alts=[f"{v['lock']}ms", f"{v['lock']} ms"],
            slot_keys=["lock_timeout_ms", "lock_timeout"])
    sk.fact("statement_timeout", "superseded_setting", "superseded", v["stmt_new"],
            "What statement_timeout, in seconds, did we end up using for the migration job?", old=v["stmt_old"],
            alts=[f"{v['stmt_new']}s", f"{v['stmt_new']} s"], slot_keys=["statement_timeout_s", "statement_timeout"])
    sk.fact("blocked_relation", "error_code", "tool_output", v["rel1"],
            f"Which relation was migration attempt {v['n1']} waiting on?")
    sk.fact("blocking_pid", "shared_id", "tool_output", v["pid"],
            f"Which backend pid was blocking both attempt {v['n1']} and attempt {v['n2']}?")
    sk.fact("replica", "host", "user", v["host"], "Which replica are we testing the migration against?")
    sk.fact("constraint_hash", "commit_hash", "tool_output", v["hash"],
            "What is the short hash of the commit that added the NOT NULL constraint on orders.region?")
    sk.fact("constraint_author", "author", "thinking", v["author"], "Who wrote the NOT NULL migration?")
    sk.fact("batch_initial", "count_initial", "assistant_text", v["initial"],
            "How many rows per statement was the backfill updating when we found the long locks?",
            slot_keys=["batch_size", "BACKFILL_BATCH"])
    sk.fact("batch_tried", "count_tried", "assistant_text", v["tried"],
            "Which backfill batch size did we try first that made things worse?")
    sk.fact("batch_final", "count_final", "superseded", v["final"], "What backfill batch size did we settle on?",
            old=v["tried"])
    sk.fact("psycopg_cap", "version_cap", "user", v["ver"], "What is the newest psycopg version we may use?")
    sk.fact("cron_host", "hidden_location", "thinking", v["hidden"], "Which host actually runs the backfill cron?")
    sk.fact("cron_schedule", "cron", "tool_output", v["cron"], "What schedule does the backfill cron entry use?")
    sk.fact("check_credentials", "credential_path", "user", v["path"],
            "Which credentials file should the migration checks use?")
    sk.fact("window_contact", "contact", "user", v["contact"], "Who should we talk to about the maintenance window?")
    sk.fact("window_time", "time", "user", v["time"], f"At what time on {v['day']} does the maintenance window start?")
    sk.lure("vacuum_factor", "lure", "What autovacuum scale factor did we set on the orders table?")
    sk.lure("pgbouncer_mode", "lure", "Which pgbouncer pool mode did we switch to?")
    sk.lure("archive_table", "lure", "What did we name the archive table for old orders?")
    sk.decoy("rel2", v["rel2"])
    sk.decoy("prev_replica", v["prev"])
    sk.decoy("on_leave", v["on_leave"])
    sk.decoy("ver_next", v["ver_next"])
    sk.decoy("runbook_host", "ops-bastion-1")
    for name in ("mig2", "rel3", "rel4", "pid3", "pin2", "host2", "contact2", "leave2", "time2", "hidden2", "cron2"):
        sk.rival(name, v[name])
    sk.rival("batch2", v["a"], v["b"], v["c"])
    return v


def _alembic_history(w: Work, head: str, head_n: int, special_msg: str) -> str:
    out = []
    slugs = w.sample(MIG_SLUGS, 10)
    nums = list(range(head_n - 10, head_n))
    out.append(f"{nums[-1]:04d}_{slugs[-1]} -> {head} (head), {special_msg}")
    for k in range(len(nums) - 1, 0, -1):
        out.append(f"{nums[k - 1]:04d}_{slugs[k - 1]} -> {nums[k]:04d}_{slugs[k]}, {slugs[k].replace('_', ' ')}")
    return "\n".join(out)


def _pg_attempt_log(w: Work, n: int, rel: str, blocker: str, at: list[int]) -> str:
    out, t = [], w.i(1, 2) * 3600 + w.i(0, 3000)
    me = str(w.i(50000, 59999))
    for k in range(n):
        t += w.i(1, 90)
        ts = f"{w.clock(t, ms=True)} UTC"
        pre = f"{ts} [{me}] migrate@orders"
        if k in at:
            lo, hi = w.i(100000, 900000), w.i(900001, 1900000)
            out += [f"{pre} LOG:  statement: UPDATE orders SET region = COALESCE(region, 'unknown') WHERE id BETWEEN {lo} AND {hi}",
                    f"{pre} ERROR:  canceling statement due to lock timeout",
                    f"{pre} DETAIL:  Process {me} waits for RowExclusiveLock on relation {rel} of database orders; "
                    f"blocked by process {blocker}.",
                    f"{pre} CONTEXT:  while updating tuple ({w.i(1000, 99999)},{w.i(1, 40)}) in relation \"{rel}\""]
        else:
            out.append(w.ch([
                f"{ts} [{w.i(10000, 39999)}] LOG:  checkpoint starting: time",
                f"{ts} [{w.i(10000, 39999)}] LOG:  checkpoint complete: wrote {w.i(100, 9000)} buffers ({w.f(0.1, 9, 1)}%); "
                f"write={w.f(1, 200, 3)} s, sync={w.f(0, 2, 3)} s",
                f"{ts} [{w.i(10000, 39999)}] LOG:  automatic vacuum of table \"orders.public.{w.ch(['shipments', 'refunds', 'line_items', 'audit_events'])}\": "
                f"index scans: {w.i(0, 2)}",
                f"{pre} LOG:  migrate: backfill progress {w.i(1, 99)}%",
                f"{ts} [{w.i(10000, 39999)}] LOG:  connection authorized: user=orders_app database=orders application_name={w.ch(['api', 'worker', 'reports'])}",
                f"{pre} LOG:  duration: {w.f(10, 9000, 3)} ms  statement: ANALYZE {w.ch(['shipments', 'refunds', 'customers'])}"]))
    return "\n".join(out)


def _mig_file(w: Work, v: dict) -> str:
    return "\n".join([
        f'"""add NOT NULL constraint to orders.region', "", f"Revision ID: {v['mig']}",
        f"Revises: {v['mig_n'] - 1:04d}_{w.ch(MIG_SLUGS)}", f'Create Date: 2026-09-{w.i(10, 20)} {w.hm()}:{w.i(10, 59)}', '"""',
        "", "from alembic import op", "import sqlalchemy as sa", "", f'revision = "{v["mig"]}"',
        f'down_revision = "{v["mig_n"] - 1:04d}"', "",
        "", "def upgrade() -> None:",
        "    # backfill first, in batches (see orders/backfill/region.py)",
        "    op.execute(\"SELECT orders_backfill_region()\")",
        "    op.execute(\"ALTER TABLE orders ADD CONSTRAINT orders_region_not_null CHECK (region IS NOT NULL) NOT VALID\")",
        "    op.execute(\"ALTER TABLE orders VALIDATE CONSTRAINT orders_region_not_null\")",
        "    op.alter_column(\"orders\", \"region\", existing_type=sa.Text(), nullable=False)",
        "", "", "def downgrade() -> None:",
        "    op.alter_column(\"orders\", \"region\", existing_type=sa.Text(), nullable=True)",
        "    op.execute(\"ALTER TABLE orders DROP CONSTRAINT IF EXISTS orders_region_not_null\")", ""])


def _mig_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    sk.add(_u("The orders migration keeps failing on staging. Can you find out why? Attempt logs are in "
              "/var/log/migrations/."))
    sk.add(_a(thinking="Which revision is the head, and what does it do?",
              calls=[_bash("alembic history | head -10", _alembic_history(w, v["mig"], v["mig_n"],
                                                                          "add NOT NULL constraint to orders.region"))]),
           plants=("failing_migration",))
    cfg = "\n".join(["migration:", f"  lock_timeout_ms: {v['lock']}", f"  statement_timeout_s: {v['stmt_old']}",
                     f"  batch_size: {v['initial']}", f"  sleep_ms: {w.ch([50, 100])}", "  concurrently: true",
                     "  analyze_after: true", f"  max_retries: {w.ch([3, 5])}"])
    sk.add(_a(thinking="Check the job's settings.", calls=[_read("db/migrate.yml", cfg)]),
           plants=("lock_timeout",), at=("statement_timeout", "batch_initial"))
    sk.add(_a(text="The head revision adds the NOT NULL constraint on orders.region, and the job's settings are in "
                   "db/migrate.yml. Before the logs: is the statement timeout deliberate?"))
    sk.add(_u("What statement_timeout is the job using?"))
    sk.add(_a(text=f"{v['stmt_old']} seconds, so slow batches don't get killed."), at=("statement_timeout",))
    sk.add(_u(f"Make it {v['stmt_new']} seconds. Anything slower than that means something is wrong."),
           plants=("statement_timeout",))
    sk.add(_a(calls=[_edit("db/migrate.yml", [(f"  statement_timeout_s: {v['stmt_old']}", f"  statement_timeout_s: {v['stmt_new']}")])],
              text=f"Done, statement_timeout_s is {v['stmt_new']} now. Reading the attempt logs."),
           at=("statement_timeout",))
    sk.gap(0.5)
    sk.add(_a(thinking=f"Read attempt {v['n1']}.",
              calls=[_read(f"/var/log/migrations/attempt-{v['n1']}.log",
                           _pg_attempt_log(w, 40, v["rel1"], v["pid"], [9, 21, 33]))]),
           plants=("blocked_relation",), at=("blocking_pid",))
    sk.add(_a(text=f"Attempt {v['n1']}: every cancelled statement waited on the same partition and hit the lock "
                   f"timeout. Comparing with attempt {v['n2']}."))
    sk.add(_a(thinking=f"Read attempt {v['n2']}.",
              calls=[_read(f"/var/log/migrations/attempt-{v['n2']}.log",
                           _pg_attempt_log(w, 40, v["rel2"], v["pid"], [14, 30]))]),
           plants=("blocking_pid",), at=("decoy:rel2",))
    sk.add(_a(text=f"Attempt {v['n2']} waits on a different partition, but the same backend blocked both attempts. "
                   f"It's probably a long-running report holding locks; noting it."))
    sk.add(_u(f"Note we test against {v['host']} now; {v['prev']} was rebuilt and is out of rotation."),
           plants=("replica",), at=("decoy:prev_replica",))
    sk.add(_a(thinking="When did the constraint change land?",
              calls=[_bash("git log --oneline -12 -- db/migrations/",
                           w.oneline(12, {w.i(0, 4): f"{v['hash']} add NOT NULL constraint to orders.region"}))]),
           plants=("constraint_hash",))
    sk.add(_a(thinking=f"{v['author']} wrote this migration, it was their PR; they should review whatever we change. "
                       f"Let me read it.",
              calls=[_read(f"db/migrations/{v['mig']}.py", _mig_file(w, v))]),
           plants=("constraint_author",), at=("failing_migration",))
    sk.add(_a(text=f"The backfill function updates {v['initial']} rows per statement. On the big partitions each "
                   f"statement holds row locks for seconds, so the validation and the reports pile up behind it."),
           plants=("batch_initial",))
    sk.add(_u("Would a bigger batch at least finish faster?"))
    sk.add(_a(calls=[_bash(f"./scripts/migrate-bench --replica {v['host']} --batch {v['tried']}",
                           f"batch    statements  max_lock_wait_s  aborted\n{v['tried']:<8} {w.i(20, 60):<11} "
                           f"{w.f(30, 60, 1):<16} {w.i(15, 30)}/40")]), at=("batch_tried", "replica"))
    sk.add(_a(text=f"{v['tried']} per statement is much worse: lock waits close to a minute. Going the other way, "
                   f"{v['final']}."), plants=("batch_tried",), at=("batch_final",))
    sk.add(_a(calls=[_bash(f"./scripts/migrate-bench --replica {v['host']} --batch {v['final']}",
                           f"batch    statements  max_lock_wait_s  aborted\n{v['final']:<8} {w.i(900, 2000):<11} "
                           f"{w.f(0.1, 0.9, 1):<16} 0/40")]), at=("batch_final", "replica"))
    sk.add(_a(text=f"{v['final']} rows per statement: no aborts and sub-second lock waits. Setting it.",
              calls=[_edit("db/migrate.yml", [(f"  batch_size: {v['initial']}", f"  batch_size: {v['final']}")])]),
           plants=("batch_final",), at=("batch_initial",))
    sk.add(_u(f"Good. Don't upgrade psycopg past {v['ver']} while you're at it; {v['ver_next']} changes the pool API."),
           plants=("psycopg_cap",), at=("decoy:ver_next",))
    cron = "\n".join(["# orders backfill jobs (installed by ansible role orders-cron)", "SHELL=/bin/bash",
                      "PATH=/usr/local/bin:/usr/bin:/bin", "",
                      f"{v['cron']} orders /opt/orders/bin/backfill --job region --resume >> /var/log/orders/backfill.log 2>&1",
                      f"{w.ch([0, 30])} 3 * * 0 orders /opt/orders/bin/vacuum-report --weekly"])
    sk.add(_a(thinking=f"The runbook says migrations and backfills run from ops-bastion-1, but the cron entry I saw "
                       f"earlier is installed on {v['hidden']}; the runbook is out of date.",
              calls=[_read("deploy/cron/orders-backfill.cron", cron)]),
           plants=("cron_host", "cron_schedule"), at=("decoy:runbook_host",))
    sk.add(_a(text="The backfill cron resumes from the last batch, so a smaller batch size won't restart the work."))
    sk.add(_u(f"For the checks, use the credentials in {v['path']}. Never the superuser."), plants=("check_credentials",))
    sk.add(_a(text="Switched the migration checks to the read-only credentials."))
    sk.gap(0.5)
    sk.add(_u(f"Thanks. Talk to {v['contact']} about the maintenance window, not {v['on_leave']}, {v['on_leave']} is "
              f"out this week. The window starts {v['day']} at {v['time']}."),
           plants=("window_contact", "window_time"), at=("decoy:on_leave",))
    sk.add(_a(text="Will do."))


def _mig_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"The refunds migration, {v['mig2']}, failed too: attempts {v['n3']} and {v['n4']}. Same thing?"),
               at=("rival:mig2",))
        sk.add(_a(calls=[_bash(f"tail -n 40 /var/log/migrations/refunds-attempt-{v['n3']}.log",
                               _pg_attempt_log(w, 30, v["rel3"], v["pid3"], [12]))]), at=("rival:rel3", "rival:pid3"))
        sk.add(_a(calls=[_bash(f"tail -n 40 /var/log/migrations/refunds-attempt-{v['n4']}.log",
                               _pg_attempt_log(w, 30, v["rel4"], v["pid3"], [19]))]), at=("rival:rel4", "rival:pid3"))
        sk.add(_a(text=f"Attempt {v['n3']} waited on {v['rel3']} and {v['n4']} on {v['rel4']}, both blocked by pid "
                       f"{v['pid3']}. That's the nightly refund export holding a lock; different cause from ours."),
               at=("rival:rel3", "rival:rel4", "rival:pid3"))
    elif r == 2:
        sk.add(_u("While you're in there, tune the refunds backfill as well."))
        sk.add(_a(text=f"The refunds backfill updates {v['a']} rows per statement. Trying {v['b']}.",
                  calls=[_bash(f"./scripts/migrate-bench --table refunds --batch {v['b']}",
                               f"batch    statements  max_lock_wait_s  aborted\n{v['b']:<8} {w.i(10, 40):<11} "
                               f"{w.f(20, 50, 1):<16} {w.i(5, 20)}/40")]), at=("rival:batch2",))
        sk.add(_a(text=f"{v['b']} is worse for refunds too. {v['c']} runs clean; using that."), at=("rival:batch2",))
        sk.add(_u(f"Also keep alembic at {v['pin2']}, the next minor changes batch_alter_table."), at=("rival:pin2",))
        sk.add(_a(text=f"Pinned alembic to {v['pin2']}."), at=("rival:pin2",))
    else:
        sk.add(_u(f"Two more things: the reporting replica is {v['host2']} now. And ask {v['contact2']} about the "
                  f"replica lag, not {v['leave2']}; the DBA sync is on {v['day']} at {v['time2']}."),
               at=("rival:host2", "rival:contact2", "rival:leave2", "rival:time2"))
        sk.add(_a(thinking=f"The wiki says the nightly backup runs on ops-backup-1, but the crontab I saw puts it on "
                           f"{v['hidden2']} at `{v['cron2']}`.",
                  text="Noted, I'll follow up with both."), at=("rival:hidden2", "rival:cron2"))


_register("migration", "tool-first", _mig_values, _mig_plant, _mig_episodes)


# =============================================================================
# Story 4: leak (relay websocket gateway memory leak; TypeScript). Skeleton
# "security-first": the user front-loads a CVE, a version cap, the canary host
# and where heap dumps may go; crash logs and the gateway config follow; the
# deadline and contact come mid-session; the buffer-limit chain, the flaky test
# and a superseded heartbeat interval close the plant phase, so the user details
# are farthest from the end.
# =============================================================================

LEAK_TESTS = ["pauses fanout under backpressure", "reuses the deflate context", "drops slow consumers",
              "keeps fanout order per channel", "resumes after reconnect", "times out silent sockets"]
LEAK_PEOPLE = ["Saskia", "Jorge", "Mirela", "Obi", "Klara", "Tariq", "Ines", "Nils"]
LEAK_AUTHORS = ["vstepanova", "dmuller", "kadebayo", "pjansen", "cromano", "hyamada"]


def _leak_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["cve"] = _distinct(rng, lambda r: f"CVE-2026-{r.randint(31000, 48999)}", used)
    mi = rng.randint(13, 17)
    v["ws"], v["ws_next"] = f"8.{mi}.{rng.randint(1, 9)}", f"8.{mi + 1}"
    n = rng.randint(3, 9)
    v["host"], v["prev"] = f"relay-canary-{n:02d}", f"relay-canary-{n - 1:02d}"
    v["path"] = f"/data/relay/heapdumps-{rng.choice(['secure', 'restricted', 'private'])}"
    v["c1"] = rng.randint(212, 260)
    v["c2"] = v["c1"] + 1
    v["d1"] = rng.randint(12, 20)

    def snap(r, d):
        return f"Heap.202609{d:02d}.{r.randint(0, 23):02d}{r.randint(0, 59):02d}{r.randint(0, 59):02d}.{r.randint(20000, 39999)}.0.001.heapsnapshot"
    v["snap1"] = _distinct(rng, lambda r: snap(r, v["d1"]), used)
    v["snap2"] = _distinct(rng, lambda r: snap(r, v["d1"] + 1), used)
    v["conn"] = _distinct(rng, lambda r: f"conn-{_hexr(r, 6)}", used)
    v["hash"] = _hexr(rng, 7)
    v["payload"] = rng.choice([72, 88, 104, 120])
    v["initial"] = rng.choice([768, 896, 1024])
    v["tried"] = rng.choice([3072, 4096])
    v["final"] = rng.choice([160, 192, 224])
    v["hb_old"], v["hb_new"] = 30000, rng.choice([15000, 20000, 25000])
    v["metric"] = f"relay_socket_{rng.choice(['pending', 'queued', 'backlog'])}_bytes"
    v["flaky"], v["real"] = rng.sample(LEAK_TESTS, 2)
    v["contact"], v["on_leave"] = rng.sample(LEAK_PEOPLE, 2)
    v["day"] = rng.choice(DAYS)
    v["time"] = f"{rng.randint(8, 17):02d}:{rng.choice(['15', '30', '45'])}"
    v["author"] = rng.choice(LEAK_AUTHORS)
    # rivals
    v["c3"] = v["c2"] + rng.randint(4, 9)
    v["c4"] = v["c3"] + 1
    v["snap3"] = _distinct(rng, lambda r: snap(r, v["d1"] + 4), used)
    v["snap4"] = _distinct(rng, lambda r: snap(r, v["d1"] + 5), used)
    v["conn3"] = _distinct(rng, lambda r: f"conn-{_hexr(r, 6)}", used)
    v["a"], v["b"], v["c"] = rng.choice([640, 1152]), rng.choice([5120, 6144]), rng.choice([96, 136])
    v["pin2"] = f"20.{rng.randint(40, 49)}.{rng.randint(1, 9)}"
    v["host2"] = f"relay-presence-{rng.randint(3, 9):02d}"
    v["contact2"], v["leave2"] = rng.sample([p for p in LEAK_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["time2"] = f"{rng.randint(8, 17):02d}:{rng.choice(['05', '50'])}"
    v["hidden2"] = f"relay_presence_{rng.choice(['dropped', 'stalled'])}_frames"

    sk.fact("cve", "cve", "user", v["cve"], "Which CVE did security flag in the websocket library?")
    sk.fact("ws_cap", "version_cap", "user", v["ws"], "What is the newest ws library version we may use?")
    sk.fact("canary_host", "host", "user", v["host"], "Which canary host are we testing on?")
    sk.fact("dump_path", "credential_path", "user", v["path"], "Where must heap dumps be written?",
            alts=[v["path"] + "/"])
    sk.fact("heap_snapshot", "error_code", "tool_output", v["snap1"], f"Which heap snapshot file did crash {v['c1']} write?")
    sk.fact("conn_both", "shared_id", "tool_output", v["conn"],
            f"Which connection id was the top slow consumer in both crash {v['c1']} and crash {v['c2']}?")
    sk.fact("deflate_hash", "commit_hash", "tool_output", v["hash"],
            "What is the short hash of the commit that enabled permessage-deflate?")
    sk.fact("deflate_author", "author", "thinking", v["author"], "Who wrote the permessage-deflate change?")
    sk.fact("max_payload", "config_value", "tool_output", v["payload"],
            "What is max_payload_kb set to in the gateway config?", alts=[f"{v['payload']} KB", f"{v['payload']}KB"],
            slot_keys=["max_payload_kb"])
    sk.fact("buffer_initial", "count_initial", "assistant_text", v["initial"],
            "What was the per-connection send buffer limit, in KB, when we found the leak?",
            slot_keys=["send_buffer_limit_kb", "sendBufferLimitKb"])
    sk.fact("buffer_tried", "count_tried", "assistant_text", v["tried"],
            "Which send buffer limit did we try first that made memory worse?")
    sk.fact("buffer_final", "count_final", "superseded", v["final"], "Which send buffer limit did we settle on?",
            old=v["tried"])
    sk.fact("heartbeat", "superseded_setting", "superseded", v["hb_new"],
            "What heartbeat (ping) interval, in milliseconds, did we end up using?", old=v["hb_old"],
            alts=[f"{v['hb_new']} ms", f"{v['hb_new']}ms"], slot_keys=["ping_interval_ms", "pingIntervalMs"])
    sk.fact("buffer_metric", "metric_name", "thinking", v["metric"],
            "What is the actual name of the metric the exporter emits for buffered socket bytes?")
    sk.fact("flaky_test", "flaky_test", "tool_output", v["flaky"], "Which gateway test is flaky?")
    sk.fact("sdk_contact", "contact", "user", v["contact"], "Who should we loop in on the client SDK side?")
    sk.fact("green_by", "time", "user", v["time"], f"By what time on {v['day']} must the canary be green?")
    sk.lure("gc_flag", "lure", "Which Node GC flag did we add to the gateway?")
    sk.lure("cdn_vendor", "lure", "Which CDN did we move the static assets to?")
    sk.lure("ip_rate_limit", "lure", "What per-IP connection rate limit did we set?")
    sk.decoy("ws_next", v["ws_next"])
    sk.decoy("prev_canary", v["prev"])
    sk.decoy("snap2", v["snap2"])
    sk.decoy("runbook_metric", "relay_buffered_bytes")
    sk.decoy("real_failure", v["real"])
    sk.decoy("on_leave", v["on_leave"])
    for name in ("snap3", "snap4", "conn3", "pin2", "host2", "contact2", "leave2", "time2", "hidden2"):
        sk.rival(name, v[name])
    sk.rival("buffer2", v["a"], v["b"], v["c"])
    return v


def _leak_crash_log(w: Work, crash: int, snap: str, conn: str, conn_rank: int) -> str:
    out, t = [], w.i(0, 20) * 3600
    for _ in range(w.i(14, 22)):
        t += w.i(20, 400)
        lvl, msg = w.log_msg()
        out.append(f"{w.clock(t)} {lvl:<7} relay.{w.area()}: {msg}")
    t += 60
    heap = w.f(1.6, 1.95, 2)
    out.append(f"{w.clock(t)} WARNING relay.metrics: heap used {heap} GB of 2.00 GB; "
               f"top slow consumers by bufferedAmount:")
    for k in range(1, 6):
        c = conn if k == conn_rank else f"conn-{w.hexs(6)}"
        out.append(f"    {k}. {c} channel={w.tenant()}:{w.hexs(4)} bufferedAmount={w.i(900000, 9000000)} bytes "
                   f"lastPong={w.i(20, 300)}s ago")
    t += w.i(5, 30)
    out += [f"{w.clock(t)} INFO    relay.metrics: wrote heap snapshot {snap}",
            "",
            "<--- Last few GCs --->",
            f"[1:0x{w.hexs(12)}] {w.i(9000000, 99999999)} ms: Mark-Compact {w.f(1900, 2047, 1)} ({w.f(2000, 2100, 1)}) -> "
            f"{w.f(1900, 2047, 1)} ({w.f(2000, 2100, 1)}) MB, {w.f(800, 2500, 1)} / 0.0 ms  allocation failure; "
            "scavenge might not succeed",
            "",
            "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory",
            f"pod relay-gateway-{w.hexs(9)}-{w.hexs(5)} OOMKilled (crash #{crash}), restarting"]
    return "\n".join(out)


def _leak_gateway_cfg(w: Work, v: dict) -> str:
    return "\n".join([
        "# relay gateway (loaded by src/config.ts)", "gateway:", f"  port: 8443", f"  max_payload_kb: {v['payload']}",
        f"  send_buffer_limit_kb: {v['initial']}", f"  ping_interval_ms: {v['hb_old']}",
        f"  ping_timeout_ms: {w.ch([10000, 12000])}", "  per_message_deflate:", "    enabled: true",
        f"    server_max_window_bits: {w.ch([10, 12, 15])}", "    client_no_context_takeover: false",
        "fanout:", f"  shards: {w.ch([16, 32, 64])}", f"  max_channels_per_conn: {w.ch([64, 128])}",
        "presence:", f"  roster_batch_ms: {w.ch([200, 500])}"])


def _leak_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    sk.add(_u(f"Two things on the relay gateway. Security flagged {v['cve']} in ws, so we have to upgrade, but don't go "
              f"past {v['ws']}: {v['ws_next']} drops the Node 20 build we still ship. And the gateway is leaking memory "
              f"again, pods get OOM-killed every few hours."), plants=("cve", "ws_cap"), at=("decoy:ws_next",))
    sk.add(_u(f"Test everything on {v['host']}, {v['prev']} was taken out of the pool. Heap dumps must go to "
              f"{v['path']}, never /tmp: they contain message payloads."),
           plants=("canary_host", "dump_path"), at=("decoy:prev_canary",))
    pkg = json.dumps({"name": "@relay/gateway", "version": f"4.{w.i(10, 30)}.0", "private": True,
                      "dependencies": {"ws": f"^8.{w.i(5, 12)}.0", "pino": "^9.4.0", "prom-client": "^15.1.3",
                                       "fastify": "^5.0.0", "ioredis": "^5.4.1"},
                      "devDependencies": {"vitest": "^3.2.4", "typescript": "^5.6.3", "tsx": "^4.19.1"}}, indent=2)
    sk.add(_a(thinking="Check the current ws range first.", calls=[_read("package.json", pkg)]))
    sk.add(_a(text="ws is on an older minor; I'll do the upgrade after the leak, within the cap. Reading the last two "
                   "crashes."))
    sk.gap(0.3)
    sk.add(_a(calls=[_bash(f"kubectl -n relay logs relay-gateway-0 --previous | tail -n 40 > crash-{v['c1']}.log; "
                           f"cat crash-{v['c1']}.log", _leak_crash_log(w, v["c1"], v["snap1"], v["conn"], 2))]),
           plants=("heap_snapshot",), at=("conn_both",))
    sk.add(_a(text=f"Crash {v['c1']} is a plain heap exhaustion; a handful of connections hold megabytes of unsent "
                   f"frames each. Comparing with crash {v['c2']}."))
    sk.add(_a(calls=[_bash(f"kubectl -n relay logs relay-gateway-1 --previous | tail -n 40 > crash-{v['c2']}.log; "
                           f"cat crash-{v['c2']}.log", _leak_crash_log(w, v["c2"], v["snap2"], v["conn"], 4))]),
           plants=("conn_both",), at=("decoy:snap2",))
    sk.add(_a(text=f"{v['c2']} wrote a different snapshot, but one connection is among the top slow consumers both "
                   f"times. Probably a client that never reads; noting it."))
    sk.add(_u(f"We need the canary green by {v['day']} {v['time']}. Loop in {v['contact']} from the client SDK team if "
              f"it's a client bug, not {v['on_leave']}, {v['on_leave']} is on leave."),
           plants=("green_by", "sdk_contact"), at=("decoy:on_leave",))
    sk.add(_a(thinking="What does the gateway allow per connection?", calls=[_read("config/gateway.yaml", _leak_gateway_cfg(w, v))]),
           plants=("max_payload",), at=("buffer_initial", "heartbeat"))
    sk.add(_a(thinking="And what changed recently in the codec?",
              calls=[_bash("git log --oneline -12 -- src/codec/",
                           w.oneline(12, {w.i(1, 6): f"{v['hash']} enable permessage-deflate with per-connection context"}))]),
           plants=("deflate_hash",))
    sk.add(_a(thinking=f"{v['author']} wrote the permessage-deflate change; their PR kept a deflate context per "
                       f"connection, which costs memory for every idle socket. Worth asking them."),
           plants=("deflate_author",))
    sk.add(_a(text=f"Each connection may buffer up to {v['initial']} KB of unsent frames, and with deflate contexts on "
                   f"top, a few thousand slow consumers are enough to exhaust the heap. I'll try a different limit."),
           plants=("buffer_initial",))
    sk.gap(0.5)
    sk.add(_a(calls=[_bash(f"pnpm tsx scripts/soak.ts --host {v['host']} --send-buffer-kb {v['tried']}",
                           f"send_buffer_kb  conns  heap_peak_mb  dropped_slow  oom\n{v['tried']:<15} 18000  "
                           f"{w.i(1900, 2040):<13} {w.i(0, 5):<13} yes")]), at=("buffer_tried", "canary_host"))
    sk.add(_a(text=f"{v['tried']} KB is worse: the heap fills faster and the pod is OOM-killed during the soak. Going "
                   f"down instead, {v['final']}."), plants=("buffer_tried",), at=("buffer_final",))
    sk.add(_a(calls=[_bash(f"pnpm tsx scripts/soak.ts --host {v['host']} --send-buffer-kb {v['final']}",
                           f"send_buffer_kb  conns  heap_peak_mb  dropped_slow  oom\n{v['final']:<15} 18000  "
                           f"{w.i(600, 900):<13} {w.i(40, 200):<13} no")]), at=("buffer_final", "canary_host"))
    sk.add(_a(text=f"{v['final']} KB keeps the heap under a gigabyte; slow consumers get dropped instead of buffered.",
              calls=[_edit("config/gateway.yaml", [(f"  send_buffer_limit_kb: {v['initial']}",
                                                    f"  send_buffer_limit_kb: {v['final']}")])]),
           plants=("buffer_final",), at=("buffer_initial",))
    sk.add(_a(thinking=f"The runbook calls the metric relay_buffered_bytes, but the exporter actually emits "
                       f"{v['metric']}; the alert should use the real name.",
              text="I'll point the memory alert at the exporter's buffered-bytes metric."),
           plants=("buffer_metric",), at=("decoy:runbook_metric",))
    names = LEAK_TESTS + w.test_names(8)
    w.rng.shuffle(names)
    suite = "test/gateway/fanout.test.ts"
    sk.add(_a(calls=[_bash("pnpm vitest run test/gateway --reporter verbose", w.vitest_run(suite, names, [v["flaky"], v["real"]]))]),
           plants=("flaky_test",), at=("decoy:real_failure",))
    sk.add(_a(text="Two failures; rerunning to see which is stable."))
    sk.add(_a(calls=[_bash("pnpm vitest run test/gateway --reporter verbose", w.vitest_run(suite, names, [v["real"]]))]),
           at=("flaky_test", "decoy:real_failure"))
    sk.add(_a(text="One passes on the rerun, so it's flaky; the other fails both times because the drop path changed. "
                   "I'll fix that test's expectation."))
    sk.add(_u("What heartbeat interval are we using? Idle sockets should go away faster."))
    sk.add(_a(text=f"{v['hb_old']} ms between pings."), at=("heartbeat",))
    sk.add(_u(f"Make it {v['hb_new']} ms."), plants=("heartbeat",))
    sk.add(_a(calls=[_edit("config/gateway.yaml", [(f"  ping_interval_ms: {v['hb_old']}", f"  ping_interval_ms: {v['hb_new']}")])],
              text="Done."), at=("heartbeat",))
    sk.gap(0.2)


def _leak_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"Crashes {v['c3']} and {v['c4']} came in overnight, on the presence pods. Same leak?"))
        sk.add(_a(calls=[_bash(f"kubectl -n relay logs relay-presence-0 --previous | tail -n 40",
                               _leak_crash_log(w, v["c3"], v["snap3"], v["conn3"], 1))]), at=("rival:snap3", "rival:conn3"))
        sk.add(_a(calls=[_bash(f"kubectl -n relay logs relay-presence-1 --previous | tail -n 40",
                               _leak_crash_log(w, v["c4"], v["snap4"], v["conn3"], 3))]), at=("rival:snap4", "rival:conn3"))
        sk.add(_a(text=f"Crash {v['c3']} wrote {v['snap3']} and {v['c4']} wrote {v['snap4']}; {v['conn3']} is a top "
                       f"consumer in both. The presence pods leak through the roster cache, not the send buffers."),
               at=("rival:snap3", "rival:snap4", "rival:conn3"))
    elif r == 2:
        sk.add(_u("While you're at it, tune the presence fanout buffer as well."))
        sk.add(_a(text=f"The presence fanout buffer is {v['a']} KB. Trying {v['b']}.",
                  calls=[_bash(f"pnpm tsx scripts/soak.ts --presence-buffer-kb {v['b']}",
                               f"presence_buffer_kb  heap_peak_mb  oom\n{v['b']:<19} {w.i(1900, 2040):<13} yes")]),
               at=("rival:buffer2",))
        sk.add(_a(text=f"{v['b']} KB is worse for presence too. {v['c']} KB is clean; using it."), at=("rival:buffer2",))
        sk.add(_u(f"Also keep uWS at {v['pin2']}, the next release changes TLS termination."), at=("rival:pin2",))
        sk.add(_a(text=f"Pinned uWS to {v['pin2']}."), at=("rival:pin2",))
    else:
        sk.add(_u(f"Two more things: the presence canary is {v['host2']} now. And for the mobile reconnect storms ask "
                  f"{v['contact2']}, not {v['leave2']}; the load test is {v['day']} at {v['time2']}."),
               at=("rival:host2", "rival:contact2", "rival:leave2", "rival:time2"))
        sk.add(_a(thinking=f"The runbook calls the presence metric relay_presence_drops, but the exporter actually emits "
                           f"{v['hidden2']}.", text="Noted."), at=("rival:hidden2",))


_register("leak", "security-first", _leak_values, _leak_plant, _leak_episodes)


# =============================================================================
# Story 5: rollback (checkout canary rollback; Python + Kubernetes). Skeleton
# "incident-handover": a handover message plants the on-call engineer and the
# release freeze together; the rollout history gives the last good image; two
# canary waves give the error fingerprint and the merchant; the rollout spec
# gives the canary weight and the bake time, which the user later supersedes;
# the rate-limit chain is in the middle; the flaky test, signing key and support
# contact come late, and the only thinking-only detail is the very last step.
# =============================================================================

RB_TESTS = ["test_tax_rounding_jpy", "test_idempotency_header", "test_cart_merge_guest", "test_coupon_stack_limit",
            "test_address_normalise", "test_payment_retry_window"]
RB_PEOPLE = ["Rosa", "Henrik", "Adaeze", "Mateo", "Ingrid", "Sanjay", "Elif", "Corin"]
RB_ONCALL = ["mschneider", "ochukwu", "lpetrov", "awilliams", "tnguyen", "fgarcia"]


def _rb_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["oncall"], v["oncall2"] = rng.sample(RB_ONCALL, 2)
    v["day"] = rng.choice(DAYS)
    v["freeze"] = f"{rng.randint(13, 18):02d}:{rng.choice(['00', '30'])}"
    m = rng.randint(41, 58)
    v["rev"] = rng.randint(31, 60)
    v["good"], v["bad"] = f"v3.{m}.{rng.randint(1, 6)}", f"v3.{m + 1}.0"
    n = rng.randint(3, 9)
    v["host"], v["prev"] = f"deploy-eu-{n:02d}", f"deploy-eu-{n - 1:02d}"
    v["w1"] = rng.randint(11, 29)
    v["w2"] = v["w1"] + 1
    v["fp1"] = _distinct(rng, lambda r: f"FP-{_hexr(r, 6).upper()}", used)
    v["fp2"] = _distinct(rng, lambda r: f"FP-{_hexr(r, 6).upper()}", used)
    v["merchant"] = _distinct(rng, lambda r: f"mrc_{_hexr(r, 8)}", used)
    v["canary"] = rng.choice([5, 8, 12, 15])
    v["bake_old"], v["bake_new"] = "30m", rng.choice(["45m", "55m", "70m"])
    mi = rng.randint(2, 6)
    v["sdk"], v["sdk_next"] = f"12.{mi}.{rng.randint(1, 9)}", f"12.{mi + 1}"
    v["hash"] = _hexr(rng, 7)
    v["initial"] = rng.choice([1200, 1500])
    v["tried"] = rng.choice([4000, 5000])
    v["final"] = rng.choice([450, 600, 750])
    v["flaky"], v["real"] = rng.sample(RB_TESTS, 2)
    v["key"] = f"/etc/checkout/release-{rng.choice(['signing', 'sign', 'codesign'])}.pem"
    v["contact"], v["on_leave"] = rng.sample(RB_PEOPLE, 2)
    v["region"] = f"eu-central-{rng.randint(1, 2)}"
    # rivals
    v["w3"] = v["w2"] + rng.randint(3, 6)
    v["w4"] = v["w3"] + 1
    v["fp3"] = _distinct(rng, lambda r: f"FP-{_hexr(r, 6).upper()}", used)
    v["fp4"] = _distinct(rng, lambda r: f"FP-{_hexr(r, 6).upper()}", used)
    v["merchant3"] = _distinct(rng, lambda r: f"mrc_{_hexr(r, 8)}", used)
    v["a"], v["b"], v["c"] = rng.choice([1100, 1300]), rng.choice([3500, 4500]), rng.choice([350, 550])
    v["pin2"] = f"4.{rng.randint(2, 9)}.{rng.randint(1, 9)}"
    v["host2"] = f"deploy-us-{rng.randint(3, 9):02d}"
    v["contact2"], v["leave2"] = rng.sample([p for p in RB_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["time2"] = f"{rng.randint(9, 12):02d}:{rng.choice(['15', '45'])}"
    v["region2"] = "eu-north-1"

    sk.fact("oncall", "oncall_login", "user", v["oncall"], "Who is on call tonight?")
    sk.fact("freeze_time", "time", "user", v["freeze"], f"At what time on {v['day']} does the release freeze start?")
    sk.fact("good_image", "image_tag", "tool_output", v["good"], "Which image tag are we rolling back to?",
            alts=[v["good"][1:]])
    sk.fact("deploy_host", "host", "user", v["host"], "Which deploy host does the hotfix go through now?")
    sk.fact("fingerprint", "error_code", "tool_output", v["fp1"],
            f"Which error fingerprint did the failing requests in canary wave {v['w1']} report?")
    sk.fact("merchant_both", "shared_id", "tool_output", v["merchant"],
            f"Which merchant id failed in both canary wave {v['w1']} and wave {v['w2']}?")
    sk.fact("canary_weight", "percent_threshold", "tool_output", v["canary"],
            "What traffic weight, in percent, does the first canary step set in the rollout spec?",
            alts=[f"{v['canary']}%", f"{v['canary']} %"],
            slot_keys=["setWeight"])
    sk.fact("bake_time", "superseded_setting", "superseded", v["bake_new"],
            "How long did we decide the canary should bake?", old=v["bake_old"],
            alts=[f"{v['bake_new'][:-1]} minutes", f"{v['bake_new'][:-1]} min"])
    sk.fact("sdk_cap", "version_cap", "user", v["sdk"], "What is the newest payments SDK version we may use?")
    sk.fact("tax_hash", "commit_hash", "tool_output", v["hash"],
            "What is the short hash of the commit that moved tax calculation into checkout-core?")
    sk.fact("rate_initial", "count_initial", "assistant_text", v["initial"],
            "What was the per-merchant rate limit (requests per second) when we found the throttling?",
            slot_keys=["per_merchant_rps"])
    sk.fact("rate_tried", "count_tried", "assistant_text", v["tried"],
            "Which per-merchant rate limit did we try first that made things worse?")
    sk.fact("rate_final", "count_final", "superseded", v["final"], "Which per-merchant rate limit did we settle on?",
            old=v["tried"])
    sk.fact("flaky_test", "flaky_test", "tool_output", v["flaky"], "Which checkout test is flaky?")
    sk.fact("signing_key", "credential_path", "user", v["key"], "Which signing key should the release use?")
    sk.fact("support_contact", "contact", "user", v["contact"], "Who in support should we tell before rolling back?")
    sk.fact("canary_region", "region", "thinking", v["region"], "Which region is the canary actually pinned to?",
            slot_keys=["region", "topology.kubernetes.io/region"])
    sk.lure("feature_kill", "lure", "Which kill switch did we flip for the new checkout page?")
    sk.lure("psp_failover", "lure", "Which payment provider did we fail over to?")
    sk.lure("db_pool", "lure", "What database pool size did we set for checkout?")
    sk.decoy("bad_image", v["bad"], v["bad"][1:])
    sk.decoy("prev_host", v["prev"])
    sk.decoy("fp2", v["fp2"])
    sk.decoy("sdk_next", v["sdk_next"])
    sk.decoy("real_failure", v["real"])
    sk.decoy("on_leave", v["on_leave"])
    sk.decoy("dashboard_region", "eu-west-2")
    for name in ("oncall2", "fp3", "fp4", "merchant3", "pin2", "host2", "contact2", "leave2", "time2", "region2"):
        sk.rival(name, v[name])
    sk.rival("rate2", v["a"], v["b"], v["c"])
    return v


def _rb_history(w: Work, v: dict) -> str:
    rows = ["deployment.apps/checkout", "REVISION  CHANGE-CAUSE"]
    m = int(v["good"].split(".")[1])
    for k in range(5, 1, -1):
        rows.append(f"{v['rev'] - k:<10}deploy checkout v3.{m - k // 2}.{w.i(0, 9)} ({w.ch(['release', 'hotfix'])}, "
                    f"by {w.login()})")
    rows.append(f"{v['rev'] - 1:<10}deploy checkout {v['good']} (release, by {w.login()})")
    rows.append(f"{v['rev']:<10}deploy checkout {v['bad']} (release, by {w.login()})")
    return "\n".join(rows)


def _rb_wave_log(w: Work, wave: int, fp: str, merchant: str, at: list[int], m_at: int) -> str:
    out, t = [], w.i(8, 14) * 3600
    for k in range(46):
        t += w.i(5, 120)
        if k in at:
            mid = merchant if k == m_at else f"mrc_{w.hexs(8)}"
            out.append(f"{w.clock(t)} ERROR   checkout.tax: quote failed for merchant {mid} cart={w.hexs(10)} "
                       f"country={w.ch(['DE', 'FR', 'JP', 'GB', 'NL'])}: TaxServiceTimeout after {w.i(2000, 4000)} ms "
                       f"(fingerprint {fp}, wave {wave})")
        else:
            lvl, msg = w.log_msg()
            out.append(f"{w.clock(t)} {lvl:<7} checkout.{w.area()}: {msg}")
    return "\n".join(out)


def _rb_rollout(v: dict, bake: str) -> str:
    return "\n".join([
        "apiVersion: argoproj.io/v1alpha1", "kind: Rollout", "metadata:", "  name: checkout", "  namespace: checkout",
        "spec:", "  replicas: 12", "  revisionHistoryLimit: 10", "  strategy:", "    canary:",
        "      canaryService: checkout-canary", "      stableService: checkout-stable", "      steps:",
        f"        - setWeight: {v['canary']}", f"        - pause: {{duration: {bake}}}",
        "        - analysis:", "            templates:", "              - templateName: checkout-error-rate",
        "        - setWeight: 50", "        - pause: {duration: 10m}", "  template:", "    spec:",
        "      containers:", "        - name: checkout", f"          image: registry.internal/checkout/checkout:{v['bad']}",
        "          ports:", "            - containerPort: 8080"])


def _rb_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    sk.add(_u(f"Handing this over to you: checkout error rates jumped after the {v['bad']} deploy and we want to roll "
              f"back cleanly. {v['oncall']} is on call tonight if you need a second pair of eyes. Everything has to be "
              f"done before the release freeze, {v['day']} at {v['freeze']}."),
           plants=("oncall", "freeze_time"), at=("decoy:bad_image",))
    sk.add(_a(thinking="Which revision was the last good one?",
              calls=[_bash("kubectl -n checkout rollout history deployment/checkout", _rb_history(w, v))]),
           plants=("good_image",), at=("decoy:bad_image",))
    sk.add(_a(text=f"Revision {v['rev'] - 1} is the last release before the bad one, so that's our rollback target. "
                   f"First I want to see what the canary waves recorded."))
    sk.add(_u(f"The hotfix goes through {v['host']} now, {v['prev']} is being rebuilt."),
           plants=("deploy_host",), at=("decoy:prev_host",))
    sk.add(_a(calls=[_read(f"/var/log/checkout/canary-wave-{v['w1']}.log", _rb_wave_log(w, v["w1"], v["fp1"], v["merchant"], [7, 15, 22, 31, 38], 22))]),
           plants=("fingerprint",), at=("merchant_both",))
    sk.add(_a(text=f"Wave {v['w1']}: the failing quotes all share one fingerprint, a tax service timeout. Comparing "
                   f"with wave {v['w2']}."))
    sk.add(_a(calls=[_read(f"/var/log/checkout/canary-wave-{v['w2']}.log", _rb_wave_log(w, v["w2"], v["fp2"], v["merchant"], [4, 19, 27, 40], 27))]),
           plants=("merchant_both",), at=("decoy:fp2",))
    sk.add(_a(text=f"Wave {v['w2']} has a different fingerprint, but one merchant failed in both waves; probably a "
                   f"high-volume JPY merchant. Noting it."))
    sk.gap(0.4)
    sk.add(_a(thinking="How is the canary configured?", calls=[_read("deploy/rollouts/checkout.yaml", _rb_rollout(v, v["bake_old"]))]),
           plants=("canary_weight",), at=("bake_time", "decoy:bad_image"))
    sk.add(_a(text="The canary takes a slice of traffic and bakes before the analysis step; the error-rate analysis "
                   "should have caught this, so the bake may be too short."))
    sk.add(_u(f"Don't bump the payments SDK past {v['sdk']} while you're in there; {v['sdk_next']} changes the "
              f"idempotency header."), plants=("sdk_cap",), at=("decoy:sdk_next",))
    sk.add(_a(calls=[_bash("git log --oneline -14 -- checkout/tax/",
                           w.oneline(14, {w.i(0, 5): f"{v['hash']} move tax calculation into checkout-core"}))]),
           plants=("tax_hash",))
    sk.add(_a(text=f"Since tax moved into checkout-core, every quote calls the tax service, and the tax client limits "
                   f"each merchant to {v['initial']} requests per second. Big merchants hit that during the canary and "
                   f"their retries time out."), plants=("rate_initial",))
    sk.add(_a(calls=[_bash(f"./scripts/loadtest.sh --per-merchant-rps {v['tried']} --merchants 40",
                           f"per_merchant_rps  quotes  tax_timeouts  p99_ms\n{v['tried']:<17} {w.i(80000, 99000):<7} "
                           f"{w.i(300, 900):<13} {w.i(3000, 6000)}")]), at=("rate_tried",))
    sk.add(_a(text=f"{v['tried']} rps is worse: the tax service itself falls over. Trying {v['final']}."),
           plants=("rate_tried",), at=("rate_final",))
    sk.add(_a(calls=[_bash(f"./scripts/loadtest.sh --per-merchant-rps {v['final']} --merchants 40",
                           f"per_merchant_rps  quotes  tax_timeouts  p99_ms\n{v['final']:<17} {w.i(80000, 99000):<7} "
                           f"0             {w.i(300, 900)}")]), at=("rate_final",))
    sk.add(_a(text=f"{v['final']} rps per merchant: no tax timeouts, p99 well under a second.",
              calls=[_edit("config/ratelimits.yaml", [(f"  per_merchant_rps: {v['initial']}", f"  per_merchant_rps: {v['final']}")])]),
           plants=("rate_final",), at=("rate_initial",))
    sk.add(_u(f"And make the canary bake {v['bake_new']} instead of {v['bake_old']}; the error-rate analysis needs more "
              f"traffic than half an hour gives it."), plants=("bake_time",))
    sk.add(_a(calls=[_edit("deploy/rollouts/checkout.yaml", [(f"        - pause: {{duration: {v['bake_old']}}}",
                                                              f"        - pause: {{duration: {v['bake_new']}}}")])],
              text="Bake time updated."), at=("bake_time",))
    sk.gap(0.3)
    names = RB_TESTS + w.test_names(8)
    w.rng.shuffle(names)
    sk.add(_a(calls=[_bash("pytest tests/checkout -v", w.pytest_run("tests/checkout/test_quotes.py", names, [v["flaky"], v["real"]]))]),
           plants=("flaky_test",), at=("decoy:real_failure",))
    sk.add(_a(text="Two failures; rerunning before trusting either."))
    sk.add(_a(calls=[_bash("pytest tests/checkout -v", w.pytest_run("tests/checkout/test_quotes.py", names, [v["real"]]))]),
           at=("flaky_test", "decoy:real_failure"))
    sk.add(_a(text="One passes on the rerun, so that one is flaky; the other is a real failure in the rounding code. "
                   "Should I skip signing for the rollback build to save time?"))
    sk.add(_u(f"No, always sign. Use the key at {v['key']}."), plants=("signing_key",))
    sk.add(_u(f"And tell {v['contact']} in support before you roll back, not {v['on_leave']}; {v['on_leave']} is off today."),
           plants=("support_contact",), at=("decoy:on_leave",))
    sk.add(_a(thinking=f"The checkout dashboard says the canary runs in eu-west-2, but the rollout manifest I checked "
                       f"pins the canary pods to {v['region']}; the dashboard label is wrong.",
              text="Will do."), plants=("canary_region",), at=("decoy:dashboard_region",))
    sk.gap(1.3)


def _rb_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"Canary waves {v['w3']} and {v['w4']} on the receipts service failed too. Related?"))
        sk.add(_a(calls=[_bash(f"tail -n 46 /var/log/receipts/canary-wave-{v['w3']}.log",
                               _rb_wave_log(w, v["w3"], v["fp3"], v["merchant3"], [9, 30], 9))]), at=("rival:fp3", "rival:merchant3"))
        sk.add(_a(calls=[_bash(f"tail -n 46 /var/log/receipts/canary-wave-{v['w4']}.log",
                               _rb_wave_log(w, v["w4"], v["fp4"], v["merchant3"], [12, 25], 25))]), at=("rival:fp4", "rival:merchant3"))
        sk.add(_a(text=f"Receipts wave {v['w3']} fails with {v['fp3']} and {v['w4']} with {v['fp4']}; {v['merchant3']} "
                       f"failed in both. It's the receipts locale fallback, a separate issue."),
               at=("rival:fp3", "rival:fp4", "rival:merchant3"))
    elif r == 2:
        sk.add(_u("While you're at it, check the shipping-quote rate limit."))
        sk.add(_a(text=f"The shipping-quote limit is {v['a']} rps per merchant. Trying {v['b']}.",
                  calls=[_bash(f"./scripts/loadtest.sh --shipping-rps {v['b']}",
                               f"shipping_rps  quotes  carrier_timeouts\n{v['b']:<13} {w.i(30000, 60000):<7} {w.i(200, 600)}")]),
               at=("rival:rate2",))
        sk.add(_a(text=f"{v['b']} is worse for shipping quotes. {v['c']} is clean; using it."), at=("rival:rate2",))
        sk.add(_u(f"Keep tax-client at {v['pin2']}, the next minor breaks JPY rounding."), at=("rival:pin2",))
        sk.add(_a(text=f"Pinned tax-client to {v['pin2']}."), at=("rival:pin2",))
    else:
        sk.add(_u(f"Two more things: the receipts deploy host is {v['host2']} now, and {v['oncall2']} takes over on call "
                  f"tomorrow. Ask {v['contact2']} about the merchant comms, not {v['leave2']}; the incident review is "
                  f"{v['day']} at {v['time2']}."),
               at=("rival:host2", "rival:oncall2", "rival:contact2", "rival:leave2", "rival:time2"))
        sk.add(_a(thinking=f"The receipts dashboard says eu-west-1, but the receipts worker manifest pins it to "
                           f"{v['region2']}.", text="Noted."), at=("rival:region2",))


_register("rollback", "incident-handover", _rb_values, _rb_plant, _rb_episodes)


# =============================================================================
# Story 6: thumbs (image thumbnail service; TypeScript + docker compose).
# Skeleton "repro-first": a local reproduction comes first (the compose port),
# then two traffic spikes (resize error code, image id), the thumbs config and
# a libvips cap; the author is a thinking-only note before the git history; the
# origin-connection chain and a superseded cache TTL with a thinking-only bucket
# follow; the flaky test, key file, dashboard panel and contact close the phase.
# =============================================================================

TH_TESTS = ["keeps srcset widths sorted", "keeps the ICC profile", "strips EXIF by default", "falls back from AVIF",
            "keeps the ETag stable", "crops towards faces"]
TH_PEOPLE = ["Wanjiru", "Stellan", "Paloma", "Ruairi", "Xenia", "Bao", "Ottilie", "Hamid"]
TH_AUTHORS = ["egreen", "tkovacs", "nabubakar", "msato", "jlarsen", "vrossi"]
TH_PANELS = ["Origin fetch p99 by POP", "Resize queue depth", "Cache hit ratio by format", "Thumbnail bytes served"]


def _th_values(sk: Skel, rng: random.Random) -> dict:
    used: set[str] = set()
    v: dict = {}
    v["port"] = _distinct(rng, lambda r: str(r.choice([18080, 18443, 28080, 38080, 48080])), used)
    n = rng.randint(3, 9)
    v["host"], v["prev"] = f"img-stg-{n:02d}", f"img-stg-{n - 1:02d}"
    v["s1"] = rng.randint(14, 40)
    v["s2"] = v["s1"] + 1
    v["code1"] = _distinct(rng, lambda r: f"RZ-{_hexr(r, 5).upper()}", used)
    v["code2"] = _distinct(rng, lambda r: f"RZ-{_hexr(r, 5).upper()}", used)
    v["img"] = _distinct(rng, lambda r: f"img-{_hexr(r, 8)}", used)
    v["swr"] = rng.choice([270, 330, 390, 450])
    mi = rng.randint(13, 16)
    v["vips"], v["vips_next"] = f"8.{mi}.{rng.randint(1, 9)}", f"8.{mi + 1}"
    v["hash"] = _hexr(rng, 7)
    v["author"] = rng.choice(TH_AUTHORS)
    v["initial"] = rng.choice([180, 220, 260])
    v["tried"] = rng.choice([800, 1000, 1200])
    v["final"] = rng.choice([60, 70, 90])
    v["ttl_old"], v["ttl_new"] = 300, rng.choice([600, 900, 1200])
    v["region"] = rng.choice(["eu", "us", "ap"])
    v["hidden"] = f"thumbs-origin-{v['region']}-{rng.randint(2, 9)}"
    v["flaky"], v["real"] = rng.sample(TH_TESTS, 2)
    v["key"] = f"/etc/thumbs/origin-{rng.choice(['reader', 'ro', 'fetch'])}.key"
    v["panel"] = rng.choice(TH_PANELS)
    v["contact"], v["on_leave"] = rng.sample(TH_PEOPLE, 2)
    # rivals
    v["s3"] = v["s2"] + rng.randint(3, 7)
    v["s4"] = v["s3"] + 1
    v["code3"] = _distinct(rng, lambda r: f"RZ-{_hexr(r, 5).upper()}", used)
    v["code4"] = _distinct(rng, lambda r: f"RZ-{_hexr(r, 5).upper()}", used)
    v["img3"] = _distinct(rng, lambda r: f"img-{_hexr(r, 8)}", used)
    v["a"], v["b"], v["c"] = rng.choice([150, 240]), rng.choice([700, 1500]), rng.choice([40, 50])
    v["pin2"] = f"0.{rng.randint(30, 34)}.{rng.randint(1, 9)}"
    v["host2"] = f"img-edge-stg-{rng.randint(3, 9):02d}"
    v["contact2"], v["leave2"] = rng.sample([p for p in TH_PEOPLE if p not in (v["contact"], v["on_leave"])], 2)
    v["panel2"] = rng.choice([p for p in TH_PANELS if p != v["panel"]])
    v["hidden2"] = f"thumbs-avatars-{rng.choice(['eu', 'us', 'ap'])}-{rng.randint(2, 9)}"
    v["port2"] = _distinct(rng, lambda r: str(r.choice([19090, 29090, 39090])), used)

    sk.fact("compose_port", "compose_port", "tool_output", v["port"],
            "Which host port does docker-compose map the thumbs service to?")
    sk.fact("staging_node", "host", "user", v["host"], "Which staging image node are we using now?")
    sk.fact("resize_code", "error_code", "tool_output", v["code1"],
            f"Which resize error code did the failed requests in spike {v['s1']} log?")
    sk.fact("image_both", "shared_id", "tool_output", v["img"],
            f"Which image id failed in both spike {v['s1']} and spike {v['s2']}?")
    sk.fact("swr", "config_value", "tool_output", v["swr"],
            "What is stale_while_revalidate_s set to in the thumbs config?", alts=[f"{v['swr']}s", f"{v['swr']} s"],
            slot_keys=["stale_while_revalidate_s"])
    sk.fact("vips_cap", "version_cap", "user", v["vips"], "What is the newest libvips version we may use?")
    sk.fact("ttl_author", "author", "thinking", v["author"], "Who wrote the cache TTL change?")
    sk.fact("ttl_hash", "commit_hash", "tool_output", v["hash"],
            "What is the short hash of the commit that lowered the thumbnail cache TTL?")
    sk.fact("origin_initial", "count_initial", "assistant_text", v["initial"],
            "What was the origin connection limit when we found the stampede?", slot_keys=["origin_max_conns"])
    sk.fact("origin_tried", "count_tried", "assistant_text", v["tried"],
            "Which origin connection limit did we try first that made things worse?")
    sk.fact("origin_final", "count_final", "superseded", v["final"], "Which origin connection limit did we settle on?",
            old=v["tried"])
    sk.fact("cache_ttl", "superseded_setting", "superseded", v["ttl_new"],
            "What thumbnail cache TTL, in seconds, did we end up using?", old=v["ttl_old"],
            alts=[f"{v['ttl_new']}s", f"{v['ttl_new']} s"], slot_keys=["cache_ttl_s"])
    sk.fact("origin_bucket", "hidden_location", "thinking", v["hidden"],
            "Which bucket are thumbnails actually fetched from at origin?", slot_keys=["origin_bucket"])
    sk.fact("flaky_test", "flaky_test", "tool_output", v["flaky"], "Which thumbs test is flaky?")
    sk.fact("origin_key", "credential_path", "user", v["key"], "Which key file should the origin fetcher use?")
    sk.fact("dash_panel", "dashboard_panel", "user", v["panel"], "Which dashboard panel should we watch during the peak?")
    sk.fact("cache_contact", "contact", "user", v["contact"], "Who should we ask about the image cache dashboards?")
    sk.lure("cdn_purge_key", "lure", "Which CDN surrogate key do we purge on edits?")
    sk.lure("webp_quality", "lure", "What WebP quality did we settle on?")
    sk.lure("worker_threads", "lure", "How many libuv threads did we give the resize workers?")
    sk.decoy("prev_node", v["prev"])
    sk.decoy("code2", v["code2"])
    sk.decoy("vips_next", v["vips_next"])
    sk.decoy("readme_bucket", "thumbs-shared-assets")
    sk.decoy("real_failure", v["real"])
    sk.decoy("on_leave", v["on_leave"])
    for name in ("code3", "code4", "img3", "pin2", "host2", "contact2", "leave2", "panel2", "hidden2", "port2"):
        sk.rival(name, v[name])
    sk.rival("conns2", v["a"], v["b"], v["c"])
    return v


def _th_spike_log(w: Work, spike: int, code: str, img: str, at: list[int], img_at: int) -> str:
    out, t = [], w.i(17, 19) * 3600
    for k in range(50):
        t += w.i(2, 40)
        if k in at:
            iid = img if k == img_at else f"img-{w.hexs(8)}"
            out.append(f"{w.clock(t)} ERROR   thumbs.resize: {iid} {w.ch([320, 640, 1280])}w "
                       f"{w.ch(['avif', 'webp', 'jpeg'])} failed: origin fetch timed out after {w.i(5000, 9000)} ms "
                       f"(code={code}, spike={spike})")
        else:
            lvl, msg = w.log_msg()
            out.append(f"{w.clock(t)} {lvl:<7} thumbs.{w.area()}: {msg}")
    return "\n".join(out)


def _th_plant(sk: Skel, v: dict) -> None:
    w = sk.w
    sk.add(_u("Thumbnail requests time out during the evening peak. Can you reproduce it locally first?"))
    compose = "\n".join([
        "services:", "  thumbs:", "    build: .", "    ports:", f"      - \"{v['port']}:8080\"", "    environment:",
        "      THUMBS_ORIGIN_URL: http://origin:9000", f"      THUMBS_LOG_LEVEL: {w.ch(['info', 'debug'])}",
        "    depends_on: [origin, redis]", "  origin:", "    image: minio/minio:RELEASE.2025-09-07T16-13-09Z",
        "    command: server /data", "    ports:", "      - \"9000:9000\"", "  redis:", "    image: redis:7.4-alpine"])
    sk.add(_a(thinking="Look at the compose setup.", calls=[_read("docker-compose.yml", compose)]), plants=("compose_port",))
    sk.add(_a(calls=[_bash("docker compose up -d && sleep 5 && ./scripts/replay-peak.sh --rps 400 --duration 60",
                           "\n".join([" Container thumbs-redis-1  Started", " Container thumbs-origin-1  Started",
                                      " Container thumbs-thumbs-1  Started",
                                      f"requests {w.i(20000, 26000)}  ok {w.i(15000, 19000)}  timeouts {w.i(2000, 5000)}",
                                      f"p50 {w.i(80, 200)}ms  p95 {w.i(4000, 9000)}ms"]))]))
    sk.add(_a(text="Reproduced: under replayed peak traffic a large share of resizes time out waiting on the origin."))
    sk.add(_u(f"For anything you need on real hardware, the staging image node is {v['host']} now; {v['prev']} is gone."),
           plants=("staging_node",), at=("decoy:prev_node",))
    sk.add(_a(calls=[_read(f"/var/log/thumbs/spike-{v['s1']}.log", _th_spike_log(w, v["s1"], v["code1"], v["img"], [6, 13, 21, 29, 44], 29))]),
           plants=("resize_code",), at=("image_both",))
    sk.add(_a(text=f"Spike {v['s1']}: every failed resize logs the same error code after an origin timeout. Comparing "
                   f"with spike {v['s2']}."))
    sk.add(_a(calls=[_read(f"/var/log/thumbs/spike-{v['s2']}.log", _th_spike_log(w, v["s2"], v["code2"], v["img"], [3, 17, 36], 17))]),
           plants=("image_both",), at=("decoy:code2",))
    sk.add(_a(text=f"Spike {v['s2']} has a different code, but one image failed in both spikes; a huge original, "
                   f"probably. Noting it."))
    cfg = "\n".join(["cache:", f"  cache_ttl_s: {v['ttl_old']}", f"  stale_while_revalidate_s: {v['swr']}",
                     f"  max_entries: {w.ch([50000, 200000])}", "origin:", f"  origin_max_conns: {v['initial']}",
                     f"  timeout_ms: {w.ch([5000, 8000])}", "resize:", f"  jpeg_quality: {w.ch([78, 82])}",
                     "  formats: [avif, webp, jpeg]"])
    sk.add(_a(thinking="Check the thumbs config.", calls=[_read("config/thumbs.yaml", cfg)]),
           plants=("swr",), at=("origin_initial", "cache_ttl"))
    sk.add(_a(text="Cached thumbnails expire quickly and the stale window is short, so every peak refetches a lot "
                   "from the origin at once."))
    sk.add(_u(f"Keep libvips at {v['vips']} at most; {v['vips_next']} changes the AVIF encoder defaults."),
           plants=("vips_cap",), at=("decoy:vips_next",))
    sk.gap(0.6)
    sk.add(_a(thinking=f"{v['author']} wrote the TTL change, the PR was theirs; they'll know why it was lowered. "
                       f"Let me find the commit."), plants=("ttl_author",))
    sk.add(_a(calls=[_bash("git log --oneline -12 -- src/cache/",
                           w.oneline(12, {w.i(0, 5): f"{v['hash']} lower thumbnail cache TTL to 5 minutes"}))]),
           plants=("ttl_hash",))
    sk.add(_a(text=f"With the short TTL, misses stampede the origin: we allow {v['initial']} concurrent origin "
                   f"connections per pod and the origin starts queueing well before that. I'll try other limits."),
           plants=("origin_initial",))
    sk.add(_a(calls=[_bash(f"./scripts/replay-peak.sh --origin-conns {v['tried']} --rps 400",
                           f"origin_conns  timeouts  origin_queue_p95_ms\n{v['tried']:<13} {w.i(5000, 9000):<9} {w.i(9000, 20000)}")]),
           at=("origin_tried",))
    sk.add(_a(text=f"{v['tried']} is much worse: the origin collapses. Going down to {v['final']}."),
           plants=("origin_tried",), at=("origin_final",))
    sk.add(_a(calls=[_bash(f"./scripts/replay-peak.sh --origin-conns {v['final']} --rps 400",
                           f"origin_conns  timeouts  origin_queue_p95_ms\n{v['final']:<13} {w.i(0, 30):<9} {w.i(100, 400)}")]),
           at=("origin_final",))
    sk.add(_a(text=f"{v['final']} connections per pod: almost no timeouts. Setting it.",
              calls=[_edit("config/thumbs.yaml", [(f"  origin_max_conns: {v['initial']}", f"  origin_max_conns: {v['final']}")])]),
           plants=("origin_final",), at=("origin_initial",))
    sk.add(_u("And the TTL? What would you set?"))
    sk.add(_a(text=f"I'd leave it at {v['ttl_old']} seconds so edits show up quickly."), at=("cache_ttl",))
    sk.add(_u(f"Make it {v['ttl_new']} seconds; we purge explicitly on edit anyway."), plants=("cache_ttl",))
    sk.add(_a(thinking=f"The README says thumbnails come from the thumbs-shared-assets bucket, but the origin fetcher "
                       f"config I saw earlier reads {v['hidden']}; the README is stale.",
              calls=[_edit("config/thumbs.yaml", [(f"  cache_ttl_s: {v['ttl_old']}", f"  cache_ttl_s: {v['ttl_new']}")])],
              text="TTL updated."), plants=("origin_bucket",), at=("cache_ttl", "decoy:readme_bucket"))
    names = TH_TESTS + w.test_names(8)
    w.rng.shuffle(names)
    suite = "test/resize/variants.test.ts"
    sk.add(_a(calls=[_bash("pnpm vitest run test/resize --reporter verbose", w.vitest_run(suite, names, [v["flaky"], v["real"]]))]),
           plants=("flaky_test",), at=("decoy:real_failure",))
    sk.add(_a(text="Two failures; rerunning."))
    sk.add(_a(calls=[_bash("pnpm vitest run test/resize --reporter verbose", w.vitest_run(suite, names, [v["real"]]))]),
           at=("flaky_test", "decoy:real_failure"))
    sk.add(_a(text="One passes on the rerun, so it's flaky; the other fails both times and needs a fix. Can I use the "
                   "admin credentials for the origin to test?"))
    sk.add(_u(f"No, use the read-only key at {v['key']}."), plants=("origin_key",))
    sk.add(_u(f"During the peak, watch the \"{v['panel']}\" panel. And ask {v['contact']} about the image cache "
              f"dashboards, not {v['on_leave']}, who is on holiday."),
           plants=("dash_panel", "cache_contact"), at=("decoy:on_leave",))
    sk.add(_a(text="Will do."))
    sk.gap(0.9)


def _th_episodes(sk: Skel, v: dict, r: int) -> None:
    w = sk.w
    if r == 1:
        sk.add(_u(f"Spikes {v['s3']} and {v['s4']} on the avatar path failed too. Same thing?"))
        sk.add(_a(calls=[_bash(f"tail -n 50 /var/log/thumbs/avatar-spike-{v['s3']}.log",
                               _th_spike_log(w, v["s3"], v["code3"], v["img3"], [8, 31], 8))]), at=("rival:code3", "rival:img3"))
        sk.add(_a(calls=[_bash(f"tail -n 50 /var/log/thumbs/avatar-spike-{v['s4']}.log",
                               _th_spike_log(w, v["s4"], v["code4"], v["img3"], [12, 40], 40))]), at=("rival:code4", "rival:img3"))
        sk.add(_a(text=f"Avatar spike {v['s3']} logs {v['code3']} and {v['s4']} logs {v['code4']}; {v['img3']} failed in "
                       f"both. Avatars go through a different fetcher; separate issue."),
               at=("rival:code3", "rival:code4", "rival:img3"))
    elif r == 2:
        sk.add(_u("While you're at it, the avatar resize pool has its own connection limit. Tune it too?"))
        sk.add(_a(text=f"The avatar pool allows {v['a']} connections. Trying {v['b']}.",
                  calls=[_bash(f"./scripts/replay-peak.sh --avatar-conns {v['b']}",
                               f"avatar_conns  timeouts\n{v['b']:<13} {w.i(2000, 5000)}")]), at=("rival:conns2",))
        sk.add(_a(text=f"{v['b']} is worse for avatars. {v['c']} is clean; using it."), at=("rival:conns2",))
        sk.add(_u(f"Also keep sharp at {v['pin2']}; the next minor changes the AVIF encoder. The avatar service's local "
                  f"compose port is {v['port2']}, by the way."), at=("rival:pin2", "rival:port2"))
        sk.add(_a(text=f"Pinned sharp to {v['pin2']}."), at=("rival:pin2",))
    else:
        sk.add(_u(f"Two more things: the avatar edge cache is {v['host2']} now. And ask {v['contact2']} about the CDN "
                  f"bill, not {v['leave2']}; watch \"{v['panel2']}\" for avatars."),
               at=("rival:host2", "rival:contact2", "rival:leave2", "rival:panel2"))
        sk.add(_a(thinking=f"The README says avatars come from the shared bucket, but the avatar fetcher actually reads "
                           f"{v['hidden2']}.", text="Noted."), at=("rival:hidden2",))


_register("thumbs", "repro-first", _th_values, _th_plant, _th_episodes)
