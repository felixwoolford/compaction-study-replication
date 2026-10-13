"""EC ("challenge") stimulus plans: a debugging story with non-salient details.

Each session replays the same story (a failing nightly export job) with values
drawn from the session seed. The details are the kind a summary tends to lose or
misread: a code read once in a log, a request id that appears in two logs, a
flaky test identified only by comparing two runs, a number mentioned in passing,
values that change twice. The compaction rounds continue the work with many
look-alike codes, hashes, hosts and names, so the details compete instead of
standing out against empty filler.

Questions are the same in every session; values, run numbers and round work
vary. Probed values never recur in the round work (asserted).
"""

from __future__ import annotations

import random
import re

from .facts import Fact, value_regex
from .filler import step_chars
from .stimuli import Plan

# Story-only vocabularies (disjoint from the round-work vocabularies below).
STORY_TESTS = ["test_rollover_at_month_end", "test_currency_rounding_eur", "test_invoice_number_gap",
               "test_vat_split_payment", "test_refund_rounding_chf", "test_ledger_close_period"]
CONTACTS = ["Marta", "Olu", "Brigid", "Teodor", "Amara", "Yusuf", "Greta", "Haruto"]
AUTHORS = ["rkowalczyk", "abernard", "nokafor", "ylindqvist", "cmoreau", "hpetrov"]
CA_DIRS = ["ledger", "billing", "finops", "payments"]
CA_FILES = ["ca-chain.pem", "internal-ca.pem", "root-bundle.crt"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Round-work vocabularies.
FIRST = ["Priya", "Tomasz", "Ines", "Kwame", "Lena", "Ravi", "Sofia", "Jonah", "Mei", "Dario", "Hana", "Felix"]
LOGINS = ["pnair", "tbrzezinski", "iferreira", "kmensah", "lvogel", "rsharma", "sgarcia", "jcohen", "mzhang", "dross"]
TESTS = ["test_invoice_totals", "test_vat_reverse_charge", "test_credit_note_link", "test_pdf_render_fonts",
         "test_export_manifest_order", "test_retry_budget", "test_upload_multipart", "test_ledger_snapshot",
         "test_currency_rounding_gbp", "test_currency_rounding_usd", "test_partial_refund", "test_tax_id_format",
         "test_timezone_cutoff", "test_duplicate_invoice_guard", "test_s3_key_layout", "test_batch_boundary",
         "test_rollover_at_year_end", "test_rollover_at_week_end", "test_dunning_schedule", "test_csv_escape"]
COMMITS = ["bump pdf renderer", "fix VAT rounding for credit notes", "add export manifest checksum",
           "refactor uploader session handling", "tune ledger writer flush interval", "add dunning email template",
           "switch uploader to multipart for >8MB", "log request ids on retry", "pin tzdata",
           "drop python 3.10 support", "parallelise PDF rendering", "add idempotency keys to uploads",
           "move S3 client to shared pool", "raise ledger snapshot retention", "fix CSV quoting for notes"]
HOSTS = [f"ledger-prod-{i}" for i in range(1, 6)] + ["ledger-batch-1", "ledger-batch-2"]
ASIDES = ["btw {h} is being drained this week.", "{n} owns the PDF renderer if you need someone.",
          "ignore alerts from {h}, it's a known disk issue.", "{n} said the invoice backlog is fine.",
          "the {h} box got a new kernel yesterday.", "ask {n} if anything looks off in billing."]
HEX = "0123456789abcdef"

LURES = [("redis_policy", "What Redis eviction policy did we set?"),
         ("admin_port", "What port does the admin console listen on?"),
         ("kafka_topic", "Which Kafka topic do the export events go to?")]


def _u(text, uid=None):
    d = {"kind": "user", "text": text}
    if uid:
        d["fact_uid"] = uid
    return d


def _a(text=None, thinking=None, calls=None, uid=None):
    d = {"kind": "assistant", "thinking": thinking, "text": text, "calls": calls or []}
    if uid:
        d["fact_uid"] = uid
    return d


def _c(name, args, result):
    return {"name": name, "args": args, "result": result, "is_error": False}


class _Gen:
    def __init__(self, rng: random.Random, forbidden_codes: set[str]):
        self.rng = rng
        self.codes = []
        while len(self.codes) < 30:
            c = f"E-{rng.randint(1000, 9999)}"
            if c not in forbidden_codes and c not in self.codes:
                self.codes.append(c)

    def hexs(self, n):
        return "".join(self.rng.choice(HEX) for _ in range(n))

    def log_block(self, run, n, fail_code=None, special=None, n_fail=6):
        rng, lines, fails = self.rng, [], 0
        special = special or {}
        for i in range(n):
            req, st, code = f"req-{self.hexs(6)}", "ok", "-"
            if i in special:
                st, code, req = "fail", fail_code, special[i]
            elif fail_code and fails < n_fail and rng.random() < 0.12:
                st, code, fails = "fail", fail_code, fails + 1
            elif not fail_code and rng.random() < 0.08:
                st, code = rng.choice(["retry", "fail"]), rng.choice(self.codes)
            elif rng.random() < 0.06:
                st = "retry"
            lines.append(f"2026-09-{rng.randint(10, 27)}T02:{rng.randint(10, 59)}:{rng.randint(10, 59)}Z "
                         f"ledgerd[{rng.randint(2000, 9000)}] run={run} req={req} invoice=INV-{rng.randint(10000, 99999)} "
                         f"status={st} code={code} latency_ms={rng.randint(120, 4800)}")
        return "\n".join(lines)

    def gitlog(self, n, special=None):
        out = []
        for i in range(n):
            out.append(special[1] if special and i == special[0] else f"{self.hexs(7)} {self.rng.choice(COMMITS)}")
        return "\n".join(out)

    def env_file(self, special=None):
        keys = ["LEDGERD_EXPORT_CONCURRENCY", "LEDGERD_UPLOAD_TIMEOUT_S", "LEDGERD_RENDER_WORKERS",
                "LEDGERD_RETRY_MAX", "LEDGERD_SNAPSHOT_EVERY_MIN", "LEDGERD_LOG_LEVEL"]
        lines = ["# ledgerd production environment"]
        for k in keys:
            v = self.rng.choice(["info", "debug"]) if k.endswith("LEVEL") else str(
                self.rng.choice([3, 4, 11, 12, 16, 20, 30, 60, 90]))
            lines.append(f"{k}={v}")
        if special:
            lines.insert(self.rng.randint(1, len(lines)), special)
        return "\n".join(lines)

    def yaml_cfg(self, name, special=None):
        keys = {"healthcheck_interval_s": [5, 10], "drain_timeout_s": [45, 60, 90], "connect_timeout_s": [3, 5, 8],
                "max_body_mb": [16, 32, 64], "keepalive_requests": [100, 500, 1000], "listen_port": [8443, 9443, 7443]}
        lines = [f"# {name}", "upstream:", f"  name: ledgerd-{self.rng.choice(['api', 'export', 'render'])}"]
        items = [f"  {k}: {self.rng.choice(v)}" for k, v in keys.items()]
        if special:
            items.insert(self.rng.randint(0, len(items)), special)
        return "\n".join(lines + items)

    def pytest_out(self, failing, n_pass):
        lines = [f"tests/test_export.py::{t} FAILED" for t in failing]
        tail = f"{len(failing)} failed, {n_pass} passed" if failing else f"{n_pass} passed"
        return "\n".join(lines + [f"{tail} in {self.rng.uniform(4, 30):.2f}s"])

    def round_unit(self, run):
        rng, k = self.rng, self.rng.random()
        aside = lambda: rng.choice(ASIDES).format(h=rng.choice(HOSTS), n=rng.choice(FIRST))  # noqa: E731
        if k < 0.3:
            c1, c2 = rng.sample(self.codes, 2)
            return [_u(f"Check run {run}." + (" " + aside() if rng.random() < 0.4 else "")),
                    _a(thinking=f"Read run {run}.", calls=[_c("read", {"path": f"/var/log/ledgerd/run-0{run}.log"},
                                                             self.log_block(run, rng.randint(30, 60)))]),
                    _a(text=f"Run {run}: a handful of retries, failures with {c1} and {c2}; nothing new.")]
        if k < 0.45:
            h = self.hexs(7)
            return [_u("What changed in the renderer recently?"),
                    _a(thinking="git log for the renderer.", calls=[_c("bash", {"command": "git log --oneline -12 -- renderer/"},
                                                                      self.gitlog(12, (3, f"{h} {rng.choice(COMMITS)}")))]),
                    _a(text=f"Mostly small fixes; {h} is the only one touching the export path.")]
        if k < 0.6:
            fail = rng.sample(TESTS, rng.randint(0, 2))
            return [_u(f"Run the {rng.choice(['render', 'tax', 'refund', 'dunning'])} tests."),
                    _a(thinking="Run tests.", calls=[_c("bash", {"command": "pytest -q"}, self.pytest_out(fail, rng.randint(30, 60)))]),
                    _a(text=(f"{len(fail)} failing: {', '.join(fail)}. Known, tracked separately." if fail else "All green."))]
        if k < 0.72:
            n = rng.choice(["deploy/render.yaml", "deploy/api-lb.yaml", "deploy/batch.yaml"])
            return [_u(f"Check {n} for anything odd."),
                    _a(thinking=f"Read {n}.", calls=[_c("read", {"path": n}, self.yaml_cfg(n, f"  idle_timeout_s: {rng.choice([30, 45, 60, 90])}"))]),
                    _a(text="Nothing odd; timeouts are consistent with the other services.")]
        if k < 0.82:
            p = rng.choice([8, 32, 64, 96])
            return [_u(f"Try the render workers at {p}."),
                    _a(thinking="Load test the renderer.", calls=[_c("bash", {"command": f"ledgerctl loadtest --render-workers {p}"},
                                                                    f"p50 {rng.randint(150, 400)}ms  p95 {rng.randint(700, 2900)}ms\nerrors {rng.randint(0, 20)}/4800")]),
                    _a(text=f"Render workers at {p}: within budget.")]
        if k < 0.9:
            return [_u("Show me the env for the batch hosts."),
                    _a(thinking="Read env.", calls=[_c("read", {"path": "deploy/batch.env"}, self.env_file())]),
                    _a(text="Batch env matches prod except for log level.")]
        who = rng.choice(LOGINS)
        blame = "\n".join(f"{self.hexs(7)} ({rng.choice(LOGINS)} 2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}) line {i}"
                          for i in range(40, 60))
        return [_u(f"Who touched the dunning scheduler last? {aside()}"),
                _a(thinking=f"Blame says {who}.", calls=[_c("bash", {"command": "git blame -L 40,60 dunning/schedule.py"}, blame)]),
                _a(text=f"Mostly {who}, a few lines by others.")]


def _story(rng: random.Random, g: _Gen, sid: str) -> tuple[list[Fact], list[dict], set[str], int]:
    """The plant phase: facts, the steps that carry them, the error codes used and the last run number."""
    run1 = rng.randint(700, 899)
    run2 = run1 + 1
    code1, code2 = rng.sample([f"E-{n}" for n in range(1000, 10000)], 2)
    req = f"req-{g.hexs(6)}"
    major, minor = rng.randint(3, 6), rng.randint(10, 29)
    client = f"{major}.{minor}.{rng.randint(1, 9)}"
    bhash = g.hexs(7)
    lb_idle = rng.choice([12, 15, 18, 20, 25, 27])
    workers = rng.choice([40, 48, 56, 64])
    pool_initial = rng.randint(10, 16)
    pool_tried = rng.randint(34, 44)
    pool_final = rng.randint(20, 28)
    batch_old = rng.choice([700, 750, 800])
    batch = rng.choice([550, 600, 650])
    k = rng.randint(2, 4)
    s3 = f"exports/v{k}-legacy/"
    flaky, real = rng.sample(STORY_TESTS, 2)
    ca = f"/etc/pki/{rng.choice(CA_DIRS)}/{rng.choice(CA_FILES)}"
    conc = rng.randint(5, 9)
    contact, on_leave = rng.sample(CONTACTS, 2)
    day = rng.choice(DAYS)
    freeze = f"{rng.randint(13, 17)}:{rng.choice(['00', '15', '30', '45'])}"
    author = rng.choice(AUTHORS)
    stg = rng.randint(3, 9)
    host = f"ledger-stg-{stg:02d}"

    facts: dict[str, Fact] = {}

    def fact(key, source, value, question, old=None, alts=None):
        acc = "|".join(value_regex(v) for v in [value] + (alts or []))
        facts[key] = Fact(uid=f"{sid}:{key}", slot=key, key=key, kind="challenge", source=source, position="early",
                          value=value, value_old=old, question=question, accept=acc,
                          intrusion=value_regex(old) if old else None)

    fact("staging_host", "user", host, "Which host is the staging box now?", alts=[f"stg-{stg:02d}"])
    fact("code_first", "tool_output", code1, f"What error code did the failed uploads in nightly run {run1} log?")
    fact("req_both", "tool_output", req, f"Which request id failed in both run {run1} and run {run2}?")
    fact("client_max", "user", client, "What is the newest version of the HTTP client library we are allowed to use?")
    fact("backoff_hash", "tool_output", bhash, "What is the short hash of the commit that switched retry backoff to jittered?")
    fact("lb_idle", "tool_output", str(lb_idle), "What is the load balancer's idle timeout, in seconds?",
         alts=[f"{lb_idle}s", f"{lb_idle} s"])
    fact("pool_initial", "assistant_text", str(pool_initial), "How many connections did the pool have when we found it was exhausted?")
    fact("pool_tried", "assistant_text", str(pool_tried), "Which pool size did we try first that made things worse?")
    fact("pool_final", "superseded", str(pool_final), "What pool size did we finally settle on?", old=str(pool_tried))
    fact("batch_rows", "superseded", str(batch), "How many rows per export batch did we end up using?", old=str(batch_old))
    fact("s3_prefix", "thinking", s3, "What S3 prefix does prod actually use for exports?", alts=[s3.rstrip("/")])
    fact("flaky_test", "tool_output", flaky, "Which test turned out to be flaky?")
    fact("ca_bundle", "user", ca, "Where is the internal CA bundle we should use?")
    fact("upload_conc", "tool_output", str(conc), "What value is LEDGERD_UPLOAD_CONCURRENCY set to in the env file?")
    fact("dash_contact", "user", contact, "Who should we contact about the dashboards?")
    fact("freeze_time", "user", freeze, f"At what time on {day} is the freeze?")
    fact("backoff_author", "thinking", author, "Who wrote the retry backoff change?")
    for key, q in LURES:
        facts[key] = Fact(uid=f"{sid}:{key}", slot=key, key=key, kind="challenge", source="lure", position="none",
                          value=None, question=q)
    uid = {k: f.uid for k, f in facts.items()}

    s = [
        _u(f"The nightly ledgerd export has been failing since Monday. Can you take a look? Logs are in "
           f"/var/log/ledgerd/. Heads-up, the staging box is {host} now, ledger-stg-{stg - 1:02d} was decommissioned "
           f"last week.", uid["staging_host"]),
        _a(thinking="Start by listing the run logs.", calls=[_c("bash", {"command": "ls /var/log/ledgerd/"},
           "\n".join(f"run-0{r}.log" for r in range(run1 - 7, run2 + 3)))]),
        _a(text=f"Runs {run1 - 7}-{run1 - 1} look clean by size; {run1} is the first one that's much larger. Reading it."),
        _a(thinking="Read the first failing run.", calls=[_c("read", {"path": f"/var/log/ledgerd/run-0{run1}.log"},
           g.log_block(run1, 70, code1, {33: req}, 8))], uid=uid["code_first"]),
        _a(text=f"Run {run1} has several failed uploads, all with the same error code, all after at least one retry. "
                f"Let me compare with {run2}."),
        _a(thinking="Compare with the next run.", calls=[_c("read", {"path": f"/var/log/ledgerd/run-0{run2}.log"},
           g.log_block(run2, 70, code2, {51: req}, 7))], uid=uid["req_both"]),
        _a(text=f"{run2} fails with a different code, but one invoice request failed in both runs. Could be a symptom "
                f"rather than the cause; I'll keep it in mind."),
        _u(f"Could it be the HTTP client upgrade? We moved to {client} last month. Don't go past {client} by the way, "
           f"{major}.{minor + 1} breaks our proxy.", uid["client_max"]),
        _a(thinking="Check what changed right before Monday.", calls=[_c("bash", {"command": "git log --oneline -14"},
           g.gitlog(14, (rng.randint(2, 9), f"{bhash} switch retry backoff to jittered (cap 30s)")))], uid=uid["backoff_hash"]),
        _a(text="The client bump is a month old, so probably not it. The retry-backoff change landed Sunday night, "
                "right before the failures started, and it caps the backoff at 30s. Let me look at the load balancer."),
        _a(thinking="If the backoff cap exceeds the LB idle timeout, reused connections would be dead.",
           calls=[_c("read", {"path": "deploy/lb.yaml"}, g.yaml_cfg("deploy/lb.yaml", f"  idle_timeout_s: {lb_idle}"))],
           uid=uid["lb_idle"]),
        _a(text="That's the mechanism: the new backoff can wait longer than the load balancer keeps idle "
                "connections, so retries go out on connections that are already dead."),
        _u("Makes sense. Anything else going on?"),
        _a(text=f"Yes, the timeouts also come from the connection pool: {workers} workers share a pool of {pool_initial} "
                f"connections, so it is exhausted during the export peak. I'll try a bigger pool.", uid=uid["pool_initial"]),
        _a(thinking=f"Run the load test with pool {pool_tried}.",
           calls=[_c("bash", {"command": f"ledgerctl loadtest --pool {pool_tried} --workers {workers}"},
                     f"p50 410ms  p95 3120ms  p99 5210ms\nlock_wait_ms(avg) 880\nerrors 37/4800")]),
        _a(text=f"Pool {pool_tried} is worse: the ledger writer serialises on a lock and contention shoots up. "
                f"Reverting and trying {pool_final}.", uid=uid["pool_tried"]),
        _a(thinking=f"Load test with pool {pool_final}.",
           calls=[_c("bash", {"command": f"ledgerctl loadtest --pool {pool_final} --workers {workers}"},
                     "p50 220ms  p95 960ms  p99 1480ms\nlock_wait_ms(avg) 95\nerrors 0/4800")]),
        _a(text=f"{pool_final} is the sweet spot: no errors and p95 under a second. Keeping pool_size: {pool_final}.",
           uid=uid["pool_final"]),
        _u("Good. What are you setting the export batch size to?"),
        _a(text=f"{batch_old} rows per batch; that keeps each upload under the multipart threshold."),
        _u(f"Make it {batch}, the downstream parser chokes somewhere above {batch + 50}.", uid["batch_rows"]),
        _a(thinking=f"Update the uploader config. The README says exports/v{k}/ but the deploy manifest I read earlier "
                    f"uses {s3} in prod; the README is stale, so don't trust it.",
           calls=[_c("edit", {"path": "config/uploader.yaml"}, "Applied 2 edits.")], uid=uid["s3_prefix"]),
        _a(text=f"Batch size set to {batch} and the uploader config updated. Running the export tests."),
        _a(thinking="Run the tests.", calls=[_c("bash", {"command": "pytest tests/test_export.py -q"},
           g.pytest_out(sorted([flaky, real]), 41))], uid=uid["flaky_test"]),
        _a(text="Two failures. I suspect one of them is timing-dependent; rerunning before I dig in."),
        _a(thinking="Rerun.", calls=[_c("bash", {"command": "pytest tests/test_export.py -q"}, g.pytest_out([real], 42))]),
        _a(text="One of the two passes on rerun, so that one is flaky; the other is a real failure that I'll fix "
                "separately. Should I run the uploader with --insecure to rule out TLS?"),
        _u(f"No, never disable TLS verification. Use the internal CA bundle at {ca}.", uid["ca_bundle"]),
        _a(thinking="Check the env file for the uploader settings.", calls=[_c("read", {"path": "deploy/ledgerd.env"},
           g.env_file(f"# parallel uploads to S3\nLEDGERD_UPLOAD_CONCURRENCY={conc}"))], uid=uid["upload_conc"]),
        _a(text="Pointed the uploader at the internal CA bundle. Env settings look sane."),
        _u(f"Thanks. Ping {contact} about the dashboards, not {on_leave}, {on_leave} is on leave. And we need all this "
           f"fixed before {day}'s {freeze} freeze.", uid["dash_contact"]),
        _a(thinking=f"{author} wrote the backoff change (I saw it in git blame); worth looping them in on the fix.",
           text="Will do.", uid=uid["backoff_author"]),
    ]
    # freeze_time is planted in the same user message as dash_contact
    return list(facts.values()), s, {code1, code2}, run2


def build_challenge_plan(scenario: dict, index: int, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    exp = scenario["experiment"]
    seed = scenario["seed"] * 1000 + index
    rng = random.Random(seed)
    sid = f"{exp.lower()}-s{index:03d}"
    g = _Gen(random.Random(seed + 17), forbidden_codes=set())
    facts, story, story_codes, last_run = _story(rng, g, sid)
    g = _Gen(g.rng, forbidden_codes=story_codes)  # round codes never equal the story's codes
    target = int(scenario["round_filler_x_keep"] * keep_recent_tokens * chars_per_token)
    phases = [{"name": "plant", "steps": story}]
    run = last_run + 10
    probed = [re.compile(value_regex(f.value)) for f in facts if f.value and len(f.value) > 2]
    for r in range(1, scenario["rounds"] + 1):
        steps, tot = [], 0
        while tot < target:
            u = g.round_unit(run)
            if any(p.search(steps_text(u)) for p in probed):
                continue  # a look-alike may compete with a probed value, never equal it
            run += 1
            steps += u
            tot += sum(step_chars(x) for x in u)
        phases.append({"name": f"round{r}", "steps": steps})
    for st in story:
        st["plant"] = True
    order = [f.uid for f in facts]
    rng.shuffle(order)
    plan = Plan(exp, sid, seed, facts, phases, order)
    _check(plan)
    return plan


def steps_text(steps: list[dict]) -> str:
    """All text a step list puts into the session (user/assistant text, thinking, tool args and results)."""
    parts = []
    for st in steps:
        parts += [st.get("text") or "", st.get("thinking") or ""]
        for c in st.get("calls", []):
            parts += [" ".join(str(v) for v in c["args"].values()), c["result"]]
    return "\n".join(parts)


def _check(plan: Plan) -> None:
    """Every planted value is in the story; no value of 3+ chars recurs in the round work."""
    story = steps_text(plan.phases[0]["steps"])
    rounds = steps_text([st for ph in plan.phases[1:] for st in ph["steps"]])
    for f in plan.facts:
        if not f.value:
            continue
        if not re.search(value_regex(f.value), story):
            raise AssertionError(f"{plan.session_id}: {f.key} value {f.value} missing from the story")
        if len(f.value) > 2 and re.search(value_regex(f.value), rounds):
            raise AssertionError(f"{plan.session_id}: {f.key} value {f.value} recurs in the round work")


# =============================================================================
# Stories 2-5 (Stage 4 deviation, DEVIATIONS.md): same 17 kinds of detail as
# story 1, different incidents, vocabularies and round work. Story 1 above is
# untouched, so its plans are byte-identical to the frozen version.
# =============================================================================


class _DomainGen:
    """Look-alike round work for one story's domain."""

    def __init__(self, rng: random.Random, d: dict):
        self.rng, self.d = rng, d

    def hexs(self, n):
        return "".join(self.rng.choice(HEX) for _ in range(n))

    def code(self):
        return self.d["code"](self.rng, self)

    def log_block(self, run, n, fail_code=None, special=None, n_fail=6):
        rng, d, lines, fails = self.rng, self.d, [], 0
        special = special or {}
        for i in range(n):
            xid, st, code = d["xid"](rng, self), "ok", "-"
            if i in special:
                st, code, xid = "fail", fail_code, special[i]
            elif fail_code and fails < n_fail and rng.random() < 0.12:
                st, code, fails = "fail", fail_code, fails + 1
            elif not fail_code and rng.random() < 0.08:
                st, code = rng.choice(["retry", "fail"]), self.code()
            elif rng.random() < 0.06:
                st = "retry"
            lines.append(f"2026-09-{rng.randint(10, 27)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}Z "
                         f"{d['service']}[{rng.randint(2000, 9000)}] {d['unit']}={run} {d['xid_label']}={xid} "
                         f"status={st} {d['code_label']}={code} dur_ms={rng.randint(120, 4800)}")
        return "\n".join(lines)

    def gitlog(self, n, special=None):
        return "\n".join(special[1] if special and i == special[0] else f"{self.hexs(7)} {self.rng.choice(self.d['commits'])}"
                         for i in range(n))

    def env_file(self, special=None):
        lines = [f"# {self.d['service']} environment"]
        for k in self.d["env_keys"]:
            lines.append(f"{k}={self.rng.choice([3, 4, 11, 12, 16, 20, 30, 60, 90])}")
        if special:
            lines.insert(self.rng.randint(1, len(lines)), special)
        return "\n".join(lines)

    def yaml_cfg(self, name, special=None):
        lines = [f"# {name}", f"{self.d['yaml_root']}:"]
        items = [f"  {k}: {self.rng.choice(v)}" for k, v in self.d["yaml_keys"].items()]
        if special:
            items.insert(self.rng.randint(0, len(items)), special)
        return "\n".join(lines + items)

    def test_out(self, failing, n_pass, suite):
        lines = [f"{suite}::{t} FAILED" for t in failing]
        tail = f"{len(failing)} failed, {n_pass} passed" if failing else f"{n_pass} passed"
        return "\n".join(lines + [f"{tail} in {self.rng.uniform(4, 30):.2f}s"])

    def round_unit(self, run):
        rng, d, k = self.rng, self.d, self.rng.random()
        aside = lambda: rng.choice(d["asides"]).format(h=rng.choice(d["hosts"]), n=rng.choice(d["people"]))  # noqa: E731
        if k < 0.3:
            c1, c2 = self.code(), self.code()
            path = d["log_path"].format(n=run)
            return [_u(f"Check {d['unit']} {run}." + (" " + aside() if rng.random() < 0.4 else "")),
                    _a(thinking=f"Read {d['unit']} {run}.", calls=[_c("read", {"path": path}, self.log_block(run, rng.randint(30, 60)))]),
                    _a(text=f"{d['unit'].capitalize()} {run}: a few retries, failures with {c1} and {c2}; nothing new.")]
        if k < 0.45:
            h = self.hexs(7)
            return [_u(f"What changed in {rng.choice(d['areas'])} recently?"),
                    _a(thinking="Check git log.", calls=[_c("bash", {"command": "git log --oneline -12"},
                                                           self.gitlog(12, (3, f"{h} {rng.choice(d['commits'])}")))]),
                    _a(text=f"Mostly small fixes; {h} is the only one that looks relevant.")]
        if k < 0.6:
            fail = rng.sample(d["round_tests"], rng.randint(0, 2))
            return [_u(f"Run the {rng.choice(d['areas'])} tests."),
                    _a(thinking="Run tests.", calls=[_c("bash", {"command": d["test_cmd"]}, self.test_out(fail, rng.randint(30, 60), d["suite"]))]),
                    _a(text=(f"{len(fail)} failing: {', '.join(fail)}. Known, tracked separately." if fail else "All green."))]
        if k < 0.72:
            n = rng.choice(d["yaml_files"])
            key, vals = d["yaml_lookalike"]
            return [_u(f"Check {n} for anything odd."),
                    _a(thinking=f"Read {n}.", calls=[_c("read", {"path": n}, self.yaml_cfg(n, f"  {key}: {rng.choice(vals)}"))]),
                    _a(text="Nothing odd; values are consistent with the other services.")]
        if k < 0.82:
            p = rng.choice(d["bench_values"])
            return [_u(d["bench_ask"].format(p=p)),
                    _a(thinking="Run the benchmark.", calls=[_c("bash", {"command": d["bench_cmd"].format(p=p)},
                                                               f"p50 {rng.randint(150, 400)}ms  p95 {rng.randint(700, 2900)}ms\nerrors {rng.randint(0, 20)}/4800")]),
                    _a(text=f"At {p}: within budget.")]
        if k < 0.9:
            return [_u(f"Show me the env on {rng.choice(d['hosts'])}."),
                    _a(thinking="Read env.", calls=[_c("read", {"path": d["env_path"]}, self.env_file())]),
                    _a(text="Env matches prod except for log level.")]
        who = rng.choice(d["logins"])
        blame = "\n".join(f"{self.hexs(7)} ({rng.choice(d['logins'])} 2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}) line {i}"
                          for i in range(40, 60))
        return [_u(f"Who touched {rng.choice(d['areas'])} last? {aside()}"),
                _a(thinking=f"Blame says {who}.", calls=[_c("bash", {"command": f"git blame -L 40,60 {d['blame_file']}"}, blame)]),
                _a(text=f"Mostly {who}, a few lines by others.")]


def _fact_table(sid, spec, v):
    facts = {}

    def fact(key, source, value, question, old=None, alts=None):
        acc = "|".join(value_regex(x) for x in [value] + (alts or []))
        facts[key] = Fact(uid=f"{sid}:{key}", slot=key, key=key, kind="challenge", source=source, position="early",
                          value=value, value_old=old, question=question, accept=acc,
                          intrusion=value_regex(old) if old else None)

    q = spec["questions"]
    fact("staging_host", "user", v["host"], q["staging_host"])
    fact("code_first", "tool_output", v["code1"], q["code_first"].format(**v))
    fact("req_both", "tool_output", v["xid"], q["req_both"].format(**v))
    fact("client_max", "user", v["version"], q["client_max"])
    fact("backoff_hash", "tool_output", v["hash"], q["backoff_hash"])
    fact("lb_idle", "tool_output", str(v["cfgval"]), q["lb_idle"], alts=v.get("cfgval_alts"))
    fact("pool_initial", "assistant_text", str(v["initial"]), q["pool_initial"])
    fact("pool_tried", "assistant_text", str(v["tried"]), q["pool_tried"])
    fact("pool_final", "superseded", str(v["final"]), q["pool_final"], old=str(v["tried"]))
    fact("batch_rows", "superseded", str(v["sup_new"]), q["batch_rows"], old=str(v["sup_old"]))
    fact("s3_prefix", "thinking", v["hidden"], q["s3_prefix"], alts=v.get("hidden_alts"))
    fact("flaky_test", "tool_output", v["flaky"], q["flaky_test"])
    fact("ca_bundle", "user", v["path"], q["ca_bundle"])
    fact("upload_conc", "tool_output", str(v["envval"]), q["upload_conc"])
    fact("dash_contact", "user", v["contact"], q["dash_contact"])
    fact("freeze_time", "user", v["time"], q["freeze_time"].format(**v))
    fact("backoff_author", "thinking", v["author"], q["backoff_author"])
    for key, qq in spec["lures"]:
        facts[key] = Fact(uid=f"{sid}:{key}", slot=key, key=key, kind="challenge", source="lure", position="none",
                          value=None, question=qq)
    return facts


def _story_steps(spec, v, g, uid):
    """The shared plot skeleton; every sentence comes from the story's spec."""
    t, d = spec["text"], spec["domain"]
    f = lambda key: t[key].format(**v)  # noqa: E731
    flaky, real = v["flaky"], v["real"]
    return [
        _u(f("open"), uid["staging_host"]),
        _a(thinking=f("list_think"), calls=[_c("bash", {"command": f("list_cmd")}, v["listing"])]),
        _a(text=f("list_text")),
        _a(thinking=f("read1_think"), calls=[_c("read", {"path": d["log_path"].format(n=v["run1"])},
           g.log_block(v["run1"], 70, v["code1"], {33: v["xid"]}, 8))], uid=uid["code_first"]),
        _a(text=f("read1_text")),
        _a(thinking=f("read2_think"), calls=[_c("read", {"path": d["log_path"].format(n=v["run2"])},
           g.log_block(v["run2"], 70, v["code2"], {51: v["xid"]}, 7))], uid=uid["req_both"]),
        _a(text=f("read2_text")),
        _u(f("version"), uid["client_max"]),
        _a(thinking=f("git_think"), calls=[_c("bash", {"command": "git log --oneline -14"},
           g.gitlog(14, (v["hash_pos"], f"{v['hash']} {t['commit_msg']}")))], uid=uid["backoff_hash"]),
        _a(text=f("git_text")),
        _a(thinking=f("cfg_think"), calls=[_c("read", {"path": t["cfg_file"]}, g.yaml_cfg(t["cfg_file"], f"  {t['cfg_key']}: {v['cfgval']}"))],
           uid=uid["lb_idle"]),
        _a(text=f("cfg_text")),
        _u(f("more")),
        _a(text=f("initial"), uid=uid["pool_initial"]),
        _a(thinking=f("tried_think"), calls=[_c("bash", {"command": f("tried_cmd")}, f("tried_out"))]),
        _a(text=f("tried_text"), uid=uid["pool_tried"]),
        _a(thinking=f("final_think"), calls=[_c("bash", {"command": f("final_cmd")}, f("final_out"))]),
        _a(text=f("final_text"), uid=uid["pool_final"]),
        _u(f("sup_ask")),
        _a(text=f("sup_old")),
        _u(f("sup_new"), uid["batch_rows"]),
        _a(thinking=f("hidden_think"), calls=[_c("edit", {"path": t["edit_file"]}, "Applied 2 edits.")], uid=uid["s3_prefix"]),
        _a(text=f("edit_text")),
        _a(thinking="Run the tests.", calls=[_c("bash", {"command": d["test_cmd"]}, g.test_out(sorted([flaky, real]), 41, d["suite"]))],
           uid=uid["flaky_test"]),
        _a(text=f("tests1_text")),
        _a(thinking="Rerun.", calls=[_c("bash", {"command": d["test_cmd"]}, g.test_out([real], 42, d["suite"]))]),
        _a(text=f("tests2_text")),
        _u(f("path"), uid["ca_bundle"]),
        _a(thinking=f("env_think"), calls=[_c("read", {"path": d["env_path"]}, g.env_file(f"{t['env_comment']}\n{t['env_key']}={v['envval']}"))],
           uid=uid["upload_conc"]),
        _a(text=f("env_text")),
        _u(f("people"), uid["dash_contact"]),
        _a(thinking=f("author_think"), text="Will do.", uid=uid["backoff_author"]),
    ]


def _hex(rng, n):
    return "".join(rng.choice(HEX) for _ in range(n))


def _base_values(rng, spec):
    """Values every story draws the same way; the spec's 'values' adds the rest."""
    v = {"run1": rng.randint(300, 899)}
    v["run2"] = v["run1"] + 1
    v["hash"] = _hex(rng, 7)
    v["hash_pos"] = rng.randint(2, 9)
    v["flaky"], v["real"] = rng.sample(spec["story_tests"], 2)
    v["contact"], v["on_leave"] = rng.sample(spec["contacts"], 2)
    v["day"] = rng.choice(DAYS)
    v["time"] = f"{rng.randint(6, 18):02d}:{rng.choice(['05', '20', '35', '50'])}"
    v["author"] = rng.choice(spec["authors"])
    v.update(spec["values"](rng))
    return v


STORY_SPECS: dict[str, dict] = {}


def _register(name, spec):
    STORY_SPECS[name] = spec


# ---------------------------------------------------------------- story 2: CI
_register("ci", {
    "story_tests": ["test_cache_restore_order", "test_lockfile_hash", "test_artifact_upload_retry",
                    "test_shard_balance", "test_matrix_expand", "test_env_redaction"],
    "contacts": ["Noor", "Pavel", "Leocadia", "Tunde", "Signe", "Rafael", "Aiko", "Bram"],
    "authors": ["jdelacroix", "mokonkwo", "tvasquez", "elindgren", "rbianchi", "skowalski"],
    "values": lambda rng: {
        "n": (n := rng.randint(3, 9)), "host": f"mac-ci-{n:02d}", "prev": f"mac-ci-{n - 1:02d}",
        "code1": f"v3-deps-{_hex(rng, 8)}", "code2": f"v3-deps-{_hex(rng, 8)}", "xid": f"agent-{_hex(rng, 6)}",
        "maj": (maj := rng.randint(18, 22)), "minr": (mi := rng.randint(4, 14)),
        "version": f"{maj}.{mi}.{rng.randint(1, 9)}", "nextv": f"{maj}.{mi + 1}",
        "cfgval": rng.choice([23, 27, 33, 37, 41]), "cores": rng.choice([8, 12]),
        "initial": rng.choice([24, 28, 32]), "tried": rng.choice([2, 3]), "final": rng.choice([6, 7, 9]),
        "sup_old": 3, "sup_new": rng.choice([1, 2]),
        "region": (reg := rng.choice(["eu", "us", "ap"])), "hidden": f"gs://atlas-ci-cache-{reg}-{rng.randint(2, 9)}/",
        "path": f"/etc/ci/keys/atlas-deploy-{rng.choice(['ed25519', 'ecdsa', 'rsa4096'])}",
        "envval": rng.randint(5, 9)},
    "questions": {
        "staging_host": "Which macOS runner host are we using now?",
        "code_first": "Which cache key did build {run1} miss on?",
        "req_both": "Which CI agent ran the failing job in both build {run1} and build {run2}?",
        "client_max": "What is the newest Node version we are allowed to use?",
        "backoff_hash": "What is the short hash of the commit that enabled test sharding?",
        "lb_idle": "What is the test step's timeout in minutes in the pipeline config?",
        "pool_initial": "How many jest workers were we running when we found the runners oversubscribed?",
        "pool_tried": "Which jest worker count did we try first that made the build too slow?",
        "pool_final": "What jest worker count did we finally settle on?",
        "batch_rows": "How many automatic retries per test did we end up allowing?",
        "s3_prefix": "Which bucket do the CI cache artifacts actually upload to?",
        "flaky_test": "Which test turned out to be flaky?",
        "ca_bundle": "Which deploy key should the pipeline use?",
        "upload_conc": "What value is TURBO_CONCURRENCY set to in the CI env file?",
        "dash_contact": "Who should we ask about the runner budget?",
        "freeze_time": "At what time on {day} is the release branch cut?",
        "backoff_author": "Who wrote the test sharding change?"},
    "lures": [("docker_registry", "Which Docker registry mirror did we switch to?"),
              ("coverage_floor", "What minimum coverage percentage did we set?"),
              ("slack_channel", "Which Slack channel do build alerts go to?")],
    "text": {
        "open": "CI on the atlas monorepo has been timing out since the weekend. Can you dig in? Build logs are under /var/log/ci/. Heads-up, the macOS runner is {host} now, {prev} was retired.",
        "list_think": "List the recent build logs.", "list_cmd": "ls /var/log/ci/",
        "list_text": "Builds before {run1} are normal size; {run1} is the first long one. Reading it.",
        "read1_think": "Read the first slow build.",
        "read1_text": "Build {run1}: every failing job missed the same cache key and fell back to a cold install. Let me compare with {run2}.",
        "read2_think": "Compare with the next build.",
        "read2_text": "{run2} misses a different key, but the same agent ran a failing job in both builds. Might be a symptom; noting it.",
        "version": "Could it be the Node bump? We went to {version} recently. Don't go past {version} by the way, {nextv} breaks our native addon.",
        "git_think": "What changed right before the weekend?",
        "git_text": "The Node bump is older than the failures. The sharding change landed Friday evening, right before things went bad. Checking the pipeline config.",
        "cfg_think": "If shards got slower, the step timeout would start firing.",
        "cfg_text": "That fits: with sharding, each shard does a cold install after the cache miss and runs into the step timeout.",
        "more": "OK. Anything else?",
        "initial": "Yes, the runners are oversubscribed: we start {initial} jest workers on {cores}-core machines. I'll try fewer.",
        "tried_think": "Try {tried} workers.", "tried_cmd": "ci-bench --jest-workers {tried}",
        "tried_out": "wall 41m12s\nshards timed out 0/6", "tried_text": "{tried} workers is far too slow: 41 minutes per build. Trying {final}.",
        "final_think": "Try {final} workers.", "final_cmd": "ci-bench --jest-workers {final}",
        "final_out": "wall 14m03s\nshards timed out 0/6", "final_text": "{final} workers: 14 minutes and no timeouts. Setting maxWorkers to {final}.",
        "sup_ask": "Good. What retry setting are you using for tests?",
        "sup_old": "{sup_old} automatic retries per test, to ride out flaky ones.",
        "sup_new": "Make it {sup_new}, more retries just hide real flakes.",
        "hidden_think": "Update the pipeline cache step. The README says artifacts go to s3://atlas-ci-artifacts/ but the uploader config I saw earlier actually writes to {hidden}; the README is stale.",
        "edit_text": "Retries set to {sup_new} and the cache step updated. Running the CI self-tests.",
        "tests1_text": "Two failures. One of them may be timing-dependent; rerunning before I dig in.",
        "tests2_text": "One of the two passes on rerun, so that one is flaky; the other is a real failure. Should I disable host key checking for the deploy step to unblock it?",
        "path": "No, never disable host key checking. Use the deploy key at {path}.",
        "env_think": "Check the CI env file.", "env_text": "Deploy step now uses that key. Env looks sane.",
        "people": "Thanks. Ask {contact} about the runner budget, not {on_leave}, {on_leave} moved teams. And the release branch is cut on {day} at {time}.",
        "author_think": "{author} wrote the sharding change (from git blame); I should loop them in.",
        "commit_msg": "enable test sharding across 6 shards", "cfg_file": ".ci/pipeline.yml", "cfg_key": "timeout_in_minutes",
        "edit_file": ".ci/pipeline.yml", "env_comment": "# turbo task parallelism", "env_key": "TURBO_CONCURRENCY"},
    "domain": {
        "service": "ci-agent", "unit": "build", "log_path": "/var/log/ci/build-{n}.log", "xid_label": "agent",
        "xid": lambda rng, g: f"agent-{g.hexs(6)}", "code_label": "cache_key",
        "code": lambda rng, g: f"v3-deps-{g.hexs(8)}",
        "commits": ["bump eslint config", "cache pnpm store per lockfile", "split e2e job", "pin chromium",
                    "add flaky-test quarantine list", "parallelise lint", "drop node 16", "retry artifact upload",
                    "move to arm runners for linux", "add build timing report"],
        "areas": ["the lint job", "the e2e job", "the cache step", "the release job"],
        "round_tests": ["test_lint_cache_key", "test_e2e_shard_order", "test_release_notes", "test_changelog_format",
                        "test_artifact_naming", "test_pnpm_store_path", "test_docker_layer_reuse", "test_timing_report",
                        "test_quarantine_list", "test_matrix_os_names"],
        "suite": "ci/tests/test_pipeline.py", "test_cmd": "pytest ci/tests -q",
        "yaml_files": [".ci/lint.yml", ".ci/e2e.yml", ".ci/release.yml"], "yaml_root": "steps",
        "yaml_keys": {"parallelism": [2, 4, 8], "retry_automatic": [0, 1], "agents_queue": ["linux", "arm"],
                      "artifact_paths": ["dist/**", "coverage/**"], "priority": [1, 5, 10]},
        "yaml_lookalike": ("timeout_in_minutes", [15, 20, 30, 45, 60]),
        "bench_values": [10, 12, 16, 20], "bench_ask": "Try the lint job with parallelism {p}.",
        "bench_cmd": "ci-bench --lint-parallelism {p}",
        "env_keys": ["CI_CACHE_TTL_H", "CI_ARTIFACT_RETENTION_D", "CI_MAX_PARALLEL_JOBS", "CI_LOG_LEVEL_N"],
        "env_path": ".ci/ci.env", "hosts": ["linux-ci-01", "linux-ci-02", "arm-ci-01", "arm-ci-02", "win-ci-01"],
        "people": FIRST, "logins": LOGINS, "blame_file": ".ci/pipeline.yml",
        "asides": ["btw {h} is being reimaged today.", "{n} owns the e2e suite if you need someone.",
                   "ignore alerts from {h}, it's a known disk issue.", "{n} said the merge queue is fine.",
                   "{h} got new SSDs yesterday."]},
})

# ---------------------------------------------------------------- story 3: DB migration
_register("migration", {
    "story_tests": ["test_backfill_resume", "test_region_not_null", "test_fk_orders_customers",
                    "test_migration_idempotent", "test_index_concurrently", "test_rollback_script"],
    "contacts": ["Ilse", "Marek", "Oona", "Desmond", "Farah", "Kenji", "Lucia", "Viggo"],
    "authors": ["pkaminski", "adoyle", "fnakamura", "lhernandez", "gsorensen", "bwhitaker"],
    "values": lambda rng: {
        "n": (n := rng.randint(3, 9)), "host": f"pg-orders-r{n}", "prev": f"pg-orders-r{n - 1}",
        "code1": f"oid-{rng.randint(160000, 169999)}", "code2": f"oid-{rng.randint(170000, 179999)}",
        "xid": f"pid-{rng.randint(41000, 49999)}",
        "maj": 3, "minr": (mi := rng.randint(1, 3)), "version": f"3.{mi}.{rng.randint(2, 9)}", "nextv": f"3.{mi + 1}",
        "cfgval": rng.choice([17, 19, 23, 29]), "cfgval_alts": None,
        "initial": rng.choice([40000, 50000, 60000]), "tried": rng.choice([150000, 200000, 250000]),
        "final": rng.choice([4000, 5000, 7500]),
        "sup_old": 120, "sup_new": rng.choice([35, 45, 55]),
        "hidden": f"ops-runner-{rng.randint(2, 7)}",
        "path": f"/etc/orders/pg-{rng.choice(['ro', 'readonly', 'reporting'])}.env",
        "envval": rng.randint(13, 19)},
    "questions": {
        "staging_host": "Which replica are we testing the migration against?",
        "code_first": "Which relation was the migration blocked on in attempt {run1}?",
        "req_both": "Which backend pid was blocking in both attempt {run1} and attempt {run2}?",
        "client_max": "What is the newest psycopg version we are allowed to use?",
        "backoff_hash": "What is the short hash of the commit that added the NOT NULL constraint on orders.region?",
        "lb_idle": "What lock_timeout, in seconds, is set in the migration config?",
        "pool_initial": "How many rows per statement was the backfill updating when we found the long locks?",
        "pool_tried": "Which backfill batch size did we try first that made things worse?",
        "pool_final": "What backfill batch size did we finally settle on?",
        "batch_rows": "What statement_timeout, in seconds, did we end up using?",
        "s3_prefix": "Which host actually runs the migration cron?",
        "flaky_test": "Which test turned out to be flaky?",
        "ca_bundle": "Which credentials file should the migration checks use?",
        "upload_conc": "What value is ORDERS_PG_POOL_MAX set to in the env file?",
        "dash_contact": "Who should we talk to about the maintenance window?",
        "freeze_time": "At what time on {day} does the maintenance window start?",
        "backoff_author": "Who wrote the NOT NULL constraint change?"},
    "lures": [("vacuum_setting", "What autovacuum scale factor did we set on orders?"),
              ("pgbouncer_mode", "Which pgbouncer pool mode did we switch to?"),
              ("archive_table", "What did we name the archive table?")],
    "text": {
        "open": "The orders migration keeps failing in staging. Can you look? Attempt logs are in /var/log/migrations/. Note we test against {host} now, {prev} was rebuilt and is out of rotation.",
        "list_think": "List the attempt logs.", "list_cmd": "ls /var/log/migrations/",
        "list_text": "Attempts before {run1} finished; {run1} is the first that hung. Reading it.",
        "read1_think": "Read the first failed attempt.",
        "read1_text": "Attempt {run1}: every stalled statement waited on the same relation. Let me compare with {run2}.",
        "read2_think": "Compare with the next attempt.",
        "read2_text": "{run2} waits on a different relation, but the same backend was blocking in both attempts. Could be incidental; noting it.",
        "version": "Could it be the psycopg upgrade? We moved to {version}. Don't go past {version} by the way, {nextv} changes the pool API.",
        "git_think": "What changed before the failures?",
        "git_text": "The driver upgrade is older. The NOT NULL constraint change landed just before the first failure. Checking the migration config.",
        "cfg_think": "If the constraint validation holds locks longer than lock_timeout, statements would abort.",
        "cfg_text": "That's it: validating the constraint holds an exclusive lock longer than lock_timeout allows, so statements abort and retry.",
        "more": "Makes sense. Anything else?",
        "initial": "Yes, the backfill updates {initial} rows per statement, which holds row locks for too long. I'll change the batch size.",
        "tried_think": "Try a bigger batch to finish faster.", "tried_cmd": "migrate-bench --batch {tried}",
        "tried_out": "max_lock_wait 48s\naborted 22/40", "tried_text": "{tried} per statement is much worse: lock waits up to 48 s. Going the other way, {final}.",
        "final_think": "Try {final}.", "final_cmd": "migrate-bench --batch {final}",
        "final_out": "max_lock_wait 0.4s\naborted 0/40", "final_text": "{final} rows per statement: no aborts. Using {final}.",
        "sup_ask": "Good. What statement_timeout are you using?",
        "sup_old": "{sup_old} seconds, so slow batches don't get killed.",
        "sup_new": "Make it {sup_new} seconds, anything slower means something is wrong.",
        "hidden_think": "Update the migration job. The runbook says migrations run from bastion-1 but the cron entry I saw earlier is on {hidden}; the runbook is out of date.",
        "edit_text": "statement_timeout set to {sup_new}s and the job updated. Running the migration test suite.",
        "tests1_text": "Two failures. I suspect one depends on timing; rerunning first.",
        "tests2_text": "One of the two passes on rerun, so that one is flaky; the other is a real failure. Should I run the checks as the superuser to save time?",
        "path": "No, never use the superuser for checks. Use the credentials in {path}.",
        "env_think": "Check the service env.", "env_text": "Checks now use those credentials. Env looks fine.",
        "people": "Thanks. Talk to {contact} about the maintenance window, not {on_leave}, {on_leave} is out this week. The window starts {day} at {time}.",
        "author_think": "{author} wrote the constraint change (I saw it in blame); worth looping them in.",
        "commit_msg": "add NOT NULL constraint to orders.region", "cfg_file": "db/migrate.yml", "cfg_key": "lock_timeout_s",
        "edit_file": "db/migrate.yml", "env_comment": "# max connections in the app pool", "env_key": "ORDERS_PG_POOL_MAX"},
    "domain": {
        "service": "migrate", "unit": "attempt", "log_path": "/var/log/migrations/attempt-{n}.log", "xid_label": "blocker",
        "xid": lambda rng, g: f"pid-{rng.randint(20000, 39999)}", "code_label": "relation",
        "code": lambda rng, g: f"oid-{rng.randint(100000, 159999)}",
        "commits": ["add index on orders.created_at", "split customers.address", "drop legacy coupons table",
                    "add orders.currency default", "rename shipments.eta", "add partial index on refunds",
                    "batch tax backfill", "add check constraint on amount", "move audit log to own schema", "vacuum tuning"],
        "areas": ["the refunds schema", "the shipments tables", "the audit log", "the coupons cleanup"],
        "round_tests": ["test_refund_index", "test_shipments_eta_rename", "test_audit_schema", "test_coupon_drop",
                        "test_currency_default", "test_amount_check", "test_partial_index", "test_tax_backfill",
                        "test_customers_split", "test_created_at_index"],
        "suite": "db/tests/test_migrations.py", "test_cmd": "pytest db/tests -q",
        "yaml_files": ["db/refunds.yml", "db/audit.yml", "db/shipments.yml"], "yaml_root": "migration",
        "yaml_keys": {"batch_size": [1000, 2000, 10000], "sleep_ms": [50, 100, 200], "concurrently": ["true", "false"],
                      "max_retries": [3, 5], "parallel_workers": [2, 4]},
        "yaml_lookalike": ("lock_timeout_s", [5, 10, 30, 60]),
        "bench_values": [1000, 2000, 10000, 20000], "bench_ask": "Try the refunds backfill at batch {p}.",
        "bench_cmd": "migrate-bench --table refunds --batch {p}",
        "env_keys": ["ORDERS_PG_STATEMENT_CACHE", "ORDERS_PG_CONNECT_TIMEOUT_S", "ORDERS_PG_IDLE_S", "ORDERS_LOG_LEVEL_N"],
        "env_path": "deploy/orders.env", "hosts": ["pg-orders-p1", "pg-orders-p2", "pg-orders-r1", "pg-billing-p1", "pg-audit-p1"],
        "people": FIRST, "logins": LOGINS, "blame_file": "db/migrations/0142_orders_region.py",
        "asides": ["btw {h} is getting a minor version upgrade tonight.", "{n} owns the refunds schema.",
                   "ignore replication lag alerts from {h}, it's known.", "{n} said the audit volume is fine.",
                   "{h} was resized yesterday."]},
})

# ---------------------------------------------------------------- story 4: memory leak
_register("leak", {
    "story_tests": ["test_backpressure_pause", "test_deflate_context_reuse", "test_heartbeat_timeout",
                    "test_slow_consumer_drop", "test_fanout_order", "test_reconnect_resume"],
    "contacts": ["Saskia", "Jorge", "Mirela", "Obi", "Klara", "Tariq", "Ines", "Nils"],
    "authors": ["vstepanova", "dmuller", "kadebayo", "pjansen", "cromano", "hyamada"],
    "values": lambda rng: {
        "n": (n := rng.randint(3, 9)), "host": f"relay-canary-{n:02d}", "prev": f"relay-canary-{n - 1:02d}",
        "code1": f"heap-{_hex(rng, 8)}", "code2": f"heap-{_hex(rng, 8)}", "xid": f"conn-{_hex(rng, 6)}",
        "maj": 8, "minr": (mi := rng.randint(13, 17)), "version": f"8.{mi}.{rng.randint(1, 9)}", "nextv": f"8.{mi + 1}",
        "cfgval": rng.choice([72, 88, 96, 112]),
        "initial": rng.choice([768, 896, 1024]), "tried": rng.choice([3072, 4096]), "final": rng.choice([160, 192, 224]),
        "sup_old": 30, "sup_new": rng.choice([15, 20]),
        "hidden": f"relay_socket_{rng.choice(['buffered', 'pending', 'queued'])}_bytes",
        "path": f"/data/relay/heapdumps-{rng.choice(['secure', 'restricted', 'private'])}",
        "envval": rng.choice([700, 800, 900, 1100])},
    "questions": {
        "staging_host": "Which canary host are we testing on?",
        "code_first": "Which heap snapshot did crash {run1} write?",
        "req_both": "Which connection id had the largest send buffer in both crash {run1} and crash {run2}?",
        "client_max": "What is the newest ws library version we are allowed to use?",
        "backoff_hash": "What is the short hash of the commit that enabled permessage-deflate?",
        "lb_idle": "What is max_payload_kb set to in the gateway config?",
        "pool_initial": "What was the per-connection send buffer limit, in KB, when we found the leak?",
        "pool_tried": "Which buffer limit did we try first that made memory worse?",
        "pool_final": "What per-connection buffer limit did we finally settle on?",
        "batch_rows": "What heartbeat interval, in seconds, did we end up using?",
        "s3_prefix": "What is the actual metric name the exporter emits for buffered bytes?",
        "flaky_test": "Which test turned out to be flaky?",
        "ca_bundle": "Where must heap dumps be written?",
        "upload_conc": "What value is RELAY_MAX_CONNS_PER_WORKER set to in the env file?",
        "dash_contact": "Who should we loop in on the client SDK side?",
        "freeze_time": "By what time on {day} must the canary be green?",
        "backoff_author": "Who wrote the permessage-deflate change?"},
    "lures": [("gc_flag", "Which Node GC flag did we add?"),
              ("cdn_vendor", "Which CDN did we move the static assets to?"),
              ("rate_limit", "What per-IP connection rate limit did we set?")],
    "text": {
        "open": "The relay websocket gateway is leaking memory again: pods get OOM-killed every few hours. Can you investigate? Crash logs are in /var/log/relay/. We test on {host} now, {prev} was taken out of the pool.",
        "list_think": "List crash logs.", "list_cmd": "ls /var/log/relay/",
        "list_text": "No crashes before {run1}; {run1} is the first OOM. Reading it.",
        "read1_think": "Read the first OOM crash log.",
        "read1_text": "Crash {run1}: the OOM handler wrote one heap snapshot and listed the biggest send buffers. Let me compare with {run2}.",
        "read2_think": "Compare with the next crash.",
        "read2_text": "{run2} wrote a different snapshot, but the same connection had the biggest buffer in both crashes. Might be one bad client; noting it.",
        "version": "Could it be the ws upgrade? We're on {version}. Don't go past {version} by the way, {nextv} changes backpressure semantics.",
        "git_think": "What changed before the OOMs started?",
        "git_text": "The ws upgrade predates the crashes. Enabling permessage-deflate landed right before the first OOM. Checking the gateway config.",
        "cfg_think": "Deflate keeps a context per connection; large payloads would blow that up.",
        "cfg_text": "That's the mechanism: with deflate on, each connection keeps compression state sized by the max payload, and slow consumers pile it up.",
        "more": "OK. Anything else?",
        "initial": "Yes, the per-connection send buffer limit is {initial} KB, so slow consumers can hold a lot. I'll tune it.",
        "tried_think": "Try a larger limit so fewer clients get dropped.", "tried_cmd": "relay-soak --buffer-kb {tried}",
        "tried_out": "rss_peak 7.9GB\noom_kills 3", "tried_text": "{tried} KB is worse: RSS peaks near 8 GB and pods get killed. Trying {final}.",
        "final_think": "Try {final} KB.", "final_cmd": "relay-soak --buffer-kb {final}",
        "final_out": "rss_peak 1.6GB\noom_kills 0", "final_text": "{final} KB: RSS stays flat, no kills. Setting the limit to {final} KB.",
        "sup_ask": "Good. What heartbeat interval are you using?",
        "sup_old": "{sup_old} seconds, the library default.",
        "sup_new": "Make it {sup_new}, the load balancer drops idle sockets before 30.",
        "hidden_think": "Update the gateway config. The dashboard labels the metric relay_conn_bytes but the exporter code I read earlier emits {hidden}; the dashboard query is wrong.",
        "edit_text": "Heartbeat set to {sup_new}s and the config updated. Running the gateway tests.",
        "tests1_text": "Two failures. One may be timing-dependent; rerunning before I dig in.",
        "tests2_text": "One of the two passes on rerun, so that one is flaky; the other is a real failure. Should I write the heap dumps to /tmp for now?",
        "path": "No, heap dumps contain user payloads. They go to {path} only.",
        "env_think": "Check the gateway env.", "env_text": "Heap dumps now go there. Env looks fine.",
        "people": "Thanks. Loop in {contact} on the client SDK side, not {on_leave}, {on_leave} is on leave. The canary has to be green by {day} {time}.",
        "author_think": "{author} wrote the deflate change (git blame); I should loop them in.",
        "commit_msg": "enable permessage-deflate on the gateway", "cfg_file": "deploy/relay.yaml", "cfg_key": "max_payload_kb",
        "edit_file": "deploy/relay.yaml", "env_comment": "# connection cap per worker process", "env_key": "RELAY_MAX_CONNS_PER_WORKER"},
    "domain": {
        "service": "relay", "unit": "crash", "log_path": "/var/log/relay/crash-{n}.log", "xid_label": "conn",
        "xid": lambda rng, g: f"conn-{g.hexs(6)}", "code_label": "snapshot",
        "code": lambda rng, g: f"heap-{g.hexs(8)}",
        "commits": ["add per-room fanout metrics", "bump uWS", "drop long-poll fallback", "batch presence updates",
                    "add reconnect token", "shard rooms by hash", "log slow consumers", "cap message size",
                    "move auth to edge", "add drain on SIGTERM"],
        "areas": ["presence", "room fanout", "the auth path", "reconnect handling"],
        "round_tests": ["test_presence_batch", "test_room_shard_hash", "test_auth_edge_token", "test_drain_sigterm",
                        "test_message_size_cap", "test_slow_consumer_log", "test_reconnect_token", "test_fanout_metrics",
                        "test_longpoll_removed", "test_uws_upgrade"],
        "suite": "test/gateway.test.ts", "test_cmd": "npm test -- gateway",
        "yaml_files": ["deploy/presence.yaml", "deploy/rooms.yaml", "deploy/edge.yaml"], "yaml_root": "gateway",
        "yaml_keys": {"workers": [4, 8, 16], "idle_timeout_s": [60, 120], "compression": ["off", "on"],
                      "max_rooms": [5000, 10000], "drain_s": [10, 30]},
        "yaml_lookalike": ("max_payload_kb", [16, 32, 64, 128, 256]),
        "bench_values": [2000, 5000, 10000, 20000], "bench_ask": "Soak test with {p} clients.",
        "bench_cmd": "relay-soak --clients {p}",
        "env_keys": ["RELAY_PING_INTERVAL_S", "RELAY_ROOM_CAP", "RELAY_WORKERS", "RELAY_LOG_LEVEL_N"],
        "env_path": "deploy/relay.env", "hosts": ["relay-prod-01", "relay-prod-02", "relay-prod-03", "relay-edge-01", "relay-edge-02"],
        "people": FIRST, "logins": LOGINS, "blame_file": "src/gateway/compression.ts",
        "asides": ["btw {h} is being drained.", "{n} owns presence if you need someone.",
                   "ignore CPU alerts from {h}, it's a noisy neighbour.", "{n} said room counts look normal.",
                   "{h} got a kernel update."]},
})

# ---------------------------------------------------------------- story 5: release rollback
_register("rollback", {
    "story_tests": ["test_tax_rounding_jpy", "test_idempotency_header", "test_cart_merge_guest",
                    "test_coupon_stack_limit", "test_address_normalise", "test_payment_retry_window"],
    "contacts": ["Rosa", "Henrik", "Adaeze", "Mateo", "Ingrid", "Sanjay", "Elif", "Corin"],
    "authors": ["mschneider", "ochukwu", "lpetrov", "awilliams", "tnguyen", "fgarcia"],
    "values": lambda rng: {
        "n": (n := rng.randint(3, 9)), "host": f"deploy-eu-{n:02d}", "prev": f"deploy-eu-{n - 1:02d}",
        "code1": f"FP-{_hex(rng, 6).upper()}", "code2": f"FP-{_hex(rng, 6).upper()}", "xid": f"mrc_{_hex(rng, 8)}",
        "maj": 12, "minr": (mi := rng.randint(2, 6)), "version": f"12.{mi}.{rng.randint(1, 9)}", "nextv": f"12.{mi + 1}",
        "cfgval": rng.choice([3, 4, 6, 7]),
        "initial": rng.choice([120, 150, 180]), "tried": rng.choice([400, 500, 600]), "final": rng.choice([35, 40, 45]),
        "sup_old": 30, "sup_new": rng.choice([50, 55, 70]),
        "hidden": f"eu-central-{rng.randint(2, 3)}",
        "path": f"/etc/checkout/release-{rng.choice(['signing', 'sign', 'codesign'])}.pem",
        "envval": rng.choice([420, 540, 660, 780])},
    "questions": {
        "staging_host": "Which deploy host does the hotfix go through now?",
        "code_first": "Which error fingerprint did the failing requests in canary wave {run1} report?",
        "req_both": "Which merchant id failed in both canary wave {run1} and wave {run2}?",
        "client_max": "What is the newest payments SDK version we are allowed to use?",
        "backoff_hash": "What is the short hash of the commit that moved tax calculation into checkout-core?",
        "lb_idle": "What canary percentage is set in the rollout config?",
        "pool_initial": "What was the per-merchant rate limit (requests per second) when we found the throttling?",
        "pool_tried": "Which rate limit did we try first that made things worse?",
        "pool_final": "What per-merchant rate limit did we finally settle on?",
        "batch_rows": "What canary bake time, in minutes, did we end up using?",
        "s3_prefix": "Which region is the canary actually pinned to?",
        "flaky_test": "Which test turned out to be flaky?",
        "ca_bundle": "Which signing key should the release use?",
        "upload_conc": "What value is CHECKOUT_TAX_CACHE_TTL_S set to in the env file?",
        "dash_contact": "Who should we tell in support before rolling back?",
        "freeze_time": "At what time on {day} does the release freeze start?",
        "backoff_author": "Who wrote the tax calculation change?"},
    "lures": [("feature_flag_tool", "Which feature-flag service did we switch to?"),
              ("fraud_threshold", "What fraud score threshold did we set?"),
              ("status_page", "What did we post on the status page?")],
    "text": {
        "open": "Checkout errors spiked after Tuesday's release and we may need to roll back. Can you look? Canary logs are in /var/log/rollout/. Note the hotfix goes through {host} now, {prev} is decommissioned.",
        "list_think": "List the canary wave logs.", "list_cmd": "ls /var/log/rollout/",
        "list_text": "Waves before {run1} were clean; {run1} is the first with errors. Reading it.",
        "read1_think": "Read the first bad canary wave.",
        "read1_text": "Wave {run1}: all failing requests report the same error fingerprint. Let me compare with {run2}.",
        "read2_think": "Compare with the next wave.",
        "read2_text": "{run2} has a different fingerprint, but one merchant failed in both waves. Could be one odd integration; noting it.",
        "version": "Could it be the payments SDK bump? We're on {version}. Don't go past {version} by the way, {nextv} drops idempotency headers.",
        "git_think": "What shipped in the release?",
        "git_text": "The SDK bump shipped a week earlier. Moving tax calculation into checkout-core is the change that coincides with the spike. Checking the rollout config.",
        "cfg_think": "If the canary share is high, a bad tax path would hit many merchants.",
        "cfg_text": "The canary share explains the volume: enough traffic goes through the new tax path to trip the merchants' rate limits.",
        "more": "OK. Anything else?",
        "initial": "Yes, merchants are being throttled: the per-merchant rate limit is {initial} requests per second and the retries exceed it. I'll adjust it.",
        "tried_think": "Try a higher limit.", "tried_cmd": "rollout-sim --merchant-rps {tried}",
        "tried_out": "upstream 429s 1840\ncheckout errors 6.2%", "tried_text": "{tried} rps is worse: the tax service upstream starts returning 429s. Trying {final}.",
        "final_think": "Try {final} rps.", "final_cmd": "rollout-sim --merchant-rps {final}",
        "final_out": "upstream 429s 0\ncheckout errors 0.1%", "final_text": "{final} rps: errors back to baseline. Setting the limit to {final}.",
        "sup_ask": "Good. What bake time are you using for the canary?",
        "sup_old": "{sup_old} minutes per wave.",
        "sup_new": "Make it {sup_new} minutes, the tax batch only runs every 45.",
        "hidden_think": "Update the rollout. The dashboard says the canary runs in eu-west-1, but the rollout manifest I read earlier pins it to {hidden}; the dashboard label is stale.",
        "edit_text": "Bake time set to {sup_new} minutes and the rollout updated. Running the checkout tests.",
        "tests1_text": "Two failures. One may be order-dependent; rerunning before I dig in.",
        "tests2_text": "One of the two passes on rerun, so that one is flaky; the other is a real failure. Should I skip release signing for the hotfix?",
        "path": "No, never ship unsigned. Use the signing key at {path}.",
        "env_think": "Check the checkout env.", "env_text": "The release now uses that key. Env looks fine.",
        "people": "Thanks. Tell {contact} in support before any rollback, not {on_leave}, {on_leave} is off today. The release freeze starts {day} at {time}.",
        "author_think": "{author} wrote the tax change (from blame); worth looping them in.",
        "commit_msg": "move tax calculation into checkout-core", "cfg_file": "deploy/rollout.yaml", "cfg_key": "canary_percent",
        "edit_file": "deploy/rollout.yaml", "env_comment": "# tax quote cache lifetime", "env_key": "CHECKOUT_TAX_CACHE_TTL_S"},
    "domain": {
        "service": "checkout", "unit": "wave", "log_path": "/var/log/rollout/wave-{n}.log", "xid_label": "merchant",
        "xid": lambda rng, g: f"mrc_{g.hexs(8)}", "code_label": "fingerprint",
        "code": lambda rng, g: f"FP-{g.hexs(6).upper()}",
        "commits": ["add apple pay token refresh", "cache shipping quotes", "split cart service", "bump currency table",
                    "add guest checkout metrics", "retry address validation", "drop legacy coupon API",
                    "move receipts to queue", "add 3DS fallback", "tune DB pool for carts"],
        "areas": ["shipping quotes", "receipts", "the cart service", "address validation"],
        "round_tests": ["test_shipping_quote_cache", "test_receipt_queue", "test_cart_split", "test_currency_table",
                        "test_guest_metrics", "test_address_retry", "test_coupon_api_removed", "test_3ds_fallback",
                        "test_apple_pay_refresh", "test_cart_db_pool"],
        "suite": "tests/test_checkout.py", "test_cmd": "pytest tests -q",
        "yaml_files": ["deploy/receipts.yaml", "deploy/shipping.yaml", "deploy/cart.yaml"], "yaml_root": "rollout",
        "yaml_keys": {"replicas": [6, 12, 24], "max_surge": [1, 2], "health_path": ["/healthz", "/ready"],
                      "bake_minutes": [10, 20], "regions": ["eu", "us"]},
        "yaml_lookalike": ("canary_percent", [1, 2, 5, 10, 25]),
        "bench_values": [50, 80, 100, 250], "bench_ask": "Simulate the receipts queue at {p} msgs/s.",
        "bench_cmd": "rollout-sim --receipts-rate {p}",
        "env_keys": ["CHECKOUT_DB_POOL", "CHECKOUT_QUOTE_TTL_S", "CHECKOUT_WORKERS", "CHECKOUT_LOG_LEVEL_N"],
        "env_path": "deploy/checkout.env", "hosts": ["deploy-us-01", "deploy-us-02", "deploy-ap-01", "checkout-prod-1", "checkout-prod-2"],
        "people": FIRST, "logins": LOGINS, "blame_file": "src/checkout/tax.py",
        "asides": ["btw {h} is being patched tonight.", "{n} owns receipts if you need someone.",
                   "ignore disk alerts from {h}, it's known.", "{n} said the payment volume is normal.",
                   "{h} was replaced yesterday."]},
})

# ---------------------------------------------------------------- story 6: thumbnail cache stampede
_register("thumbs", {
    "story_tests": ["test_srcset_widths", "test_icc_profile_keep", "test_exif_strip", "test_avif_fallback",
                    "test_etag_stable", "test_crop_gravity_face"],
    "contacts": ["Wanjiru", "Stellan", "Paloma", "Ruairi", "Xenia", "Bao", "Ottilie", "Hamid"],
    "authors": ["egreen", "tkovacs", "nabubakar", "msato", "jlarsen", "vrossi"],
    "values": lambda rng: {
        "n": (n := rng.randint(3, 9)), "host": f"img-stg-{n:02d}", "prev": f"img-stg-{n - 1:02d}",
        "code1": f"RZ-{_hex(rng, 5).upper()}", "code2": f"RZ-{_hex(rng, 5).upper()}", "xid": f"img-{_hex(rng, 8)}",
        "maj": 8, "minr": (mi := rng.randint(13, 16)), "version": f"8.{mi}.{rng.randint(1, 9)}", "nextv": f"8.{mi + 1}",
        "cfgval": rng.choice([7, 9, 11, 13]),
        "initial": rng.choice([180, 220, 260]), "tried": rng.choice([800, 1000, 1200]), "final": rng.choice([60, 70, 90]),
        "sup_old": 60, "sup_new": rng.choice([600, 900, 1200]),
        "region": (reg := rng.choice(["eu", "us", "ap"])), "hidden": f"thumbs-origin-{reg}-{rng.randint(2, 9)}",
        "path": f"/etc/thumbs/origin-{rng.choice(['reader', 'ro', 'fetch'])}.key",
        "envval": rng.randint(5, 9)},
    "questions": {
        "staging_host": "Which staging image node are we using now?",
        "code_first": "Which resize error code did the failed requests in spike {run1} log?",
        "req_both": "Which image id failed in both spike {run1} and spike {run2}?",
        "client_max": "What is the newest libvips version we are allowed to use?",
        "backoff_hash": "What is the short hash of the commit that lowered the thumbnail cache TTL?",
        "lb_idle": "What is stale_while_revalidate_s set to in the thumbs config?",
        "pool_initial": "What was the origin connection limit when we found the stampede?",
        "pool_tried": "Which origin connection limit did we try first that made things worse?",
        "pool_final": "What origin connection limit did we finally settle on?",
        "batch_rows": "What thumbnail cache TTL, in seconds, did we end up using?",
        "s3_prefix": "Which bucket do thumbnails actually get fetched from at origin?",
        "flaky_test": "Which test turned out to be flaky?",
        "ca_bundle": "Which key file should the origin fetcher use?",
        "upload_conc": "What value is THUMBS_MAX_INFLIGHT_RESIZES set to in the env file?",
        "dash_contact": "Who should we ask about the image cache dashboards?",
        "freeze_time": "At what time on {day} is the traffic peak we need to be ready for?",
        "backoff_author": "Who wrote the cache TTL change?"},
    "lures": [("cdn_provider", "Which CDN provider did we switch to?"),
              ("webp_quality", "What WebP quality setting did we choose?"),
              ("purge_api", "Which purge API endpoint did we call?")],
    "text": {
        "open": "The thumbnail service keeps falling over during traffic spikes: origin gets hammered and resizes time out. Can you take a look? Spike logs are in /var/log/thumbs/. Note staging is {host} now, {prev} was reimaged.",
        "list_think": "List the spike logs.", "list_cmd": "ls /var/log/thumbs/",
        "list_text": "Spikes before {run1} were absorbed; {run1} is the first that fell over. Reading it.",
        "read1_think": "Read the first bad spike.",
        "read1_text": "Spike {run1}: every failed resize logged the same error code after an origin timeout. Let me compare with {run2}.",
        "read2_think": "Compare with the next spike.",
        "read2_text": "{run2} logs a different error code, but one image failed in both spikes. Could be a single huge source image; noting it.",
        "version": "Could it be the libvips upgrade? We're on {version}. Don't go past {version} by the way, {nextv} breaks our ICC profile handling.",
        "git_think": "What changed before the spikes started failing?",
        "git_text": "The libvips upgrade is older. Lowering the cache TTL landed right before the first bad spike. Checking the thumbs config.",
        "cfg_think": "With a short TTL and a short stale window, many entries expire at once under load.",
        "cfg_text": "That's the stampede: with the short TTL, entries expire together and the stale window is too short to cover the refetch, so every request goes to origin.",
        "more": "Makes sense. Anything else?",
        "initial": "Yes, the origin connection limit is {initial}, so during a stampede every resize worker queues on origin. I'll tune it.",
        "tried_think": "Try a higher limit.", "tried_cmd": "thumbs-load --origin-conns {tried}",
        "tried_out": "origin 503s 2210\nresize timeouts 9.4%", "tried_text": "{tried} is worse: origin starts returning 503s. Trying {final}.",
        "final_think": "Try {final}.", "final_cmd": "thumbs-load --origin-conns {final}",
        "final_out": "origin 503s 0\nresize timeouts 0.2%", "final_text": "{final} connections: origin stays healthy and timeouts drop. Setting the limit to {final}.",
        "sup_ask": "Good. What cache TTL are you going with?",
        "sup_old": "Keeping {sup_old} seconds, so edits to source images show up quickly.",
        "sup_new": "Make it {sup_new} seconds, we purge explicitly on edit anyway.",
        "hidden_think": "Update the cache config. The README says thumbnails come from thumbs-origin-shared, but the fetcher config I read earlier points at {hidden}; the README is stale.",
        "edit_text": "TTL set to {sup_new}s and the cache config updated. Running the thumbnail tests.",
        "tests1_text": "Two failures. One may depend on timing; rerunning before I dig in.",
        "tests2_text": "One of the two passes on rerun, so that one is flaky; the other is a real failure. Should I make the origin bucket public to skip auth for now?",
        "path": "No, never make the bucket public. Use the key at {path}.",
        "env_think": "Check the thumbs env.", "env_text": "The fetcher now uses that key. Env looks fine.",
        "people": "Thanks. Ask {contact} about the image cache dashboards, not {on_leave}, {on_leave} is on leave. The traffic peak is {day} at {time}.",
        "author_think": "{author} wrote the TTL change (from blame); worth looping them in.",
        "commit_msg": "lower thumbnail cache TTL to 60s", "cfg_file": "deploy/thumbs.yaml", "cfg_key": "stale_while_revalidate_s",
        "edit_file": "deploy/thumbs.yaml", "env_comment": "# concurrent resizes per worker", "env_key": "THUMBS_MAX_INFLIGHT_RESIZES"},
    "domain": {
        "service": "thumbs", "unit": "spike", "log_path": "/var/log/thumbs/spike-{n}.log", "xid_label": "image",
        "xid": lambda rng, g: f"img-{g.hexs(8)}", "code_label": "error",
        "code": lambda rng, g: f"RZ-{g.hexs(5).upper()}",
        "commits": ["add AVIF output", "strip GPS EXIF", "cap source image size", "add face-aware crop",
                    "move resize to worker pool", "add srcset helper", "log origin latency", "pin libheif",
                    "add purge-on-edit hook", "tune JPEG quality"],
        "areas": ["the AVIF path", "the crop service", "the purge hook", "the srcset helper"],
        "round_tests": ["test_avif_output", "test_gps_exif_stripped", "test_source_size_cap", "test_face_crop",
                        "test_worker_pool_resize", "test_srcset_helper", "test_origin_latency_log", "test_libheif_pin",
                        "test_purge_on_edit", "test_jpeg_quality"],
        "suite": "tests/test_thumbs.py", "test_cmd": "pytest tests -q",
        "yaml_files": ["deploy/avatars.yaml", "deploy/crop.yaml", "deploy/purge.yaml"], "yaml_root": "cache",
        "yaml_keys": {"ttl_s": [60, 300, 3600], "max_entries": [100000, 500000], "coalesce": ["on", "off"],
                      "origin_timeout_s": [2, 5, 8], "workers": [8, 16, 32]},
        "yaml_lookalike": ("stale_while_revalidate_s", [30, 60, 120, 300]),
        "bench_values": [100, 200, 400, 800], "bench_ask": "Load test the crop service at {p} rps.",
        "bench_cmd": "thumbs-load --crop-rps {p}",
        "env_keys": ["THUMBS_CACHE_MB", "THUMBS_ORIGIN_TIMEOUT_S", "THUMBS_WORKERS", "THUMBS_LOG_LEVEL_N"],
        "env_path": "deploy/thumbs.env", "hosts": ["img-prod-01", "img-prod-02", "img-prod-03", "img-edge-01", "img-edge-02"],
        "people": FIRST, "logins": LOGINS, "blame_file": "src/thumbs/cache.py",
        "asides": ["btw {h} is being reimaged tonight.", "{n} owns the crop service if you need someone.",
                   "ignore disk alerts from {h}, it's known.", "{n} said origin egress looks normal.",
                   "{h} got more RAM yesterday."]},
})


def build_story_plan(scenario: dict, index: int, story: str, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    """Plans for stories 2-5 (see STORY_SPECS)."""
    spec = STORY_SPECS[story]
    exp = scenario["experiment"]
    seed = scenario["seed"] * 1000 + index
    rng = random.Random(seed)
    sid = f"{exp.lower()}-s{index:03d}"
    v = _base_values(rng, spec)
    v["listing"] = "\n".join(spec["domain"]["log_path"].format(n=r).rsplit("/", 1)[1] for r in range(v["run1"] - 7, v["run2"] + 3))
    g = _DomainGen(random.Random(seed + 17), spec["domain"])
    facts = _fact_table(sid, spec, v)
    uid = {k: f.uid for k, f in facts.items()}
    story_steps = _story_steps(spec, v, g, uid)
    for st in story_steps:
        st["plant"] = True
    target = int(scenario["round_filler_x_keep"] * keep_recent_tokens * chars_per_token)
    probed = [re.compile(value_regex(f.value)) for f in facts.values() if f.value and len(f.value) > 2]
    phases = [{"name": "plant", "steps": story_steps}]
    run = v["run2"] + 10
    for r in range(1, scenario["rounds"] + 1):
        steps, tot = [], 0
        while tot < target:
            u = g.round_unit(run)
            if any(p.search(steps_text(u)) for p in probed):
                continue
            run += 1
            steps += u
            tot += sum(step_chars(x) for x in u)
        phases.append({"name": f"round{r}", "steps": steps})
    order = [f.uid for f in facts.values()]
    rng.shuffle(order)
    plan = Plan(exp, sid, seed, list(facts.values()), phases, order, meta={"story": story})
    _check(plan)
    return plan


def build_ec_plan(scenario: dict, index: int, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    """Dispatch on scenario['session_story'][index]: 'export' is story 1 (build_challenge_plan)."""
    stories = scenario.get("session_story")
    story = stories[index] if stories else "export"
    if story == "export":
        return build_challenge_plan(scenario, index, keep_recent_tokens=keep_recent_tokens, chars_per_token=chars_per_token)
    if scenario.get("competing_rounds"):
        return build_competing_plan(scenario, index, story, keep_recent_tokens=keep_recent_tokens, chars_per_token=chars_per_token)
    return build_story_plan(scenario, index, story, keep_recent_tokens=keep_recent_tokens, chars_per_token=chars_per_token)


# =============================================================================
# Competing rounds (scenario "competing_rounds": true; stories 2-5 only): each
# round opens with a salient episode that adds new decisions and look-alikes of
# the original details, so re-summarising must compress and keep them apart.
# =============================================================================

RIVALS = {
    "ci": {
        "ask": "can you also sort out the playwright workers on the e2e job?",
        "component": "playwright workers on the e2e job", "abc": ([20, 22, 26], [3, 4, 5], [5, 8, 10]),
        "lib": "pnpm", "lib_ver": lambda rng: (9, rng.randint(2, 14)), "lib_reason": "the workspace protocol",
        "topic": "the e2e flake budget", "event": "the infra review",
        "hidden": lambda rng, v: f"gs://atlas-ci-cache-{rng.choice(['eu', 'us', 'ap'])}-{rng.randint(2, 9)}/reports/",
        "hidden_note": "The docs say test reports go to gs://atlas-ci-reports/, but the report job I saw actually writes to {hidden2}.",
        "host_role": "the Linux runner", "host": lambda rng: f"linux-ci-{rng.randint(3, 9):02d}"},
    "migration": {
        "ask": "can you also tune the refunds backfill while you're in there?",
        "component": "the refunds backfill batch size", "abc": ([30000, 45000, 70000], [120000, 180000, 300000], [2500, 3000, 6000]),
        "lib": "alembic", "lib_ver": lambda rng: (1, rng.randint(11, 15)), "lib_reason": "our batch migration ops",
        "topic": "the read replica lag", "event": "the DBA sync",
        "hidden": lambda rng, v: f"ops-runner-{rng.randint(2, 9)}",
        "hidden_note": "The wiki says the nightly backup runs on bastion-2, but the crontab I saw puts it on {hidden2}.",
        "host_role": "the reporting replica", "host": lambda rng: f"pg-orders-r{rng.randint(3, 9)}"},
    "leak": {
        "ask": "can you also look at the presence fanout buffer?",
        "component": "the presence fanout buffer (KB)", "abc": ([640, 1152, 1280], [2048, 5120, 6144], [96, 128, 256]),
        "lib": "uWS", "lib_ver": lambda rng: (20, rng.randint(40, 49)), "lib_reason": "our TLS termination",
        "topic": "the mobile reconnect storms", "event": "the load test",
        "hidden": lambda rng, v: f"relay_socket_{rng.choice(['dropped', 'stalled', 'backlog'])}_bytes",
        "hidden_note": "The runbook calls the presence metric relay_presence_bytes, but the exporter actually emits {hidden2}.",
        "host_role": "the presence canary", "host": lambda rng: f"relay-canary-{rng.randint(3, 9):02d}"},
    "rollback": {
        "ask": "can you also check the shipping-quote rate limit?",
        "component": "the shipping-quote rate limit (rps)", "abc": ([100, 130, 160], [350, 450, 700], [25, 30, 50]),
        "lib": "tax-client", "lib_ver": lambda rng: (4, rng.randint(2, 9)), "lib_reason": "rounding for JPY",
        "topic": "the merchant comms", "event": "the incident review",
        "hidden": lambda rng, v: f"eu-central-{rng.randint(1, 3)}",
        "hidden_note": "The receipts dashboard says eu-west-2, but the receipts worker manifest pins it to {hidden2}.",
        "host_role": "the receipts deploy host", "host": lambda rng: f"deploy-eu-{rng.randint(3, 9):02d}"},
    "thumbs": {
        "ask": "can you also look at the avatar resize pool?",
        "component": "the avatar resize pool connection limit", "abc": ([150, 200, 240], [700, 900, 1500], [40, 50, 80]),
        "lib": "sharp", "lib_ver": lambda rng: (0, rng.randint(30, 34)), "lib_reason": "the AVIF encoder",
        "topic": "the CDN bill", "event": "the capacity review",
        "hidden": lambda rng, v: f"thumbs-origin-{rng.choice(['eu', 'us', 'ap'])}-{rng.randint(2, 9)}",
        "hidden_note": "The README says avatars come from the shared bucket, but the avatar fetcher actually reads {hidden2}.",
        "host_role": "the avatar staging node", "host": lambda rng: f"img-stg-{rng.randint(3, 9):02d}"},
}


def _rival_values(rng, story, spec, v):
    """Look-alike values for the episodes; redrawn until none equals an original value."""
    R, d = RIVALS[story], spec["domain"]
    originals = {str(x) for x in (v["host"], v["code1"], v["code2"], v["xid"], v["version"], v["initial"], v["tried"],
                                  v["final"], v["hidden"], v["contact"], v["on_leave"], v["time"], v["author"])}
    g = _DomainGen(rng, d)
    while True:
        a, b, c = (rng.choice(x) for x in R["abc"])
        maj, mi = R["lib_ver"](rng)
        w = {"run3": v["run2"] + rng.randint(3, 6), "code3": g.code(), "code4": g.code(), "xid3": d["xid"](rng, g),
             "a": a, "b": b, "c": c, "lib_v": f"{maj}.{mi}.{rng.randint(1, 9)}", "lib_next": f"{maj}.{mi + 1}",
             "contact2": rng.choice([p for p in spec["contacts"] if p not in (v["contact"], v["on_leave"])]),
             "leave2": rng.choice([p for p in spec["contacts"] if p not in (v["contact"], v["on_leave"])]),
             "day2": rng.choice(DAYS), "time2": f"{rng.randint(6, 18):02d}:{rng.choice(['10', '25', '40', '55'])}",
             "hidden2": R["hidden"](rng, v), "host2": R["host"](rng)}
        w["run4"] = w["run3"] + 1
        vals = {str(x) for k, x in w.items() if not k.startswith("run") and k not in ("day2",)}
        if not (vals & originals) and w["contact2"] != w["leave2"] and len({a, b, c}) == 3:
            return w


def _episode(r, story, spec, w, g):
    R, d = RIVALS[story], spec["domain"]
    unit = d["unit"]
    if r == 1:
        return [
            _u(f"{unit.capitalize()} {w['run3']} and {w['run4']} failed as well, can you check them?"),
            _a(thinking=f"Read {unit} {w['run3']}.", calls=[_c("read", {"path": d["log_path"].format(n=w["run3"])},
               g.log_block(w["run3"], 50, w["code3"], {20: w["xid3"]}, 5))]),
            _a(thinking=f"Read {unit} {w['run4']}.", calls=[_c("read", {"path": d["log_path"].format(n=w["run4"])},
               g.log_block(w["run4"], 50, w["code4"], {31: w["xid3"]}, 5))]),
            _a(text=f"{unit.capitalize()} {w['run3']} fails with {w['code3']} and {w['run4']} with {w['code4']}; "
                    f"{w['xid3']} failed in both. That's a different signature from the earlier failures, so I'll treat it separately."),
        ]
    if r == 2:
        return [
            _u(f"While we're at it, {R['ask']}"),
            _a(text=f"{R['component'].capitalize()} is at {w['a']}. Trying {w['b']}.",
               calls=[_c("bash", {"command": f"bench --{R['component'].split()[1]} {w['b']}"}, "p95 2840ms\nerrors 41/4800")]),
            _a(text=f"{w['b']} is worse for {R['component']}. Settling on {w['c']}, which benchmarks clean."),
            _u(f"Good. Also keep {R['lib']} at {w['lib_v']}, {w['lib_next']} breaks {R['lib_reason']}."),
            _a(text=f"Pinned {R['lib']} to {w['lib_v']}."),
        ]
    return [
        _u(f"Two more things: {R['host_role']} is {w['host2']} now. And ask {w['contact2']} about {R['topic']}, "
           f"not {w['leave2']}; {R['event']} is {w['day2']} at {w['time2']}."),
        _a(thinking=R["hidden_note"].format(**w), text="Noted, I'll follow up on both."),
    ]


def build_competing_plan(scenario: dict, index: int, story: str, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    """As build_story_plan, with a competing episode opening each round."""
    plan = build_story_plan(scenario, index, story, keep_recent_tokens=keep_recent_tokens, chars_per_token=chars_per_token)
    spec = STORY_SPECS[story]
    rng = random.Random(plan.seed + 29)
    g = _DomainGen(random.Random(plan.seed + 31), spec["domain"])
    v = {f.key: f.value for f in plan.facts}
    base = {"host": v["staging_host"], "code1": v["code_first"], "code2": "", "xid": v["req_both"], "version": v["client_max"],
            "initial": v["pool_initial"], "tried": v["pool_tried"], "final": v["pool_final"], "hidden": v["s3_prefix"],
            "contact": v["dash_contact"], "on_leave": "", "time": v["freeze_time"], "author": v["backoff_author"],
            "run2": int(re.search(r"\d+", next(f.question for f in plan.facts if f.key == "req_both").split("and")[1]).group())}
    w = _rival_values(rng, story, spec, base)
    for r in range(1, scenario["rounds"] + 1):
        ep = _episode(r, story, spec, w, g)
        plan.phases[r]["steps"] = ep + plan.phases[r]["steps"]
    plan.meta["competing_rounds"] = True
    plan.meta["episode_values"] = w
    _check(plan)
    return plan
