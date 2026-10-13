"""Build E1/E2 stimulus plans: phases of steps with planted facts.

A plan is pure data (JSON-serialisable). The runner turns it into session
files with pisession.SessionWriter, so the same plan is replayed identically
in every arm.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import facts as F
from .filler import NOUNS, ATTRS, filler

USER_STATE = [
    "Note for later: the {slot} is {value}.",
    "FYI, we're using {value} as the {slot}.",
    "Quick note: set the {slot} to {value}.",
]
ASK = ["What should we use for the {slot}?", "Can you pick the {slot} for us?"]
ASSIST_STATE = ["Let's use {value} for the {slot}.", "I'd go with {value} as the {slot}."]
TOOL_ASK = ["Check what the {slot} is set to in {cfg}.", "Look up the {slot} in {cfg}, please."]
TOOL_ACK = ["Checked {cfg}; the {slot} is set in there.", "Found the {slot} in {cfg}."]
THINK_ASK = [
    "Choose the {slot} yourself and keep it to yourself for now. I'll ask later.",
    "Pick a value for the {slot} on your own, don't tell me yet.",
]
THINK_THOUGHT = "The user wants me to choose the {slot} privately. I'll go with {value}."
THINK_ACK = ["Done, I've picked one and will keep it in mind.", "Picked. I'll remember it."]
SUPERSEDE = [
    "Correction: the {slot} is now {value}, not {old}.",
    "Change of plan: use {value} for the {slot} instead of {old}.",
]


@dataclass
class Plan:
    experiment: str
    session_id: str
    seed: int
    facts: list[F.Fact]
    phases: list[dict]  # {"name": str, "steps": [step, ...]}
    probe_order: list[str]  # fact uids, the order questions are asked in
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"experiment": self.experiment, "session_id": self.session_id, "seed": self.seed,
                "facts": [f.to_dict() for f in self.facts], "phases": self.phases,
                "probe_order": self.probe_order, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan(d["experiment"], d["session_id"], d["seed"], [F.Fact(**f) for f in d["facts"]],
                    d["phases"], d["probe_order"], d.get("meta", {}))


def _config_file(rng: random.Random, key: str, value: str) -> tuple[str, str]:
    noun = rng.choice(NOUNS)
    path = f"config/{noun}.toml"
    lines = [f"[{noun}]"]
    names = list(dict.fromkeys(f"{rng.choice(NOUNS)}_{rng.choice(ATTRS)}" for _ in range(rng.randint(4, 9))))
    keys = [f"{n} = {rng.randint(1, 99)}" for n in names]
    pos = rng.randint(0, len(keys))
    keys.insert(pos, f'{key} = "{value}"')
    return path, "\n".join(lines + keys)


def plant_steps(rng: random.Random, fact: F.Fact, part: str = "main") -> list[dict]:
    """Steps that plant one fact. The step(s) carrying the value get 'fact_uid'."""
    s, v = fact.slot, fact.value
    if fact.source in ("user",) or (fact.source == "superseded" and part == "v1"):
        val = fact.value_old if fact.source == "superseded" else v
        return [{"kind": "user", "text": rng.choice(USER_STATE).format(slot=s, value=val), "fact_uid": fact.uid}]
    if fact.source == "superseded" and part == "v2":
        return [{"kind": "user", "text": rng.choice(SUPERSEDE).format(slot=s, value=v, old=fact.value_old),
                 "fact_uid": fact.uid}]
    if fact.source == "assistant_text":
        return [{"kind": "user", "text": rng.choice(ASK).format(slot=s)},
                {"kind": "assistant", "thinking": f"The user wants a choice for the {s}.",
                 "text": rng.choice(ASSIST_STATE).format(slot=s, value=v), "calls": [], "fact_uid": fact.uid}]
    if fact.source == "tool_output":
        path, content = _config_file(rng, fact.key, v)
        return [{"kind": "user", "text": rng.choice(TOOL_ASK).format(slot=s, cfg=path)},
                {"kind": "assistant", "thinking": f"Read {path} to find the {s}.", "text": None,
                 "calls": [{"name": "read", "args": {"path": path}, "result": content, "is_error": False}],
                 "fact_uid": fact.uid},
                {"kind": "assistant", "thinking": None, "text": rng.choice(TOOL_ACK).format(slot=s, cfg=path),
                 "calls": []}]
    if fact.source == "thinking":
        return [{"kind": "user", "text": rng.choice(THINK_ASK).format(slot=s)},
                {"kind": "assistant", "thinking": THINK_THOUGHT.format(slot=s, value=v),
                 "text": rng.choice(THINK_ACK), "calls": [], "fact_uid": fact.uid}]
    raise ValueError(f"cannot plant {fact.source}/{part}")


def _interleave(rng: random.Random, items: list[list[dict]], filler_chars: int) -> list[dict]:
    """Plant groups separated by small filler chunks (total ≈ filler_chars)."""
    per_gap = max(1, filler_chars // max(1, len(items) + 1))
    steps: list[dict] = filler(rng, per_gap)
    for group in items:
        steps += group
        steps += filler(rng, per_gap)
    return steps


def _plant_items(rng: random.Random, fs: list[F.Fact]) -> list[list[dict]]:
    parts: list[tuple[F.Fact, str]] = []
    for f in fs:
        if f.source == "superseded":
            parts += [(f, "v1"), (f, "v2")]
        else:
            parts.append((f, "main"))
    rng.shuffle(parts)
    # Supersession must come after the original, with at least two items between.
    for f in [f for f in fs if f.source == "superseded"]:
        i1 = next(i for i, (g, p) in enumerate(parts) if g is f and p == "v1")
        i2 = next(i for i, (g, p) in enumerate(parts) if g is f and p == "v2")
        if i2 < i1 + 3:
            item = parts.pop(i2)
            i1 = next(i for i, (g, p) in enumerate(parts) if g is f and p == "v1")
            parts.insert(min(len(parts), i1 + 3), item)
    return [plant_steps(rng, f, p) for f, p in parts]


def _check_no_leaks(plan: Plan) -> None:
    """Fact values and slot phrases may appear only in the steps that plant them."""
    forbidden = F.forbidden_filler_tokens()
    tok_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, forbidden), key=len, reverse=True)) + r")\b")
    for phase in plan.phases:
        for st in phase["steps"]:
            if st.get("fact_uid") or st.get("plant"):
                continue
            blob = _step_text(st)
            m = tok_re.search(blob)
            if m:
                raise AssertionError(f"{plan.session_id}: filler contains fact vocabulary {m.group(0)!r}")
            for f in plan.facts:
                if f.value and re.search(F.value_regex(f.value), blob):
                    raise AssertionError(f"{plan.session_id}: value of {f.uid} leaked into filler")
                if f.is_lure and f.slot in blob:
                    raise AssertionError(f"{plan.session_id}: lure slot {f.slot!r} appears in filler")


def _step_text(st: dict) -> str:
    parts = [st.get("text") or "", st.get("thinking") or ""]
    for c in st.get("calls", []):
        parts += [str(c["args"]), c["result"]]
    return "\n".join(parts)


def _mark_plant_steps(groups: list[list[dict]]) -> None:
    for g in groups:
        for st in g:
            st["plant"] = True


def scenario_indices(scenario: dict) -> list[int]:
    """The plan indices a scenario runs: its explicit `sessions` list (Study 1b
    runs S1 indices 0, 2, 4, 6, 8, 10), else 0 .. n_sessions-1."""
    if "sessions" in scenario:
        idx = [int(i) for i in scenario["sessions"]]
        if len(idx) != scenario.get("n_sessions", len(idx)):
            raise ValueError("scenario: n_sessions must equal len(sessions)")
        return idx
    return list(range(scenario["n_sessions"]))


def build_plan(scenario: dict, index: int, *, keep_recent_tokens: int, chars_per_token: float) -> Plan:
    exp = scenario["experiment"]
    seed = scenario["seed"] * 1000 + index
    rng = random.Random(seed)
    sid = f"{exp.lower()}-s{index:03d}"
    fs = F.make_facts(rng, sid, scenario["facts_per_source"], scenario["lures"], scenario.get("late_controls", 0))
    early = [f for f in fs if f.position == "early"]
    late = [f for f in fs if f.position == "late"]

    def chars(tokens: float) -> int:
        return int(tokens * chars_per_token)

    early_groups = _plant_items(rng, early)
    _mark_plant_steps(early_groups)
    phases = [{"name": "plant", "steps": _interleave(rng, early_groups, chars(scenario["plant_filler_tokens"]))}]

    if exp == "E1":
        for r in range(1, scenario["rounds"] + 1):
            phases.append({"name": f"round{r}",
                           "steps": filler(rng, chars(scenario["round_filler_x_keep"] * keep_recent_tokens))})
    elif exp == "E2":
        phases.append({"name": "bulk", "steps": filler(rng, chars(scenario["bulk_filler_x_keep"] * keep_recent_tokens))})
        late_groups = [plant_steps(rng, f) for f in late]
        _mark_plant_steps(late_groups)
        phases.append({"name": "tail",
                       "steps": _interleave(rng, late_groups, chars(scenario["tail_filler_x_keep"] * keep_recent_tokens))})
    else:
        raise ValueError(exp)

    order = [f.uid for f in fs]
    rng.shuffle(order)
    plan = Plan(exp, sid, seed, fs, phases, order)
    _check_no_leaks(plan)
    return plan
