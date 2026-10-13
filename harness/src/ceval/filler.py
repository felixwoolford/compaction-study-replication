"""Deterministic filler: realistic-looking agent work with no fact content.

A "step" is one of:
  {"kind": "user", "text": str}
  {"kind": "assistant", "thinking": str|None, "text": str|None,
   "calls": [{"name": str, "args": dict, "result": str, "is_error": bool}]}
Assistant steps with calls are expanded by the session writer into an assistant
message with toolCall blocks followed by one toolResult message per call.
"""

from __future__ import annotations

import random

# Filler-only vocabulary, disjoint from facts.ADJ / facts.ANIMAL / slot keys.
NOUNS = [
    "ledger", "parcel", "rivet", "spindle", "tally", "beacon", "chisel", "gasket",
    "harbor", "lantern", "mortar", "nozzle", "pylon", "ratchet", "sprocket", "trellis",
    "valve", "anvil", "bracket", "dynamo", "filament", "girder", "hinge", "kiln",
    "lathe", "mallet", "piston", "rotor", "sluice", "turbine", "winch", "cistern",
]
VERBS = [
    "load", "parse", "merge", "flush", "scan", "emit", "resolve", "encode",
    "render", "sync", "prune", "sort", "split", "fetch", "store", "check",
]
ATTRS = ["ready", "dirty", "stale", "pinned", "valid", "locked", "active", "queued"]


def _ident(rng: random.Random) -> str:
    return f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}"


def code_block(rng: random.Random, n_funcs: int) -> str:
    noun = rng.choice(NOUNS)
    lines = [f'"""Utilities for {noun} handling."""', "", "from __future__ import annotations", "",
             f"class {noun.capitalize()}Store:", f"    def __init__(self, items=None):",
             "        self._items = dict(items or {})", ""]
    for _ in range(n_funcs):
        name = _ident(rng)
        arg = rng.choice(NOUNS)
        var = rng.choice(NOUNS) + "s"
        attr = rng.choice(ATTRS)
        lines += [
            f"    def {name}(self, {arg}):",
            f'        """{name.replace("_", " ").capitalize()} for the given {arg}."""',
            f"        {var} = self._items.get({arg})",
            f"        if {var} is None:",
            "            return []",
            f"        return [x for x in {var} if getattr(x, {attr!r}, False)]",
            "",
        ]
    return "\n".join(lines)


def test_output(rng: random.Random, module: str, n: int) -> str:
    lines = [f"tests/test_{module}.py::test_{_ident(rng)} PASSED" for _ in range(n)]
    lines.append(f"{n} passed in {rng.uniform(0.2, 3.0):.2f}s")
    return "\n".join(lines)


def unit(rng: random.Random) -> list[dict]:
    module = f"{rng.choice(NOUNS)}_{rng.choice(NOUNS)}"
    kind = rng.choice(["read", "read", "test"])
    if kind == "read":
        code = code_block(rng, rng.randint(4, 12))
        n_funcs = code.count("    def ") - 1
        return [
            {"kind": "user", "text": f"Look at src/{module}.py and tell me briefly what it does."},
            {"kind": "assistant", "thinking": f"I should read src/{module}.py first.", "text": None,
             "calls": [{"name": "read", "args": {"path": f"src/{module}.py"}, "result": code, "is_error": False}]},
            {"kind": "assistant", "thinking": None,
             "text": f"src/{module}.py is a small store class with {n_funcs} filter-style helpers. Nothing unusual.",
             "calls": []},
        ]
    n = rng.randint(5, 25)
    return [
        {"kind": "user", "text": f"Run the tests for {module}."},
        {"kind": "assistant", "thinking": "Run the module's test file.", "text": None,
         "calls": [{"name": "bash", "args": {"command": f"pytest tests/test_{module}.py -q"},
                    "result": test_output(rng, module, n), "is_error": False}]},
        {"kind": "assistant", "thinking": None, "text": f"All {n} tests pass.", "calls": []},
    ]


def step_chars(step: dict) -> int:
    n = len(step.get("text") or "") + len(step.get("thinking") or "")
    for c in step.get("calls", []):
        n += len(str(c["args"])) + len(c["result"])
    return n


def filler(rng: random.Random, target_chars: int) -> list[dict]:
    """Whole units until at least target_chars of content."""
    steps: list[dict] = []
    total = 0
    while total < target_chars:
        u = unit(rng)
        steps += u
        total += sum(step_chars(s) for s in u)
    return steps
