"""Analyse Study 4: per-probe pass rate and confident-wrong rate per arm over the repetitions (blind judge scores),
on the probes not skipped by the control rule; failed turns and probes an arm could not run count as failures.
Paired two-sided sign-flip tests across probes for every pair of arms (PROTOCOL 18.6 style).
  uv run python scripts/analyze_s4.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from itertools import combinations

from common import ROOT, rcfg

from ceval.stats import sign_flip_test  # noqa: E402

spec = importlib.util.spec_from_file_location("s4a", ROOT / "scripts/s4_analyze.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def main() -> int:
    c4 = rcfg()["study4"]
    probes, turns, kappa = A.load(ROOT)
    keep = sorted(p for p in probes if p not in set(c4["skip_probes"]))
    reps = len(c4["seeds"])
    passes, conf = defaultdict(int), defaultdict(int)
    for t in turns:
        if t["kind"] == "probe" and t["ok"] and t["pass"] is not None:
            passes[(t["probe"], t["arm"])] += bool(t["pass"])
            conf[(t["probe"], t["arm"])] += (not t["pass"]) and t.get("confident") is True
    arms = [a for a in c4["arms"] if any(t["arm"] == a for t in turns)]
    rate = lambda d, p, a: d[(p, a)] / reps  # missing or failed turns count as failures
    out = {"probes": len(keep), "repetitions": reps,
           "pass_rate": {a: round(sum(rate(passes, p, a) for p in keep) / len(keep), 4) for a in arms},
           "confident_wrong_rate": {a: round(sum(rate(conf, p, a) for p in keep) / len(keep), 4) for a in arms}, "tests": {}}
    for x, y in combinations(arms, 2):
        for d, name in ((passes, "pass"), (conf, "confident_wrong")):
            diff = [rate(d, p, x) - rate(d, p, y) for p in keep]
            t = sign_flip_test(diff)
            out["tests"][f"{name}: {x} - {y}"] = {"mean_diff": round(float(t["statistic"]), 4), "p": round(float(t["p"]), 6),
                                                   "probes_x_higher": sum(v > 0 for v in diff), "probes_y_higher": sum(v < 0 for v in diff)}
    (ROOT / "results/S4/replication-analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
