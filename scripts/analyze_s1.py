"""Score and analyse Study 1: rule-based scoring (`ceval score`), then per-session recall and confident-wrong rates
over rounds 1-3 and paired sign-flip tests of verbatim against each other arm (PROTOCOL sections 9.1 and 9.4).
  uv run python scripts/analyze_s1.py"""
from __future__ import annotations

import json
import sys

from common import ROOT, rcfg

import pandas as pd  # noqa: E402

from ceval import stats as S  # noqa: E402
from ceval.analysis import story_map  # noqa: E402


def main() -> int:
    exp = rcfg()["study1"]["experiment"]
    from ceval.cli import main as ceval_main
    ceval_main(["score", exp])
    res = ROOT / "results" / exp
    rows = pd.DataFrame([json.loads(x) for x in (res / "scored.jsonl").read_text().splitlines() if x.strip()])
    kept, excl = S.prepare_recall(rows)
    story_of, _ = story_map(exp, sorted(rows["session_id"].unique()), ROOT)
    arms = [a for a in rcfg()["study1"]["arms"] if a in set(rows["arm"])]
    rec = S.session_rates(kept, "recall", [1, 2, 3]); cw = S.session_rates(kept, "cw", [1, 2, 3])
    out = {"recall_mean": {a: round(float(rec[a].mean()), 4) for a in arms},
           "confident_wrong_mean": {a: round(float(cw[a].mean()), 4) for a in arms}, "tests": []}
    if "verbatim" in arms:
        for other in (a for a in arms if a != "verbatim"):
            for col, d in (("recall", 1), ("cw", -1)):
                r = S.h_arm_diff(f"{col}: verbatim - {other}", kept, col, "verbatim", other, [1, 2, 3], story_of, d)
                out["tests"].append({k: r[k] for k in ("hypothesis", "estimate", "ci_lo", "ci_hi", "p", "n_sessions")} | {"story_p": r["story"]["p"]})
    (res / "replication-analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
