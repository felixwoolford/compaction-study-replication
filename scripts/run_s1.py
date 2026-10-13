"""Run Study 1 (planted facts under repeated compaction) with the configured model and arms.
Resumable: rerun to continue; finished session lines are skipped. Results: results/<experiment>/.
  uv run python scripts/run_s1.py"""
from __future__ import annotations

import json
import sys

from common import ROOT, rcfg, require_setup

require_setup()
from ceval.config import load_config  # noqa: E402
from ceval.runner import Runner  # noqa: E402


def main() -> int:
    c = rcfg()["study1"]
    sc = json.loads((ROOT / "materials/scenarios/s1.json").read_text())
    sc.update(experiment=c["experiment"], arms=c["arms"], description="Study 1 replication")
    lo, hi = c["sessions"]
    ok = Runner(load_config(), sc).run(arms=c["arms"], indices=list(range(lo, hi + 1)), log=print)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
