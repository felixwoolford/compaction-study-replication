"""Check that the model-free Study 4 replays (verbatim, pi-vcc) reproduce the paper's compactions exactly
(same compaction points, cuts and summary sizes; materials/s4/expected_replays.json). Run after `run_s4.py replay`.
  uv run python scripts/verify_replays.py"""
import json
import sys

from common import ROOT

exp = json.loads((ROOT / "materials/s4/expected_replays.json").read_text())
ok = True
for m in ("verbatim", "pi-vcc"):
    f = ROOT / f"runs/S4/replay/rows-{m}-48k.jsonl"
    if not f.exists():
        print(f"{m}: no replay yet"); ok = False; continue
    got = sorted([r["sid"], r["k"], r["entry_index"], r["summary_chars"], r.get("cut_entry_index")] for r in map(json.loads, f.read_text().splitlines()))
    same = got == exp[m]
    ok &= same
    print(f"{m}: {len(got)} compactions, {'identical to the paper' if same else 'DIFFERENT from the paper'}")
sys.exit(0 if ok else 1)
