"""Download the 18 real pi sessions Study 4 uses from the public Hugging Face dataset
MaxDevv/real-pi-coding-agent-traces-sessions (each session stays under its source's licence; all 18 are MIT),
verify their SHA-256 against materials/s4/selection.json, and put them in runs/realpi/hf/.
  uv run python scripts/download_sessions.py"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sel = json.loads((ROOT / "materials/s4/selection.json").read_text())
out = ROOT / "runs/realpi/hf"
out.mkdir(parents=True, exist_ok=True)
base = f"https://huggingface.co/datasets/{sel['dataset']['repo']}/resolve/main/"
bad = 0
for s in sel["selected"]:
    f = out / s["file"]
    if not f.exists():
        print("downloading", s["file"], flush=True)
        urllib.request.urlretrieve(base + urllib.request.quote(s["file"]), f)
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != s["sha256"]:
        print(f"SHA-256 MISMATCH {s['file']}: got {h}", flush=True)
        bad += 1
print(f"{len(sel['selected']) - bad} of {len(sel['selected'])} sessions verified in {out}")
sys.exit(1 if bad else 0)
