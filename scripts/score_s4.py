"""Blind-score Study 4 replies with an LLM judge (any OpenAI-compatible chat-completions endpoint, from
replication.config.json), using the frozen brief materials/s4/scorer.md. Items carry no arm; each repetition is
packed and shuffled by the harness (study3.cmd_blind_pack). Writes runs/S4/blind-rep<i>/key.json and
primary_scores.json, the format the analysis reads. Resumable (scores are cached per item).
  JUDGE_API_KEY=... uv run python scripts/score_s4.py"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import urllib.request

from common import ROOT, rcfg

from pathlib import Path  # noqa: E402

from ceval import study3 as S  # noqa: E402

S.BASE, S.RES = Path("runs/S4"), Path("results/S4")

J = rcfg()["judge"]
BRIEF = (ROOT / "materials/s4/scorer.md").read_text()
INSTR = ("Score this one item. Reply with ONLY a JSON object: {\"pass\": true|false, \"confident\": true|false|null, "
         "\"note\": \"one short abstract reason\"}. confident is null when pass is true.\n\nITEM:\n")


def judge(item: dict) -> dict:
    body = {"model": J["model"], "temperature": J.get("temperature", 0), "messages": [
        {"role": "system", "content": BRIEF},
        {"role": "user", "content": INSTR + json.dumps({k: v for k, v in item.items() if k != "item"}, ensure_ascii=False)}]}
    req = urllib.request.Request(J["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[J['api_key_env']]}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            v = json.loads(m.group(0))
            return {"pass": bool(v["pass"]), "confident": v.get("confident"), "note": str(v.get("note", ""))[:200]}
        except Exception as e:  # transient API or parse failure: retry, then give up loudly
            err = e
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"judge failed on {item['item']}: {err!r}")


def main() -> int:
    if J["api_key_env"] not in os.environ:
        raise SystemExit(f"set {J['api_key_env']} to the judge API key")
    reps = sorted(int(p.stem.split("rep")[1]) for p in (ROOT / "results/S4").glob("runs-rep*.jsonl"))
    for i in reps:
        out = ROOT / f"runs/S4/blind-rep{i}"
        if (out / "primary_scores.json").exists():
            continue
        scratch = ROOT / f"runs/S4/.pack-rep{i}"
        shutil.rmtree(scratch, ignore_errors=True)
        (scratch / "runs/S4").mkdir(parents=True); (scratch / "results/S4").mkdir(parents=True)
        shutil.copy(ROOT / "runs/S4/probes.json", scratch / "runs/S4/probes.json")
        shutil.copy(ROOT / "results/S4/probes.sha256", scratch / "results/S4/probes.sha256")
        shutil.copy(ROOT / f"results/S4/runs-rep{i}.jsonl", scratch / "results/S4/runs.jsonl")
        (scratch / "runs/S4/replies").symlink_to(ROOT / f"runs/S4/replies-rep{i}")
        S.cmd_blind_pack(scratch)
        items = json.loads((scratch / "runs/S4/blind/primary_items.json").read_text())
        cache_f = scratch / "scores.partial.json"
        scores = json.loads(cache_f.read_text()) if cache_f.exists() else {}
        for n, it in enumerate(items, 1):
            if it["item"] not in scores:
                scores[it["item"]] = judge(it)
                cache_f.write_text(json.dumps(scores))
            print(f"repetition {i}: {n}/{len(items)}", end="\r", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(scratch / "runs/S4/blind/key.json", out / "key.json")
        (out / "primary_scores.json").write_text(json.dumps(scores, indent=1))
        print(f"\nrepetition {i}: {sum(v['pass'] for v in scores.values())}/{len(scores)} passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
