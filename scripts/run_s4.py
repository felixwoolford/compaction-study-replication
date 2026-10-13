"""Run Study 4 (commitment probes on real pi sessions) with the configured model and arms.
  uv run python scripts/run_s4.py replay     # compact the 18 sessions: verbatim and pi-vcc without the model,
                                             # pi-default with the model (pi writes its summaries)
  uv run python scripts/run_s4.py run        # probe turns, all repetitions (resumable)
Needs the sessions (scripts/download_sessions.py). Results: results/S4/runs-rep<i>.jsonl, replies in runs/S4/.
The probe pipeline is the harness's study3 module (Study 4 ran its probes through Study 3's pipeline); its working
folders are set to runs/S4 and results/S4 here."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from common import ROOT, rcfg, require_setup

require_setup()
from ceval import driver as D  # noqa: E402
from ceval import realpi_replay as R  # noqa: E402
from ceval import schedule, usageproxy  # noqa: E402
from ceval import study3 as S  # noqa: E402

S.BASE, S.RES = Path("runs/S4"), Path("results/S4")  # the pipeline's working folders

BASE = ROOT / "runs/S4/replay"
DATA = ROOT / "runs/realpi/hf"
CFG = rcfg()
R.WINDOWS["48k"] = {**R.WINDOWS["48k"], "models_file": "config/pi-models-48k.json"}


def stage_materials() -> None:
    """The pipeline reads its frozen probes from runs/S4 and their hash from results/S4."""
    (ROOT / "runs/S4/replay").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/S4").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "materials/s4/probes.json", ROOT / "runs/S4/probes.json")
    shutil.copy(ROOT / "materials/s4/probes.sha256", ROOT / "results/S4/probes.sha256")
    shutil.copy(ROOT / "materials/s4/selection.json", BASE / "selection.json")


def sessions() -> list[dict]:
    return json.loads((BASE / "selection.json").read_text())["selected"]


def replay_one(cfg, method: str, proxy) -> None:
    done_f, rows_f = BASE / f"replays-{method}-48k.jsonl", BASE / f"rows-{method}-48k.jsonl"
    done = {json.loads(x)["sid"] for x in done_f.read_text().splitlines()} if done_f.exists() else set()
    for s in sessions():
        if s["sid"] in done:
            continue
        rep = R.replay(cfg, DATA, s, method, "48k", rows_f, proxy)
        with done_f.open("a") as f:
            f.write(json.dumps(rep) + "\n")
        print(method, "REPLAY", s["sid"], "compactions", rep["compactions"], "failed", rep["failed"], flush=True)


def replay() -> None:
    stage_materials()
    missing = [s["file"] for s in sessions() if not (DATA / s["file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sessions missing: run scripts/download_sessions.py")
    R.METHODS["pi-default"] = "pi-default"
    model_free = [a for a in CFG["study4"]["arms"] if a != "pi-default"]
    if model_free:  # verbatim and pi-vcc compact without a model: point pi at a dead upstream
        cfg = R._cfg_for("48k", BASE)
        schedule.require_llm = lambda _c: None
        proxy = usageproxy.UsageProxy(R.DEAD).start()
        D.set_usage_proxy(cfg, proxy, BASE / ".proxy-models.json")
        try:
            for m in model_free:
                replay_one(cfg, m, proxy)
        finally:
            proxy.stop()
    if "pi-default" in CFG["study4"]["arms"]:
        R.WINDOWS["48k"] = {**R.WINDOWS["48k"], "window": 72_576, "reserve": 32_768, "models_file": "config/pi-models-48k-r32k.json"}
        cfg = R._cfg_for("48k", BASE)
        cfg.raw.setdefault("usage_proxy", {})["enabled"] = False
        D.set_usage_proxy(cfg, None)
        replay_one(cfg, "pi-default", None)


def run() -> None:
    import os
    os.environ["S3_MODEL"] = "qwen"         # "use the configured model" (the harness's name for its local path)
    os.environ.setdefault("S3_MEMORY_NOTE", "1")
    c4 = CFG["study4"]
    skip = set(c4["skip_probes"])
    probes = json.loads((ROOT / "runs/S4/probes.json").read_text())
    failed = set()
    for a in c4["arms"]:
        f = BASE / f"replays-{a}-48k.jsonl"
        if not f.exists():
            raise SystemExit(f"no replay for {a}: run `run_s4.py replay` first")
        failed |= {(a, r["sid"]) for r in map(json.loads, f.read_text().splitlines()) if r["failed"] or r["entries_replayed"] < r["entries"]}
    if failed:
        print("replays that failed (their probes are left out for that arm and count as failures in the analysis):", sorted(failed), flush=True)
    S.KINDS = ("probe",)
    _fp, base_cfg = S.frozen_probes, S.probe_cfg
    m = CFG["model"]
    for i, seed in enumerate(c4["seeds"], 1):
        done = ROOT / S.RES / f"runs-rep{i}.jsonl"
        if done.exists():
            continue
        models = json.loads((ROOT / "config/pi-models.json").read_text())
        if m["kind"] != "builtin":
            for prov in models["providers"].values():
                for e in prov.get("models", []):
                    e["samplingParams"] = {**e.get("samplingParams", {}), "temperature": m["s4_temperature"], "seed": seed}
        mf = ROOT / S.BASE / f"pi-models-s4-s{seed}.json"
        mf.write_text(json.dumps(models, indent=2) + "\n")

        def cfg(r, mf=mf):
            c = base_cfg(r)
            c.raw["pi"]["models_file"] = str(mf)
            if m["kind"] == "builtin":
                c.raw["usage_proxy"]["enabled"] = False
            return c
        S.probe_cfg = cfg
        n = 0
        for arm in c4["arms"]:
            S.ARMS = (arm,)
            arm_skip = skip | {p["pid"] for p in probes if (arm, p["sid"]) in failed}
            S.frozen_probes = lambda r, s=arm_skip: [p for p in _fp(r) if p["pid"] not in s]
            n += len(S.frozen_probes(ROOT))
            print(f"repetition {i} (seed {seed}) {arm}", S.cmd_run(ROOT, workers=1), flush=True)
        rows = (ROOT / S.RES / "runs.jsonl").read_text().splitlines() if (ROOT / S.RES / "runs.jsonl").exists() else []
        if n == 0 or len(rows) < n:
            raise SystemExit(f"repetition {i} incomplete ({len(rows)} of {n} turns); rerun to resume")
        shutil.move(ROOT / S.RES / "runs.jsonl", done)
        shutil.move(ROOT / S.BASE / "replies", ROOT / S.BASE / f"replies-rep{i}")


if __name__ == "__main__":
    {"replay": replay, "run": run}[sys.argv[1]]()
