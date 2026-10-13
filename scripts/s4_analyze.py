"""Study 4 analysis (PROTOCOL.md section 18), fixed before any Study 4 run.
  python s4_analyze.py <run copy root> <out.json>
Reads runs/S4/probes.json, results/S4/runs-rep<i>.jsonl and runs/S4/blind-rep<i>/ (key.json,
primary_scores.json, second_scores.json) in the run copy."""
import json
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ARMS = ("verbatim", "pi-vcc")
MIN_REPS = 3


def sign_test(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(root: Path):
    probes = {p["pid"]: p for p in json.loads((root / "runs/S4/probes.json").read_text())}
    turns = []  # one dict per scored reply
    kappa = []
    for f in sorted((root / "results/S4").glob("runs-rep*.jsonl")):
        rep = int(f.stem.split("rep")[1])
        b = root / "runs/S4" / f"blind-rep{rep}"
        key = json.loads((b / "key.json").read_text())
        prim = json.loads((b / "primary_scores.json").read_text())
        sc = {(key[i]["probe"], key[i]["arm"], key[i]["kind"]): v for i, v in prim.items() if i in key}
        if (b / "second_scores.json").exists():
            sec = json.loads((b / "second_scores.json").read_text())
            for i, v in sec.items():
                k = (key[i]["probe"], key[i]["arm"], key[i]["kind"]) if i in key else None
                if k in sc:
                    kappa.append((bool(sc[k]["pass"]), bool(v["pass"])))
        for line in f.read_text().splitlines():
            r = json.loads(line)
            s = sc.get((r["probe"], r["arm"], r["kind"]))
            turns.append({**r, "rep": rep, "pass": (bool(s["pass"]) if (r["ok"] and s) else (False if not r["ok"] and r["kind"] == "control" else None)),
                          "confident": (s.get("confident") if s else None)})
    return probes, turns, kappa


def cohen_kappa(pairs):
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def analyze(probes: dict, turns: list, kappa_pairs=None) -> dict:
    from ceval.stats import sign_flip_test
    ctl = {(t["probe"], t["arm"]): t["pass"] for t in turns if t["kind"] == "control"}
    per = defaultdict(list)
    for t in turns:
        if t["kind"] == "probe" and t["pass"] is not None:
            per[(t["probe"], t["arm"])].append(t)
    keep, excl = [], {"control_failed_or_missing": 0, "too_few_valid_reps": 0}
    for pid in sorted(probes):
        if not all(ctl.get((pid, a)) for a in ARMS):
            excl["control_failed_or_missing"] += 1
            continue
        if any(len(per[(pid, a)]) < MIN_REPS for a in ARMS):
            excl["too_few_valid_reps"] += 1
            continue
        keep.append(pid)

    def rate(pid, arm, f):
        ts = per[(pid, arm)]
        return sum(f(t) for t in ts) / len(ts)
    passr = {(p, a): rate(p, a, lambda t: t["pass"]) for p in keep for a in ARMS}
    conf = {(p, a): rate(p, a, lambda t: (not t["pass"]) and t["confident"] is True) for p in keep for a in ARMS}

    def flip(sel, values, **kw):
        d = [values[(p, ARMS[0])] - values[(p, ARMS[1])] for p in sel]
        if not d:
            return None
        r = sign_flip_test(d, **kw)
        return {"probes": len(d), "mean_diff": round(float(r["statistic"]), 4), "p": round(float(r["p"]), 6), "method": r["method"],
                "higher_verbatim": sum(x > 0 for x in d), "higher_pi_vcc": sum(x < 0 for x in d), "equal": sum(x == 0 for x in d)}

    def majority(sel):
        b = sum(passr[(p, ARMS[0])] > .5 and not passr[(p, ARMS[1])] > .5 for p in sel)
        c = sum(passr[(p, ARMS[1])] > .5 and not passr[(p, ARMS[0])] > .5 for p in sel)
        return {"verbatim_only": b, "pi_vcc_only": c, "p_sign": round(sign_test(b, c), 6)}

    sid = [probes[p]["sid"] for p in keep]
    unrepaired = [p for p in keep if not probes[p].get("repaired")]
    fresh = [p for p in keep if probes[p].get("group") == "fresh"]
    not_s3 = [p for p in keep if probes[p].get("group") != "s3"]
    out = {
        "probes_frozen": len(probes), "probes_analysed": len(keep), "exclusions": excl,
        "pass_rate_mean": {a: round(st.mean(passr[(p, a)] for p in keep), 4) if keep else None for a in ARMS},
        "H15_pass_rate": flip(keep, passr),
        "secondary": {
            "session_clustered": flip(keep, passr, clusters=sid),
            "majority_sign_test": majority(keep),
            "confident_wrong_rate_mean": {a: round(st.mean(conf[(p, a)] for p in keep), 4) if keep else None for a in ARMS},
            "confident_wrong": flip(keep, conf),
            "without_repaired_probes": flip(unrepaired, passr),
            "fresh_sessions_only": flip(fresh, passr),
            "without_study3_sessions": flip(not_s3, passr),
            "by_type": {ty: flip([p for p in keep if probes[p]["type"] == ty], passr) for ty in ("constraint", "decision")},
        },
        "scorer_agreement": {"pairs": len(kappa_pairs or []), "kappa": cohen_kappa(kappa_pairs or [])},
    }
    cost = {}
    for a in ARMS:
        ts = [t for t in turns if t["arm"] == a and t["kind"] == "probe" and t["ok"]]
        u = [t["usage"] for t in ts if t.get("usage")]
        cost[a] = {"turns": len(ts), "median_wall_s": st.median(t["wall_s"] for t in ts) if ts else None,
                   "total_wall_min": round(sum(t["wall_s"] for t in ts) / 60, 1),
                   "mean_output_tokens": round(st.mean(x["completion_tokens"] for x in u)) if u else None,
                   "mean_fresh_prompt_tokens": round(st.mean(x["prompt_eval_tokens"] for x in u)) if u else None,
                   "recovery_calls": sum(t.get("recovery_calls") or 0 for t in ts),
                   "errors": sum(1 for t in turns if t["arm"] == a and not t["ok"])}
    out["cost"] = cost
    return out


if __name__ == "__main__":
    root, outp = Path(sys.argv[1]), Path(sys.argv[2])
    probes, turns, kp = load(root)
    res = analyze(probes, turns, kp)
    outp.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
