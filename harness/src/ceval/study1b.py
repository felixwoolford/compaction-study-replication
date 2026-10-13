"""Study 1b analysis (PROTOCOL.md, "Study 1b: extended compaction"): one 1..8
curve per session from S1 (rounds 1-3) and S1B (rounds 4-8), the H6-H8 family,
and the report.

Merging. S1B continues S1's lines, keeping S1's session ids (s1-s000, ...) and
recording the real round number, so the scored rows of the two experiments
concatenate directly: `merge_rounds` takes S1's pi-default and mech rows for
rounds 1-3 and S1B's for rounds 4-8, restricted to the sessions S1B ran. Each
row keeps its `experiment` (S1 or S1B).

    from ceval.study1b import load_merged
    df = load_merged(Path("results"))   # results/S1/scored.jsonl + results/S1B/scored.jsonl

Tests: the same session-level paired sign-flip tests as Study 1
(stats.sign_flip_test via stats._result), Holm within H6-H8. Decision rules as
for Study 1 (docs/build/analysis.md): a directional hypothesis is supported
when its Holm-adjusted p < 0.05 and the estimate has the predicted sign.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import stats

ARMS = ("pi-default", "mech")
S1_ROUNDS = (1, 2, 3)
S1B_ROUNDS = (4, 5, 6, 7, 8)
FAMILY = ["H6", "H7", "H8"]


def _frame(rows) -> pd.DataFrame:
    return rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))


def merge_rounds(s1_rows, s1b_rows, sessions=None) -> pd.DataFrame:
    """S1 rounds 1-3 and S1B rounds 4-8 for pi-default and mech, on the
    sessions S1B ran (or `sessions`)."""
    a, b = _frame(s1_rows), _frame(s1b_rows)
    if a.empty or b.empty:
        return pd.concat([a, b], ignore_index=True)
    a = a[a["arm"].isin(ARMS) & a["round"].isin(S1_ROUNDS)]
    b = b[b["arm"].isin(ARMS) & b["round"].isin(S1B_ROUNDS)]
    keep = set(sessions) if sessions is not None else set(b["session_id"])
    a = a[a["session_id"].isin(keep)]
    b = b[b["session_id"].isin(keep)]
    return pd.concat([a, b], ignore_index=True)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# Rows written by an erroneous resume (2026-10-07 14:25-15:58 JST): it re-ran
# pi-default lines of sessions 0-7 that had already stopped at a failed
# compaction. Only first attempts count (DEVIATIONS.md); these rows are dropped.
RETRY_WINDOW_UTC = ("2026-10-07T05:25", "2026-10-07T06:59")


def is_retry(row: dict) -> bool:
    ts = str(row.get("ts", ""))
    return (row.get("experiment") == "S1B" and row.get("arm") == "pi-default"
            and int(str(row.get("session_id", "s1-s999"))[-3:]) <= 7
            and RETRY_WINDOW_UTC[0] <= ts < RETRY_WINDOW_UTC[1])


def load_merged(results_root: Path, name: str = "scored.jsonl") -> pd.DataFrame:
    """Merged scored rows (or `name`, e.g. compactions.jsonl) of results/S1 and results/S1B,
    without the retried rows (is_retry)."""
    s1b = [r for r in _read_jsonl(results_root / "S1B" / name) if not is_retry(r)]
    return merge_rounds(_read_jsonl(results_root / "S1" / name), s1b)


# -----------------------------------------------------------------------------
# Hypotheses
# -----------------------------------------------------------------------------


def hypotheses(kept: pd.DataFrame, story_of=None, **kw) -> list[dict]:
    """H6: recall at round 8, mech − pi-default > 0.
    H7: confident-wrong rate over rounds 4-8, mech − pi-default < 0 (per
        session: confident wrong answers / planted-detail trials kept).
    H8: the recall gap (mech − pi-default) grows from round 3 to round 8:
        [mech(8) − pi-default(8)] − [mech(3) − pi-default(3)] > 0."""
    out = [stats.h_arm_diff("H6", kept, "recall", "mech", "pi-default", [8], story_of, +1, **kw),
           stats.h_arm_diff("H7", kept, "cw", "mech", "pi-default", list(S1B_ROUNDS), story_of, -1, **kw)]
    r3, r8 = stats.session_rates(kept, "recall", [3]), stats.session_rates(kept, "recall", [8])
    contrast = "[mech(r8) − pi-default(r8)] − [mech(r3) − pi-default(r3)]"
    if all(a in r3 and a in r8 for a in ARMS):
        t = pd.concat({"m3": r3["mech"], "p3": r3["pi-default"], "m8": r8["mech"], "p8": r8["pi-default"]},
                      axis=1).dropna()
    else:
        t = pd.DataFrame()
    if t.empty:
        out.append(stats._empty("H8", contrast, "recall gap, round 8 − round 3", "no session with both arms at rounds 3 and 8", +1))
    else:
        out.append(stats._result("H8", contrast, "recall gap, round 8 − round 3",
                                 (t["m8"] - t["p8"]) - (t["m3"] - t["p3"]), story_of, direction=+1, **kw))
    return out


def decide(results: list[dict]) -> list[dict]:
    """Holm within H6-H8, then the decision, at session and at story level."""
    for level in ("session", "story"):
        ps = [r["p"] if level == "session" else r["story"]["p"] for r in results]
        for r, pa in zip(results, stats.holm(ps)):
            sup = stats._supported(r, pa)
            text = _decision(r, sup, pa)
            if level == "session":
                r.update(p_holm=pa, supported=sup, decision=text)
            else:
                r["story"].update(p_holm=pa, supported=sup, decision=text)
    return results


def _decision(r: dict, supported: bool, p_adj: float) -> str:
    if math.isnan(p_adj):
        return "not tested (" + (r.get("note") or "no data") + ")"
    if supported:
        return "supported"
    if p_adj < stats.ALPHA:
        return "not supported (significant, opposite direction)"
    return "not supported"


def run_family(kept: pd.DataFrame, story_of=None, **kw) -> list[dict]:
    return decide(hypotheses(kept, story_of, **kw))


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def report(results_root: Path, fit_models: bool = False) -> Path:
    """results/S1B/report.md: confirmatory H6-H8 and the 1..8 curves."""
    from .analysis import _exclusions_md, _repo_root, confirmatory_md, md_table, story_map

    res = results_root / "S1B"
    rows = load_merged(results_root)
    kept, excl = stats.prepare_recall(rows)
    story_of, src = story_map("S1B", sorted(rows["session_id"].unique()), _repo_root(res))
    results = run_family(kept, story_of)
    curve = kept.groupby(["arm", "round"]).agg(recall=("recall", "mean"), confident_wrong=("cw", "mean"),
                                               trials=("recall", "size")).reset_index()
    comps = load_merged(results_root, "compactions.jsonl")
    parts = ["# Report: S1B (Study 1b, with S1 rounds 1-3)", "",
             f"Confirmatory family: {', '.join(FAMILY)} (Holm within the family). Sessions: "
             f"{rows['session_id'].nunique()}. Planted-detail trials analysed: {len(kept)}. "
             "With 6 sessions the exact two-sided sign-flip test has 64 sign patterns: the smallest possible p is "
             "2/64 = 0.031 (see PROTOCOL.md, Study 1b).", "",
             confirmatory_md(results, src), _exclusions_md(excl, "trials"),
             "## Recall and confident wrong answers by arm × round (S1 rounds 1-3, S1B rounds 4-8)", "",
             md_table(curve), ""]
    if not comps.empty:
        cols = [c for c in ("summary_chars", "real_tokens_before", "real_tokens_after", "compact_ms") if c in comps]
        size = comps.groupby(["arm", "round"])[cols].mean().reset_index()
        parts += ["## Summary / checkpoint size and compaction by arm × round (means)", "", md_table(size, "{:.0f}"), ""]
        if "mech_checkpoint" in comps:
            m = comps[(comps["arm"] == "mech") & comps["mech_checkpoint"].notna()]
            if len(m):
                trim = pd.DataFrame([{"session_id": s, "round": r, **c} for s, r, c in
                                     zip(m["session_id"], m["round"], m["mech_checkpoint"])])
                parts += ["## mech's size guard: trimming by round", "",
                          md_table(trim.groupby("round")[["span_chars", "lines_dropped", "user_lines_stubbed",
                                                           "spans_trimmed", "trimmed"]].mean().reset_index(), "{:.2f}"), ""]
    out = res / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
