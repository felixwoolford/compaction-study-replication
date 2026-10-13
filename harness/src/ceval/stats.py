"""Confirmatory statistics (PROTOCOL.md §9.1): paired session-level sign-flip
permutation tests, Holm correction, bootstrap intervals, per-session outcome
builders and one function per hypothesis.

Decision rules, exclusion handling and the choices made where the protocol is
silent are documented in `docs/build/analysis.md`. In short:

- Every test is a two-sided sign-flip test of the per-session mean difference
  at alpha = 0.05 (PROTOCOL §9.1), except H1m, which is one-sided
  non-inferiority with margin -0.05. A directional hypothesis is *supported*
  when its Holm-adjusted p < 0.05 **and** the estimate has the predicted sign.
- Story-level sensitivity: the sessions of one story flip together (one sign
  per story); the statistic is still the mean over sessions.
- H3 is an intersection-union test: p(H3) = max(p_a, p_b), supported only when
  both parts go in the predicted direction.
- H6 is two-sided with a three-way outcome (mech lower / mech higher / no
  difference detected); "no difference detected" is not evidence of equality.
- If H4 is not supported, H5 and H6 are labelled descriptive.

Scoring is blind to arm; these functions read the arm only to form contrasts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .scoring import HEDGE_RE

ALPHA = 0.05
SEED = 20261005
EXACT_MAX_UNITS = 16  # exact enumeration up to 2^16 sign patterns
NI_MARGIN = -0.05  # H1m non-inferiority margin (PROTOCOL §3)

FAMILIES = {"S1": ["H1", "H2", "H3", "H1m"], "S2": ["H4", "H5", "H6"]}
# Study 1b (PROTOCOL.md, "Study 1b"): its own family, tested by ceval.study1b
# on S1 rounds 1-3 merged with S1B rounds 4-8. Its H6 is not Study 2's H6.
FAMILIES["S1B"] = ["H6", "H7", "H8"]
STUDY_OF_EXP = {"S1": "S1", "P1": "S1", "P1B": "S1", "P1C": "S1", "S2": "S2", "P2": "S2"}

REORIENT = ("mech-nolookup", "mech-nosig")  # re-orient block on
NO_REORIENT = ("mech-noreorient", "narration-only")  # re-orient block off
S2_ARMS = ("none", "pi-default", "mech", "mech-nolookup", "mech-noreorient", "mech-nosig", "narration-only")


# -----------------------------------------------------------------------------
# Tests and intervals
# -----------------------------------------------------------------------------


def _signs(n: int) -> np.ndarray:
    """All 2^n sign patterns as a (2^n, n) array of +-1."""
    if n == 0:
        return np.ones((1, 0))
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    return 1.0 - 2.0 * bits


def sign_flip_test(diffs: Sequence[float], alternative: str = "two-sided", clusters: Sequence | None = None,
                   margin: float = 0.0, n_mc: int = 100_000, seed: int = SEED) -> dict:
    """Paired sign-flip permutation test of mean(diffs) against `margin`.

    Under H0 each unit's centred difference (d - margin) is symmetric about 0,
    so its sign is exchangeable. Units are sessions, or, with `clusters`, groups
    of sessions that flip together (story-level sensitivity: one sign per story,
    the statistic stays the mean over sessions).

    Exact enumeration when 2^units <= 2^16, else seeded Monte Carlo with
    p = (1 + #extreme) / (1 + n_mc). Returns statistic (mean of the raw
    diffs), p, n_sessions, n_units, n_patterns, min_p (the smallest p any sign
    assignment of these data could give: zero units cannot change the
    statistic) and design_min_p (all units non-zero).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative: {alternative!r}")
    d = np.asarray(diffs, dtype=float)
    if d.size == 0 or np.isnan(d).any():
        if d.size == 0:
            return {"statistic": math.nan, "p": math.nan, "n_sessions": 0, "n_units": 0, "n_patterns": 0,
                    "min_p": math.nan, "design_min_p": math.nan, "method": "none", "alternative": alternative,
                    "margin": margin}
        raise ValueError("diffs contain NaN")
    c = d - margin
    if clusters is None:
        unit = c
    else:
        labels = pd.Series(list(clusters)).astype(str).to_numpy()
        if len(labels) != len(d):
            raise ValueError("clusters must have one label per diff")
        unit = np.array([c[labels == u].sum() for u in dict.fromkeys(labels)])
    n = len(d)
    k = len(unit)
    t_obs = unit.sum() / n
    tol = 1e-9 * max(1.0, float(np.abs(unit).sum()) / n)
    if k <= EXACT_MAX_UNITS:
        S = _signs(k)
        method, n_pat = "exact", 2**k
    else:
        rng = np.random.default_rng(seed)
        S = np.vstack([np.ones((1, k)), rng.choice([-1.0, 1.0], size=(n_mc, k))])
        method, n_pat = "monte-carlo", n_mc
    T = S @ unit / n
    if alternative == "two-sided":
        hits = np.abs(T) >= abs(t_obs) - tol
    elif alternative == "greater":
        hits = T >= t_obs - tol
    else:
        hits = T <= t_obs + tol
    if method == "exact":
        p = float(hits.mean())
    else:
        p = float(hits.sum()) / (n_mc + 1)  # identity pattern included
    k_eff = int((np.abs(unit) > tol).sum())
    sides = 2 if alternative == "two-sided" else 1
    min_p = min(1.0, sides / 2**k_eff)
    return {"statistic": float(d.mean()), "p": min(1.0, p), "n_sessions": n, "n_units": k, "n_patterns": n_pat,
            "min_p": min_p, "design_min_p": min(1.0, sides / 2**k), "method": method,
            "alternative": alternative, "margin": margin}


def holm(pvalues: Sequence[float | None]) -> list[float]:
    """Holm step-down adjusted p-values. NaN/None entries stay NaN and do not
    count towards the family size."""
    p = np.array([math.nan if v is None else float(v) for v in pvalues])
    ok = np.flatnonzero(~np.isnan(p))
    out = np.full(len(p), math.nan)
    m = len(ok)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def bootstrap_mean_ci(values: Sequence[float], clusters: Sequence | None = None, n_boot: int = 10_000,
                      level: float = 0.95, seed: int = SEED) -> tuple[float, float, float]:
    """Percentile bootstrap interval for the mean of per-session values,
    resampling sessions (or, with `clusters`, whole clusters) with replacement."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boots = v[idx].mean(axis=1)
    else:
        labels = pd.Series(list(clusters)).astype(str).to_numpy()
        groups = list(dict.fromkeys(labels))
        sums = np.array([v[labels == g].sum() for g in groups])
        sizes = np.array([(labels == g).sum() for g in groups], dtype=float)
        idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        boots = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_diff_ci(per_unit_a: Sequence[float], per_unit_b: Sequence[float], clusters: Sequence | None = None,
                      n_boot: int = 10_000, level: float = 0.95, seed: int = SEED) -> tuple[float, float, float]:
    """Paired bootstrap interval for mean(a - b): units are resampled as pairs."""
    a = np.asarray(per_unit_a, dtype=float)
    b = np.asarray(per_unit_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have the same length")
    return bootstrap_mean_ci(a - b, clusters=clusters, n_boot=n_boot, level=level, seed=seed)


# -----------------------------------------------------------------------------
# Per-session outcomes (Study 1)
# -----------------------------------------------------------------------------

EXCLUSION_REASONS = ("turn_failed", "cut_violation", "leak", "a0_compacted")


def _flag(df: pd.DataFrame, col: str, value) -> pd.Series:
    """True where df[col] equals value; None/NaN/missing column never match."""
    if col not in df:
        return pd.Series(False, index=df.index)
    return df[col].map(lambda x: x is not None and bool(x == value)).astype(bool)


def confident_wrong_of(row: Mapping) -> int | None:
    """`confident_wrong` as scoring.py defines it, for rows scored before the field existed."""
    cat = row.get("category")
    if cat == "needs_review":
        return None
    return int(cat in ("incorrect", "intrusion") and not HEDGE_RE.search(str(row.get("answer") or "")))


def prepare_recall(rows: pd.DataFrame | Sequence[Mapping], unanswered: str = "not_recalled") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Planted-fact trials ready for per-session outcomes, plus exclusion counts by arm.

    Exclusions (PROTOCOL §9.4), in this order, each row counted once:
    `turn_ok` false (failed bridge call or timeout), `cut_ok` false, `leak` true.
    A `needs_review` row from a successful turn is a question still unanswered
    after the follow-up (§5.3): with unanswered="not_recalled" (the protocol's
    rule) it counts as recall 0 and confident-wrong 0; with "exclude" it is
    dropped (sensitivity only). Lures and late controls are not planted facts.
    """
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if "is_lure" in df:
        df = df[~df["is_lure"].fillna(False).astype(bool)]
    if "position" in df:
        df = df[df["position"].fillna("early") != "late"]
    df = df.copy()
    failed = ~df["turn_ok"].fillna(False).astype(bool) if "turn_ok" in df else pd.Series(False, index=df.index)
    cut = _flag(df, "cut_ok", False) & ~failed
    leak = _flag(df, "leak", True) & ~failed & ~cut
    a0c = _flag(df, "a0_compacted", True) & ~failed & ~cut & ~leak  # A0 silently compacted (§6.1)
    review = (df["category"] == "needs_review") & ~failed & ~cut & ~leak & ~a0c
    df["exclusion"] = None
    df.loc[failed, "exclusion"] = "turn_failed"
    df.loc[cut, "exclusion"] = "cut_violation"
    df.loc[leak, "exclusion"] = "leak"
    df.loc[a0c, "exclusion"] = "a0_compacted"
    if unanswered == "exclude":
        df.loc[review, "exclusion"] = "unanswered"
    elif unanswered != "not_recalled":
        raise ValueError(f"unanswered: {unanswered!r}")
    if "confident_wrong" not in df:
        df["confident_wrong"] = [confident_wrong_of(r) for r in df.to_dict("records")]
    df["recall"] = (df["category"] == "correct").astype(int)
    cw = pd.to_numeric(df["confident_wrong"], errors="coerce")
    df["cw"] = cw.fillna(0).astype(int)  # unanswered: not confident-wrong
    df["unanswered"] = review.astype(int)
    counts = []
    for arm, g in df.groupby("arm", sort=False):
        row = {"arm": arm, "trials": len(g)}
        for r in EXCLUSION_REASONS + (("unanswered",) if unanswered == "exclude" else ()):
            row[r] = int((g["exclusion"] == r).sum())
        row["excluded"] = int(g["exclusion"].notna().sum())
        row["excluded_pct"] = 100.0 * row["excluded"] / max(1, len(g))
        row["unanswered_scored_not_recalled"] = int(g["unanswered"].sum()) if unanswered == "not_recalled" else 0
        row["over_10pct"] = row["excluded_pct"] > 10
        counts.append(row)
    kept = df[df["exclusion"].isna()].copy()
    return kept, pd.DataFrame(counts)


def session_rates(df: pd.DataFrame, col: str, rounds: Sequence[int] | None = None) -> pd.DataFrame:
    """session_id x arm table of the pooled mean of `col` over the given rounds."""
    d = df if rounds is None else df[df["round"].isin(list(rounds))]
    return d.groupby(["session_id", "arm"])[col].mean().unstack("arm")


def _paired(table: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Sessions with every named arm observed (pairing, PROTOCOL §8)."""
    if any(a not in table for a in arms):
        return table.iloc[0:0].reindex(columns=list(arms))
    return table[list(arms)].dropna()


# -----------------------------------------------------------------------------
# Hypotheses
# -----------------------------------------------------------------------------


def _result(hyp: str, contrast: str, outcome: str, values: pd.Series, story_of: Mapping | None, *,
            alternative: str = "two-sided", direction: int = 0, margin: float = 0.0, n_boot: int = 10_000,
            seed: int = SEED) -> dict:
    """Test one per-session contrast at session level and at story level.

    direction: +1 / -1 for the predicted sign (0: two-sided, no prediction)."""
    sids = list(values.index)
    v = values.to_numpy(dtype=float)
    stories = [str((story_of or {}).get(s, s)) for s in sids]
    t = sign_flip_test(v, alternative=alternative, margin=margin, seed=seed)
    ts = sign_flip_test(v, alternative=alternative, margin=margin, clusters=stories, seed=seed)
    est, lo, hi = bootstrap_mean_ci(v, n_boot=n_boot, seed=seed)
    _, slo, shi = bootstrap_mean_ci(v, clusters=stories, n_boot=n_boot, seed=seed)
    return {"hypothesis": hyp, "contrast": contrast, "outcome": outcome, "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "p": t["p"], "alternative": alternative, "margin": margin, "direction": direction,
            "n_sessions": t["n_sessions"], "n_patterns": t["n_patterns"], "min_p": t["min_p"],
            "design_min_p": t["design_min_p"], "method": t["method"],
            "story": {"p": ts["p"], "n_units": ts["n_units"], "n_patterns": ts["n_patterns"], "min_p": ts["min_p"],
                      "ci_lo": slo, "ci_hi": shi},
            "per_session": dict(zip(sids, v.tolist())), "note": ""}


def _empty(hyp: str, contrast: str, outcome: str, note: str, direction: int = 0, alternative: str = "two-sided",
           margin: float = 0.0) -> dict:
    nan = math.nan
    return {"hypothesis": hyp, "contrast": contrast, "outcome": outcome, "estimate": nan, "ci_lo": nan, "ci_hi": nan,
            "p": nan, "alternative": alternative, "margin": margin, "direction": direction, "n_sessions": 0,
            "n_patterns": 0, "min_p": nan, "design_min_p": nan, "method": "none",
            "story": {"p": nan, "n_units": 0, "n_patterns": 0, "min_p": nan, "ci_lo": nan, "ci_hi": nan},
            "per_session": {}, "note": note}


def h_arm_diff(hyp: str, kept: pd.DataFrame, col: str, arm_a: str, arm_b: str, rounds: Sequence[int],
               story_of: Mapping | None, direction: int, alternative: str = "two-sided", margin: float = 0.0,
               **kw) -> dict:
    tab = _paired(session_rates(kept, col, rounds), [arm_a, arm_b])
    name = {"recall": "recall", "cw": "confident-wrong rate"}.get(col, col)
    contrast = f"{arm_a} − {arm_b}"
    outcome = f"{name}, round{'s' if len(rounds) > 1 else ''} {','.join(map(str, rounds))}"
    if tab.empty:
        return _empty(hyp, contrast, outcome, f"no session has both {arm_a} and {arm_b}", direction, alternative, margin)
    return _result(hyp, contrast, outcome, tab[arm_a] - tab[arm_b], story_of, alternative=alternative,
                   direction=direction, margin=margin, **kw)


def h3(kept: pd.DataFrame, story_of: Mapping | None, **kw) -> dict:
    """H3: (a) pi-default round3 − round1 confident-wrong > 0 and (b) the
    difference in slopes, mech − pi-default, < 0 (operationalising "≤ 0").
    Intersection-union: p = max(p_a, p_b); both parts must have the predicted sign."""
    r1 = session_rates(kept, "cw", [1])
    r3 = session_rates(kept, "cw", [3])
    contrast_b = "[mech(r3) − mech(r1)] − [pi-default(r3) − pi-default(r1)]"
    pa = _paired(pd.concat({"r1": r1.get("pi-default"), "r3": r3.get("pi-default")}, axis=1)
                 if "pi-default" in r1 and "pi-default" in r3 else pd.DataFrame(), ["r1", "r3"])
    if pa.empty:
        a = _empty("H3a", "pi-default: round 3 − round 1", "confident-wrong rate", "no pi-default rounds 1 and 3", 1)
    else:
        a = _result("H3a", "pi-default: round 3 − round 1", "confident-wrong rate", pa["r3"] - pa["r1"], story_of,
                    direction=1, **kw)
    need = ["mech", "pi-default"]
    if all(x in r1 and x in r3 for x in need):
        sl = pd.concat({"m1": r1["mech"], "m3": r3["mech"], "p1": r1["pi-default"], "p3": r3["pi-default"]}, axis=1).dropna()
    else:
        sl = pd.DataFrame()
    if sl.empty:
        b = _empty("H3b", contrast_b, "confident-wrong rate", "no session with mech and pi-default rounds 1 and 3", -1)
    else:
        b = _result("H3b", contrast_b, "confident-wrong rate", (sl["m3"] - sl["m1"]) - (sl["p3"] - sl["p1"]),
                    story_of, direction=-1, **kw)
    p = max(a["p"], b["p"]) if not (math.isnan(a["p"]) or math.isnan(b["p"])) else math.nan
    sp = max(a["story"]["p"], b["story"]["p"]) if not math.isnan(p) else math.nan
    out = dict(b)
    out.update({"hypothesis": "H3", "contrast": "H3a: " + a["contrast"] + "; H3b: " + contrast_b,
                "outcome": "confident-wrong rate (round 3 − round 1)", "p": p, "direction": 0,
                "min_p": max(a["min_p"], b["min_p"]), "parts": [a, b],
                "note": "intersection-union: p = max(p_a, p_b); supported only if H3a > 0 and H3b < 0"})
    out["story"] = dict(b["story"], p=sp, min_p=max(a["story"]["min_p"], b["story"]["min_p"]))
    return out


def recall_hypotheses(kept: pd.DataFrame, story_of: Mapping | None = None, **kw) -> list[dict]:
    """Study 1: H1, H2, H3, H1m. Rounds 1–3; A0 (`none`) only in H1m."""
    rounds = [1, 2, 3]
    return [h_arm_diff("H1", kept, "recall", "mech", "pi-default", rounds, story_of, +1, **kw),
            h_arm_diff("H2", kept, "cw", "mech", "pi-default", rounds, story_of, -1, **kw),
            h3(kept, story_of, **kw),
            h_arm_diff("H1m", kept, "recall", "mech", "none", [1], story_of, +1, alternative="greater",
                       margin=NI_MARGIN, **kw)]


# -----------------------------------------------------------------------------
# Study 2
# -----------------------------------------------------------------------------


def prepare_episodes(rows: pd.DataFrame | Sequence[Mapping]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Episodes for the S2 tests, plus exclusion counts by arm (§9.4): failed
    turns (`turn_ok` false or `timeout`), `cut_ok` false, `leak` true, and
    `fabricated_progress` null (needs_review not resolved by the coder)."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df = df.copy()
    failed = ~df["turn_ok"].fillna(False).astype(bool) if "turn_ok" in df else pd.Series(False, index=df.index)
    if "timeout" in df:
        failed |= df["timeout"].fillna(False).astype(bool)
    cut = _flag(df, "cut_ok", False) & ~failed
    leak = _flag(df, "leak", True) & ~failed & ~cut
    a0c = _flag(df, "a0_compacted", True) & ~failed & ~cut & ~leak  # A0 silently compacted (§6.1)
    fp = pd.to_numeric(df.get("fabricated_progress"), errors="coerce") if "fabricated_progress" in df else pd.Series(math.nan, index=df.index)
    review = fp.isna() & ~failed & ~cut & ~leak & ~a0c
    df["exclusion"] = None
    df.loc[failed, "exclusion"] = "turn_failed"
    df.loc[cut, "exclusion"] = "cut_violation"
    df.loc[leak, "exclusion"] = "leak"
    df.loc[a0c, "exclusion"] = "a0_compacted"
    df.loc[review, "exclusion"] = "needs_review"
    counts = []
    for arm, g in df.groupby("arm", sort=False):
        row = {"arm": arm, "episodes": len(g)}
        for r in EXCLUSION_REASONS + ("needs_review",):
            row[r] = int((g["exclusion"] == r).sum())
        row["excluded"] = int(g["exclusion"].notna().sum())
        row["excluded_pct"] = 100.0 * row["excluded"] / max(1, len(g))
        row["over_10pct"] = row["excluded_pct"] > 10
        counts.append(row)
    kept = df[df["exclusion"].isna()].copy()
    kept["fabricated_progress"] = pd.to_numeric(kept["fabricated_progress"]).astype(int)
    return kept, pd.DataFrame(counts)


def continuation_hypotheses(kept: pd.DataFrame, story_of: Mapping | None = None, **kw) -> list[dict]:
    """H4, H5, H6 on per-session fabricated progress (0/1). Study 2 has one
    session per task, so the story-level analysis equals the session level
    unless `story_of` groups tasks."""
    tab = kept.groupby(["session_id", "arm"])["fabricated_progress"].mean().unstack("arm")
    out = []
    t = _paired(tab, ["narration-only", "none"])
    out.append(_empty("H4", "narration-only − none", "fabricated progress", "missing arms", 1) if t.empty else
               _result("H4", "narration-only − none", "fabricated progress", t["narration-only"] - t["none"],
                       story_of, direction=1, **kw))
    t = _paired(tab, list(REORIENT + NO_REORIENT))
    c5 = "((mech-nolookup + mech-nosig) − (mech-noreorient + narration-only)) / 2"
    out.append(_empty("H5", c5, "fabricated progress", "missing arms", -1) if t.empty else
               _result("H5", c5, "fabricated progress",
                       (t[list(REORIENT)].sum(axis=1) - t[list(NO_REORIENT)].sum(axis=1)) / 2, story_of,
                       direction=-1, **kw))
    t = _paired(tab, ["mech", "pi-default"])
    out.append(_empty("H6", "mech − pi-default", "fabricated progress", "missing arms", 0) if t.empty else
               _result("H6", "mech − pi-default", "fabricated progress", t["mech"] - t["pi-default"], story_of,
                       direction=0, **kw))
    return out


# -----------------------------------------------------------------------------
# Decisions
# -----------------------------------------------------------------------------


def _supported(r: dict, p_adj: float) -> bool:
    if math.isnan(p_adj) or p_adj >= ALPHA:
        return False
    if r["hypothesis"] == "H3":
        a, b = r["parts"]
        return a["estimate"] > 0 and b["estimate"] < 0
    if r["direction"] == 0 or r["alternative"] != "two-sided":
        return True  # one-sided tests (H1m) carry their direction in the test
    return math.copysign(1, r["estimate"]) == r["direction"] and r["estimate"] != 0


def decide(results: list[dict]) -> list[dict]:
    """Holm within the family (the list passed), then the decision per
    hypothesis, at session level and at story level. Mutates and returns."""
    for level, get, put in (("session", lambda r: r["p"], "p_holm"),
                            ("story", lambda r: r["story"]["p"], None)):
        adj = holm([get(r) for r in results])
        for r, pa in zip(results, adj):
            if level == "session":
                r["p_holm"] = pa
                r["supported"] = _supported(r, pa)
            else:
                r["story"]["p_holm"] = pa
                r["story"]["supported"] = _supported(r, pa)
    gate_failed = any(r["hypothesis"] == "H4" and not r["supported"] for r in results)
    for r in results:
        r["decision"] = _decision_text(r, r["supported"], r["p_holm"])
        r["story"]["decision"] = _decision_text(r, r["story"]["supported"], r["story"]["p_holm"])
        if gate_failed and r["hypothesis"] in ("H5", "H6"):
            r["supported"] = None
            r["story"]["supported"] = None
            r["decision"] = "descriptive (H4 not supported): " + r["decision"]
            r["story"]["decision"] = "descriptive (H4 not supported): " + r["story"]["decision"]
    return results


def _decision_text(r: dict, supported: bool, p_adj: float) -> str:
    if math.isnan(p_adj):
        return "not tested (" + (r.get("note") or "no data") + ")"
    if r["hypothesis"] == "H6":
        if p_adj >= ALPHA:
            return "no difference detected"
        return "mech lower" if r["estimate"] < 0 else "mech higher (H6 contradicted)"
    if supported:
        return "supported"
    if p_adj < ALPHA:
        return "not supported (significant, opposite direction)"
    return "not supported"


def run_family(study: str, kept: pd.DataFrame, story_of: Mapping | None = None, **kw) -> list[dict]:
    """All confirmatory tests of one study's family, Holm-adjusted and decided."""
    if study == "S2":
        return decide(continuation_hypotheses(kept, story_of, **kw))
    return decide(recall_hypotheses(kept, story_of, **kw))


def results_table(results: list[dict], level: str = "session") -> pd.DataFrame:
    rows = []
    for r in results:
        if level == "session":
            rows.append({"hypothesis": r["hypothesis"], "contrast": r["contrast"], "outcome": r["outcome"],
                         "n": r["n_sessions"], "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "p": r["p"], "p_holm": r["p_holm"], "min_p": r["min_p"], "test": _test_label(r),
                         "decision": r["decision"]})
        else:
            s = r["story"]
            rows.append({"hypothesis": r["hypothesis"], "stories": s["n_units"], "estimate": r["estimate"],
                         "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "p": s["p"], "p_holm": s["p_holm"],
                         "min_p": s["min_p"], "decision": s["decision"]})
    return pd.DataFrame(rows)


def _test_label(r: dict) -> str:
    if r["alternative"] == "two-sided":
        return "two-sided"
    return f"one-sided {r['alternative']}, margin {r['margin']:+.2f}"

