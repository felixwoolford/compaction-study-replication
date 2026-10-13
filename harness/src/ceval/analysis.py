"""Descriptives, mixed models and the report (PROTOCOL.md §9)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots, stats

REFERENCE_ARM = "pi-default"
# Table order: the figure arms, then the Study 2 ablations (tables only, no figure colours).
ANALYSIS_ARM_ORDER = plots.ARM_ORDER + ["mech-noreorient", "mech-nosig", "narration-only"]


# -----------------------------------------------------------------------------
# Loading and descriptives
# -----------------------------------------------------------------------------


def load(scored_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame([json.loads(x) for x in scored_path.read_text(encoding="utf-8").splitlines() if x.strip()])
    n0 = len(df)
    excl = {
        "turn_failed": int((~df["turn_ok"].astype(bool)).sum()),
        "cut_violation": int((df["cut_ok"] == False).sum()),  # noqa: E712  (None = no compaction)
        "needs_review": int((df["category"] == "needs_review").sum()),
    }
    keep = df["turn_ok"].astype(bool) & (df["cut_ok"] != False) & (df["category"] != "needs_review")  # noqa: E712
    df = df[keep].copy()
    df["correct"] = df["correct"].astype(int)
    df["confab"] = df["confab"].astype(int)
    excl["kept"] = len(df)
    excl["total"] = n0
    return df, excl


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate_table(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        p, lo, hi = wilson(int(g[col].sum()), len(g))
        rows.append({**dict(zip(by, key)), "n": len(g), "p": p, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    if "arm" in out:
        order = {a: i for i, a in enumerate(ANALYSIS_ARM_ORDER)}
        out = out.sort_values([c for c in by if c != "arm"] + ["arm"],
                              key=lambda s: s.map(order) if s.name == "arm" else s)
    return out


def md_table(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floats.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def laplace_fixed_effects(model, vb) -> tuple[np.ndarray, np.ndarray]:
    """Fixed effects and their standard errors by the Laplace approximation at
    the VB variance components (PROTOCOL.md §9).

    With the random-effect sds fixed at the VB estimates, find the joint
    conditional mode of (beta, u) by Newton's method and take beta's covariance
    from the inverse joint Hessian, as lme4 computes vcov. The mean-field VB
    fe_sd ignores the posterior correlation between fixed and random effects and
    is anti-conservative here (Stage 4 null simulation: ~19% Type I for the H1
    term, against ~5.6% for these intervals).
    """
    X = np.asarray(model.exog, dtype=float)
    Z = model.exog_vc.toarray() if hasattr(model.exog_vc, "toarray") else np.asarray(model.exog_vc, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    p = X.shape[1]
    A = np.hstack([X, Z])
    prior = np.concatenate([np.zeros(p), np.exp(-2 * np.asarray(vb.vcp_mean))[model.ident]])
    theta = np.concatenate([vb.fe_mean, vb.vc_mean])

    def hessian(theta):
        mu = 1 / (1 + np.exp(-(A @ theta)))
        return mu, (A * (mu * (1 - mu))[:, None]).T @ A + np.diag(prior)

    for _ in range(100):
        mu, H = hessian(theta)
        step = np.linalg.solve(H, A.T @ (mu - y) + prior * theta)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise RuntimeError("Laplace conditional mode did not converge")
    cov = np.linalg.inv(hessian(theta)[1])[:p, :p]
    return theta[:p], np.sqrt(np.diag(cov))


def fit_mixed(df: pd.DataFrame, outcome: str, rhs: str) -> str:
    """Mixed logistic model with random intercepts for fact and session: VB fit
    for the variance components, Laplace intervals for the fixed effects."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    if df[outcome].nunique() < 2:
        return f"(not fitted: {outcome} has no variation in this subset)"
    ref = REFERENCE_ARM if REFERENCE_ARM in set(df["arm"]) else sorted(df["arm"].unique())[0]
    formula = f"{outcome} ~ {rhs.replace('ARM', f'C(arm, Treatment({ref!r}))')}"
    vc = {"fact": "0 + C(fact_uid)", "session": "0 + C(session_id)"}
    try:
        model = BinomialBayesMixedGLM.from_formula(formula, vc, df)
        res = model.fit_vb()
        est, se = laplace_fixed_effects(model, res)
    except Exception as e:  # report, don't crash the whole report
        return f"(model failed: {type(e).__name__}: {e})"
    names = model.exog_names
    rows = []
    for name, m, s in zip(names, est, se):
        lo, hi = m - 1.96 * s, m + 1.96 * s
        rows.append(f"| {name} | {m:+.3f} | {s:.3f} | [{lo:+.3f}, {hi:+.3f}] | {'*' if lo > 0 or hi < 0 else ''} |")
    vcs = ", ".join(f"{n}: sd={math.exp(v):.3f}" for n, v in zip(model.vcp_names, res.vcp_mean))
    return ("`" + formula + "` + (1|fact) + (1|session), reference arm " + ref + "\n\n"
            "| term | estimate (logit) | se | 95% interval | excl. 0 |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + f"\n\nRandom effects (VB): {vcs}. Intervals: Laplace at the VB variance components.")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def report(exp: str, res_dir: Path, fit_models: bool = True, story_of: dict | None = None) -> Path:
    """Write `res_dir/report.md`. S1 (and pilot P1) and S2 (P2) get the
    confirmatory report (PROTOCOL.md §9); E1/E2/EC keep the v1 report.

    `story_of` maps session_id -> story for the story-level sensitivity
    analysis; by default it is read from the plans or the scenario."""
    if exp in stats.STUDY_OF_EXP:
        study = stats.STUDY_OF_EXP[exp]
        if study == "S2":
            return report_continuation(exp, res_dir, story_of=story_of)
        return report_recall(exp, res_dir, fit_models=fit_models, story_of=story_of)
    df, excl = load(res_dir / "scored.jsonl")
    planted = df[~df["is_lure"].astype(bool) & (df["position"] == "early")]
    lures = df[df["is_lure"].astype(bool)]
    parts = [f"# Report: {exp}", "",
             f"Trials: {excl['total']} total, {excl['kept']} analysed. Excluded: "
             f"{excl['turn_failed']} failed turns, {excl['cut_violation']} cut violations, "
             f"{excl['needs_review']} uncoded reviews.", ""]

    if exp in ("E1", "EC"):
        acc = rate_table(planted, ["arm", "round"], "correct")
        plots.line_by_round(acc, res_dir / f"fig_{exp.lower()}_recall_by_round.png",
                            f"{exp}: recall of early facts across compaction rounds")
        parts += ["## Recall of planted facts by arm × round", "", md_table(acc), "",
                  f"![](fig_{exp.lower()}_recall_by_round.png)", ""]
        parts += ["## Confident wrong answers on planted facts by arm × round", "",
                  md_table(rate_table(planted, ["arm", "round"], "confab")), ""]
        parts += ["## Recall by source × arm (all rounds)", "",
                  md_table(rate_table(planted, ["source", "arm"], "correct")), ""]
        if "lookup_used" in planted:
            parts += ["## Lookup use per probe turn by arm × round", "",
                      md_table(rate_table(planted.drop_duplicates(["session_id", "arm", "round"]), ["arm", "round"], "lookup_used")), ""]
        parts += ["## False memories (lures) by arm × round", "",
                  md_table(rate_table(lures, ["arm", "round"], "confab")), ""]
        if fit_models:
            p = planted.copy()
            p["round_c"] = p["round"] - p["round"].mean()
            parts += ["## Model: H1", "", fit_mixed(p, "correct", "ARM * round_c"), ""]
    else:
        acc = rate_table(planted, ["source", "arm"], "correct")
        plots.dots_by_group(acc, "source", res_dir / "fig_e2_recall_by_source.png",
                            "E2: recall by where the fact lived", "accuracy")
        parts += ["## Recall of planted facts by source × arm", "", md_table(acc), "",
                  "![](fig_e2_recall_by_source.png)", ""]
        late = df[df["position"] == "late"]
        parts += ["## Manipulation check: late controls (retained tail; expect ≈ 1)", "",
                  md_table(rate_table(late, ["arm"], "correct")), ""]
        parts += ["## False memories (lures) by arm", "", md_table(rate_table(lures, ["arm"], "confab")), ""]
        look = df[df["arm"].isin(["pi-default+lookup", "mech", "mech-matched"])]
        if len(look):
            parts += ["## Lookup use by source (lookup arms; per turn in batched mode)", "",
                      md_table(rate_table(look[~look["is_lure"].astype(bool) & (look["position"] == "early")], ["source", "arm"], "lookup_used")), ""]
        if fit_models:
            parts += ["## Model: H2", "", fit_mixed(planted, "correct", "ARM * C(source)"), ""]

    conf = rate_table(planted, ["arm"], "confab")
    fa = rate_table(lures, ["arm"], "confab").assign(kind="lure false alarm")
    plots.dots_by_group(pd.concat([conf.assign(kind="planted: confident wrong"), fa]), "kind",
                        res_dir / f"fig_{exp.lower()}_confabulation.png", f"{exp}: confabulation by arm", "rate")
    parts += ["## Calibration (H3)", "", f"![](fig_{exp.lower()}_confabulation.png)", ""]
    if fit_models:
        parts += ["Planted facts:", "", fit_mixed(planted, "confab", "ARM"), "",
                  "Lures:", "", fit_mixed(lures, "confab", "ARM"), ""]

    comp_path = res_dir / "compactions.jsonl"
    if comp_path.exists():
        comp = pd.DataFrame([json.loads(x) for x in comp_path.read_text().splitlines() if x.strip()])
        cost = comp.groupby("arm").agg(n=("ok", "size"), ok=("ok", "mean"), summary_chars_median=("summary_chars", "median"),
                                       compact_ms_median=("compact_ms", "median"), cut_ok=("cut_ok", "mean")).reset_index()
        parts += ["## Compaction cost and cut placement", "", md_table(cost, "{:.1f}"), "",
                  "`summary_chars_median` for pi-default is the value for `run.matched_summary_chars` (A5b).", ""]
    tok = df.groupby("arm").agg(tokens_in=("tokens_in", "median"), tokens_out=("tokens_out", "median"),
                                probe_ms=("probe_ms", "median")).reset_index()
    parts += ["## Probe cost (medians per turn)", "", md_table(tok, "{:.0f}"), ""]
    out = res_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# v2 confirmatory reports (PROTOCOL.md §9; decision rules in docs/build/analysis.md)
# -----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()])


def _repo_root(start: Path) -> Path | None:
    for d in [start.resolve(), *start.resolve().parents]:
        if (d / "materials" / "scenarios").is_dir():
            return d
    return None


def story_map(exp: str, session_ids, root: Path | None) -> tuple[dict, str]:
    """session_id -> story: plan meta (`runs/<exp>/plans/<sid>.json`) first, else
    the scenario's `session_story` by the session index (`<exp>-sNNN`). Sessions
    with neither are their own story."""
    out, src = {}, set()
    sc_path = root / "materials" / "scenarios" / f"{exp.lower()}.json" if root else None
    sc = json.loads(sc_path.read_text(encoding="utf-8")) if sc_path and sc_path.exists() else {}
    for sid in session_ids:
        plan = root / "runs" / exp / "plans" / f"{sid}.json" if root else None
        if plan and plan.exists():
            meta = json.loads(plan.read_text(encoding="utf-8")).get("meta") or {}
            if meta.get("story"):
                out[sid] = str(meta["story"])
                src.add("plan meta")
                continue
        try:
            idx = int(str(sid).rsplit("-s", 1)[1])
            out[sid] = str(sc["session_story"][idx])
            src.add("scenario session_story")
        except (IndexError, KeyError, ValueError):
            out[sid] = str(sid)
            src.add("session (no story found)")
    return out, ", ".join(sorted(src)) or "none"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


def _fmt_ci(lo: float, hi: float) -> str:
    return "—" if math.isnan(lo) else f"[{lo:+.3f}, {hi:+.3f}]"


def confirmatory_md(results: list[dict], story_src: str) -> str:
    lines = ["## Confirmatory tests", "",
             "Paired sign-flip permutation tests over sessions (PROTOCOL.md §9.1); estimate = mean per-session "
             "difference; 95% CI = session-level paired percentile bootstrap (10,000 resamples); Holm within this "
             "study's family; `min p` = smallest p these data could give (sessions with a zero difference cannot "
             "change the statistic). Decision rules: `docs/build/analysis.md`.", "",
             "| hypothesis | contrast | outcome | sessions | estimate | 95% CI | test | p | Holm p | min p | decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['hypothesis']} | {r['contrast']} | {r['outcome']} | {r['n_sessions']} | {r['estimate']:+.3f} | "
                     f"{_fmt_ci(r['ci_lo'], r['ci_hi'])} | {stats._test_label(r)} | {_fmt_p(r['p'])} | "
                     f"{_fmt_p(r['p_holm'])} | {_fmt_p(r['min_p'])} | {r['decision']} |")
        for part in r.get("parts", []):
            lines.append(f"| ↳ {part['hypothesis']} | {part['contrast']} | {part['outcome']} | {part['n_sessions']} | "
                         f"{part['estimate']:+.3f} | {_fmt_ci(part['ci_lo'], part['ci_hi'])} | two-sided | "
                         f"{_fmt_p(part['p'])} | | {_fmt_p(part['min_p'])} | predicted sign "
                         f"{'> 0' if part['direction'] > 0 else '< 0'} |")
    lines += ["", "### Story-level sensitivity", "",
              f"The same tests with the sessions of each story flipped together, and the bootstrap resampling whole "
              f"stories. Stories from: {story_src}. A result that holds only at the session level is reported as such.", "",
              "| hypothesis | stories | estimate | 95% CI (story bootstrap) | p | Holm p | min p | decision |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["story"]
        lines.append(f"| {r['hypothesis']} | {s['n_units']} | {r['estimate']:+.3f} | {_fmt_ci(s['ci_lo'], s['ci_hi'])} | "
                     f"{_fmt_p(s['p'])} | {_fmt_p(s['p_holm'])} | {_fmt_p(s['min_p'])} | {s['decision']} |")
    only_session = [r["hypothesis"] for r in results if r["supported"] and not r["story"]["supported"]]
    if only_session:
        lines += ["", f"Holds at session level only: {', '.join(only_session)}."]
    return "\n".join(lines) + "\n"


def _exclusions_md(excl: pd.DataFrame, unit: str) -> str:
    flagged = excl[excl["over_10pct"]]["arm"].tolist() if len(excl) else []
    tab = excl.drop(columns=["over_10pct"]) if len(excl) else excl
    out = ["## Exclusions by arm (PROTOCOL.md §9.4)", "", md_table(tab, "{:.1f}"), "",
           f"Arms losing more than 10% of their {unit} to exclusions (reported as a result): "
           f"{', '.join(flagged) if flagged else 'none'}."]
    return "\n".join(out) + "\n"


def cells_ok(df: pd.DataFrame, col: str, min_events: int = 5) -> tuple[bool, str]:
    """PROTOCOL §9.2: no model when any arm × round cell has < 5 events (or non-events)."""
    g = df.groupby(["arm", "round"])[col].agg(["sum", "size"])
    bad = g[(g["sum"] < min_events) | (g["size"] - g["sum"] < min_events)]
    if len(bad):
        cells = ", ".join(f"{a}×r{r} ({int(x['sum'])}/{int(x['size'])})" for (a, r), x in bad.iterrows())
        return False, (f"(not fitted: arm × round cells with fewer than {min_events} events or non-events: {cells}. "
                       "Risk differences only, PROTOCOL.md §9.2.)")
    return True, ""


def _turns(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("session_id", "arm", "round", "rep", "probe_mode", "batch_index") if c in df]
    return df.drop_duplicates(keys)


def report_recall(exp: str, res_dir: Path, fit_models: bool = True, story_of: dict | None = None) -> Path:
    """Study 1 report (and pilot P1)."""
    study = stats.STUDY_OF_EXP[exp]
    rows = _read_jsonl(res_dir / "scored.jsonl")
    kept, excl = stats.prepare_recall(rows)
    src = "parameter"
    if story_of is None:
        story_of, src = story_map(exp, sorted(rows["session_id"].unique()), _repo_root(res_dir))
    results = stats.run_family(study, kept, story_of)
    n_planted = int((~rows["is_lure"].astype(bool)).sum())
    parts = [f"# Report: {exp}", "",
             f"Study {study[1]}. Confirmatory family: {', '.join(stats.FAMILIES[study])}. Planted-detail trials: "
             f"{n_planted} total, {len(kept)} analysed. Questions still unanswered after the follow-up "
             f"(needs_review on a successful turn) count as not recalled and not confident-wrong (§5.3, §9.4).", "",
             confirmatory_md(results, src)]
    sess = kept.groupby(["session_id", "arm"])["recall"].mean().unstack("arm")
    sess = sess[[a for a in ANALYSIS_ARM_ORDER if a in sess] + [a for a in sess if a not in ANALYSIS_ARM_ORDER]]
    sess = sess.reset_index()
    sess.insert(1, "story", sess["session_id"].map(story_of))
    parts += ["## Per-session recall (all analysed rounds)", "", md_table(sess), ""]

    acc = rate_table(kept, ["arm", "round"], "recall")
    fig = f"fig_{exp.lower()}_recall_by_round.png"
    plots.line_by_round(acc, res_dir / fig, f"{exp}: recall of planted details across compaction rounds")
    parts += ["## Recall of planted details by arm × round (Wilson 95%)", "", md_table(acc), "", f"![]({fig})", ""]
    parts += ["## Confident wrong answers by arm × round", "", md_table(rate_table(kept, ["arm", "round"], "cw")), ""]
    if "source" in kept:
        parts += ["## Recall by source × arm (secondary)", "", md_table(rate_table(kept, ["source", "arm"], "recall")), ""]
    lures = rows[rows["is_lure"].astype(bool) & rows["turn_ok"].fillna(False).astype(bool)
                 & (rows["category"] != "needs_review")]
    if len(lures) and "confab" in lures:
        lures = lures.assign(confab=lures["confab"].astype(int))
        parts += ["## False memories (lures; descriptive) by arm × round", "",
                  md_table(rate_table(lures, ["arm", "round"], "confab")), ""]

    parts += ["## Models (effect sizes, PROTOCOL.md §9.2)", "",
              "A0 (`none`) is excluded from the round models (it has round 1 only)."]
    rm = kept[kept["arm"] != "none"].copy()
    if not fit_models:
        parts += ["", "(models not fitted on request)"]
    elif len(rm):
        rm["round_c"] = rm["round"] - rm["round"].mean()
        rm["correct"], rm["confident_wrong"] = rm["recall"], rm["cw"]
        for title, col in (("Recall", "correct"), ("Confident wrong", "confident_wrong")):
            ok, why = cells_ok(rm, col)
            parts += ["", f"{title}:", "", fit_mixed(rm, col, "ARM * round_c") if ok else why]
    parts.append("")

    turns = _turns(kept)
    if "lookup_used" in kept:
        t = turns.assign(lookup_used=turns["lookup_used"].fillna(False).astype(int))
        parts += ["## Lookup use per probe turn by arm × round", "", md_table(rate_table(t, ["arm", "round"], "lookup_used")), ""]
    cost_cols = {c: (c, "median") for c in ("tokens_in", "tokens_out", "probe_ms", "n_tool_calls") if c in kept}
    if cost_cols:
        tok = turns.groupby("arm").agg(turns=("arm", "size"), **cost_cols).reset_index()
        parts += ["## Probe cost (medians per probe turn)", "", md_table(tok, "{:.0f}"), ""]
    comp_path = res_dir / "compactions.jsonl"
    if comp_path.exists():
        comp = _read_jsonl(comp_path)
        if len(comp):
            agg = {"n": ("arm", "size")}
            for c, f in (("ok", "mean"), ("summary_chars", "median"), ("compact_ms", "median"), ("cut_ok", "mean")):
                if c in comp:
                    agg[f"{c}_{f}"] = (c, f)
            by = ["arm", "round"] if "round" in comp else ["arm"]
            parts += ["## Compaction size and cost by arm × round", "",
                      md_table(comp.groupby(by).agg(**agg).reset_index(), "{:.1f}"), ""]
    parts += [_exclusions_md(excl, "planted-detail trials")]
    out = res_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


S2_OUTCOMES = ("fabricated_progress", "inherited_false_claim", "success", "verified_false_step")
S2_2X2 = (("kept", "yes", "mech-nolookup"), ("kept", "no", "mech-noreorient"),
          ("dropped", "yes", "mech-nosig"), ("dropped", "no", "narration-only"))


def report_continuation(exp: str, res_dir: Path, story_of: dict | None = None) -> Path:
    """Study 2 report (and pilot P2) from `episodes.jsonl`."""
    rows = _read_jsonl(res_dir / "episodes.jsonl")
    kept, excl = stats.prepare_episodes(rows)
    src = "parameter"
    if story_of is None:
        story_of, src = {s: s for s in rows["session_id"].unique()}, "one task per session (story = session)"
    results = stats.run_family("S2", kept, story_of)
    parts = [f"# Report: {exp}", "", f"Study 2. Confirmatory family: H4 (gate for H5–H6), H5, H6. Episodes: "
             f"{len(rows)} total, {len(kept)} analysed.", "", confirmatory_md(results, src)]
    parts += ["## Outcomes by arm (Wilson 95%)", ""]
    for col in S2_OUTCOMES:
        if col in kept:
            d = kept[pd.to_numeric(kept[col], errors="coerce").notna()]
            if len(d):
                d = d.assign(**{col: pd.to_numeric(d[col]).astype(int)})
                parts += [f"### {col}", "", md_table(rate_table(d, ["arm"], col)), ""]
    two = []
    for sig, reo, arm in S2_2X2:
        g = kept[kept["arm"] == arm]
        k, n = int(g["fabricated_progress"].sum()), len(g)
        p, lo, hi = wilson(k, n)
        two.append({"signatures": sig, "re-orient": reo, "arm": arm, "fabricated": k, "n": n, "p": p, "lo": lo, "hi": hi})
    parts += ["## The 2 × 2: fabricated progress by signatures × re-orient (no-lookup arms)", "",
              md_table(pd.DataFrame(two)), ""]
    cols = [c for c in ("unsupported_reports", "repeated_work", "turns", "followups", "n_tool_calls", "tokens_in",
                        "tokens_out", "duration_ms", "summary_chars") if c in kept]
    if cols:
        cost = kept.groupby("arm")[cols].median()
        cost = cost.reindex([a for a in ANALYSIS_ARM_ORDER if a in cost.index]
                            + [a for a in cost.index if a not in ANALYSIS_ARM_ORDER]).reset_index()
        parts += ["## Secondary counts and cost (medians per episode)", "", md_table(cost, "{:.1f}"), ""]
    parts += [_exclusions_md(excl, "episodes")]
    out = res_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
