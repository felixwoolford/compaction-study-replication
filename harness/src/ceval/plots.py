"""Static report figures (PNG). Colour follows the arm, in a fixed order, in every figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (light), fixed order; arm -> slot never changes.
ARM_ORDER = ["none", "pi-default", "designed", "pi-default+lookup", "mech-nolookup", "mech", "mech-matched", "pi-lcm", "pi-fold"]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#9c5a2e", "#6b7280"]
ARM_COLOR = dict(zip(ARM_ORDER, SLOTS))
GROUP_ORDER = ["user", "assistant_text", "tool_output", "thinking", "superseded"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _arms(df: pd.DataFrame) -> list[str]:
    return [a for a in ARM_ORDER if a in set(df["arm"])]


def line_by_round(summary: pd.DataFrame, out: Path, title: str) -> None:
    """summary: arm, round, p, lo, hi."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    rounds = sorted(summary["round"].unique())
    for arm in _arms(summary):
        d = summary[summary["arm"] == arm].sort_values("round")
        c = ARM_COLOR[arm]
        ax.fill_between(d["round"], d["lo"], d["hi"], color=c, alpha=0.10, linewidth=0)
        ax.plot(d["round"], d["p"], color=c, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm)
        last = d.iloc[-1]
        ax.annotate(arm, (last["round"], last["p"]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xticks(rounds)
    ax.set_xlim(min(rounds) - 0.2, max(rounds) + 0.9)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Compaction round")
    ax.set_ylabel("Recall accuracy (planted facts)")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def dots_by_group(summary: pd.DataFrame, group: str, out: Path, title: str, xlabel: str) -> None:
    """Small multiples: one panel per group value; arms on y, estimate ± 95% CI on x."""
    seen = list(dict.fromkeys(summary[group]))
    groups = [g for g in GROUP_ORDER if g in seen] + [g for g in seen if g not in GROUP_ORDER]
    arms = _arms(summary)
    fig, axes = plt.subplots(1, len(groups), figsize=(2.3 * len(groups) + 1.4, 0.38 * len(arms) + 1.3),
                             sharey=True, facecolor=SURFACE)
    axes = [axes] if len(groups) == 1 else list(axes)
    for ax, g in zip(axes, groups):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        d = summary[summary[group] == g].set_index("arm")
        for y, arm in enumerate(arms):
            if arm not in d.index:
                continue
            r = d.loc[arm]
            c = ARM_COLOR[arm]
            ax.plot([r["lo"], r["hi"]], [y, y], color=c, linewidth=2, solid_capstyle="round")
            ax.plot([r["p"]], [y], "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(str(g), fontsize=9, color=INK)
        ax.set_xlabel(xlabel, fontsize=8)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels(arms)
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
