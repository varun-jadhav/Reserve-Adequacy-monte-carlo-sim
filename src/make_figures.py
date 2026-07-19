"""
Step 6 -- Visualize results.

Produces in results/figures/:
  pipeline_stages_{lob}.png      one company end to end, for any line
  cross_line_comparison.png      shortfall risk and uncertainty across all
                                 six lines of business

    python src/make_figures.py wkcomp    # pipeline example for one line
    python src/make_figures.py all       # examples for all lines + comparison
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    LINES_OF_BUSINESS, cleaned_path, results_path, COMBINED_RESULTS_PATH,
)
from build_triangle import build_upper_triangle, triangle_to_frame
from link_ratios import link_ratios_by_column
from simulate import simulate_ultimates, carried_reserve_and_paid

BLUE = "#2b6cb0"
LINE_COLORS = {
    "wkcomp": "#2b6cb0", "ppauto": "#dd6b20", "comauto": "#38a169",
    "medmal": "#805ad5", "prodliab": "#d53f8c", "othliab": "#718096",
}


def figure_pipeline_example(lob: str):
    """Three-panel end-to-end visual for the largest company in one line."""
    cleaned = pd.read_csv(cleaned_path(lob))
    results = pd.read_csv(results_path(lob))
    label = LINES_OF_BUSINESS[lob]["label"]

    grcode = results.loc[results.carried_total.idxmax(), "GRCODE"]
    name = results.loc[results.carried_total.idxmax(), "GRNAME"]

    tri = build_upper_triangle(cleaned, grcode)
    tri_df = triangle_to_frame(tri)
    pools = link_ratios_by_column(tri)
    sims = simulate_ultimates(tri, pools)
    paid, reserve = carried_reserve_and_paid(cleaned, grcode)
    carried = paid + reserve

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    ax = axes[0]
    mask = tri_df.isna()
    ax.imshow(np.where(mask, np.nan, tri_df.values), cmap="Blues", aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels(tri_df.columns, rotation=45)
    ax.set_yticks(range(10)); ax.set_yticklabels(tri_df.index)
    vmax = np.nanmax(tri_df.values)
    for i in range(10):
        for j in range(10):
            if not mask.iloc[i, j]:
                v = tri_df.iloc[i, j]
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > vmax * 0.5 else "black")
            else:
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=8,
                        color="lightgray")
    ax.set_title(f"Stage 1: upper triangle\n{name} ({label})\n"
                 "cumulative paid loss, \u2014 = unobserved")

    ax = axes[1]
    ax.boxplot([p if len(p) else [1.0] for p in pools],
               positions=range(1, 10), widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#63b3ed", alpha=0.7))
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"{i}\u2192{i+1}" for i in range(1, 10)])
    ax.set_xlabel("Development step")
    ax.set_ylabel("Link ratio")
    ax.set_title("Stage 2: empirical link ratio pools\n(resampled by the bootstrap)")

    ax = axes[2]
    ax.hist(sims, bins=50, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(carried, color="red", linestyle="--", linewidth=2,
               label=f"Carried total = {carried:,.0f}")
    ax.axvline(sims.mean(), color="black", linewidth=1.5,
               label=f"Simulated mean = {sims.mean():,.0f}")
    ax.set_xlabel("Simulated ultimate loss")
    ax.set_ylabel("Frequency")
    p_short = float(np.mean(sims > carried))
    ax.set_title(f"Stage 3: bootstrap simulations\nP(shortfall) = {p_short:.1%}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"results/figures/pipeline_stages_{lob}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def figure_cross_line_comparison():
    """Compare shortfall risk and reserve uncertainty across all lines."""
    combined = pd.read_csv(COMBINED_RESULTS_PATH)
    lobs = [l for l in LINES_OF_BUSINESS if l in set(combined.line)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: mean P(shortfall) and share of companies above 50%
    ax = axes[0]
    stats = combined.groupby("line").agg(
        mean_ps=("prob_reserve_shortfall", "mean"),
        pct50=("prob_reserve_shortfall", lambda x: (x > 0.5).mean()),
    ).reindex(lobs)
    x = np.arange(len(lobs))
    w = 0.38
    ax.bar(x - w/2, stats.mean_ps, w, label="Mean P(shortfall)",
           color=[LINE_COLORS[l] for l in lobs], alpha=0.85)
    ax.bar(x + w/2, stats.pct50, w, label="Share of companies P > 50%",
           color=[LINE_COLORS[l] for l in lobs], alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([LINES_OF_BUSINESS[l]["label"] for l in lobs],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Probability / share")
    ax.set_title("Reserve shortfall risk by line of business")
    ax.legend()

    # Panel 2: distribution of CV per line (boxplot) -- long-tail vs short-tail
    ax = axes[1]
    data = [combined[combined.line == l].coeff_of_variation.dropna() for l in lobs]
    bp = ax.boxplot(data, positions=range(len(lobs)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, l in zip(bp["boxes"], lobs):
        patch.set_facecolor(LINE_COLORS[l])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(lobs)))
    ax.set_xticklabels([LINES_OF_BUSINESS[l]["label"] for l in lobs],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient of variation of simulated ultimate")
    ax.set_title("Reserve uncertainty by line of business\n"
                 "(long-tail lines develop with more uncertainty)")

    plt.tight_layout()
    out = "results/figures/cross_line_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        for lob in LINES_OF_BUSINESS:
            figure_pipeline_example(lob)
        figure_cross_line_comparison()
    else:
        figure_pipeline_example(target)
