"""
Step 3 -- Extract empirical age-to-age link ratios from an upper triangle.

For each development step j -> j+1, collect the individual per-accident-year
ratios actually observed in the triangle:

    ratio = cumulative_paid[ay, j+1] / cumulative_paid[ay, j]

These per-step ratio pools are what the Monte Carlo simulation resamples
from (bootstrap), instead of collapsing each step to a single averaged
chain-ladder factor. Keeping the whole empirical distribution is what turns
a point-estimate chain ladder into a distributional simulation.

Guards:
  - Cells are already floored at >= 0 upstream (build_triangle.py), so only
    zero denominators need excluding (they make the ratio undefined).
  - Single-step ratios above MAX_PLAUSIBLE_RATIO (50x) are dropped as data
    artifacts: they arise from near-zero denominators, not real development.

Run directly to print an example company's ratio pools:

    python src/link_ratios.py medmal
"""

import sys

import numpy as np
import pandas as pd

from config import MAX_PLAUSIBLE_RATIO, cleaned_path


def link_ratios_by_column(tri: np.ndarray) -> list:
    """Return a list of 9 arrays; element j holds the observed link ratios
    for development step j+1 -> j+2 (0-indexed columns j -> j+1)."""
    ratios_by_col = []
    n_steps = tri.shape[1] - 1
    for j in range(n_steps):
        col_j = tri[:, j]
        col_j1 = tri[:, j + 1]
        mask = ~np.isnan(col_j) & ~np.isnan(col_j1) & (col_j > 0)
        ratios = col_j1[mask] / col_j[mask]
        ratios = ratios[np.isfinite(ratios) & (ratios < MAX_PLAUSIBLE_RATIO)]
        ratios_by_col.append(ratios)
    return ratios_by_col


if __name__ == "__main__":
    from build_triangle import build_upper_triangle

    lob = sys.argv[1] if len(sys.argv) > 1 else "wkcomp"
    cleaned = pd.read_csv(cleaned_path(lob))
    example = cleaned.GRCODE.iloc[0]
    name = cleaned[cleaned.GRCODE == example].GRNAME.iloc[0]
    tri = build_upper_triangle(cleaned, example)
    pools = link_ratios_by_column(tri)
    print(f"[{lob}] Link ratio pools for GRCODE={example} ({name}):\n")
    for j, pool in enumerate(pools):
        pretty = ", ".join(f"{r:.3f}" for r in pool)
        print(f"  Lag {j+1} -> {j+2}:  n={len(pool)}   [{pretty}]")
