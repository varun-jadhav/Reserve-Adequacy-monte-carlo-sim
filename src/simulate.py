"""
Step 4 -- Bootstrap Monte Carlo simulation of ultimate losses.

Method: bootstrap chain ladder (non-parametric). For each simulation:
  1. At every development step, draw one link ratio at random (with
     replacement) from the company's empirical ratio pool for that step.
  2. Use the drawn ratios to fill the unobserved lower triangle, projecting
     every accident year to full development (lag 10).
  3. Sum the final column across accident years = one simulated total
     ultimate loss.

Across thousands of draws this yields a distribution of ultimate losses,
capturing development uncertainty that a single-point chain ladder hides.

Reserve adequacy: each simulated ultimate is compared against the company's
carried total (paid-to-date + posted reserve at year-end 2007). The fraction
of simulations exceeding the carried total estimates the probability of
reserve shortfall.

Note: this is neither an Expected Loss Ratio (ELR) nor Bornhuetter-Ferguson
(BF) method -- no premium-based prior expected losses are used anywhere.
It is a resampling variant of the chain-ladder family.

Run directly to simulate one example company:

    python src/simulate.py comauto
"""

import sys

import numpy as np
import pandas as pd

from config import N_SIMS, RANDOM_SEED, LATEST_ACCIDENT_YEAR, cleaned_path


def simulate_ultimates(tri: np.ndarray, ratios_by_col: list,
                       n_sims: int = N_SIMS,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Return an array of n_sims simulated total ultimate losses for one
    company's triangle, using bootstrap-resampled link ratios.

    Vectorized: for each accident year, ultimate = latest observed value
    x product of the drawn link ratios for the remaining development steps.
    All n_sims draws are generated in one shot per development step.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_years = tri.shape[0]
    n_steps = n_years - 1

    # Draw factors for every (simulation, development step) at once
    factors = np.ones((n_sims, n_steps))
    for j in range(n_steps):
        pool = ratios_by_col[j]
        if len(pool) > 0:
            factors[:, j] = rng.choice(pool, size=n_sims, replace=True)

    # For each accident year, the latest observed cell sits at column
    # (n_years - 1 - ay_idx); its ultimate applies the remaining factors.
    sim_totals = np.zeros(n_sims)
    for ay_idx in range(n_years):
        last_obs_col = n_years - 1 - ay_idx
        latest_value = tri[ay_idx, last_obs_col]
        if np.isnan(latest_value):
            continue
        if last_obs_col == n_years - 1:
            sim_totals += latest_value  # already fully developed
        else:
            remaining = factors[:, last_obs_col:]          # (n_sims, k)
            sim_totals += latest_value * remaining.prod(axis=1)

    return sim_totals


def carried_reserve_and_paid(df: pd.DataFrame, grcode) -> tuple:
    """Return (paid_to_date, posted_reserve) for one company. Paid-to-date
    sums the latest observed diagonal, floored at zero per accident year for
    consistency with the triangle treatment."""
    sub = df[df.GRCODE == grcode]
    latest_diag = sub[
        sub.AccidentYear + sub.DevelopmentLag - 1 == LATEST_ACCIDENT_YEAR
    ]
    paid_to_date = latest_diag.CumPaidLoss.clip(lower=0).sum()
    posted_reserve = sub.PostedReserves2007.iloc[0]
    return paid_to_date, posted_reserve


if __name__ == "__main__":
    from build_triangle import build_upper_triangle
    from link_ratios import link_ratios_by_column

    lob = sys.argv[1] if len(sys.argv) > 1 else "wkcomp"
    cleaned = pd.read_csv(cleaned_path(lob))
    example = cleaned.GRCODE.iloc[0]
    name = cleaned[cleaned.GRCODE == example].GRNAME.iloc[0]

    tri = build_upper_triangle(cleaned, example)
    pools = link_ratios_by_column(tri)
    sims = simulate_ultimates(tri, pools)

    paid, reserve = carried_reserve_and_paid(cleaned, example)
    carried = paid + reserve
    p_short = float(np.mean(sims > carried))

    print(f"[{lob}] Example simulation: GRCODE={example} ({name})")
    print(f"  Paid to date:        {paid:>15,.0f}")
    print(f"  Posted reserve:      {reserve:>15,.0f}")
    print(f"  Carried total:       {carried:>15,.0f}")
    print(f"  Simulated mean ult.: {np.mean(sims):>15,.0f}")
    print(f"  Simulated P95 ult.:  {np.percentile(sims, 95):>15,.0f}")
    print(f"  P(reserve shortfall): {p_short:.1%}")
