"""
Step 5 -- Run the full pipeline for one or all lines of business.

For each company in each requested line:
  triangle -> link ratio pools -> bootstrap simulations -> adequacy metrics

Outputs:
  results/{lob}_simulation_results.csv        one file per line
  results/all_lines_simulation_results.csv    all lines concatenated,
                                              with a `line` column

    python src/run_pipeline.py wkcomp     # one line
    python src/run_pipeline.py all        # everything (~5 min)
"""

import sys

import numpy as np
import pandas as pd

from config import (
    LINES_OF_BUSINESS, N_SIMS, RANDOM_SEED,
    results_path, COMBINED_RESULTS_PATH,
)
from clean_data import load_and_clean
from build_triangle import build_upper_triangle
from link_ratios import link_ratios_by_column
from simulate import simulate_ultimates, carried_reserve_and_paid


def run_line(lob: str, n_sims: int = N_SIMS) -> pd.DataFrame:
    label = LINES_OF_BUSINESS[lob]["label"]
    print(f"[{lob}] {label}: cleaning...")
    cleaned, companies, _ = load_and_clean(lob)
    print(f"[{lob}] simulating {len(companies)} companies "
          f"({n_sims} sims each)...")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i, grcode in enumerate(companies, 1):
        tri = build_upper_triangle(cleaned, grcode)
        pools = link_ratios_by_column(tri)
        if all(len(p) == 0 for p in pools):
            continue  # no usable development history

        sims = simulate_ultimates(tri, pools, n_sims=n_sims, rng=rng)
        paid, reserve = carried_reserve_and_paid(cleaned, grcode)
        carried = paid + reserve
        mean_ult = float(np.mean(sims))

        rows.append({
            "line": lob,
            "line_label": label,
            "GRCODE": grcode,
            "GRNAME": cleaned[cleaned.GRCODE == grcode].GRNAME.iloc[0],
            "paid_to_date": paid,
            "posted_reserve": reserve,
            "carried_total": carried,
            "sim_mean_ultimate": mean_ult,
            "sim_p50_ultimate": float(np.percentile(sims, 50)),
            "sim_p75_ultimate": float(np.percentile(sims, 75)),
            "sim_p90_ultimate": float(np.percentile(sims, 90)),
            "sim_p95_ultimate": float(np.percentile(sims, 95)),
            "coeff_of_variation": (float(np.std(sims) / mean_ult)
                                   if mean_ult > 0 else np.nan),
            "prob_reserve_shortfall": float(np.mean(sims > carried)),
        })
        if i % 25 == 0:
            print(f"[{lob}]   ...{i}/{len(companies)}")

    results = pd.DataFrame(rows)
    results.to_csv(results_path(lob), index=False)
    print(f"[{lob}] done: {len(results)} companies -> {results_path(lob)}")
    return results


def summarize(all_results: pd.DataFrame) -> pd.DataFrame:
    return (
        all_results.groupby("line_label")
        .agg(
            n_companies=("GRCODE", "count"),
            mean_prob_shortfall=("prob_reserve_shortfall", "mean"),
            median_prob_shortfall=("prob_reserve_shortfall", "median"),
            pct_gt50pct_risk=("prob_reserve_shortfall",
                              lambda x: (x > 0.5).mean()),
            mean_cv=("coeff_of_variation", "mean"),
        )
        .round(4)
        .sort_values("mean_cv", ascending=False)
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    lobs = list(LINES_OF_BUSINESS) if target == "all" else [target]

    frames = [run_line(lob) for lob in lobs]

    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(COMBINED_RESULTS_PATH, index=False)
        print(f"\nCombined results -> {COMBINED_RESULTS_PATH}")
        print("\n=== Cross-line summary ===")
        print(summarize(combined).to_string())
