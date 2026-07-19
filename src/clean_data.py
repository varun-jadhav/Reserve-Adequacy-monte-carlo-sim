"""
Step 1 -- Data cleaning, parameterized by line of business.

Applies three filters to a raw Schedule P file:
  1. Complete triangle: each company (GRCODE) must have exactly 100 rows
     (10 accident years x 10 development lags).
  2. Real current book: positive net earned premium in the most recent
     accident year (2007).
  3. Non-degenerate paid-to-date: positive total cumulative paid loss along
     the latest observed diagonal (otherwise the reserve-adequacy ratio has
     a meaningless denominator).

Every dropped company is logged with the reason(s) it failed.

Run for one line or all lines:

    python src/clean_data.py wkcomp
    python src/clean_data.py all
"""

import sys

import pandas as pd

from config import (
    LINES_OF_BUSINESS, LATEST_ACCIDENT_YEAR, ROWS_PER_FULL_TRIANGLE,
    cleaned_path, dropped_path,
)


def load_and_clean(lob: str):
    """Clean one line of business. Returns (cleaned_df, kept_codes, dropped_df)."""
    raw_file = LINES_OF_BUSINESS[lob]["raw_file"]
    df = pd.read_csv(raw_file)
    df.columns = [c.strip() for c in df.columns]

    # Filter 1: complete 10x10 triangle
    counts = df.groupby("GRCODE").size()
    full_triangle = set(counts[counts == ROWS_PER_FULL_TRIANGLE].index)

    # Filter 2: positive net earned premium in the most recent accident year
    latest_prem = (
        df[df.AccidentYear == LATEST_ACCIDENT_YEAR]
        .groupby("GRCODE")["EarnedPremNet"]
        .first()
    )
    prem_ok = set(latest_prem[latest_prem > 0].index)

    base_clean = full_triangle & prem_ok

    # Filter 3: positive total paid-to-date along the latest diagonal
    sub = df[df.GRCODE.isin(base_clean)]
    latest_diag = sub[
        sub.AccidentYear + sub.DevelopmentLag - 1 == LATEST_ACCIDENT_YEAR
    ]
    paid_by_co = latest_diag.groupby("GRCODE")["CumPaidLoss"].sum()
    degenerate_paid = set(paid_by_co[paid_by_co <= 0].index)

    keep = base_clean - degenerate_paid

    dropped_rows = []
    for code in sorted(set(df.GRCODE.unique()) - keep):
        reasons = []
        if code not in full_triangle:
            reasons.append("incomplete triangle")
        if code not in prem_ok:
            reasons.append(f"{LATEST_ACCIDENT_YEAR} premium <= 0")
        if code in base_clean and code in degenerate_paid:
            reasons.append("paid-to-date <= 0")
        dropped_rows.append({
            "GRCODE": code,
            "GRNAME": df[df.GRCODE == code].GRNAME.iloc[0],
            "reason": "; ".join(reasons),
        })
    dropped = pd.DataFrame(dropped_rows)

    cleaned = df[df.GRCODE.isin(keep)].copy()
    return cleaned, sorted(keep), dropped


def clean_and_save(lob: str) -> None:
    cleaned, kept, dropped = load_and_clean(lob)
    cleaned.to_csv(cleaned_path(lob), index=False)
    dropped.to_csv(dropped_path(lob), index=False)
    label = LINES_OF_BUSINESS[lob]["label"]
    total = len(kept) + len(dropped)
    print(f"[{lob}] {label}: kept {len(kept)}/{total} companies "
          f"({len(cleaned)} rows), dropped {len(dropped)}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    lobs = list(LINES_OF_BUSINESS) if target == "all" else [target]
    for lob in lobs:
        clean_and_save(lob)
