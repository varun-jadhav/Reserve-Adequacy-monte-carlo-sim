"""
Step 2 -- Build the loss development (upper) triangle for a company.

Line-of-business agnostic: operates on any cleaned Schedule P dataframe,
since all six lines share an identical schema.

Reshapes a company's long-format rows into a 10x10 matrix:
  rows    = accident years (1998 ... 2007)
  columns = development lags (1 ... 10)
  values  = cumulative paid losses

Cells not yet observable as of year-end 2007 (the "lower triangle") are set
to NaN: a cell for accident year AY at development lag L is observed in
calendar year AY + L - 1, so anything past 2007 hasn't happened yet from the
model's point of view.

Negative cumulative paid losses -- a genuine feature of Schedule P data
caused by subrogation/salvage recoveries, reinsurance recoveries, voided
payments, or deductible collections exceeding paid-to-date -- are floored at
zero, the standard actuarial treatment for chain-ladder methods (negative
cells distort development factors). See NAIC / American Academy of Actuaries
guidance on negative accident-year payment percentages.

Run directly to print an example triangle:

    python src/build_triangle.py wkcomp
"""

import sys

import numpy as np
import pandas as pd

from config import FIRST_ACCIDENT_YEAR, LATEST_ACCIDENT_YEAR, N_YEARS, cleaned_path


def build_upper_triangle(df: pd.DataFrame, grcode) -> np.ndarray:
    """Return a 10x10 array of cumulative paid losses for one company, with
    unobserved (lower-triangle) cells set to NaN and negative cells floored
    at zero."""
    sub = df[df.GRCODE == grcode].sort_values(["AccidentYear", "DevelopmentLag"])
    tri = np.full((N_YEARS, N_YEARS), np.nan)
    for _, row in sub.iterrows():
        ay_idx = int(row.AccidentYear) - FIRST_ACCIDENT_YEAR
        lag_idx = int(row.DevelopmentLag) - 1
        tri[ay_idx, lag_idx] = max(row.CumPaidLoss, 0.0)

    for ay_idx in range(N_YEARS):
        for lag_idx in range(N_YEARS):
            calendar_year = FIRST_ACCIDENT_YEAR + ay_idx + lag_idx
            if calendar_year > LATEST_ACCIDENT_YEAR:
                tri[ay_idx, lag_idx] = np.nan
    return tri


def triangle_to_frame(tri: np.ndarray) -> pd.DataFrame:
    """Wrap a triangle array in a labeled DataFrame for display/export."""
    return pd.DataFrame(
        tri,
        index=[f"AY{FIRST_ACCIDENT_YEAR + i}" for i in range(N_YEARS)],
        columns=[f"Lag{j + 1}" for j in range(N_YEARS)],
    )


if __name__ == "__main__":
    lob = sys.argv[1] if len(sys.argv) > 1 else "wkcomp"
    cleaned = pd.read_csv(cleaned_path(lob))
    example = cleaned.GRCODE.iloc[0]
    name = cleaned[cleaned.GRCODE == example].GRNAME.iloc[0]
    tri = build_upper_triangle(cleaned, example)
    print(f"[{lob}] Upper triangle for GRCODE={example} ({name}):\n")
    print(triangle_to_frame(tri).round(0).to_string())
