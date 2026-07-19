"""
Central configuration: the six NAIC Schedule P lines of business.

Every pipeline module is parameterized by a line-of-business key from
LINES_OF_BUSINESS. All six raw files share an identical schema, differing
only in filename and the set of companies they contain.
"""

FIRST_ACCIDENT_YEAR = 1998
LATEST_ACCIDENT_YEAR = 2007
N_YEARS = 10                    # accident years, and development lags
ROWS_PER_FULL_TRIANGLE = 100    # 10 x 10
N_SIMS = 5000
RANDOM_SEED = 42
MAX_PLAUSIBLE_RATIO = 50.0

LINES_OF_BUSINESS = {
    "wkcomp": {
        "label": "Workers' Compensation",
        "raw_file": "data/raw/wkcomp_pos_98-07.csv",
    },
    "ppauto": {
        "label": "Private Passenger Auto",
        "raw_file": "data/raw/ppauto_pos_98-07.csv",
    },
    "comauto": {
        "label": "Commercial Auto",
        "raw_file": "data/raw/comauto_pos_98-07.csv",
    },
    "medmal": {
        "label": "Medical Malpractice (Claims-Made)",
        "raw_file": "data/raw/medmal_pos_98-07.csv",
    },
    "prodliab": {
        "label": "Product Liability (Occurrence)",
        "raw_file": "data/raw/prodliab_pos_98-07.csv",
    },
    "othliab": {
        "label": "Other Liability (Occurrence)",
        "raw_file": "data/raw/othliab_pos_98-07.csv",
    },
}


def cleaned_path(lob: str) -> str:
    return f"data/processed/{lob}_cleaned.csv"


def dropped_path(lob: str) -> str:
    return f"data/processed/{lob}_dropped_companies.csv"


def results_path(lob: str) -> str:
    return f"results/{lob}_simulation_results.csv"


COMBINED_RESULTS_PATH = "results/all_lines_simulation_results.csv"
