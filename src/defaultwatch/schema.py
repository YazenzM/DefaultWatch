"""Freddie Mac Single-Family Loan-Level Dataset file layouts.

Both files are pipe-delimited with no header row. Column order follows the Freddie Mac
"General User Guide", Release 48: sample_YYYY.zip holds sample_orig_YYYY.txt and sample_perf_YYYY.txt.

Release 48 changed both layouts. Servicer Name and the MI Cancellation Indicator moved from the
origination file to the monthly performance file. VantageScore 4.0 was appended to the origination
file and Bankruptcy Cramdown Costs to the performance file. The Payment Deferral flag became
C (current period) / P (prior period) instead of Y.

Earlier releases (performance file sample_svcg_YYYY.txt) are still read. Their origination layout
is LEGACY_ORIGINATION_COLUMNS, and their performance layout is a prefix of PERFORMANCE_COLUMNS.
"""

from __future__ import annotations

LEGACY_ORIGINATION_COLUMNS: list[str] = [
    "credit_score",               # 1  300-850; 9999 = not available
    "first_payment_date",         # 2  YYYYMM
    "first_time_homebuyer",       # 3  Y / N / 9
    "maturity_date",              # 4  YYYYMM
    "msa",                        # 5  MSA or metropolitan division code
    "mi_pct",                     # 6  0-55; 999 = not available
    "num_units",                  # 7  1-4; 99 = not available
    "occupancy_status",           # 8  P / I / S / 9
    "orig_cltv",                  # 9  999 = not available
    "orig_dti",                   # 10 999 = not available
    "orig_upb",                   # 11
    "orig_ltv",                   # 12 999 = not available
    "orig_interest_rate",         # 13
    "channel",                    # 14 R / B / C / T / 9
    "ppm_flag",                   # 15
    "amortization_type",          # 16 FRM / ARM
    "property_state",             # 17
    "property_type",              # 18 SF / CO / PU / MH / CP / 99
    "postal_code",                # 19
    "loan_id",                    # 20 loan sequence number
    "loan_purpose",               # 21 P / C / N / R / 9
    "orig_loan_term",             # 22 months
    "num_borrowers",              # 23
    "seller_name",                # 24
    "servicer_name",              # 25
    "super_conforming",           # 26
    "pre_harp_loan_id",           # 27
    "program_indicator",          # 28
    "harp_indicator",             # 29
    "property_valuation_method",  # 30
    "interest_only",              # 31
    "mi_cancellation",            # 32
]

# Release 48: fields 1-24 unchanged, servicer name and MI cancellation moved to the performance file.
ORIGINATION_COLUMNS: list[str] = LEGACY_ORIGINATION_COLUMNS[:24] + [
    "super_conforming",           # 25 Y / N
    "pre_harp_loan_id",           # 26
    "program_indicator",          # 27 H / F / R; blank = not applicable
    "harp_indicator",             # 28 Y / N
    "property_valuation_method",  # 29 1-4; 7 = not available
    "interest_only",              # 30 Y / N
    "vantage_score_4",            # 31 300-850; 9999 = not available
]


def origination_columns(first_row: list[str]) -> list[str]:
    """Column names for an origination file, recognised from its first row.

    Release 48 has 31 fields with the Super Conforming flag (Y / N / blank) in field 25. Earlier
    releases had the Servicer Name there and only ever appended fields, so a shorter legacy file
    uses the first N legacy names.
    """
    n = len(first_row)
    if n == len(ORIGINATION_COLUMNS) and first_row[24].strip() in {"Y", "N", ""}:
        return ORIGINATION_COLUMNS
    if n <= len(LEGACY_ORIGINATION_COLUMNS):
        return LEGACY_ORIGINATION_COLUMNS[:n]
    return ORIGINATION_COLUMNS + [f"extra_{i}" for i in range(n - len(ORIGINATION_COLUMNS))]


PERFORMANCE_COLUMNS: list[str] = [
    "loan_id",                        # 1
    "period",                         # 2  monthly reporting period YYYYMM
    "current_upb",                    # 3
    "dlq_status",                     # 4  0,1,2,... months delinquent; RA = REO acquisition; blank/XX = unknown
    "loan_age",                       # 5
    "remaining_months",               # 6
    "defect_settlement_date",         # 7
    "modification_flag",              # 8
    "zero_balance_code",              # 9  01 prepaid, 02 third-party sale, 03 short sale/charge-off, 09 REO, 15/16 note sale, 96 repurchase
    "zero_balance_date",              # 10
    "current_interest_rate",          # 11
    "non_interest_bearing_upb",       # 12
    "ddlpi",                          # 13 due date of last paid installment
    "mi_recoveries",                  # 14
    "net_sale_proceeds",              # 15
    "non_mi_recoveries",              # 16
    "total_expenses",                 # 17
    "legal_costs",                    # 18
    "maintenance_costs",              # 19
    "taxes_insurance",                # 20
    "misc_expenses",                  # 21
    "actual_loss",                    # 22
    "cumulative_mod_cost",            # 23
    "step_mod_flag",                  # 24
    "payment_deferral",               # 25 C = granted this period, P = prior period (Y before Release 48)
    "eltv",                           # 26
    "zero_balance_removal_upb",       # 27
    "delinquent_accrued_interest",    # 28
    "disaster_delinquency",           # 29 Y = delinquency due to a declared disaster
    "borrower_assistance_status",     # 30 F = forbearance, R = repayment plan, T = trial modification
    "current_month_mod_cost",         # 31
    "interest_bearing_upb",           # 32 (last field before Release 48)
    "mi_cancellation",                # 33 Y / N; 7 = not applicable (moved from origination)
    "servicer_name",                  # 34 (moved from origination)
    "bankruptcy_cramdown_costs",      # 35
]

# Performance-file kinds in member names: "perf" from Release 48, "svcg" before.
PERFORMANCE_KINDS: tuple[str, ...] = ("perf", "svcg")

# Performance fields the target builder needs.
PERFORMANCE_USECOLS: list[str] = [
    "loan_id",
    "period",
    "dlq_status",
    "zero_balance_code",
    "payment_deferral",
    "disaster_delinquency",
    "borrower_assistance_status",
]

# Sentinel codes that mean "not available" in the origination file.
NUMERIC_SENTINELS: dict[str, set[float]] = {
    "credit_score": {9999},
    "vantage_score_4": {9999},
    "mi_pct": {999},
    "num_units": {99},
    "orig_cltv": {999},
    "orig_dti": {999},
    "orig_ltv": {999},
    "num_borrowers": {99},
}

CATEGORICAL_MISSING: set[str] = {"9", "99", ""}

# Plain-language labels for categorical codes.
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "loan_purpose": {
        "P": "purchase",
        "C": "cash-out refinance",
        "N": "no-cash-out refinance",
        "R": "refinance (type not specified)",
    },
    "occupancy_status": {
        "P": "primary residence",
        "I": "investment property",
        "S": "second home",
    },
    "property_type": {
        "SF": "single-family home",
        "CO": "condominium",
        "PU": "planned unit development",
        "MH": "manufactured home",
        "CP": "co-op",
    },
    "channel": {
        "R": "retail",
        "B": "broker",
        "C": "correspondent",
        "T": "third-party origination",
    },
}

ASSISTANCE_RELIEF_CODES: set[str] = {"F", "R", "T"}

# Payment deferral granted in the reporting month: C from Release 48, Y before. P (prior period) stays
# on every later month, so it does not mark that month's delinquency as relief.
PAYMENT_DEFERRAL_RELIEF_CODES: set[str] = {"C", "Y"}
