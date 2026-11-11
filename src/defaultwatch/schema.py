"""Freddie Mac Single-Family Loan-Level Dataset file layouts.

Both files are pipe-delimited with no header row. Column order follows the
Freddie Mac "General User Guide" for the standard dataset. Older releases have
fewer columns; new fields have always been appended at the end, so a file with
N columns uses the first N names below.
"""

from __future__ import annotations

ORIGINATION_COLUMNS: list[str] = [
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
    "payment_deferral",               # 25 Y = payment deferral this month
    "eltv",                           # 26
    "zero_balance_removal_upb",       # 27
    "delinquent_accrued_interest",    # 28
    "disaster_delinquency",           # 29 Y = delinquency due to a declared disaster
    "borrower_assistance_status",     # 30 F = forbearance, R = repayment plan, T = trial modification
    "current_month_mod_cost",         # 31
    "interest_bearing_upb",           # 32
]

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
