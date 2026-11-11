"""Principal reasons for a high-risk score, in plain language.

Regulation B (12 CFR 1002.9) requires a creditor that denies credit to give the
specific principal reasons. CFPB Circular 2023-03 adds that a creditor using a
complex model cannot pick the closest item from a sample checklist if that item
does not describe the actual reason. So every reason here:

  * comes from the model's own per-applicant attribution (SHAP for the GBM,
    points-below-maximum for the scorecard), ranked by size;
  * names the factor and the applicant's actual value; and
  * compares that value with typical approved borrowers only when the comparison
    is true for this applicant.

The Regulation B category is carried alongside for notice templates, never in
place of the specific text.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import FEATURE_LABELS, MISSING, OTHER
from .schema import CATEGORY_LABELS

REG_B_CATEGORY = {
    "credit_score": "Credit history (credit score)",
    "orig_dti": "Excessive obligations in relation to income",
    "orig_ltv": "Value or type of collateral not sufficient",
    "orig_cltv": "Value or type of collateral not sufficient",
    "orig_loan_term": "We do not grant credit on the terms and conditions you request",
    "num_units": "Value or type of collateral not sufficient",
    "loan_purpose": "We do not grant credit on the terms and conditions you request",
    "occupancy_status": "Value or type of collateral not sufficient",
    "property_type": "Value or type of collateral not sufficient",
}

REASON_CODES = {
    "credit_score": "R01",
    "orig_dti": "R02",
    "orig_ltv": "R03",
    "orig_cltv": "R04",
    "orig_loan_term": "R05",
    "num_units": "R06",
    "loan_purpose": "R07",
    "occupancy_status": "R08",
    "property_type": "R09",
}

CATEGORY_TEXT = {
    ("loan_purpose", "C"): "The loan is a cash-out refinance, which increases the amount owed against the home.",
    ("occupancy_status", "I"): "The property is an investment property rather than the borrower's home.",
    ("occupancy_status", "S"): "The property is a second home rather than the borrower's primary residence.",
    ("property_type", "MH"): "The property is a manufactured home.",
    ("property_type", "CO"): "The property is a condominium.",
}


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def reason_text(feature: str, value, reference) -> str:
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == MISSING:
        return f"{label} was not available on the application."

    if feature == "credit_score":
        if value < reference:
            return f"Credit score of {_fmt(value)} is below the typical approved borrower's {_fmt(reference)}."
        return f"Credit score of {_fmt(value)} is below the scores of the lowest-risk borrowers."
    if feature == "orig_dti":
        if value > reference:
            return (f"Monthly debt payments are {_fmt(value)}% of income, above the typical approved "
                    f"borrower's {_fmt(reference)}%.")
        return f"Monthly debt payments are {_fmt(value)}% of income, above the lowest-risk borrowers."
    if feature == "orig_ltv":
        equity = max(0.0, 100.0 - value)
        if value > reference:
            return (f"The loan is {_fmt(value)}% of the property's value (a {_fmt(equity)}% down payment or "
                    f"equity stake); the typical approved loan is {_fmt(reference)}%.")
        return (f"The loan is {_fmt(value)}% of the property's value ({_fmt(equity)}% equity), "
                f"above the lowest-risk loans.")
    if feature == "orig_cltv":
        if value > reference:
            return (f"All loans secured by the property total {_fmt(value)}% of its value, above the "
                    f"typical {_fmt(reference)}%.")
        return f"All loans secured by the property total {_fmt(value)}% of its value, above the lowest-risk loans."
    if feature == "orig_loan_term":
        years = value / 12
        return f"A {years:.0f}-year term repays the balance more slowly than a shorter term."
    if feature == "num_units":
        return f"The property has {_fmt(value)} units; multi-unit properties default more often."

    if isinstance(value, str):
        specific = CATEGORY_TEXT.get((feature, value))
        if specific:
            return specific
        if value == OTHER:
            return f"{label} is an uncommon category with a higher default rate in the development data."
        names = CATEGORY_LABELS.get(feature, {})
        return f"{label} ({names.get(value, value)}) has a higher default rate than average in the development data."
    return f"{label} of {value} adds to the estimated risk."


def top_reasons(contribs: pd.DataFrame, X: pd.DataFrame, reference: dict, top_n: int = 4) -> list[list[dict]]:
    """For each row, the top_n factors that raised risk (contribution > 0), largest first."""
    values = contribs.to_numpy(dtype=float)
    cols = list(contribs.columns)
    order = np.argsort(-values, axis=1, kind="stable")
    out: list[list[dict]] = []
    for i, row_idx in enumerate(contribs.index):
        reasons = []
        for j in order[i, :top_n]:
            c = values[i, j]
            if not c > 0:
                break
            feature = cols[j]
            raw = X.at[row_idx, feature]
            value = None if (isinstance(raw, float) and math.isnan(raw)) else (
                raw.item() if hasattr(raw, "item") else raw)
            reasons.append({
                "rank": len(reasons) + 1,
                "code": REASON_CODES.get(feature, "R99"),
                "feature": feature,
                "value": value,
                "contribution": round(float(c), 4),
                "reason": reason_text(feature, value, reference.get(feature)),
                "reg_b_category": REG_B_CATEGORY.get(feature, "Other"),
            })
        out.append(reasons)
    return out
