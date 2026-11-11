"""Turn raw origination fields into model features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_MISSING, NUMERIC_SENTINELS

MISSING = "Missing"
OTHER = "Other"

FEATURE_LABELS: dict[str, str] = {
    "credit_score": "Credit score",
    "orig_dti": "Debt-to-income ratio",
    "orig_ltv": "Loan-to-value ratio",
    "orig_cltv": "Combined loan-to-value ratio",
    "orig_loan_term": "Loan term",
    "num_units": "Number of units",
    "loan_purpose": "Loan purpose",
    "occupancy_status": "Occupancy",
    "property_type": "Property type",
    "mi_pct": "Mortgage insurance coverage",
    "orig_upb": "Loan amount",
    "num_borrowers": "Number of borrowers",
    "first_time_homebuyer": "First-time homebuyer",
    "channel": "Origination channel",
    "orig_interest_rate": "Interest rate",
}


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    monotone: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureSpec":
        f = cfg["features"]
        return cls(list(f["numeric"]), list(f["categorical"]), dict(f.get("monotone") or {}))

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Clean raw origination fields: sentinels -> NaN, categorical codes normalized."""
        out = pd.DataFrame(index=raw.index)
        for col in self.numeric:
            values = pd.to_numeric(raw[col], errors="coerce") if col in raw else pd.Series(np.nan, index=raw.index)
            sentinels = NUMERIC_SENTINELS.get(col)
            if sentinels:
                values = values.where(~values.isin(list(sentinels)))
            out[col] = values.astype(float)
        for col in self.categorical:
            if col in raw:
                values = raw[col].astype("string").fillna("").str.strip().str.upper()
                values = values.where(~values.isin(list(CATEGORICAL_MISSING)), MISSING)
                out[col] = values.astype(str)
            else:
                out[col] = MISSING
        return out


def reference_values(X: pd.DataFrame, spec: FeatureSpec) -> dict:
    """Typical values of approved (funded) borrowers, used to word reasons."""
    ref: dict[str, float | str] = {}
    for col in spec.numeric:
        ref[col] = float(X[col].median()) if X[col].notna().any() else float("nan")
    for col in spec.categorical:
        mode = X[col].mode()
        ref[col] = str(mode.iloc[0]) if len(mode) else MISSING
    return ref
