"""The deployable model: features + model + cutoff + reasons + monitoring baseline in one object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FeatureSpec
from .monitor import MonitoringBaseline
from .reasons import top_reasons


@dataclass
class ModelBundle:
    model_type: str                 # "scorecard" or "gbm"
    model: object                   # Scorecard | GBMModel
    spec: FeatureSpec
    pd_cutoff: float                # decline when PD > cutoff
    reference: dict                 # typical approved-borrower values, for reason wording
    baseline: MonitoringBaseline
    scaling: dict                   # pdo, base_score, base_odds
    top_n: int = 4
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ scoring
    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.spec.transform(raw)

    def predict_pd(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_pd(self.features(raw))

    def to_score(self, pd_values: np.ndarray) -> np.ndarray:
        """Map PD to points on the scorecard scale (higher = lower risk)."""
        p = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
        factor = self.scaling["pdo"] / np.log(2)
        offset = self.scaling["base_score"] - factor * np.log(self.scaling["base_odds"])
        return offset + factor * np.log((1 - p) / p)

    def score_frame(self, raw: pd.DataFrame, reasons_for: str = "all") -> pd.DataFrame:
        """PD, score, decision and principal reasons for each row.

        reasons_for: "all" (every row) or "declined" (only rows above the cutoff).
        """
        X = self.features(raw)
        pd_values = self.model.predict_pd(X)
        decline = pd_values > self.pd_cutoff
        out = pd.DataFrame(index=raw.index)
        out["pd"] = pd_values
        out["score"] = np.round(self.to_score(pd_values)).astype(int)
        out["decision"] = np.where(decline, "decline", "approve")
        rows = X.index if reasons_for == "all" else X.index[decline]
        reasons = pd.Series([[] for _ in range(len(X))], index=X.index, dtype=object)
        if len(rows):
            contribs = self.model.contributions(X.loc[rows])
            reasons.loc[rows] = pd.Series(top_reasons(contribs, X.loc[rows], self.reference, self.top_n),
                                          index=rows, dtype=object)
        out["reasons"] = reasons
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)
