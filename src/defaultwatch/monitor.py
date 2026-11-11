"""Population stability index (PSI) monitoring for scores and features.

PSI = sum((a - e) * ln(a / e)) over bins, where e and a are the bin shares in the
baseline (training) and the new population. Bins are deciles of the baseline,
plus a separate bin for missing values. Rules of thumb:
  < 0.10 stable, 0.10-0.25 watch, >= 0.25 alert (the population has shifted
  enough that the model's validation may no longer hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class NumericBaseline:
    cuts: list[float]
    expected: list[float]            # shares per bin; last entry is the missing bin

    @classmethod
    def fit(cls, x: np.ndarray, bins: int = 10) -> "NumericBaseline":
        x = np.asarray(x, dtype=float)
        present = x[~np.isnan(x)]
        cuts = list(np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1]))) if len(present) else []
        return cls([float(c) for c in cuts], cls._shares(x, cuts).tolist())

    @staticmethod
    def _shares(x: np.ndarray, cuts: list[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        missing = np.isnan(x)
        idx = np.searchsorted(np.asarray(cuts, dtype=float), x[~missing], side="left")
        counts = np.bincount(idx, minlength=len(cuts) + 1).astype(float)
        counts = np.r_[counts, missing.sum()]
        return counts / max(len(x), 1)

    def psi(self, x: np.ndarray) -> float:
        return _psi(self.expected, self._shares(x, self.cuts))


@dataclass
class CategoricalBaseline:
    expected: dict[str, float]

    @classmethod
    def fit(cls, x: pd.Series) -> "CategoricalBaseline":
        return cls(x.astype(str).value_counts(normalize=True).to_dict())

    def psi(self, x: pd.Series) -> float:
        actual = x.astype(str).value_counts(normalize=True).to_dict()
        keys = sorted(set(self.expected) | set(actual))
        return _psi([self.expected.get(k, 0.0) for k in keys], [actual.get(k, 0.0) for k in keys])


@dataclass
class MonitoringBaseline:
    score: NumericBaseline
    features: dict[str, NumericBaseline | CategoricalBaseline] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)
    warn: float = 0.10
    alert: float = 0.25
    baseline_label: str = "training"

    @classmethod
    def fit(cls, pd_scores: np.ndarray, X: pd.DataFrame, numeric: list[str], categorical: list[str],
            top_features: list[str], bins: int = 10, warn: float = 0.10, alert: float = 0.25,
            baseline_label: str = "training") -> "MonitoringBaseline":
        feats: dict[str, NumericBaseline | CategoricalBaseline] = {}
        for c in numeric:
            feats[c] = NumericBaseline.fit(X[c].to_numpy(dtype=float), bins)
        for c in categorical:
            feats[c] = CategoricalBaseline.fit(X[c])
        return cls(NumericBaseline.fit(pd_scores, bins), feats, list(top_features), warn, alert, baseline_label)

    def status(self, value: float) -> str:
        if value >= self.alert:
            return "alert"
        if value >= self.warn:
            return "watch"
        return "stable"

    def check(self, pd_scores: np.ndarray, X: pd.DataFrame) -> dict:
        score_psi = self.score.psi(pd_scores)
        feature_psi = {c: b.psi(X[c] if isinstance(b, CategoricalBaseline) else X[c].to_numpy(dtype=float))
                       for c, b in self.features.items()}
        watched = {"score": score_psi, **{c: feature_psi[c] for c in self.top_features}}
        worst = max(watched.values())
        alerts = [k for k, v in watched.items() if self.status(v) == "alert"]
        warnings = [k for k, v in watched.items() if self.status(v) == "watch"]
        return {
            "score_psi": score_psi,
            "score_status": self.status(score_psi),
            "feature_psi": feature_psi,
            "feature_status": {c: self.status(v) for c, v in feature_psi.items()},
            "top_features": self.top_features,
            "max_watched_psi": worst,
            "status": self.status(worst),
            "alerts": alerts,
            "warnings": warnings,
            "thresholds": {"watch": self.warn, "alert": self.alert},
            "baseline": self.baseline_label,
        }
