"""Weight-of-evidence logistic-regression scorecard.

Each feature is binned (quantile bins merged until they are large enough and,
for features with a known risk direction, until the bad rate is monotone),
replaced by its weight of evidence, and fed to a logistic regression. The
coefficients are scaled into points: higher score = lower risk.

Reasons use the "points below maximum" method: for each feature, the gap between
the points the applicant received and the most points that feature can give.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import MISSING, OTHER, FeatureSpec

log = logging.getLogger(__name__)

SMOOTH = 0.5
MIN_MISSING_COUNT = 30


@dataclass
class FeatureBins:
    feature: str
    kind: str                                  # "numeric" | "categorical"
    cuts: list[float] = field(default_factory=list)          # numeric: interior cut points, bins are (c[i-1], c[i]]
    categories: dict[str, int] = field(default_factory=dict)  # categorical: category -> bin index
    labels: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    bads: list[int] = field(default_factory=list)
    woe: list[float] = field(default_factory=list)
    iv: float = 0.0
    missing_bin: int | None = None
    fallback_bin: int = 0                      # used for unseen categories / rare missing values

    def index(self, x: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            values = x.to_numpy(dtype=float)
            idx = np.searchsorted(np.asarray(self.cuts, dtype=float), values, side="left")
            missing = np.isnan(values)
            idx[missing] = self.missing_bin if self.missing_bin is not None else self.fallback_bin
            return idx.astype(int)
        mapped = x.astype(str).map(self.categories)
        return mapped.fillna(self.categories.get(OTHER, self.fallback_bin)).to_numpy().astype(int)

    def transform(self, x: pd.Series) -> np.ndarray:
        return np.asarray(self.woe, dtype=float)[self.index(x)]


def _woe(goods: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, float]:
    g = (goods + SMOOTH) / (goods.sum() + SMOOTH * len(goods))
    b = (bads + SMOOTH) / (bads.sum() + SMOOTH * len(bads))
    woe = np.log(g / b)
    return woe, float(np.sum((g - b) * woe))


def _merge_adjacent(counts: list[int], bads: list[int], cuts: list[float], i: int) -> None:
    """Merge bin i with bin i+1 (removes cut i)."""
    counts[i] += counts.pop(i + 1)
    bads[i] += bads.pop(i + 1)
    cuts.pop(i)


def bin_numeric(x: pd.Series, y: np.ndarray, feature: str, direction: int, max_bins: int, min_frac: float) -> FeatureBins:
    values = x.to_numpy(dtype=float)
    present = ~np.isnan(values)
    v, t = values[present], y[present]
    n_total = len(values)

    cuts = list(np.unique(np.quantile(v, np.linspace(0, 1, max_bins + 1)[1:-1]))) if len(v) else []
    idx = np.searchsorted(np.asarray(cuts), v, side="left")
    counts = [int((idx == i).sum()) for i in range(len(cuts) + 1)]
    bads = [int(t[idx == i].sum()) for i in range(len(cuts) + 1)]

    # Drop empty bins (quantile ties), then merge bins below the minimum size.
    i = 0
    while i < len(counts) and len(counts) > 1:
        if counts[i] == 0:
            j = i if i < len(counts) - 1 else i - 1
            _merge_adjacent(counts, bads, cuts, j)
        else:
            i += 1
    min_count = max(1, int(min_frac * n_total))
    while len(counts) > 1 and min(counts) < min_count:
        i = int(np.argmin(counts))
        if i == 0:
            j = 0
        elif i == len(counts) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i
        _merge_adjacent(counts, bads, cuts, j)

    # Pool adjacent violators so the bad rate moves in the expected direction.
    if direction in (1, -1):
        changed = True
        while changed and len(counts) > 1:
            changed = False
            rates = [b / c for b, c in zip(bads, counts)]
            for i in range(len(rates) - 1):
                if (direction == 1 and rates[i] > rates[i + 1]) or (direction == -1 and rates[i] < rates[i + 1]):
                    _merge_adjacent(counts, bads, cuts, i)
                    changed = True
                    break

    n_missing = int((~present).sum())
    bad_missing = int(y[~present].sum())
    labels = []
    edges = [-np.inf] + cuts + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(lo):
            labels.append(f"<= {hi:g}")
        elif np.isinf(hi):
            labels.append(f"> {lo:g}")
        else:
            labels.append(f"({lo:g}, {hi:g}]")

    all_counts, all_bads = list(counts), list(bads)
    missing_bin = None
    if n_missing >= MIN_MISSING_COUNT:
        missing_bin = len(all_counts)
        all_counts.append(n_missing)
        all_bads.append(bad_missing)
        labels.append(MISSING)

    counts_arr, bads_arr = np.asarray(all_counts), np.asarray(all_bads)
    woe, iv = _woe(counts_arr - bads_arr, bads_arr)
    worst = int(np.argmin(woe[: len(counts)]))
    fb = FeatureBins(
        feature=feature, kind="numeric", cuts=[float(c) for c in cuts], labels=labels,
        counts=[int(c) for c in all_counts], bads=[int(b) for b in all_bads],
        woe=[float(w) for w in woe], iv=iv, missing_bin=missing_bin, fallback_bin=worst,
    )
    return fb


def bin_categorical(x: pd.Series, y: np.ndarray, feature: str, rare_frac: float) -> FeatureBins:
    x = x.astype(str)
    share = x.value_counts(normalize=True)
    keep = [c for c in share.index if share[c] >= rare_frac]
    pooled = x.where(x.isin(keep), OTHER)
    groups = sorted(pooled.unique())
    categories = {c: groups.index(c) for c in groups}
    for c in share.index:
        if c not in keep:
            categories[c] = categories[OTHER]
    codes = pooled.map({g: i for i, g in enumerate(groups)}).to_numpy()
    counts = np.bincount(codes, minlength=len(groups))
    bads = np.bincount(codes, weights=y, minlength=len(groups)).astype(int)
    woe, iv = _woe(counts - bads, bads)
    worst = int(np.argmin(woe))
    if OTHER not in categories:
        categories[OTHER] = worst
    return FeatureBins(
        feature=feature, kind="categorical", categories=categories, labels=list(groups),
        counts=[int(c) for c in counts], bads=[int(b) for b in bads],
        woe=[float(w) for w in woe], iv=iv, fallback_bin=worst,
    )


class Scorecard:
    model_type = "scorecard"

    def __init__(self, spec: FeatureSpec, max_bins: int = 10, min_bin_fraction: float = 0.05,
                 rare_category_fraction: float = 0.01, pdo: float = 20, base_score: float = 600,
                 base_odds: float = 50, C: float = 1.0, min_iv: float = 0.01):
        self.spec = spec
        self.max_bins, self.min_bin_fraction = max_bins, min_bin_fraction
        self.rare_category_fraction = rare_category_fraction
        self.pdo, self.base_score, self.base_odds, self.C, self.min_iv = pdo, base_score, base_odds, C, min_iv
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.bins: dict[str, FeatureBins] = {}
        self.selected: list[str] = []
        self.dropped: dict[str, str] = {}
        self.coef: dict[str, float] = {}
        self.intercept = 0.0
        self.points: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, spec: FeatureSpec, cfg: dict) -> "Scorecard":
        s = cfg["scorecard"]
        return cls(spec, s["max_bins"], s["min_bin_fraction"], s["rare_category_fraction"],
                   s["pdo"], s["base_score"], s["base_odds"], s["C"])

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Scorecard":
        y = np.asarray(y, dtype=int)
        for col in self.spec.numeric:
            self.bins[col] = bin_numeric(X[col], y, col, self.spec.monotone.get(col, 0),
                                         self.max_bins, self.min_bin_fraction)
        for col in self.spec.categorical:
            self.bins[col] = bin_categorical(X[col], y, col, self.rare_category_fraction)

        candidates = [c for c in self.spec.features if self.bins[c].iv >= self.min_iv]
        for c in self.spec.features:
            if c not in candidates:
                self.dropped[c] = f"information value {self.bins[c].iv:.4f} below {self.min_iv}"
        woe = self._woe_matrix(X, self.spec.features)

        # Refit until every coefficient has the expected (negative) sign on WOE = ln(good/bad).
        while candidates:
            lr = LogisticRegression(C=self.C, max_iter=1000)
            lr.fit(woe[candidates].to_numpy(), y)
            coefs = dict(zip(candidates, lr.coef_[0]))
            wrong = [c for c, b in coefs.items() if b >= 0]
            if not wrong:
                break
            worst = max(wrong, key=lambda c: coefs[c])
            self.dropped[worst] = f"coefficient sign flipped ({coefs[worst]:+.3f}) given correlated features"
            candidates.remove(worst)
        if not candidates:
            raise RuntimeError("Scorecard has no usable features")

        self.selected = candidates
        self.coef = {c: float(b) for c, b in coefs.items()}
        self.intercept = float(lr.intercept_[0])
        n = len(self.selected)
        for c in self.selected:
            w = np.asarray(self.bins[c].woe)
            self.points[c] = -self.factor * (self.coef[c] * w + self.intercept / n) + self.offset / n
        return self

    def _woe_matrix(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: self.bins[c].transform(X[c]) for c in cols}, index=X.index)

    # ------------------------------------------------------------------ predict
    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        woe = self._woe_matrix(X, self.selected)
        return self.intercept + woe.to_numpy() @ np.array([self.coef[c] for c in self.selected])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return sum(self.points[c][self.bins[c].index(X[c])] for c in self.selected)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points below the feature's maximum (>= 0); larger = stronger reason."""
        out = pd.DataFrame(0.0, index=X.index, columns=self.spec.features)
        for c in self.selected:
            pts = self.points[c][self.bins[c].index(X[c])]
            out[c] = self.points[c].max() - pts
        return out

    def importance(self) -> dict[str, float]:
        return {c: float(self.bins[c].iv) for c in self.spec.features}

    def points_table(self) -> pd.DataFrame:
        rows = []
        for c in self.spec.features:
            b = self.bins[c]
            for i, label in enumerate(b.labels):
                rows.append({
                    "feature": c, "bin": label, "count": b.counts[i], "bads": b.bads[i],
                    "bad_rate": b.bads[i] / b.counts[i] if b.counts[i] else np.nan,
                    "woe": b.woe[i], "iv": b.iv,
                    "points": float(self.points[c][i]) if c in self.selected else np.nan,
                    "in_model": c in self.selected,
                })
        return pd.DataFrame(rows)
