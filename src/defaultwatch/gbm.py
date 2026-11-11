"""Monotone-constrained LightGBM with Platt calibration and SHAP reason contributions."""

from __future__ import annotations

import inspect
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import MISSING, FeatureSpec

log = logging.getLogger(__name__)


class GBMModel:
    model_type = "gbm"

    def __init__(self, spec: FeatureSpec, params: dict, random_state: int = 42):
        self.spec = spec
        self.params = dict(params)
        self.early_stopping_rounds = int(self.params.pop("early_stopping_rounds", 200))
        self.random_state = random_state
        self.categories: dict[str, list[str]] = {}
        self.model: lgb.LGBMClassifier | None = None
        self.platt_a, self.platt_b = 1.0, 0.0
        self.best_iteration = None
        self._importance: dict[str, float] = {}

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.spec.numeric:
            out[c] = X[c].astype(float)
        for c in self.spec.categorical:
            out[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "GBMModel":
        for c in self.spec.categorical:
            self.categories[c] = sorted(set(X[c].astype(str)) | {MISSING})
        Xf, Xv = self._prep(X), self._prep(X_val)
        monotone = [int(self.spec.monotone.get(c, 0)) for c in Xf.columns]
        self.model = lgb.LGBMClassifier(
            objective="binary",
            monotone_constraints=monotone,
            monotone_constraints_method="advanced",
            random_state=self.random_state,
            verbose=-1,
            **self.params,
        )
        yv = np.asarray(y_val, dtype=int)
        # LightGBM >= 4.6 takes eval_X/eval_y; older releases only take eval_set.
        if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
            eval_kwargs = {"eval_X": (Xv,), "eval_y": (yv,)}
        else:
            eval_kwargs = {"eval_set": [(Xv, yv)]}
        self.model.fit(
            Xf, np.asarray(y, dtype=int),
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            **eval_kwargs,
        )
        self.best_iteration = int(self.model.best_iteration_ or self.model.n_estimators)
        log.info("GBM stopped at %s trees", self.best_iteration)

        # Platt scaling on the holdout: PD = sigmoid(a * raw + b). A positive affine map of the
        # raw log-odds keeps the ranking and keeps SHAP contributions additive in log-odds.
        raw_val = self._raw(Xv)
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(raw_val.reshape(-1, 1), np.asarray(y_val, dtype=int))
        a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
        if a <= 0:
            log.warning("Platt slope %.3f is not positive; keeping the uncalibrated output", a)
            a, b = 1.0, 0.0
        self.platt_a, self.platt_b = a, b

        contrib = self.contributions(X_val)
        self._importance = contrib.abs().mean().to_dict()
        return self

    def _raw(self, Xp: pd.DataFrame) -> np.ndarray:
        return self.model.predict(Xp, raw_score=True, num_iteration=self.best_iteration)

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.platt_a * self._raw(self._prep(X)) + self.platt_b

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(X)))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP contributions in calibrated log-odds; positive = raises risk."""
        Xp = self._prep(X)
        shap = self.model.predict(Xp, pred_contrib=True, num_iteration=self.best_iteration)
        shap = np.asarray(shap)[:, : len(Xp.columns)] * self.platt_a
        return pd.DataFrame(shap, index=X.index, columns=list(Xp.columns))

    def importance(self) -> dict[str, float]:
        return {c: float(self._importance.get(c, 0.0)) for c in self.spec.features}
