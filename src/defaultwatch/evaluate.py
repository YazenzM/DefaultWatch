"""Discrimination, calibration and approval-rate metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def ks(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def roc_points(y: np.ndarray, p: np.ndarray, n: int = 200) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(np.asarray(y, dtype=int), p)
    grid = np.linspace(0, 1, n)
    return pd.DataFrame({"fpr": grid, "tpr": np.interp(grid, fpr, tpr)})


def calibration_by_decile(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rank loans by predicted PD into equal-count bins; compare mean PD with the observed rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ranks = pd.Series(p).rank(method="first")
    decile = pd.qcut(ranks, n_bins, labels=False) + 1
    df = pd.DataFrame({"decile": decile, "y": y, "p": p})
    table = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"),
                                     observed_rate=("y", "mean"), min_pd=("p", "min"), max_pd=("p", "max"))
    table["abs_error"] = (table["observed_rate"] - table["mean_pd"]).abs()
    table["ratio_observed_to_predicted"] = table["observed_rate"] / table["mean_pd"]
    return table.reset_index()


def calibration_summary(y: np.ndarray, p: np.ndarray, table: pd.DataFrame) -> dict:
    weights = table["n"] / table["n"].sum()
    expected = table["mean_pd"] * table["n"]
    observed = table["defaults"]
    # Hosmer-Lemeshow statistic over the deciles.
    var = table["n"] * table["mean_pd"] * (1 - table["mean_pd"])
    hl = float((((observed - expected) ** 2) / var.replace(0, np.nan)).sum())
    dof = max(len(table) - 2, 1)
    return {
        "ece_decile": float((weights * table["abs_error"]).sum()),
        "max_decile_error": float(table["abs_error"].max()),
        "observed_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(p)),
        "calibration_in_the_large": float(np.mean(y) / np.mean(p)),
        "brier": float(brier_score_loss(np.asarray(y, dtype=int), p)),
        "hosmer_lemeshow": hl,
        "hosmer_lemeshow_p": float(stats.chi2.sf(hl, dof)),
    }


def approval_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Approve the lowest-PD loans first; bad rate among approved at each approval rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    n = len(ys)
    cum_bad_rate = np.cumsum(ys) / np.arange(1, n + 1)
    # Only cut between distinct PD values: tied loans are approved or declined together.
    ends = np.r_[ps[1:] != ps[:-1], True]
    return pd.DataFrame({
        "approval_rate": (np.arange(1, n + 1) / n)[ends],
        "bad_rate_approved": cum_bad_rate[ends],
        "cutoff_pd": ps[ends],
    })


def approval_at_bad_rate(y: np.ndarray, p: np.ndarray, target_bad_rate: float) -> dict:
    """Largest approval rate whose bad rate among approved loans stays at or below the target."""
    curve = approval_curve(y, p)
    ok = curve[curve["bad_rate_approved"] <= target_bad_rate]
    if ok.empty:
        return {"approval_rate": 0.0, "cutoff_pd": float("nan"), "bad_rate_approved": float("nan"),
                "target_bad_rate": target_bad_rate}
    row = ok.iloc[-1]
    return {
        "approval_rate": float(row["approval_rate"]),
        "cutoff_pd": float(row["cutoff_pd"]),
        "bad_rate_approved": float(row["bad_rate_approved"]),
        "target_bad_rate": float(target_bad_rate),
    }


def evaluate(y: np.ndarray, p: np.ndarray) -> tuple[dict, pd.DataFrame]:
    table = calibration_by_decile(y, p)
    metrics = {"n": int(len(y)), "defaults": int(np.sum(y)), "auc": auc(y, p), "gini": 2 * auc(y, p) - 1,
               "ks": ks(y, p)}
    metrics.update(calibration_summary(y, p, table))
    return metrics, table
