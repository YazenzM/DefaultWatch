"""Static report charts (PNG). Every chart has a CSV twin in the reports folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"scorecard": "#2a78d6", "gbm": "#eb6834", "third": "#1baf7a"}
LABELS = {"scorecard": "Logistic scorecard", "gbm": "Gradient boosting"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def roc_chart(rocs: dict[str, pd.DataFrame], aucs: dict[str, float], title: str, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for name, df in rocs.items():
        ax.plot(df["fpr"], df["tpr"], color=SERIES[name], label=f"{LABELS[name]} (AUC {aucs[name]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positive rate (good loans flagged)")
    ax.set_ylabel("True positive rate (defaults caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, path)


def calibration_chart(tables: dict[str, pd.DataFrame], title: str, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lo = min(min(t["mean_pd"].min(), t["observed_rate"].replace(0, np.nan).min()) for t in tables.values())
    hi = max(max(t["mean_pd"].max(), t["observed_rate"].max()) for t in tables.values())
    lo, hi = max(lo * 0.7, 1e-5), hi * 1.4
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1)
    ax.text(hi / 1.15, hi / 1.15, "perfect calibration", color=MUTED, fontsize=8, ha="right", va="bottom",
            rotation=0)
    for name, t in tables.items():
        obs = t["observed_rate"].clip(lower=lo)
        ax.plot(t["mean_pd"], obs, color=SERIES[name], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    pct = plt.FuncFormatter(lambda v, _: f"{v * 100:.2g}%")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Mean predicted default rate (by decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, path)


def approval_chart(curves: dict[str, pd.DataFrame], target: float, title: str, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, c in curves.items():
        step = max(1, len(c) // 400)
        cc = c.iloc[::step]
        ax.plot(cc["approval_rate"] * 100, cc["bad_rate_approved"] * 100, color=SERIES[name], label=LABELS[name])
    ax.axhline(target * 100, color=MUTED, linewidth=1)
    ax.text(2, target * 100, f" target default rate {target * 100:.2f}%", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Default rate among approved loans (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, path)


def psi_chart(summary: pd.DataFrame, warn: float, alert: float, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(summary))
    ax.bar(x, summary["score_psi"], width=0.6, color=SERIES["scorecard"], edgecolor=SURFACE, linewidth=2)
    for xi, v in zip(x, summary["score_psi"]):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    for level, name in ((warn, "watch"), (alert, "alert")):
        ax.axhline(level, color=MUTED, linewidth=1)
        ax.text(-0.45, level, f"{name} {level:.2f}", ha="left", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [str(v) for v in summary["vintage"]])
    ax.set_ylim(0, max(alert * 1.25, float(summary["score_psi"].max()) * 1.2))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Vintage")
    ax.set_ylabel("PSI of the score vs. training")
    ax.set_title("Score stability by vintage")
    return _save(fig, path)


def stress_chart(rows: pd.DataFrame, path: Path) -> Path:
    """Grouped bars per vintage: mean predicted PD, observed 90+ rate, observed rate excluding relief."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    measures = [("mean_pd", "Mean predicted PD", SERIES["scorecard"]),
                ("observed_rate", "Observed 90+ rate (all)", SERIES["gbm"]),
                ("observed_real_rate", "Observed 90+ excl. relief", SERIES["third"])]
    x = np.arange(len(rows))
    width = 0.26
    for k, (col, label, color) in enumerate(measures):
        vals = rows[col].to_numpy() * 100
        pos = x + (k - 1) * width
        ax.bar(pos, vals, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(pos, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [str(v) for v in rows["label"]])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Rate within 24 months (%)")
    ax.set_title("Predicted vs. observed default: out-of-time and stress vintages")
    ax.legend(loc="upper left")
    return _save(fig, path)
