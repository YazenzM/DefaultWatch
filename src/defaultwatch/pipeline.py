"""End-to-end training, validation, stress test and monitoring."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import __version__
from .bundle import ModelBundle
from .evaluate import approval_at_bad_rate, approval_curve, evaluate, roc_points
from .features import FeatureSpec, reference_values
from .gbm import GBMModel
from .ingest import available_vintages, label_to_index, load_vintages, month_label, processed_is_synthetic
from .monitor import MonitoringBaseline
from .reasons import top_reasons
from .scorecard import Scorecard

log = logging.getLogger(__name__)

BUNDLE_NAME = "defaultwatch_bundle.joblib"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _modeling_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["has_performance"].astype(bool) & df["window_complete"].astype(bool)].copy()


def bundle_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["model_dir"]) / BUNDLE_NAME


# --------------------------------------------------------------------------- stress test
def outcome_breakdown(df: pd.DataFrame, p: np.ndarray, cfg: dict) -> dict:
    """Separate relief-driven delinquency from real default and score both."""
    from .evaluate import auc, calibration_by_decile, calibration_summary

    y = df["bad"].astype(int).to_numpy()
    y_real = df["bad_real"].astype(int).to_numpy()
    covid_idx = label_to_index(cfg["target"]["covid_relief_start"])
    first_bad_idx = df["fp_idx"] + df["first_bad_msfp"]
    pre_covid_bad = (df["bad"].astype(bool) & (first_bad_idx < covid_idx)).to_numpy()
    table_real = calibration_by_decile(y_real, p)
    return {
        "n": int(len(df)),
        "observed_rate": float(y.mean()),
        "observed_real_rate": float(y_real.mean()),
        "relief_only_rate": float(df["bad_relief_only"].mean()),
        "relief_share_of_defaults": float(df["bad_relief_only"].sum() / max(int(y.sum()), 1)),
        "pre_covid_default_rate": float(pre_covid_bad.mean()),
        "relief_flag_rate": float(df["relief_in_window"].mean()),
        "mean_pd": float(np.mean(p)),
        "auc_all": auc(y, p),
        "auc_real": auc(y_real, p),
        "calibration_in_the_large_all": float(y.mean() / np.mean(p)),
        "calibration_in_the_large_real": float(y_real.mean() / np.mean(p)),
        "ece_decile_real": calibration_summary(y_real, p, table_real)["ece_decile"],
        "window_end_max": month_label(float(df["window_end_idx"].max())),
    }


# --------------------------------------------------------------------------- reasons
def reason_diagnostics(model, X: pd.DataFrame, p: np.ndarray, cutoff: float, reference: dict, top_n: int,
                       spec: FeatureSpec, sample: int = 2000, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Reason coverage and faithfulness on declined loans.

    Faithfulness: set the #1 reason's feature to the typical approved value and re-score.
    A faithful reason lowers the PD when that factor is improved.
    """
    declined = X.index[p > cutoff]
    if len(declined) == 0:
        return {"declined": 0}, pd.DataFrame()
    rng = np.random.default_rng(seed)
    idx = declined if len(declined) <= sample else pd.Index(rng.choice(declined, sample, replace=False))
    Xd = X.loc[idx]
    reasons = top_reasons(model.contributions(Xd), Xd, reference, top_n)

    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    improved = 0
    tested = 0
    X_cf = Xd.copy()
    top_feature = []
    for row_idx, rs in zip(Xd.index, reasons):
        for r in rs:
            counts[r["feature"]] = counts.get(r["feature"], 0) + 1
        if rs:
            f = rs[0]["feature"]
            first[f] = first.get(f, 0) + 1
            top_feature.append((row_idx, f))
    for row_idx, f in top_feature:
        X_cf.at[row_idx, f] = reference[f]
    if top_feature:
        rows = [r for r, _ in top_feature]
        before = model.predict_pd(Xd.loc[rows])
        after = model.predict_pd(X_cf.loc[rows])
        improved = int(np.sum(after < before))
        tested = len(rows)

    n = len(Xd)
    table = pd.DataFrame({
        "feature": list(spec.features),
        "share_of_declines_citing": [counts.get(f, 0) / n for f in spec.features],
        "share_as_top_reason": [first.get(f, 0) / n for f in spec.features],
    }).sort_values("share_as_top_reason", ascending=False)
    summary = {
        "declined": int(len(declined)),
        "decline_rate": float(len(declined) / len(X)),
        "sampled": int(n),
        "with_at_least_one_reason": float(np.mean([len(r) > 0 for r in reasons])),
        "mean_reasons": float(np.mean([len(r) for r in reasons])),
        "faithfulness_top_reason": float(improved / tested) if tested else float("nan"),
    }
    return summary, table


# --------------------------------------------------------------------------- training
def train(cfg: dict, use_mlflow: bool = True, make_plots: bool = True) -> dict:
    spec = FeatureSpec.from_config(cfg)
    V = cfg["vintages"]
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(exist_ok=True)
    synthetic = processed_is_synthetic(cfg)
    if synthetic:
        log.warning("Processed data was built from SYNTHETIC files: results are for testing only.")

    dev = _modeling_rows(load_vintages(cfg, V["train"]))
    oot = _modeling_rows(load_vintages(cfg, [V["oot"]]))
    available = available_vintages(cfg)
    stress = _modeling_rows(load_vintages(cfg, [V["stress"]])) if V["stress"] in available else None

    X_dev, y_dev = spec.transform(dev), dev["bad"].astype(int).to_numpy()
    X_oot, y_oot = spec.transform(oot), oot["bad"].astype(int).to_numpy()
    log.info("development: %s loans, %.3f%% default; OOT %s: %s loans, %.3f%% default",
             f"{len(dev):,}", 100 * y_dev.mean(), V["oot"], f"{len(oot):,}", 100 * y_oot.mean())

    # Every OOT outcome window must close before COVID payment relief.
    covid_idx = label_to_index(cfg["target"]["covid_relief_start"])
    oot_after_covid = int((oot["window_end_idx"] >= covid_idx).sum())
    if oot_after_covid:
        log.warning("%s OOT loans have windows that reach %s", oot_after_covid, cfg["target"]["covid_relief_start"])

    split = cfg["split"]
    idx_fit, idx_hold = train_test_split(np.arange(len(dev)), test_size=split["holdout_fraction"],
                                         stratify=y_dev, random_state=split["random_state"])
    X_fit, y_fit = X_dev.iloc[idx_fit], y_dev[idx_fit]
    X_hold, y_hold = X_dev.iloc[idx_hold], y_dev[idx_hold]
    reference = reference_values(X_fit, spec)

    log.info("fitting scorecard")
    scorecard = Scorecard.from_config(spec, cfg).fit(X_fit, y_fit)
    log.info("fitting gradient boosting")
    gbm = GBMModel(spec, cfg["gbm"], random_state=split["random_state"]).fit(X_fit, y_fit, X_hold, y_hold)
    models = {"scorecard": scorecard, "gbm": gbm}

    target_bad_rate = cfg["approval"]["target_bad_rate"]
    if target_bad_rate is None:
        target_bad_rate = float(cfg["approval"]["target_bad_rate_ratio"] * y_fit.mean())

    results: dict = {}
    preds_oot, preds_hold, rocs, cal_tables, curves = {}, {}, {}, {}, {}
    for name, m in models.items():
        p_hold, p_oot = m.predict_pd(X_hold), m.predict_pd(X_oot)
        preds_oot[name], preds_hold[name] = p_oot, p_hold
        hold_metrics, _ = evaluate(y_hold, p_hold)
        oot_metrics, table = evaluate(y_oot, p_oot)
        cal_tables[name] = table
        rocs[name] = roc_points(y_oot, p_oot)
        curves[name] = approval_curve(y_oot, p_oot)
        table.assign(model=name).to_csv(reports / f"calibration_oot_{name}.csv", index=False)
        results[name] = {
            "holdout": hold_metrics,
            "oot": oot_metrics,
            "approval_oot": approval_at_bad_rate(y_oot, p_oot, target_bad_rate),
        }
    thinned = [c.iloc[sorted(set(range(0, len(c), max(1, len(c) // 1000))) | {len(c) - 1})].assign(model=n)
               for n, c in curves.items()]
    pd.concat(thinned).to_csv(reports / "approval_curve_oot.csv", index=False)
    pd.concat([r.assign(model=n) for n, r in rocs.items()]).to_csv(reports / "roc_oot.csv", index=False)
    scorecard.points_table().to_csv(reports / "scorecard_points.csv", index=False)

    gain = results["gbm"]["oot"]["auc"] - results["scorecard"]["oot"]["auc"]
    champion = "gbm" if gain >= cfg["champion"]["min_auc_gain"] else "scorecard"
    model = models[champion]
    log.info("champion: %s (OOT AUC gain of GBM over scorecard %+.4f)", champion, gain)

    # Decision cutoff from the in-time holdout, so the OOT sample is never used to set it.
    cutoff_info = approval_at_bad_rate(y_hold, preds_hold[champion], target_bad_rate)
    pd_cutoff = cutoff_info["cutoff_pd"]
    if not np.isfinite(pd_cutoff):
        pd_cutoff = float(np.quantile(preds_hold[champion], 0.9))
        log.warning("no holdout cutoff meets the target default rate; declining the riskiest 10%% instead")

    importance = model.importance()
    in_model = scorecard.selected if champion == "scorecard" else spec.features
    top_features = [f for f, _ in sorted(importance.items(), key=lambda kv: -kv[1]) if f in in_model]
    top_features = top_features[: cfg["monitoring"]["top_features"]]
    p_dev = model.predict_pd(X_dev)
    mcfg = cfg["monitoring"]
    baseline = MonitoringBaseline.fit(p_dev, X_dev, spec.numeric, spec.categorical, top_features,
                                      mcfg["psi_bins"], mcfg["psi_warn"], mcfg["psi_alert"],
                                      baseline_label=f"{min(V['train'])}-{max(V['train'])} development sample")

    bundle = ModelBundle(
        model_type=champion,
        model=model,
        spec=spec,
        pd_cutoff=float(pd_cutoff),
        reference=reference,
        baseline=baseline,
        scaling={k: cfg["scorecard"][k] for k in ("pdo", "base_score", "base_odds")},
        top_n=cfg["reasons"]["top_n"],
        metadata={
            "version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_vintages": V["train"],
            "oot_vintage": V["oot"],
            "target": f"90+ days delinquent within {cfg['target']['window_months']} months of first payment",
            "synthetic": synthetic,
            "features": spec.features,
        },
    )
    path = bundle.save(bundle_path(cfg))
    log.info("saved model bundle to %s", path)

    # Reasons on the OOT loans the champion would decline.
    reason_summary, reason_table = reason_diagnostics(model, X_oot, preds_oot[champion], pd_cutoff, reference,
                                                      cfg["reasons"]["top_n"], spec)
    reason_table.to_csv(reports / "reason_frequency_oot.csv", index=False)

    # Stress test and relief decomposition.
    breakdown_rows = []
    stress_results = {}
    oot_breakdown = outcome_breakdown(oot, preds_oot[champion], cfg)
    breakdown_rows.append({"label": f"{V['oot']} (OOT)", **oot_breakdown})
    if stress is not None and len(stress):
        X_st = spec.transform(stress)
        for name, m in models.items():
            stress_results[name] = outcome_breakdown(stress, m.predict_pd(X_st), cfg)
        p_st = model.predict_pd(X_st)
        stress_results["psi_vs_training"] = baseline.check(p_st, X_st)
        stress_results["approval_real_champion"] = approval_at_bad_rate(
            stress["bad_real"].astype(int).to_numpy(), p_st, target_bad_rate)
        breakdown_rows.append({"label": f"{V['stress']} (stress)", **stress_results[champion]})
    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv(reports / "outcome_breakdown.csv", index=False)

    monitoring_summary = monitor(cfg, bundle=bundle, make_plots=make_plots)

    metrics = {
        "synthetic_data": synthetic,
        "trained_at": bundle.metadata["trained_at"],
        "champion": champion,
        "data": {
            "train_vintages": V["train"],
            "oot_vintage": V["oot"],
            "stress_vintage": V["stress"] if stress is not None else None,
            "n_dev": int(len(dev)), "n_fit": int(len(idx_fit)), "n_holdout": int(len(idx_hold)),
            "n_oot": int(len(oot)), "n_stress": int(len(stress)) if stress is not None else 0,
            "dev_default_rate": float(y_dev.mean()), "oot_default_rate": float(y_oot.mean()),
            "oot_windows_reaching_covid": oot_after_covid,
            "oot_last_window_end": month_label(float(oot["window_end_idx"].max())),
        },
        "models": results,
        "auc_gain_gbm_vs_scorecard": gain,
        "target_bad_rate": target_bad_rate,
        "decision_cutoff": {"pd_cutoff": float(pd_cutoff), "holdout": cutoff_info},
        "scorecard": {"selected": scorecard.selected, "dropped": scorecard.dropped,
                      "information_value": scorecard.importance()},
        "gbm": {"trees": gbm.best_iteration, "platt_a": gbm.platt_a, "platt_b": gbm.platt_b,
                "mean_abs_shap": gbm.importance()},
        "top_features": top_features,
        "reasons": reason_summary,
        "outcome_breakdown_oot": oot_breakdown,
        "stress": stress_results,
        "monitoring": monitoring_summary.to_dict(orient="records") if len(monitoring_summary) else [],
        "headline": {},
    }
    champ = results[champion]
    metrics["headline"] = {
        "oot_auc": champ["oot"]["auc"],
        "oot_auc_scorecard": results["scorecard"]["oot"]["auc"],
        "oot_auc_gbm": results["gbm"]["oot"]["auc"],
        "calibration_error_by_decile": champ["oot"]["ece_decile"],
        "calibration_error_by_decile_scorecard": results["scorecard"]["oot"]["ece_decile"],
        "calibration_error_by_decile_gbm": results["gbm"]["oot"]["ece_decile"],
        "psi_stress_vs_training": stress_results.get("psi_vs_training", {}).get("score_psi"),
        "approval_rate_at_target": champ["approval_oot"]["approval_rate"],
        "approval_rate_at_target_scorecard": results["scorecard"]["approval_oot"]["approval_rate"],
        "approval_rate_at_target_gbm": results["gbm"]["approval_oot"]["approval_rate"],
        "target_bad_rate": target_bad_rate,
    }
    metrics = _jsonable(metrics)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if make_plots:
        from . import plots

        figs = reports / "figures"
        plots.roc_chart(rocs, {n: results[n]["oot"]["auc"] for n in models},
                        f"Out-of-time ROC, {V['oot']} vintage", figs / "roc_oot.png")
        plots.calibration_chart(cal_tables, f"Calibration by decile, {V['oot']} vintage", figs / "calibration_oot.png")
        plots.approval_chart(curves, target_bad_rate, f"Approval rate vs. default rate, {V['oot']} vintage",
                             figs / "approval_curve_oot.png")
        if len(breakdown):
            plots.stress_chart(breakdown, figs / "stress_breakdown.png")

    from .report import write_model_document

    doc_path = write_model_document(cfg, metrics)
    log.info("model document written to %s", doc_path)

    if use_mlflow:
        from .tracking import log_training_run

        params = {
            "features": spec.features, "monotone": spec.monotone, "target": cfg["target"],
            "vintages": V, "scorecard": cfg["scorecard"], "gbm": cfg["gbm"], "champion_rule": cfg["champion"],
            "approval": cfg["approval"], "monitoring": cfg["monitoring"],
        }
        example = oot[spec.features].head(5).reset_index(drop=True)
        metrics["mlflow"] = log_training_run(cfg, path, bundle, metrics["headline"] | {
            "holdout_auc": champ["holdout"]["auc"], "oot_ks": champ["oot"]["ks"],
            "oot_brier": champ["oot"]["brier"], "reason_faithfulness": reason_summary.get("faithfulness_top_reason"),
        }, params, reports, doc_path, example)
        (reports / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    return metrics


# --------------------------------------------------------------------------- monitoring
def monitor(cfg: dict, bundle: ModelBundle | None = None, vintages: list[int] | None = None,
            make_plots: bool = True) -> pd.DataFrame:
    """PSI of the score and top features for each new vintage, plus outcomes where they exist."""
    from .evaluate import auc

    bundle = bundle or ModelBundle.load(bundle_path(cfg))
    reports = Path(cfg["paths"]["reports_dir"]) / "monitoring"
    reports.mkdir(parents=True, exist_ok=True)
    train_years = set(cfg["vintages"]["train"])
    candidates = vintages or [v for v in cfg["vintages"]["monitor"] if v not in train_years]
    present = set(available_vintages(cfg))
    rows = []
    for year in candidates:
        if year not in present:
            continue
        df = load_vintages(cfg, [year])
        X = bundle.spec.transform(df)
        p = bundle.model.predict_pd(X)
        result = bundle.baseline.check(p, X)
        result["vintage"] = year
        result["n"] = int(len(df))
        result["mean_pd"] = float(np.mean(p))
        complete = (df["has_performance"].astype(bool) & df["window_complete"].astype(bool)).to_numpy()
        if complete.sum() >= 100:
            y = df.loc[complete, "bad"].astype(int).to_numpy()
            y_real = df.loc[complete, "bad_real"].astype(int).to_numpy()
            result["outcomes"] = {
                "n_complete": int(complete.sum()),
                "observed_rate": float(y.mean()),
                "observed_real_rate": float(y_real.mean()),
                "mean_pd_complete": float(np.mean(p[complete])),
                "auc": auc(y, p[complete]),
                "auc_real": auc(y_real, p[complete]),
            }
        (reports / f"psi_{year}.json").write_text(json.dumps(_jsonable(result), indent=2))
        level = {"alert": logging.WARNING, "watch": logging.INFO}.get(result["status"], logging.INFO)
        log.log(level, "vintage %s: score PSI %.3f, status %s%s", year, result["score_psi"], result["status"].upper(),
                f" ({', '.join(result['alerts'])})" if result["alerts"] else "")
        row = {"vintage": year, "n": result["n"], "mean_pd": result["mean_pd"], "score_psi": result["score_psi"],
               "max_watched_psi": result["max_watched_psi"], "status": result["status"],
               "alerts": ";".join(result["alerts"]), "warnings": ";".join(result["warnings"])}
        row.update({f"psi_{c}": v for c, v in result["feature_psi"].items()})
        if "outcomes" in result:
            row.update({f"outcome_{k}": v for k, v in result["outcomes"].items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary.to_csv(reports / "psi_summary.csv", index=False)
        if make_plots:
            from . import plots

            plots.psi_chart(summary, bundle.baseline.warn, bundle.baseline.alert,
                            Path(cfg["paths"]["reports_dir"]) / "figures" / "psi_by_vintage.png")
    return summary
