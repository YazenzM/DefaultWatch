import numpy as np
import pandas as pd

from defaultwatch.bundle import ModelBundle
from defaultwatch.evaluate import approval_at_bad_rate, calibration_by_decile
from defaultwatch.monitor import CategoricalBaseline, NumericBaseline
from defaultwatch.pipeline import bundle_path

RISKY = {"credit_score": 640, "orig_dti": 47, "orig_ltv": 95, "orig_cltv": 95, "orig_loan_term": 360,
         "num_units": 1, "loan_purpose": "C", "occupancy_status": "I", "property_type": "CO"}
SAFE = {"credit_score": 800, "orig_dti": 20, "orig_ltv": 55, "orig_cltv": 55, "orig_loan_term": 180,
        "num_units": 1, "loan_purpose": "P", "occupancy_status": "P", "property_type": "SF"}


def test_scorecard_bins_are_monotone(trained):
    table = pd.read_csv(f"{trained['cfg']['paths']['reports_dir']}/scorecard_points.csv")
    for feature, direction in trained["cfg"]["features"]["monotone"].items():
        rates = table[(table.feature == feature) & (table.bin != "Missing")]["bad_rate"].to_numpy()
        diffs = np.diff(rates)
        assert (diffs * direction >= -1e-12).all(), (feature, rates)


def _toy(n=4000, seed=0):
    from defaultwatch.config import load_config
    from defaultwatch.features import FeatureSpec

    cfg = load_config()
    spec = FeatureSpec.from_config(cfg)
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        "credit_score": rng.normal(740, 50, n).round(), "orig_dti": rng.uniform(10, 50, n).round(),
        "orig_ltv": rng.uniform(40, 97, n).round(), "orig_cltv": rng.uniform(40, 97, n).round(),
        "orig_loan_term": rng.choice([180, 360], n), "num_units": 1,
        "loan_purpose": rng.choice(["P", "C", "N"], n), "occupancy_status": rng.choice(["P", "I"], n),
        "property_type": rng.choice(["SF", "CO"], n),
    })
    X = spec.transform(raw)
    logit = -3.5 - 0.03 * (X.credit_score - 740) + 0.03 * (X.orig_dti - 30) + 0.5 * (X.occupancy_status == "I")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return cfg, spec, X, y


def test_points_reproduce_pd():
    from defaultwatch.scorecard import Scorecard

    cfg, spec, X, y = _toy()
    sc = Scorecard.from_config(spec, cfg).fit(X, y)
    p = sc.predict_pd(X)
    score = sc.score(X)
    implied = sc.offset + sc.factor * np.log((1 - p) / p)
    assert np.allclose(score, implied)
    assert (sc.contributions(X) >= -1e-9).all().all()


def test_gbm_respects_monotone_constraints():
    from defaultwatch.gbm import GBMModel

    cfg, spec, X, y = _toy(8000, seed=3)
    params = {**cfg["gbm"], "n_estimators": 300, "min_child_samples": 50}
    gbm = GBMModel(spec, params).fit(X.iloc[:6000], y[:6000], X.iloc[6000:], y[6000:])
    for base, feature, grid, sign in [(RISKY, "credit_score", np.linspace(600, 840, 41), -1),
                                      (SAFE, "orig_dti", np.linspace(5, 50, 41), 1),
                                      (SAFE, "orig_ltv", np.linspace(30, 97, 41), 1)]:
        rows = pd.DataFrame([base] * len(grid))
        rows[feature] = grid
        p = gbm.predict_pd(spec.transform(rows))
        assert (np.diff(p) * sign >= -1e-12).all(), feature
    # SHAP contributions plus the expected value reproduce the calibrated log-odds.
    sample = X.iloc[:200]
    raw = gbm.model.predict(gbm._prep(sample), pred_contrib=True, num_iteration=gbm.best_iteration)
    logit = gbm.platt_a * raw.sum(axis=1) + gbm.platt_b
    assert np.allclose(logit, gbm.predict_logit(sample))


def test_reasons_are_specific(trained):
    bundle = ModelBundle.load(bundle_path(trained["cfg"]))
    out = bundle.score_frame(pd.DataFrame([RISKY, SAFE]))
    assert out.loc[0, "pd"] > out.loc[1, "pd"]
    assert out.loc[0, "decision"] == "decline" and out.loc[1, "decision"] == "approve"
    reasons = out.loc[0, "reasons"]
    assert 1 <= len(reasons) <= trained["cfg"]["reasons"]["top_n"]
    assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
    contribs = [r["contribution"] for r in reasons]
    assert all(c > 0 for c in contribs) and contribs == sorted(contribs, reverse=True)
    text = " ".join(r["reason"] for r in reasons)
    assert "640" in text or "47%" in text or "95%" in text, text
    declined_only = bundle.score_frame(pd.DataFrame([RISKY, SAFE]), reasons_for="declined")
    assert declined_only.loc[1, "reasons"] == []


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    base = NumericBaseline.fit(rng.normal(0, 1, 20000))
    assert base.psi(rng.normal(0, 1, 20000)) < 0.01
    assert base.psi(rng.normal(1, 1, 20000)) > 0.25
    cat = CategoricalBaseline.fit(pd.Series(["P"] * 80 + ["C"] * 20))
    assert cat.psi(pd.Series(["P"] * 80 + ["C"] * 20)) < 1e-6
    assert cat.psi(pd.Series(["P"] * 30 + ["C"] * 50 + ["N"] * 20)) > 0.25


def test_approval_at_fixed_default_rate():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    r = approval_at_bad_rate(y, p, target_bad_rate=0.2)
    # Approving the 7 lowest-PD loans gives 1/7 = 14% bad; 8 gives 2/8 = 25%.
    assert r["approval_rate"] == 0.7 and abs(r["cutoff_pd"] - 0.07) < 1e-12
    table = calibration_by_decile(y, p, n_bins=5)
    assert table["n"].sum() == 10 and len(table) == 5
