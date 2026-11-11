import json
from pathlib import Path

from defaultwatch.cli import main


def test_metrics_cover_the_brief(trained):
    m = trained["metrics"]
    h = m["headline"]
    for key in ["oot_auc", "calibration_error_by_decile", "psi_stress_vs_training", "approval_rate_at_target"]:
        assert h[key] is not None, key
    assert 0.5 < h["oot_auc_scorecard"] < 1 and 0.5 < h["oot_auc_gbm"] < 1
    assert m["synthetic_data"] is True
    assert m["data"]["oot_windows_reaching_covid"] == 0
    # The stress vintage separates relief from real default.
    st = m["stress"][m["champion"]]
    assert st["observed_rate"] > st["observed_real_rate"] > 0
    assert 0 < st["relief_share_of_defaults"] < 1
    assert m["reasons"]["with_at_least_one_reason"] == 1.0


def test_reports_and_document_written(trained):
    cfg = trained["cfg"]
    reports = Path(cfg["paths"]["reports_dir"])
    for name in ["metrics.json", "calibration_oot_gbm.csv", "calibration_oot_scorecard.csv", "scorecard_points.csv",
                 "approval_curve_oot.csv", "outcome_breakdown.csv", "monitoring/psi_summary.csv",
                 "figures/roc_oot.png", "figures/calibration_oot.png", "figures/psi_by_vintage.png"]:
        assert (reports / name).exists(), name
    doc = (Path(cfg["paths"]["docs_dir"]) / "model_document.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in doc
    for section in ["## 2. Data", "## 3. Target", "## 6. Validation", "## 9. Limitations", "## 10. Monitoring plan"]:
        assert section in doc


def test_monitor_alerts_on_shifted_vintage(trained):
    rows = {r["vintage"]: r for r in trained["metrics"]["monitoring"]}
    assert rows[2017]["status"] == "stable"
    assert rows[2021]["status"] == "alert"  # synthetic refinance boom shifts LTV and loan purpose
    assert main(["--config", str(trained["cfg_path"]), "monitor", "--vintage", "2017"]) == 0
    assert main(["--config", str(trained["cfg_path"]), "monitor", "--fail-on-alert"]) == 2
    saved = json.loads((Path(trained["cfg"]["paths"]["reports_dir"]) / "monitoring" / "psi_2021.json").read_text())
    assert saved["alerts"]
