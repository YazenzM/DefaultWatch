from pathlib import Path

import pytest
import yaml

import defaultwatch
from tests.conftest import make_config

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
SCRIPT = str(Path(defaultwatch.__file__).with_name("dashboard.py"))
TABS = ["Overview", "Validation", "Stress test", "Monitoring", "Reason codes", "Score a loan"]


def _by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_dashboard_renders_every_tab(trained, monkeypatch):
    monkeypatch.setenv("DEFAULTWATCH_CONFIG", str(trained["cfg_path"]))
    at = AppTest.from_file(SCRIPT, default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == TABS
    assert any("Synthetic data" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Out-of-time AUC") for label in labels)
    assert any(label.startswith("PSI") for label in labels)
    assert any("Approval rate" in label for label in labels)
    assert any("Alert on vintage" in e.value for e in at.error)  # synthetic 2021 shift


def test_score_tab_reacts_to_inputs(trained, monkeypatch):
    monkeypatch.setenv("DEFAULTWATCH_CONFIG", str(trained["cfg_path"]))
    at = AppTest.from_file(SCRIPT, default_timeout=120).run()
    _by_label(at.slider, "Credit score").set_value(800)
    _by_label(at.slider, "Debt-to-income ratio (%)").set_value(20)
    _by_label(at.slider, "Loan-to-value ratio (%)").set_value(60)
    at.run()
    assert _by_label(at.metric, "Decision").value == "✅ Approve"
    _by_label(at.slider, "Credit score").set_value(620)
    _by_label(at.slider, "Debt-to-income ratio (%)").set_value(49)
    _by_label(at.slider, "Loan-to-value ratio (%)").set_value(97)
    _by_label(at.selectbox, "Occupancy").set_value("I")
    at.run()
    assert not at.exception, at.exception
    assert _by_label(at.metric, "Decision").value == "🔴 Decline"
    assert any(m.value.startswith("1. ") for m in at.markdown)  # numbered principal reasons


def test_dashboard_without_outputs(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("DEFAULTWATCH_CONFIG", str(path))
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("metrics.json" in i.value for i in at.info)
