import numpy as np
import pandas as pd

from defaultwatch.config import load_config
from defaultwatch.features import MISSING, FeatureSpec
from defaultwatch.ingest import compute_targets, find_source, read_origination
from defaultwatch.synth import generate

CFG = load_config()


def _rows(loan, statuses, zb=None, assist=None, disaster=None, deferral=None):
    n = len(statuses)
    zb = zb or {}
    return pd.DataFrame({
        "loan_id": [loan] * n,
        "msfp": np.arange(n),
        "dlq_status": [str(s) for s in statuses],
        "zero_balance_code": [zb.get(k, "") for k in range(n)],
        "borrower_assistance_status": [(assist or {}).get(k, "") for k in range(n)],
        "disaster_delinquency": [(disaster or {}).get(k, "") for k in range(n)],
        "payment_deferral": [(deferral or {}).get(k, "") for k in range(n)],
    })


def _targets():
    full = 24
    perf = pd.concat([
        _rows("A_real", [0, 0, 1, 2, 3] + [4] * (full - 5)),
        _rows("B_forbearance", [0] * 5 + [1, 2, 3, 4] + [0] * (full - 9), assist={5: "F", 6: "F", 7: "F", 8: "F"}),
        _rows("C_prepaid", [0] * 7, zb={6: "01"}),
        _rows("D_incomplete", [0] * 11),
        _rows("E_disaster", [0, 0, 0, 1, 2, 3] + [0] * (full - 6), disaster={3: "Y"}),
        _rows("F_after_relief", [0, 0, 1, 0, 0, 0, 0, 1, 2, 3] + [0] * (full - 10), assist={2: "F"}),
        _rows("G_short_sale", [0] * 12 + [2], zb={12: "03"}),
        _rows("H_reo", [0] * 10 + ["RA"]),
        _rows("I_clean", [0] * full),
        _rows("J_unknown_status", ["", "XX"] + [0] * (full - 2)),
    ], ignore_index=True)
    return compute_targets(perf, CFG).set_index("loan_id")


def test_target_definitions():
    t = _targets()
    assert t.loc["A_real", "bad"] and t.loc["A_real", "bad_real"]
    assert t.loc["A_real", "first_bad_msfp"] == 4
    # 90+ during forbearance is relief, not real default.
    assert t.loc["B_forbearance", "bad"] and not t.loc["B_forbearance", "bad_real"]
    assert t.loc["B_forbearance", "bad_relief_only"]
    # Prepayment inside the window: good and complete.
    assert not t.loc["C_prepaid", "bad"] and t.loc["C_prepaid", "window_complete"]
    # Data stops early with no termination: excluded.
    assert not t.loc["D_incomplete", "window_complete"]
    # Disaster flag 2 months before reaching 90+ is inside the 3-month lookback.
    assert t.loc["E_disaster", "bad_relief_only"]
    # Relief 7 months before the 90+ month does not excuse it.
    assert t.loc["F_after_relief", "bad_real"]
    assert t.loc["G_short_sale", "bad"] and t.loc["G_short_sale", "window_complete"]
    assert t.loc["H_reo", "bad"]
    assert not t.loc["I_clean", "bad"] and t.loc["I_clean", "window_complete"]
    assert not t.loc["J_unknown_status", "bad"]


def test_reads_zip_and_text_layouts(tmp_path):
    generate(tmp_path / "z", vintages=[2015], origination_only=[], loans_per_vintage=50, as_zip=True)
    generate(tmp_path / "t", vintages=[2015], origination_only=[], loans_per_vintage=50, as_zip=False)
    z = find_source(tmp_path / "z", "orig", 2015)
    t = find_source(tmp_path / "t", "orig", 2015)
    assert z[1] == "sample_orig_2015.txt" and t[1] is None
    a, b = read_origination(z), read_origination(t)
    assert len(a) == len(b) == 50
    assert list(a.columns[:3]) == ["credit_score", "first_payment_date", "first_time_homebuyer"]
    assert find_source(tmp_path / "z", "svcg", 2015) is not None
    assert find_source(tmp_path / "z", "orig", 2099) is None


def test_sentinels_become_missing():
    spec = FeatureSpec.from_config(CFG)
    raw = pd.DataFrame({
        "credit_score": ["9999", "720"], "orig_dti": ["999", "36"], "orig_ltv": ["80", "999"],
        "orig_cltv": ["80", "90"], "orig_loan_term": ["360", "180"], "num_units": ["99", "1"],
        "loan_purpose": ["9", "C"], "occupancy_status": ["P", " i "], "property_type": ["99", "SF"],
    })
    X = spec.transform(raw)
    assert np.isnan(X.loc[0, "credit_score"]) and X.loc[1, "credit_score"] == 720
    assert np.isnan(X.loc[0, "orig_dti"]) and np.isnan(X.loc[1, "orig_ltv"]) and np.isnan(X.loc[0, "num_units"])
    assert X.loc[0, "loan_purpose"] == MISSING and X.loc[0, "property_type"] == MISSING
    assert X.loc[1, "occupancy_status"] == "I"
