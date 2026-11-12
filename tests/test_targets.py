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
        # Release 48 payment deferral: C in the month granted, P on every month after it.
        _rows("K_deferral", [0, 0, 0, 0, 1, 2, 3] + [0] * (full - 7),
              deferral={4: "C"} | {k: "P" for k in range(5, full)}),
        _rows("L_prior_deferral_only", [0] * 8 + [1, 2, 3] + [0] * (full - 11),
              deferral={2: "C"} | {k: "P" for k in range(3, full)}),
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
    # A deferral granted two months before the 90+ month is inside the lookback: relief.
    assert t.loc["K_deferral", "bad_relief_only"]
    # P only says a deferral happened earlier; 90+ six months later is real default.
    assert t.loc["L_prior_deferral_only", "bad_real"]


def test_reads_zip_and_text_layouts(tmp_path):
    generate(tmp_path / "z", vintages=[2015], origination_only=[], loans_per_vintage=50, as_zip=True)
    generate(tmp_path / "t", vintages=[2015], origination_only=[], loans_per_vintage=50, as_zip=False)
    z = find_source(tmp_path / "z", "orig", 2015)
    t = find_source(tmp_path / "t", "orig", 2015)
    assert z[1] == "sample_orig_2015.txt" and t[1] is None
    a, b = read_origination(z), read_origination(t)
    assert len(a) == len(b) == 50
    assert list(a.columns[:3]) == ["credit_score", "first_payment_date", "first_time_homebuyer"]
    assert (a["super_conforming"] == "N").all() and (a["vantage_score_4"] == "9999").all()
    assert find_source(tmp_path / "z", "perf", 2015)[1] == "sample_perf_2015.txt"
    assert find_source(tmp_path / "t", "perf", 2015)[0].name == "sample_perf_2015.txt"
    assert find_source(tmp_path / "z", "orig", 2099) is None


def test_reads_pre_release_48_files(tmp_path):
    """Older downloads: sample_svcg_YYYY.txt, and servicer name in origination field 25."""
    legacy = ["700", "201503", "N", "204502", "", "0", "1", "P", "80", "36", "200000", "80", "4.0", "R", "N",
              "FRM", "TX", "SF", "75000", "F15Q10000001", "P", "360", "1", "BANK A", "SERVICER B", "", "", "9",
              "", "9", "N", "7"]
    (tmp_path / "sample_orig_2015.txt").write_text("|".join(legacy) + "\n")
    (tmp_path / "sample_svcg_2015.txt").write_text("F15Q10000001|201503|200000|0|1|359" + "|" * 26 + "\n")
    orig = read_origination(find_source(tmp_path, "orig", 2015))
    assert orig.loc[0, "seller_name"] == "BANK A" and orig.loc[0, "servicer_name"] == "SERVICER B"
    assert orig.loc[0, "mi_cancellation"] == "7" and orig.loc[0, "credit_score"] == "700"
    assert find_source(tmp_path, "perf", 2015)[0].name == "sample_svcg_2015.txt"


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
