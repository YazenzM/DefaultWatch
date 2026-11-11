"""Synthetic files in the exact Freddie Mac sample layout, for tests and a dry run.

Nothing here resembles real Freddie Mac performance. The generator exists so the
pipeline can be exercised end to end before the real files are downloaded, and
every output built from it is stamped SYNTHETIC.

What it imitates:
  * the 32-column origination and performance layouts, sentinels and codes;
  * defaults driven by credit score, DTI, LTV, purpose, occupancy and property type;
  * 2017 hurricane disaster delinquencies in TX/FL (flagged, then deferred);
  * COVID forbearance from April 2020, mostly cured by payment deferral;
  * a refinance-boom population shift in 2021 so PSI has something to catch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from .ingest import SYNTHETIC_MARKER
from .schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS

STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "AZ", "WA", "CO", "NJ", "VA"]
STATE_P = np.array([14, 10, 9, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=float)
COVID_START = 2020 * 12 + 3        # April 2020 (month index, Jan = 0)
HURRICANE_START = 2017 * 12 + 8    # September 2017


def _yyyymm(idx: int) -> str:
    return f"{idx // 12:04d}{idx % 12 + 1:02d}"


def _origination(rng: np.random.Generator, n: int, year: int) -> dict[str, np.ndarray]:
    refi_boom = year >= 2021
    cs = np.clip(rng.normal(758 if refi_boom else 746, 42, n), 600, 840).round()
    dti = np.clip(rng.normal(33, 9, n), 5, 50).round()
    ltv = np.where(rng.random(n) < 0.3, 80, np.clip(rng.normal(74, 15, n), 20, 97)).round()
    if refi_boom:
        ltv = np.clip(ltv - rng.integers(0, 12, n), 20, 97)
    second_lien = rng.random(n) < 0.08
    cltv = np.minimum(ltv + np.where(second_lien, rng.integers(5, 16, n), 0), 105)
    purpose = rng.choice(["P", "N", "C"], n, p=[0.30, 0.45, 0.25] if refi_boom else [0.52, 0.30, 0.18])
    occ = rng.choice(["P", "I", "S"], n, p=[0.88, 0.08, 0.04])
    ptype = rng.choice(["SF", "PU", "CO", "MH", "CP"], n, p=[0.62, 0.26, 0.09, 0.02, 0.01])
    units = rng.choice([1, 2, 3, 4], n, p=[0.965, 0.025, 0.006, 0.004])
    term = rng.choice([360, 180, 240], n, p=[0.84, 0.13, 0.03])
    state = rng.choice(STATES, n, p=STATE_P / STATE_P.sum())
    channel = rng.choice(["R", "C", "B"], n, p=[0.48, 0.42, 0.10])
    fp_month = rng.integers(0, 12, n)
    fp_idx = year * 12 + fp_month
    mi = np.where(ltv > 80, np.where(ltv > 90, 30, np.where(ltv > 85, 25, 12)), 0)

    # Sentinels for missing values.
    cs_out = cs.astype(int).astype(str)
    cs_out[rng.random(n) < 0.004] = "9999"
    dti_out = dti.astype(int).astype(str)
    dti_out[rng.random(n) < 0.006] = "999"

    return {
        "cs": cs, "dti": dti, "ltv": ltv, "cltv": cltv, "purpose": purpose, "occ": occ, "ptype": ptype,
        "units": units, "term": term, "state": state, "channel": channel, "fp_idx": fp_idx, "mi": mi,
        "cs_out": cs_out, "dti_out": dti_out,
    }


def _risk_logit(o: dict, year: int) -> np.ndarray:
    cs, dti, ltv, cltv = o["cs"], o["dti"], o["ltv"], o["cltv"]
    z = (-4.3
         - 0.024 * (cs - 745)
         + 0.030 * (dti - 33)
         + 0.035 * (ltv - 75)
         + 0.025 * (cltv - ltv)
         + 0.55 * (o["purpose"] == "C")
         + 0.65 * (o["occ"] == "I") + 0.30 * (o["occ"] == "S")
         + 0.60 * (o["ptype"] == "MH") + 0.20 * (o["ptype"] == "CO")
         + 0.35 * (o["units"] > 1)
         + 0.45 * (o["term"] == 360)
         + 0.25 * (o["channel"] == "B")
         # A non-linear pocket a scorecard cannot express exactly.
         + 0.70 * ((cs < 680) & (ltv > 90))
         + 0.40 * ((dti > 43) & (cs < 700)))
    return z + {2014: 0.05, 2015: 0.0, 2016: -0.05, 2017: 0.05}.get(year, 0.1)


def _performance_lines(rng: np.random.Generator, o: dict, loan_ids: np.ndarray, year: int,
                       months: int, ncols: int) -> list[str]:
    n = len(loan_ids)
    p_default = 1 / (1 + np.exp(-_risk_logit(o, year)))
    goes_bad = rng.random(n) < p_default
    lines: list[str] = []
    empty = [""] * ncols
    for i in range(n):
        fp = int(o["fp_idx"][i])
        status = np.zeros(months, dtype=int)
        deferral = np.zeros(months, dtype=bool)
        disaster = np.zeros(months, dtype=bool)
        assist = np.array([""] * months, dtype=object)
        zb_code = ""
        end = months

        prepay = rng.geometric(0.009)
        if prepay < months:
            end = prepay
            zb_code = "01"

        if goes_bad[i]:
            start = int(rng.integers(2, 21))
            cures = rng.random() < 0.15
            run = int(rng.integers(1, 3)) if cures else months
            for k in range(start, min(months, start + run)):
                status[k] = k - start + 1
            if not cures:
                end = months
                zb_code = ""
                if start + 10 < months and rng.random() < 0.3:
                    end = start + 10
                    zb_code = rng.choice(["03", "09"])
        elif rng.random() < 0.03:
            start = int(rng.integers(1, months - 2))
            status[start] = 1

        # Disaster delinquency: 2017 hurricanes in TX and FL.
        if o["state"][i] in ("TX", "FL") and rng.random() < 0.05:
            k0 = HURRICANE_START - fp
            if 0 <= k0 < months - 6 and status[k0] == 0:
                for j in range(4):
                    status[k0 + j] = j + 1
                    disaster[k0 + j] = True
                deferral[k0 + 4] = True
                status[k0 + 4:k0 + 6] = 0

        # COVID forbearance: deep delinquency while flagged, then a payment deferral.
        k0 = COVID_START - fp
        if 0 <= k0 < months and rng.random() < 0.03 + 1.0 * p_default[i]:
            length = int(rng.integers(4, 13))
            k_end = min(months, k0 + length)
            for j, k in enumerate(range(k0, k_end)):
                status[k] = j + 1
                assist[k] = "F"
            stays_bad = rng.random() < 0.12
            for k in range(k_end, months):
                status[k] = status[k - 1] + 1 if stays_bad else 0
            if not stays_bad and k_end < months:
                deferral[k_end] = True
            if end < months and zb_code == "01" and end > k0:
                end, zb_code = months, ""

        upb = 250000.0
        n_rows = end + 1 if zb_code else months
        for k in range(n_rows):
            terminal = bool(zb_code) and k == end
            row = empty.copy()
            row[0] = loan_ids[i]
            row[1] = _yyyymm(fp + k)
            row[2] = "0.00" if terminal else f"{upb * (1 - k / 400):.2f}"
            row[3] = "0" if (terminal and zb_code == "01") else str(status[k])
            row[4] = str(k + 1)
            row[5] = str(int(o["term"][i]) - k - 1)
            row[10] = "4.250"
            row[11] = "0.00"
            if terminal:
                row[8] = str(zb_code)
                row[9] = _yyyymm(fp + k)
            if deferral[k]:
                row[24] = "Y"
            if disaster[k]:
                row[28] = "Y"
            if assist[k]:
                row[29] = assist[k]
            lines.append("|".join(row[:ncols]))
    return lines


def generate(out_dir: str | Path, vintages: list[int] | None = None, loans_per_vintage: int = 5000,
             months: int = 30, seed: int = 7, as_zip: bool = True,
             origination_only: list[int] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vintages = vintages or [2014, 2015, 2016, 2017, 2018, 2019]
    origination_only = origination_only if origination_only is not None else [2020, 2021]
    (out / SYNTHETIC_MARKER).write_text(
        "Files in this folder were produced by defaultwatch.synth. They are NOT Freddie Mac data.\n")
    for year in sorted(set(vintages) | set(origination_only)):
        rng = np.random.default_rng(seed + year)
        o = _origination(rng, loans_per_vintage, year)
        ids = np.array([f"F{year % 100:02d}Q{(int(m) % 12) // 3 + 1}{i:07d}"
                        for i, m in enumerate(o["fp_idx"])])
        upb = (rng.lognormal(12.3, 0.45, loans_per_vintage) // 1000 * 1000).clip(40000, 800000)
        orig_rows = []
        for i in range(loans_per_vintage):
            row = [""] * len(ORIGINATION_COLUMNS)
            row[0] = o["cs_out"][i]
            row[1] = _yyyymm(int(o["fp_idx"][i]))
            row[2] = rng.choice(["Y", "N"], p=[0.35, 0.65]) if o["purpose"][i] == "P" else "N"
            row[3] = _yyyymm(int(o["fp_idx"][i]) + int(o["term"][i]) - 1)
            row[4] = str(rng.integers(10000, 49999))
            row[5] = str(int(o["mi"][i]))
            row[6] = str(int(o["units"][i]))
            row[7] = o["occ"][i]
            row[8] = str(int(o["cltv"][i]))
            row[9] = o["dti_out"][i]
            row[10] = str(int(upb[i]))
            row[11] = str(int(o["ltv"][i]))
            row[12] = f"{4.0 + rng.normal(0, 0.3):.3f}"
            row[13] = o["channel"][i]
            row[14] = "N"
            row[15] = "FRM"
            row[16] = o["state"][i]
            row[17] = o["ptype"][i]
            row[18] = f"{rng.integers(100, 999)}00"
            row[19] = ids[i]
            row[20] = o["purpose"][i]
            row[21] = str(int(o["term"][i]))
            row[22] = str(rng.choice([1, 2], p=[0.45, 0.55]))
            row[23] = "Other sellers"
            row[24] = "Other servicers"
            row[25] = ""
            row[26] = ""
            row[27] = "9"
            row[28] = ""
            row[29] = "9"
            row[30] = "N"
            row[31] = "7"
            orig_rows.append("|".join(row))
        orig_text = "\n".join(orig_rows) + "\n"
        files = {f"sample_orig_{year}.txt": orig_text}
        if year not in origination_only:
            perf =_performance_lines(rng, o, ids, year, months, len(PERFORMANCE_COLUMNS))
            files[f"sample_svcg_{year}.txt"] = "\n".join(perf) + "\n"
        if as_zip:
            with zipfile.ZipFile(out / f"sample_{year}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, text in files.items():
                    zf.writestr(name, text)
        else:
            for name, text in files.items():
                (out / name).write_text(text)
    return out
