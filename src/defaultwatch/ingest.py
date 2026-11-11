"""Read Freddie Mac sample files and build the loan-level modeling table.

One output row per loan: origination fields plus the 24-month outcome.

Target definitions (months counted from the first payment date, month 0..W-1):
  bad              90+ days delinquent, REO acquisition, or a default-type zero-balance
                   code at any point in the window. This is the modeling target.
  bad_real         the same, counting only events NOT covered by a relief flag
                   (payment deferral, disaster delinquency, forbearance / repayment /
                   trial-modification status) in the event month or the lookback months before it.
  bad_relief_only  bad and not bad_real: the delinquency is explained by relief.
  window_complete  the full window was observed, the loan terminated inside it,
                   or it went bad (label known). Incomplete loans are excluded from modeling.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .schema import (
    ASSISTANCE_RELIEF_CODES,
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_USECOLS,
)

log = logging.getLogger(__name__)

SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"

Source = tuple[Path, "str | None"]  # (file path, zip member name or None)


# --------------------------------------------------------------------------- dates
def month_index(yyyymm: pd.Series) -> pd.Series:
    """YYYYMM -> months since year 0 (float, NaN when unparseable)."""
    s = pd.to_numeric(yyyymm, errors="coerce")
    return (s // 100) * 12 + (s % 100) - 1


def month_label(idx: float | int | None) -> str | None:
    if idx is None or (isinstance(idx, float) and np.isnan(idx)):
        return None
    idx = int(idx)
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def label_to_index(label: str) -> int:
    year, month = label.split("-")
    return int(year) * 12 + int(month) - 1


# --------------------------------------------------------------------------- files
def find_source(raw_dir: str | Path, kind: str, year: int) -> Source | None:
    """Locate the origination ("orig") or performance ("svcg") file for a vintage.

    Accepts extracted text files anywhere under raw_dir, or the zip archives as
    downloaded from Freddie Mac (sample_YYYY.zip).
    """
    raw = Path(raw_dir)
    if not raw.exists():
        return None
    for pattern in (f"sample_{kind}_{year}.txt", f"*_{kind}_{year}.txt"):
        hits = sorted(raw.rglob(pattern))
        if hits:
            return hits[0], None
    for archive in sorted(raw.rglob(f"*{year}*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base.endswith(".txt") and f"_{kind}_" in base and str(year) in base:
                    return archive, name
    return None


@contextlib.contextmanager
def _open(src: Source):
    path, member = src
    if member is None:
        with open(path, "rb") as f:
            yield f
    else:
        with zipfile.ZipFile(path) as zf, zf.open(member) as f:
            yield f


def _column_names(src: Source, layout: list[str]) -> list[str]:
    with _open(src) as f:
        first = f.readline().decode("latin-1").rstrip("\r\n")
    ncols = first.count("|") + 1
    if ncols <= len(layout):
        return layout[:ncols]
    return layout + [f"extra_{i}" for i in range(ncols - len(layout))]


def read_origination(src: Source) -> pd.DataFrame:
    names = _column_names(src, ORIGINATION_COLUMNS)
    with _open(src) as f:
        raw = f.read()
    df = pd.read_csv(
        io.BytesIO(raw),
        sep="|",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        encoding="latin-1",
    )
    for col in df.columns:
        df[col] = df[col].str.strip()
    for col in ORIGINATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def iter_performance(src: Source, block_size: int = 1 << 26) -> Iterator[pd.DataFrame]:
    """Stream the monthly performance file in blocks, keeping only the columns the target needs."""
    names = _column_names(src, PERFORMANCE_COLUMNS)
    include = [c for c in PERFORMANCE_USECOLS if c in names]
    read_opts = pacsv.ReadOptions(column_names=names, block_size=block_size)
    parse_opts = pacsv.ParseOptions(delimiter="|", quote_char=False)
    conv_opts = pacsv.ConvertOptions(
        include_columns=include,
        column_types={c: pa.string() for c in include},
        strings_can_be_null=False,
    )
    with _open(src) as f:
        reader = pacsv.open_csv(f, read_options=read_opts, parse_options=parse_opts, convert_options=conv_opts)
        for batch in reader:
            df = batch.to_pandas()
            for col in PERFORMANCE_USECOLS:
                if col not in df.columns:
                    df[col] = ""
                else:
                    df[col] = df[col].fillna("").astype(str).str.strip()
            yield df


# --------------------------------------------------------------------------- targets
def compute_targets(perf: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Loan-level outcome from window rows.

    `perf` holds performance rows already restricted to the window, with an
    integer column `msfp` (months since first payment, 0..W-1).
    """
    tcfg = cfg["target"]
    window = int(tcfg["window_months"])
    threshold = int(tcfg["dpd_threshold"])
    lookback = int(tcfg["relief_lookback_months"])
    default_codes = set(tcfg["default_zero_balance_codes"])

    status = perf["dlq_status"].astype(str).str.strip().str.upper()
    dlq = pd.to_numeric(status, errors="coerce")
    zb = perf["zero_balance_code"].astype(str).str.strip()
    bad_event = (dlq >= threshold).fillna(False) | status.eq("RA") | zb.isin(default_codes)
    terminated = zb.ne("")
    relief = (
        perf["payment_deferral"].astype(str).str.strip().str.upper().eq("Y")
        | perf["disaster_delinquency"].astype(str).str.strip().str.upper().eq("Y")
        | perf["borrower_assistance_status"].astype(str).str.strip().str.upper().isin(ASSISTANCE_RELIEF_CODES)
    )

    base = pd.DataFrame(
        {
            "loan_id": perf["loan_id"].to_numpy(),
            "msfp": perf["msfp"].to_numpy().astype(int),
            "bad_event": bad_event.to_numpy(dtype=bool),
            "relief": relief.to_numpy(dtype=bool),
            "terminated": terminated.to_numpy(dtype=bool),
        }
    )

    # A bad month is "relief" if any relief flag appears in that month or the lookback months before it.
    relief_keys = base.loc[base["relief"], ["loan_id", "msfp"]]
    covered = pd.concat(
        [relief_keys.assign(msfp=relief_keys["msfp"] + k) for k in range(lookback + 1)],
        ignore_index=True,
    ).drop_duplicates()
    covered["relief_cover"] = True
    events = base.loc[base["bad_event"], ["loan_id", "msfp"]].merge(covered, on=["loan_id", "msfp"], how="left")
    events["relief_cover"] = events["relief_cover"].astype("boolean").fillna(False).astype(bool)

    grouped = base.groupby("loan_id")
    out = pd.DataFrame(
        {
            "last_msfp": grouped["msfp"].max(),
            "terminated_in_window": grouped["terminated"].any(),
            "relief_in_window": grouped["relief"].any(),
        }
    )
    out["first_bad_msfp"] = events.groupby("loan_id")["msfp"].min().reindex(out.index)
    out["first_real_bad_msfp"] = (
        events.loc[~events["relief_cover"]].groupby("loan_id")["msfp"].min().reindex(out.index)
    )
    out["bad"] = out["first_bad_msfp"].notna()
    out["bad_real"] = out["first_real_bad_msfp"].notna()
    out["bad_relief_only"] = out["bad"] & ~out["bad_real"]
    out["window_complete"] = (out["last_msfp"] >= window - 1) | out["terminated_in_window"] | out["bad"]
    out.index.name = "loan_id"
    return out.reset_index()


def build_vintage(orig_src: Source, perf_src: Source | None, year: int, cfg: dict) -> pd.DataFrame:
    """Origination fields plus outcome columns for one vintage."""
    orig = read_origination(orig_src)
    orig["vintage"] = int(year)
    orig["fp_idx"] = month_index(orig["first_payment_date"])
    window = int(cfg["target"]["window_months"])
    orig["window_end_idx"] = orig["fp_idx"] + window - 1

    target_cols = [
        "last_msfp", "terminated_in_window", "relief_in_window", "first_bad_msfp",
        "first_real_bad_msfp", "bad", "bad_real", "bad_relief_only", "window_complete",
    ]
    if perf_src is None:
        for col in target_cols:
            orig[col] = np.nan
        orig["has_performance"] = False
        return orig

    fp = orig.set_index("loan_id")["fp_idx"]
    parts: list[pd.DataFrame] = []
    n_rows = 0
    for chunk in iter_performance(perf_src):
        n_rows += len(chunk)
        msfp = month_index(chunk["period"]) - chunk["loan_id"].map(fp)
        keep = ((msfp >= 0) & (msfp < window)).fillna(False).to_numpy(dtype=bool)
        if keep.any():
            sub = chunk.loc[keep, PERFORMANCE_USECOLS].copy()
            sub["msfp"] = msfp[keep].astype(int).to_numpy()
            parts.append(sub)
    log.info("vintage %s: read %s performance rows, %s inside the %s-month window",
             year, f"{n_rows:,}", f"{sum(len(p) for p in parts):,}", window)

    if parts:
        targets = compute_targets(pd.concat(parts, ignore_index=True), cfg)
    else:
        targets = pd.DataFrame(columns=["loan_id"] + target_cols)
    loans = orig.merge(targets, on="loan_id", how="left")
    for col in ["terminated_in_window", "relief_in_window", "bad", "bad_real", "bad_relief_only", "window_complete"]:
        loans[col] = loans[col].astype("boolean").fillna(False).astype(bool)
    loans["has_performance"] = True
    return loans


def all_vintages(cfg: dict) -> list[int]:
    v = cfg["vintages"]
    return sorted(set(v["train"]) | {v["oot"], v["stress"]} | set(v.get("monitor", [])))


def is_synthetic(raw_dir: str | Path) -> bool:
    return (Path(raw_dir) / SYNTHETIC_MARKER).exists()


def prepare(cfg: dict, vintages: list[int] | None = None) -> dict[int, Path]:
    """Build data/processed/loans_YYYY.parquet for every vintage whose files are present."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for year in vintages or all_vintages(cfg):
        orig_src = find_source(raw_dir, "orig", year)
        if orig_src is None:
            log.info("vintage %s: no origination file under %s, skipping", year, raw_dir)
            continue
        perf_src = find_source(raw_dir, "svcg", year)
        loans = build_vintage(orig_src, perf_src, year, cfg)
        path = out_dir / f"loans_{year}.parquet"
        loans.to_parquet(path, index=False)
        written[year] = path
        if perf_src is not None:
            complete = loans["window_complete"]
            log.info(
                "vintage %s: %s loans, %s complete windows, default rate %.3f%%",
                year, f"{len(loans):,}", f"{int(complete.sum()):,}",
                100 * loans.loc[complete, "bad"].mean() if complete.any() else float("nan"),
            )
        else:
            log.info("vintage %s: %s loans (origination only, no outcomes)", year, f"{len(loans):,}")
    meta = {"synthetic": is_synthetic(raw_dir), "vintages": sorted(written)}
    (out_dir / "data_source.json").write_text(json.dumps(meta, indent=2))
    return written


def load_vintages(cfg: dict, years: list[int]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["processed_dir"])
    frames = []
    for year in years:
        path = out_dir / f"loans_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `defaultwatch prepare` after placing the Freddie Mac files in {cfg['paths']['raw_dir']}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def available_vintages(cfg: dict) -> list[int]:
    out_dir = Path(cfg["paths"]["processed_dir"])
    return sorted(int(p.stem.split("_")[1]) for p in out_dir.glob("loans_*.parquet"))


def processed_is_synthetic(cfg: dict) -> bool:
    meta = Path(cfg["paths"]["processed_dir"]) / "data_source.json"
    return bool(json.loads(meta.read_text()).get("synthetic")) if meta.exists() else False
