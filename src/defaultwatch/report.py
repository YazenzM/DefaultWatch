"""Render the two-page model document from reports/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from .features import FEATURE_LABELS


def _pct(x, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _f(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _n(x) -> str:
    return "n/a" if x is None else f"{int(x):,}"


DIRECTION = {1: "higher value → higher risk", -1: "higher value → lower risk"}


def render(cfg: dict, m: dict) -> str:
    d = m["data"]
    sc, gb = m["models"]["scorecard"], m["models"]["gbm"]
    champ = m["champion"]
    champ_name = "gradient boosting (LightGBM)" if champ == "gbm" else "logistic-regression scorecard"
    stress = m.get("stress") or {}
    st = stress.get(champ, {})
    psi_st = stress.get("psi_vs_training", {})
    reasons = m.get("reasons", {})
    window = cfg["target"]["window_months"]
    banner = ""
    if m.get("synthetic_data"):
        banner = ("> **SYNTHETIC DATA.** These numbers come from the built-in synthetic generator and exist only to "
                  "exercise the pipeline. They are not results on Freddie Mac data.\n\n")

    feat_rows = "\n".join(
        f"| {FEATURE_LABELS.get(f, f)} | `{f}` | {DIRECTION.get(cfg['features']['monotone'].get(f), 'no constraint')} |"
        for f in cfg["features"]["numeric"] + cfg["features"]["categorical"]
    )
    excluded = "; ".join(f"`{k}` ({v})" for k, v in cfg["features"]["excluded"].items())
    dropped = m["scorecard"]["dropped"]
    dropped_txt = ("The scorecard dropped " + "; ".join(f"`{k}` ({v})" for k, v in dropped.items()) + ".") if dropped else \
        "The scorecard kept every feature."

    def row(label, key, fmt, src="oot"):
        return f"| {label} | {fmt(sc[src].get(key))} | {fmt(gb[src].get(key))} |"

    validation = "\n".join([
        "| Metric | Scorecard | Gradient boosting |",
        "|---|---|---|",
        row("AUC, in-time holdout", "auc", _f, "holdout"),
        row(f"**AUC, out-of-time {d['oot_vintage']}**", "auc", _f),
        row("KS, out-of-time", "ks", _f),
        row("**Calibration error by decile** (weighted mean abs. gap)", "ece_decile", _pct),
        row("Largest single-decile gap", "max_decile_error", _pct),
        row("Observed ÷ predicted default rate", "calibration_in_the_large", lambda v: _f(v, 2)),
        row("Brier score", "brier", lambda v: _f(v, 5)),
        f"| **Approval rate at a {_pct(m['target_bad_rate'])} default rate** | "
        f"{_pct(sc['approval_oot']['approval_rate'], 1)} | {_pct(gb['approval_oot']['approval_rate'], 1)} |",
    ])

    if st:
        stress_txt = f"""The {d['stress_vintage']} vintage's {window}-month windows run into COVID-era forbearance. Payment-deferral, \
disaster-delinquency and borrower-assistance (forbearance, repayment, trial modification) flags separate relief from real default: \
a 90+ month counts as relief if any flag appears in that month or the {cfg['target']['relief_lookback_months']} before it.

| {d['stress_vintage']} vintage, champion model | Value |
|---|---|
| Loans with complete windows | {_n(st.get('n'))} |
| Observed 90+ rate, all causes | {_pct(st.get('observed_rate'))} |
| Observed 90+ rate excluding relief (real default) | {_pct(st.get('observed_real_rate'))} |
| Share of 90+ explained by relief | {_pct(st.get('relief_share_of_defaults'), 1)} |
| Mean predicted PD | {_pct(st.get('mean_pd'))} |
| AUC vs. all 90+ / vs. real default | {_f(st.get('auc_all'))} / {_f(st.get('auc_real'))} |
| Observed ÷ predicted, all / real | {_f(st.get('calibration_in_the_large_all'), 2)} / {_f(st.get('calibration_in_the_large_real'), 2)} |
| **PSI of {d['stress_vintage']} scores vs. training** | **{_f(psi_st.get('score_psi'))}** ({psi_st.get('score_status', 'n/a')}) |
"""
    else:
        stress_txt = f"The {cfg['vintages']['stress']} vintage files were not present, so the stress test was skipped.\n"

    mon = m.get("monitoring") or []
    if mon:
        def known(x) -> bool:
            return x is not None and x == x   # not missing / NaN

        def outcome(r: dict) -> str:
            ratio, real = r.get("outcome_observed_to_predicted"), r.get("outcome_observed_real_to_predicted")
            if known(ratio) and known(real):
                return f"{ratio:.2f} / {real:.2f} | {_f(r.get('outcome_auc'))}"
            share = r.get("complete_share")
            return (f"pending ({share:.0%} of windows closed) | " if known(share) and share > 0 else "no outcomes yet | ")

        mon_rows = "\n".join(
            f"| {r['vintage']} | {_n(r['n'])} | {_f(r['score_psi'])} | {_f(r['max_watched_psi'])} | {r['status']} "
            f"| {outcome(r)} |" for r in mon)
        mon_table = ("| Vintage | Loans | Score PSI | Worst watched PSI | Status | Observed ÷ predicted, all / excl. "
                     "relief | AUC |\n|---|---|---|---|---|---|---|\n" + mon_rows)
        flagged = [f"{r['vintage']} ({r['outcome_observed_real_to_predicted']:.2f})" for r in mon
                   if known(r.get("outcome_observed_real_to_predicted"))
                   and not 0.75 <= r["outcome_observed_real_to_predicted"] <= 1.33]
        if flagged:
            mon_table += ("\n\nExcluding relief, observed ÷ predicted is outside the 0.75–1.33 band for "
                          + ", ".join(flagged) + ", which triggers review.")
    else:
        mon_table = "_No post-training vintages were available to monitor._"

    freq = Path(cfg["paths"]["reports_dir"]) / "reason_frequency_oot.csv"
    reason_rows = ""
    if freq.exists():
        import pandas as pd

        t = pd.read_csv(freq).head(5)
        reason_rows = "\n".join(
            f"| {FEATURE_LABELS.get(r.feature, r.feature)} | {_pct(r.share_as_top_reason, 1)} | {_pct(r.share_of_declines_citing, 1)} |"
            for r in t.itertuples())
        reason_rows = "| Factor | #1 reason | Cited at all |\n|---|---|---|\n" + reason_rows

    mcfg = cfg["monitoring"]
    return f"""# DefaultWatch model document

**Model:** probability of 90+ day delinquency within {window} months, Freddie Mac conforming fixed-rate mortgages  \\
**Champion:** {champ_name} · **Trained:** {m.get('trained_at', '')} · **Package version:** {__version__}

{banner}## 1. Purpose and use
The model ranks newly originated mortgages by their chance of becoming seriously delinquent early in life. It supports \
approve/decline decisions at a cutoff set for a target default rate, produces the specific principal reasons Regulation B \
requires for a decline, and gives a baseline for monitoring each new vintage. It is not a pricing, loss-severity or \
lifetime-loss model.

## 2. Data
Freddie Mac Single-Family Loan-Level Dataset, sample files (50,000 random loans per vintage). The licence does not allow \
redistribution, so raw files stay out of version control and only code and aggregate results are published. \
Development: {', '.join(str(v) for v in d['train_vintages'])} vintages, {_n(d['n_dev'])} loans with complete windows \
({_pct(d['dev_default_rate'])} default rate), split {_n(d['n_fit'])} fit / {_n(d['n_holdout'])} holdout (stratified). \
Out-of-time test: {d['oot_vintage']} vintage, {_n(d['n_oot'])} loans ({_pct(d['oot_default_rate'])}). \
Stress test: {d.get('stress_vintage') or 'n/a'} vintage, {_n(d.get('n_stress'))} loans.

## 3. Target
A loan is **bad** if, in months 0–{window - 1} after its first payment date, it is reported 90+ days delinquent, is \
acquired as REO, or ends with a default-type zero-balance code (third-party sale, short sale or charge-off, REO \
disposition, non-performing note sale). Prepayment before month {window} is good. Loans whose window was not fully \
observed are excluded. The last out-of-time window closes in {d.get('oot_last_window_end')}; \
{_n(d.get('oot_windows_reaching_covid'))} out-of-time loans have windows reaching {cfg['target']['covid_relief_start']}, \
when COVID payment relief began.

## 4. Features
Only application-time attributes of the borrower's credit, the loan terms and the collateral are used, so that every \
driver of a decision can be stated to the applicant as a reason.

| Feature | Field | Monotone constraint |
|---|---|---|
{feat_rows}

Excluded on purpose: {excluded}. {dropped_txt}

## 5. Method
**Scorecard (baseline).** Quantile bins merged until each holds ≥{_pct(cfg['scorecard']['min_bin_fraction'], 0)} of loans \
and the bad rate is monotone where a direction is known; weight-of-evidence encoding; L2 logistic regression refit until \
every coefficient has the expected sign; points scaled to {cfg['scorecard']['base_score']} at \
{cfg['scorecard']['base_odds']}:1 odds, {cfg['scorecard']['pdo']} points to double the odds. \
**Gradient boosting.** LightGBM with the monotone constraints above, early stopping on holdout log-loss \
({m['gbm']['trees']} trees), then Platt scaling on the holdout. \
**Champion rule.** Gradient boosting is preferred only if it beats the scorecard's out-of-time AUC by \
≥{cfg['champion']['min_auc_gain']}; observed gain {_f(m['auc_gain_gbm_vs_scorecard'], 4)}. \
**Cutoff.** The PD cutoff ({_pct(m['decision_cutoff']['pd_cutoff'])}) is set on the in-time holdout as the most \
approvals whose default rate stays at or below {_pct(m['target_bad_rate'])}, so the out-of-time sample is never used to set it.

## 6. Validation (out-of-time, {d['oot_vintage']} vintage)
{validation}

Calibration error by decile ranks loans into ten equal groups by predicted PD and averages |observed − predicted| \
weighted by group size. Charts: `reports/figures/roc_oot.png`, `calibration_oot.png`, `approval_curve_oot.png`.

## 7. Stress test
{stress_txt}
## 8. Reason codes
Each decline gets up to {cfg['reasons']['top_n']} reasons, taken from the model's own attribution for that applicant \
(SHAP values for gradient boosting, points below each characteristic's maximum for the scorecard), ranked by size and \
worded with the applicant's actual value. Following CFPB Circular 2023-03, reasons are never mapped to the nearest \
generic checklist item. On out-of-time declines ({_n(reasons.get('declined'))} loans, {_pct(reasons.get('decline_rate'), 1)}): \
{_pct(reasons.get('with_at_least_one_reason'), 1)} received at least one reason, {_f(reasons.get('mean_reasons'), 1)} on average. \
**Faithfulness check:** setting the #1 reason's factor to the typical approved value lowered the PD for \
{_pct(reasons.get('faithfulness_top_reason'), 1)} of sampled declines.

{reason_rows}

## 9. Limitations
- **Funded loans only.** Rejected applicants are not in the data, so approval rates are relative to Freddie Mac \
purchases, and no reject inference was done.
- **Thin feature set.** No credit bureau attributes beyond the score, income, reserves or employment, so absolute \
accuracy is below a production underwriting model.
- **Early default only.** A {window}-month window captures early payment problems, not lifetime loss.
- **Relief flags.** The relief split depends on how fully servicers reported forbearance, deferral and disaster codes.
- **Fair lending.** Known proxies are excluded, but the data has no protected-class fields, so disparate-impact testing \
(e.g. with BISG proxies) remains to be done before any lending use.
- **Model selection.** The champion is chosen on out-of-time AUC; a further later vintage gives the unbiased read.

## 10. Monitoring plan
For every new vintage, compute PSI of the score and the top {mcfg['top_features']} features \
({', '.join(m.get('top_features', []))}) against the development sample, using development deciles plus a missing bin. \
**Stable** below {mcfg['psi_warn']}; **watch** from {mcfg['psi_warn']} (review drivers, compare feature mix); \
**alert** at {mcfg['psi_alert']} or above (`defaultwatch monitor --fail-on-alert` exits non-zero; model owner reviews \
within one cycle and decides on recalibration or redevelopment). Once a vintage's {window}-month window closes, also \
compare AUC and observed ÷ predicted default with this document; an AUC drop of more than 0.03 or a ratio outside \
0.75–1.33 triggers review.

{mon_table}
"""


def render_results(m: dict) -> str:
    """The four headline measures as a Markdown table, for the README."""
    d, sc, gb = m["data"], m["models"]["scorecard"], m["models"]["gbm"]
    psi = (m.get("stress") or {}).get("psi_vs_training", {})
    champ = "Gradient boosting" if m["champion"] == "gbm" else "Scorecard"
    note = "\n_Synthetic data: pipeline check only, not Freddie Mac results._\n" if m.get("synthetic_data") else ""
    return f"""| Measure | Scorecard | Gradient boosting |
|---|---|---|
| Out-of-time AUC ({d['oot_vintage']}) | {_f(sc['oot']['auc'])} | {_f(gb['oot']['auc'])} |
| Calibration error by decile ({d['oot_vintage']}) | {_pct(sc['oot']['ece_decile'])} | {_pct(gb['oot']['ece_decile'])} |
| Approval rate at a {_pct(m['target_bad_rate'])} default rate ({d['oot_vintage']}) | {_pct(sc['approval_oot']['approval_rate'], 1)} | {_pct(gb['approval_oot']['approval_rate'], 1)} |
| PSI of {d.get('stress_vintage') or 'stress'} vs. training scores (champion: {champ}) | {_f(psi.get('score_psi'))} ({psi.get('score_status', 'n/a')}) | |
{note}"""


def write_model_document(cfg: dict, metrics: dict | None = None) -> Path:
    reports = Path(cfg["paths"]["reports_dir"])
    if metrics is None:
        metrics = json.loads((reports / "metrics.json").read_text())
    out = Path(cfg["paths"]["docs_dir"]) / "model_document.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cfg, metrics), encoding="utf-8")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.md").write_text(render_results(metrics), encoding="utf-8")
    return out
