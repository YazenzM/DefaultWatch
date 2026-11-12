# DefaultWatch model document

**Model:** probability of 90+ day delinquency within 24 months, Freddie Mac conforming fixed-rate mortgages  \
**Champion:** logistic-regression scorecard · **Trained:** 2026-11-12T03:50:38+00:00 · **Package version:** 0.1.0

## 1. Purpose and use
The model ranks newly originated mortgages by their chance of becoming seriously delinquent early in life. It supports approve/decline decisions at a cutoff set for a target default rate, produces the specific principal reasons Regulation B requires for a decline, and gives a baseline for monitoring each new vintage. It is not a pricing, loss-severity or lifetime-loss model.

## 2. Data
Freddie Mac Single-Family Loan-Level Dataset, sample files (50,000 random loans per vintage). The licence does not allow redistribution, so raw files stay out of version control and only code and aggregate results are published. Development: 2014, 2015, 2016 vintages, 147,616 loans with complete windows (0.56% default rate), split 118,092 fit / 29,524 holdout (stratified). Out-of-time test: 2017 vintage, 49,891 loans (0.80%). Stress test: 2019 vintage, 49,970 loans.

## 3. Target
A loan is **bad** if, in months 0–23 after its first payment date, it is reported 90+ days delinquent, is acquired as REO, or ends with a default-type zero-balance code (third-party sale, short sale or charge-off, REO disposition, non-performing note sale). Prepayment before month 24 is good. Loans whose window was not fully observed are excluded. The last out-of-time window closes in 2022-04; 79 out-of-time loans have windows reaching 2020-03, when COVID payment relief began.

## 4. Features
Only application-time attributes of the borrower's credit, the loan terms and the collateral are used, so that every driver of a decision can be stated to the applicant as a reason.

| Feature | Field | Monotone constraint |
|---|---|---|
| Credit score | `credit_score` | higher value → lower risk |
| Debt-to-income ratio | `orig_dti` | higher value → higher risk |
| Loan-to-value ratio | `orig_ltv` | higher value → higher risk |
| Combined loan-to-value ratio | `orig_cltv` | higher value → higher risk |
| Loan term | `orig_loan_term` | higher value → higher risk |
| Number of units | `num_units` | no constraint |
| Loan purpose | `loan_purpose` | no constraint |
| Occupancy | `occupancy_status` | no constraint |
| Property type | `property_type` | no constraint |

Excluded on purpose: `property_state` (Geography; a redlining risk and not a reason a borrower can act on); `msa` (Geography; same concern as state); `postal_code` (Geography; same concern as state); `num_borrowers` (Proxy for marital status, a prohibited basis under ECOA); `first_time_homebuyer` (Correlates with applicant age, a prohibited basis under ECOA); `orig_interest_rate` (Output of risk-based pricing rather than an applicant attribute; not a valid adverse-action reason); `mi_pct` (Set mechanically by LTV and term; duplicates LTV); `orig_upb` (Small-balance effect proxies neighborhood income; kept out pending a fair-lending review); `channel` (Lender origination channel, not an attribute of the applicant); `seller_name` (Lender identity, not an attribute of the applicant); `servicer_name` (Lender identity, not an attribute of the applicant); `property_valuation_method` (Populated only for recent vintages); `program_indicator` (Populated only for recent vintages); `vantage_score_4` (Added in Release 48 but not populated (9999) for any 2014-2025 sample loan). The scorecard dropped `num_units` (information value 0.0000 below 0.01); `property_type` (information value 0.0066 below 0.01).

## 5. Method
**Scorecard (baseline).** Quantile bins merged until each holds ≥5% of loans and the bad rate is monotone where a direction is known; weight-of-evidence encoding; L2 logistic regression refit until every coefficient has the expected sign; points scaled to 600 at 50:1 odds, 20 points to double the odds. **Gradient boosting.** LightGBM with the monotone constraints above, early stopping on holdout log-loss (272 trees), then Platt scaling on the holdout. **Champion rule.** Gradient boosting is preferred only if it beats the scorecard's out-of-time AUC by ≥0.005; observed gain 0.0024. **Cutoff.** The PD cutoff (0.94%) is set on the in-time holdout as the most approvals whose default rate stays at or below 0.28%, so the out-of-time sample is never used to set it.

## 6. Validation (out-of-time, 2017 vintage)
| Metric | Scorecard | Gradient boosting |
|---|---|---|
| AUC, in-time holdout | 0.811 | 0.815 |
| **AUC, out-of-time 2017** | 0.775 | 0.777 |
| KS, out-of-time | 0.429 | 0.446 |
| **Calibration error by decile** (weighted mean abs. gap) | 0.26% | 0.21% |
| Largest single-decile gap | 0.84% | 0.56% |
| Observed ÷ predicted default rate | 1.43 | 1.35 |
| Brier score | 0.00783 | 0.00782 |
| **Approval rate at a 0.28% default rate** | 61.8% | 68.1% |

Calibration error by decile ranks loans into ten equal groups by predicted PD and averages |observed − predicted| weighted by group size. Charts: `reports/figures/roc_oot.png`, `calibration_oot.png`, `approval_curve_oot.png`.

## 7. Stress test
The 2019 vintage's 24-month windows run into COVID-era forbearance. Payment-deferral, disaster-delinquency and borrower-assistance (forbearance, repayment, trial modification) flags separate relief from real default: a 90+ month counts as relief if any flag appears in that month or the 3 before it.

| 2019 vintage, champion model | Value |
|---|---|
| Loans with complete windows | 49,970 |
| Observed 90+ rate, all causes | 4.59% |
| Observed 90+ rate excluding relief (real default) | 0.30% |
| Share of 90+ explained by relief | 93.5% |
| Mean predicted PD | 0.49% |
| AUC vs. all 90+ / vs. real default | 0.702 / 0.678 |
| Observed ÷ predicted, all / real | 9.29 / 0.60 |
| **PSI of 2019 scores vs. training** | **0.010** (stable) |

## 8. Reason codes
Each decline gets up to 4 reasons, taken from the model's own attribution for that applicant (SHAP values for gradient boosting, points below each characteristic's maximum for the scorecard), ranked by size and worded with the applicant's actual value. Following CFPB Circular 2023-03, reasons are never mapped to the nearest generic checklist item. On out-of-time declines (8,308 loans, 16.7%): 100.0% received at least one reason, 4.0 on average. **Faithfulness check:** setting the #1 reason's factor to the typical approved value lowered the PD for 100.0% of sampled declines.

| Factor | #1 reason | Cited at all |
|---|---|---|
| Credit score | 99.2% | 100.0% |
| Debt-to-income ratio | 0.8% | 99.4% |
| Loan-to-value ratio | 0.0% | 95.9% |
| Combined loan-to-value ratio | 0.0% | 95.9% |
| Loan term | 0.0% | 4.4% |

## 9. Limitations
- **Funded loans only.** Rejected applicants are not in the data, so approval rates are relative to Freddie Mac purchases, and no reject inference was done.
- **Thin feature set.** No credit bureau attributes beyond the score, income, reserves or employment, so absolute accuracy is below a production underwriting model.
- **Early default only.** A 24-month window captures early payment problems, not lifetime loss.
- **Relief flags.** The relief split depends on how fully servicers reported forbearance, deferral and disaster codes.
- **Fair lending.** Known proxies are excluded, but the data has no protected-class fields, so disparate-impact testing (e.g. with BISG proxies) remains to be done before any lending use.
- **Model selection.** The champion is chosen on out-of-time AUC; a further later vintage gives the unbiased read.

## 10. Monitoring plan
For every new vintage, compute PSI of the score and the top 5 features (credit_score, orig_dti, orig_cltv, orig_ltv, orig_loan_term) against the development sample, using development deciles plus a missing bin. **Stable** below 0.1; **watch** from 0.1 (review drivers, compare feature mix); **alert** at 0.25 or above (`defaultwatch monitor --fail-on-alert` exits non-zero; model owner reviews within one cycle and decides on recalibration or redevelopment). Once a vintage's 24-month window closes, also compare AUC and observed ÷ predicted default with this document; an AUC drop of more than 0.03 or a ratio outside 0.75–1.33 triggers review.

| Vintage | Loans | Score PSI | Worst watched PSI | Status | Observed ÷ predicted, all / excl. relief | AUC |
|---|---|---|---|---|---|---|
| 2017 | 50,000 | 0.010 | 0.081 | stable | 1.43 / 0.91 | 0.775 |
| 2018 | 50,000 | 0.015 | 0.265 | alert | 4.74 / 0.90 | 0.722 |
| 2019 | 50,000 | 0.010 | 0.515 | alert | 9.29 / 0.60 | 0.702 |
| 2020 | 50,000 | 0.090 | 0.671 | alert | 4.29 / 0.59 | 0.722 |
| 2021 | 50,000 | 0.030 | 0.622 | alert | 2.15 / 0.77 | 0.765 |
| 2022 | 50,000 | 0.039 | 0.781 | alert | 3.29 / 1.85 | 0.771 |
| 2023 | 50,000 | 0.030 | 0.945 | alert | 3.21 / 2.10 | 0.782 |
| 2024 | 50,000 | 0.014 | 0.917 | alert | pending (53% of windows closed) |  |
| 2025 | 50,000 | 0.021 | 0.876 | alert | pending (10% of windows closed) |  |

Excluding relief, observed ÷ predicted is outside the 0.75–1.33 band for 2019 (0.60), 2020 (0.59), 2022 (1.85), 2023 (2.10), which triggers review.
