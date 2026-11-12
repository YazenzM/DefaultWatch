# DefaultWatch

A mortgage default model built the way bank risk teams need it: **calibrated, explainable and monitored**.
It predicts whether a new Freddie Mac loan goes 90+ days delinquent within 24 months of its first payment.
It gives specific, plain-language reasons for every high-risk score and watches each new vintage for drift.
Training is tracked and registered with MLflow on Databricks, and scores are served from a FastAPI container.

## Results

| Measure | Scorecard | Gradient boosting |
|---|---|---|
| Out-of-time AUC (2017) | _pending real-data run_ | |
| Calibration error by decile (2017) | | |
| Approval rate at a fixed default rate (2017) | | |
| PSI of 2019 vs. training scores | | |

Run the pipeline on the Freddie Mac files and paste `reports/results.md` here. The full write-up is generated
at [docs/model_document.md](docs/model_document.md).

## How it works

```
Freddie Mac sample files ──► prepare ──► train ─────────────────────────────► model bundle ──► FastAPI (Docker)
 (orig + monthly perf)        │           ├ WOE logistic scorecard (baseline)      │
                              │           ├ monotone LightGBM + Platt calibration  ├─► MLflow run + registry
 24-month target per loan ◄───┘           ├ out-of-time test on 2017               │    (Databricks / Unity Catalog)
 relief vs. real default                  ├ 2019 stress test                       │
                                          ├ reasons + faithfulness check           └─► monitor: PSI per vintage
                                          └ model document                              (exit 2 on alert)
```

**Target.** A loan is bad if, in months 0–23 after its first payment, it is reported 90+ days delinquent,
is acquired as REO, or ends with a default-type zero-balance code. Loans whose window was not fully observed are dropped.

**Out-of-time design.** The models are developed on the 2014–2016 vintages and tested on 2017. Every 2017 window
closes before COVID payment relief began in March 2020, and the pipeline checks this and records it.

**2019 stress test.** The 2019 vintage's windows run straight into forbearance. Payment-deferral,
disaster-delinquency and borrower-assistance flags separate relief from real default. A 90+ month counts as
relief if a flag appears in that month or the three before it. The model is then scored against both outcomes,
and the PSI of 2019 scores against training is reported.

**Features chosen for decisions, not just lift.** The model uses only application-time attributes that can be
stated to an applicant as a reason: credit score, DTI, LTV, CLTV, term, units, purpose, occupancy and property
type. Some fields are excluded on purpose, and [config/config.yaml](config/config.yaml) records why:
- geography (redlining risk)
- number of borrowers (marital-status proxy)
- first-time-buyer flag (age proxy)
- interest rate (an output of risk-based pricing)
- channel and seller (not about the applicant)

**Monotone where the direction is known.** Credit score can only lower risk; DTI, LTV, CLTV and term can only
raise it. The scorecard's bins and the GBM's trees both enforce this, so a reason never says "your credit score
is too high".

**Reason codes.** Regulation B requires the specific principal reasons for a denial. CFPB Circular 2023-03 says
a lender using a complex model cannot fall back on generic checklist reasons. Each reason here comes from the
model's own attribution for that applicant: SHAP for the GBM, points-below-maximum for the scorecard. Reasons are
ranked and worded with the applicant's actual values, for example:

> Monthly debt payments are 47% of income, above the typical approved borrower's 33%.

A faithfulness check sets each declined loan's #1 reason to the typical approved value and confirms the PD drops.

**Decision cutoff.** The cutoff is set on the in-time holdout as the most approvals whose default rate stays
at or below a target. It is never set on the out-of-time sample. The headline "approval rate at a fixed default
rate" compares both models at that same target on 2017.

## Quickstart: dry run on synthetic data

The Freddie Mac files need a free registration, so the repo includes a generator that writes synthetic files in
the exact same layout. Use it to check the setup before downloading. Every output built from it is stamped
**SYNTHETIC**.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

defaultwatch synth --out data/raw --loans 5000          # synthetic files + SYNTHETIC_DATA.txt marker
defaultwatch all --no-mlflow                            # prepare + train
pytest                                                  # 20 tests
defaultwatch dashboard                                  # explore the outputs (or: dashboard --demo)
```

Delete `data/raw/*`, `data/processed/*`, `reports/` and `models/*.joblib` before the real run.

## Run on Freddie Mac data

1. Register on Freddie Mac's Clarity data site and download the **Single-Family Loan-Level Dataset sample
   files** for 2014–2017 and 2019 (add 2018 and 2020+ to monitor more vintages). See [data/raw/README.md](data/raw/README.md).
2. Drop the zips into `data/raw/`. The licence forbids redistribution, and `.gitignore` keeps raw data,
   loan-level parquet and fitted models out of git.
3. Run:

```bash
defaultwatch prepare            # loan-level table with the 24-month target and relief split
defaultwatch train              # both models, OOT test, stress test, reasons, monitoring, MLflow run + registry
defaultwatch monitor --fail-on-alert
```

`train` writes the following:
- `reports/metrics.json`: every metric
- `reports/results.md`: the table above
- CSV tables behind every chart (`calibration_oot_*.csv`, `scorecard_points.csv`, `reason_frequency_oot.csv`, `monitoring/psi_summary.csv`)
- `reports/figures/*.png`
- `docs/model_document.md`

Without Databricks credentials, MLflow logs to a local `mlflow.db`. Browse it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Databricks Free Edition

Free Edition replaced Community Edition in 2025 and is where the MLflow docs send new users.

1. Clone this repo as a Git folder in your workspace.
2. Open [notebooks/databricks_train.py](notebooks/databricks_train.py) and run the first cells. They create the
   volume `workspace.default.freddie`.
3. Upload the `sample_YYYY.zip` files to `/Volumes/workspace/default/freddie/raw/`, then run all.

Runs go to the experiment `/Shared/defaultwatch`. The champion is registered in Unity Catalog as
`workspace.default.defaultwatch_pd` with the alias `champion`. The registered pyfunc returns PD, score, decision
and reasons, so `mlflow.pyfunc.load_model("models:/workspace.default.defaultwatch_pd@champion")` is all a
consumer needs.

To log from your own machine instead, set `MLFLOW_TRACKING_URI=databricks`, `MLFLOW_REGISTRY_URI=databricks-uc`,
`DATABRICKS_HOST` and `DATABRICKS_TOKEN`, then run `defaultwatch train`.

## Scoring API (Docker)

```bash
docker build -t defaultwatch-api .          # copies models/defaultwatch_bundle.joblib into the image
docker run --rm -p 8000:8000 defaultwatch-api
```

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "credit_score": 664, "dti": 46, "ltv": 95, "loan_purpose": "cash_out_refinance",
  "occupancy": "primary_residence", "property_type": "condo"}'
```

The response carries `pd`, `score` (higher = safer), `decision` against the stored cutoff, and up to four
`reasons`. Each reason has a code, factor, value, contribution, plain-language text and the Regulation B
category. `POST /score/batch` takes up to 1,000 applications, `GET /model` shows the cutoff and features, and
interactive docs are at `/docs`. To serve straight from the registry instead of the baked-in bundle, set
`MODEL_URI=models:/...@champion` and install `mlflow`.

The image pins the same library versions as training ([requirements-serve.txt](requirements-serve.txt)),
because a pickled model must be loaded with the versions it was saved with.

## Monitoring

`defaultwatch monitor` scores every vintage after training and compares it with the 2014–2016 development
sample. It reports PSI on the score and the top five features, using development deciles plus a missing bin.
A vintage only needs its origination file, so monitoring works long before outcomes arrive.

| PSI | Status | Action |
|---|---|---|
| < 0.10 | stable | none |
| 0.10–0.25 | watch | review drivers |
| ≥ 0.25 | alert | `--fail-on-alert` exits 2 for a scheduler; review for recalibration or redevelopment |

Once a vintage's 24-month window closes, its AUC and observed ÷ predicted default are added to the same report.

## Dashboard

```bash
defaultwatch dashboard          # the outputs of your last `defaultwatch train`
defaultwatch dashboard --demo   # builds SYNTHETIC outputs under demo/ first, then opens them
```

This opens a Streamlit app at http://localhost:8650, or the next free port if that one is taken
(`--port` picks another). It reads `reports/` and the saved model and has six tabs:

| Tab | What it shows |
|---|---|
| Overview | the four headline measures and a scorecard vs. gradient boosting comparison |
| Validation | out-of-time ROC, calibration by decile, approval rate vs. default rate |
| Stress test | 2019 predicted vs. observed default, with relief separated from real default |
| Monitoring | score PSI by vintage, a feature-by-vintage PSI heatmap, and alerts |
| Reason codes | how often each factor is cited, faithfulness, and scorecard points by bin |
| Score a loan | live PD, decision and principal reasons for an application you enter, plus a what-if curve |

Charts are interactive and adapt to light or dark mode, and each has a table view. Outputs built from
synthetic data carry a warning banner.

### Hosted demo (Streamlit Community Cloud)

[app/streamlit_app.py](app/streamlit_app.py) runs the same dashboard as a public demo. The Freddie Mac
files can't be redistributed and fitted models aren't in git, so the hosted app builds its own outputs on first
start. It generates synthetic loans, trains both models (about a minute) and keeps them until the app restarts.
The demo always shows synthetic results.

To deploy, sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose
**Create app → Deploy a public app from GitHub**:
- **Repository and branch:** this repo, `main`
- **Main file path:** `app/streamlit_app.py`
- **App URL:** pick your own name
- **Python:** 3.12, under **Advanced settings**

The cloud installs [app/requirements.txt](app/requirements.txt), a light list without MLflow or the test
tools, plus `libgomp1` from [packages.txt](packages.txt) for LightGBM.

## Repo layout

```
config/config.yaml          vintages, target, features (+ exclusion reasons), model and monitoring settings
src/defaultwatch/
  ingest.py                 file discovery (zip or txt), streaming reader, 24-month target, relief split
  features.py               sentinels -> missing, categorical clean-up
  scorecard.py              monotone WOE binning, logistic regression, points, points-below-max reasons
  gbm.py                    monotone LightGBM, Platt calibration, SHAP contributions
  reasons.py                plain-language reason text and Regulation B categories
  evaluate.py               AUC, KS, calibration by decile, Hosmer-Lemeshow, approval at fixed default rate
  monitor.py                PSI baselines and status
  pipeline.py               train / stress test / monitor orchestration
  tracking.py, mlflow_model.py   MLflow logging, pyfunc wrapper, Unity Catalog registration
  api.py                    FastAPI service
  dashboard.py              Streamlit dashboard (validation, stress test, monitoring, reasons, live scoring)
  report.py, plots.py       model document, results table, charts
  synth.py                  synthetic files in the Freddie Mac layout (tests and dry runs only)
app/streamlit_app.py        hosted demo entry point (Streamlit Community Cloud); app/requirements.txt
notebooks/databricks_train.py   Databricks Free Edition runbook
tests/                      target edge cases, monotonicity, SHAP additivity, PSI, pipeline, API, dashboard
```

## Limitations

- The data holds funded loans only, so approval rates are relative and no reject inference was done.
- The feature set is thin: no bureau attributes beyond the score, and no income or reserves.
- A 24-month window measures early default, not lifetime loss.
- No disparate-impact testing has been done. The data has no protected-class fields; BISG would be the next step.

[docs/model_document.md](docs/model_document.md) covers these limitations in more detail.
