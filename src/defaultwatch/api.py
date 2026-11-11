"""FastAPI scoring service.

The model is loaded from MODEL_PATH (a saved bundle, default models/defaultwatch_bundle.joblib)
or, if MODEL_URI is set, from MLflow (e.g. models:/workspace.default.defaultwatch_pd@champion).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .bundle import ModelBundle

PURPOSE = {"purchase": "P", "cash_out_refinance": "C", "no_cash_out_refinance": "N"}
OCCUPANCY = {"primary_residence": "P", "second_home": "S", "investment": "I"}
PROPERTY = {"single_family": "SF", "condo": "CO", "pud": "PU", "manufactured_home": "MH", "co_op": "CP"}


class LoanApplication(BaseModel):
    credit_score: int | None = Field(None, ge=300, le=850, description="Credit score; null if unavailable")
    dti: float | None = Field(None, ge=0, le=65, description="Debt-to-income ratio, percent; null if unavailable")
    ltv: float = Field(..., gt=0, le=105, description="Loan-to-value ratio, percent")
    cltv: float | None = Field(None, gt=0, le=200, description="Combined LTV incl. other liens, percent; defaults to ltv")
    loan_term_months: int = Field(360, ge=60, le=480)
    units: int = Field(1, ge=1, le=4)
    loan_purpose: Literal["purchase", "cash_out_refinance", "no_cash_out_refinance"] = "purchase"
    occupancy: Literal["primary_residence", "second_home", "investment"] = "primary_residence"
    property_type: Literal["single_family", "condo", "pud", "manufactured_home", "co_op"] = "single_family"

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "credit_score": 664, "dti": 46, "ltv": 95, "cltv": 95, "loan_term_months": 360, "units": 1,
                "loan_purpose": "cash_out_refinance", "occupancy": "primary_residence", "property_type": "condo",
            }]
        }
    }

    def to_raw(self) -> dict:
        return {
            "credit_score": self.credit_score,
            "orig_dti": self.dti,
            "orig_ltv": self.ltv,
            "orig_cltv": self.cltv if self.cltv is not None else self.ltv,
            "orig_loan_term": self.loan_term_months,
            "num_units": self.units,
            "loan_purpose": PURPOSE[self.loan_purpose],
            "occupancy_status": OCCUPANCY[self.occupancy],
            "property_type": PROPERTY[self.property_type],
        }


class Reason(BaseModel):
    rank: int
    code: str
    feature: str
    value: float | str | None
    contribution: float
    reason: str
    reg_b_category: str


class ScoreResponse(BaseModel):
    pd: float = Field(description="Probability of 90+ days delinquency within 24 months of first payment")
    score: int = Field(description="Points scale; higher = lower risk")
    decision: Literal["approve", "decline"]
    cutoff_pd: float
    reasons: list[Reason] = Field(description="Principal reasons that raised risk, largest first")
    model_type: str
    synthetic_model: bool


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    uri = os.getenv("MODEL_URI")
    if uri:
        import mlflow.pyfunc

        return mlflow.pyfunc.load_model(uri).unwrap_python_model().bundle
    path = Path(os.getenv("MODEL_PATH", "models/defaultwatch_bundle.joblib"))
    if not path.exists():
        raise FileNotFoundError(f"No model bundle at {path}. Run `defaultwatch train` first or set MODEL_URI.")
    return ModelBundle.load(path)


def _bundle() -> ModelBundle:
    try:
        return get_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _score(apps: list[LoanApplication]) -> list[ScoreResponse]:
    bundle = _bundle()
    frame = bundle.score_frame(pd.DataFrame([a.to_raw() for a in apps]))
    synthetic = bool(bundle.metadata.get("synthetic"))
    return [
        ScoreResponse(pd=round(float(r.pd), 6), score=int(r.score), decision=r.decision,
                      cutoff_pd=round(bundle.pd_cutoff, 6), reasons=r.reasons, model_type=bundle.model_type,
                      synthetic_model=synthetic)
        for r in frame.itertuples()
    ]


app = FastAPI(
    title="DefaultWatch",
    version=__version__,
    description="Mortgage default probability with specific principal reasons for high-risk scores.",
)


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
    except FileNotFoundError as exc:
        return {"status": "no_model", "detail": str(exc)}
    return {"status": "ok", "model_type": bundle.model_type, "version": __version__}


@app.get("/model")
def model_info() -> dict:
    bundle = _bundle()
    return {
        "model_type": bundle.model_type,
        "features": bundle.spec.features,
        "pd_cutoff": bundle.pd_cutoff,
        "monitoring_top_features": bundle.baseline.top_features,
        **bundle.metadata,
    }


@app.post("/score", response_model=ScoreResponse)
def score(app_in: LoanApplication) -> ScoreResponse:
    return _score([app_in])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(req: BatchRequest) -> list[ScoreResponse]:
    return _score(req.applications)
