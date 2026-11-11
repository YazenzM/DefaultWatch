"""MLflow experiment tracking and model registration.

Where runs go:
  * Inside a Databricks notebook (Free Edition included): the workspace tracking
    server, with the model registered in Unity Catalog (catalog.schema.model).
  * Locally with MLFLOW_TRACKING_URI=databricks (plus DATABRICKS_HOST / DATABRICKS_TOKEN):
    the same, driven from your machine.
  * Otherwise: a local SQLite store (mlflow.db) with artifacts under mlruns/.
    Browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT

log = logging.getLogger(__name__)

RUNTIME_PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib", "pyyaml"]


def mlflow_targets(cfg: dict) -> dict:
    mcfg = cfg["mlflow"]
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    on_databricks = bool(os.getenv("DATABRICKS_RUNTIME_VERSION")) or env_uri.startswith("databricks")
    if on_databricks:
        experiment = mcfg["experiment"]
        if not experiment.startswith("/"):
            experiment = f"/Shared/{experiment}"
        return {
            "tracking_uri": env_uri or "databricks",
            "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"),
            "experiment": experiment,
            "model_name": mcfg["registered_model_uc"],
            "artifact_location": None,
            "on_databricks": True,
        }
    tracking = env_uri or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    return {
        "tracking_uri": tracking,
        "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", tracking),
        "experiment": mcfg["experiment"],
        "model_name": mcfg["registered_model_local"],
        "artifact_location": (REPO_ROOT / "mlruns").as_uri() if not env_uri else None,
        "on_databricks": False,
    }


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _pins() -> list[str]:
    pins = []
    for pkg in RUNTIME_PACKAGES:
        try:
            pins.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pins.append(pkg)
    return pins


def log_training_run(cfg: dict, bundle_path: Path, bundle, metrics: dict, params: dict,
                     reports_dir: Path, model_doc: Path | None, input_example: pd.DataFrame) -> dict:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    from .mlflow_model import DefaultWatchPyfunc

    t = mlflow_targets(cfg)
    mlflow.set_tracking_uri(t["tracking_uri"])
    mlflow.set_registry_uri(t["registry_uri"])
    if mlflow.get_experiment_by_name(t["experiment"]) is None:
        mlflow.create_experiment(t["experiment"], artifact_location=t["artifact_location"])
    mlflow.set_experiment(t["experiment"])

    example = input_example.copy()
    for c in example.columns:
        example[c] = example[c].astype(float) if c in bundle.spec.numeric else example[c].astype(object)
    output = bundle.score_frame(example)
    output["reasons"] = output["reasons"].map(json.dumps)
    signature = infer_signature(example, output.reset_index(drop=True))

    numeric_metrics = {k: float(v) for k, v in _flatten(metrics).items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
    with mlflow.start_run(run_name=f"train-{bundle.model_type}") as run:
        mlflow.set_tags({"champion": bundle.model_type, "synthetic_data": str(bundle.metadata.get("synthetic"))})
        flat_params = {k: str(v)[:500] for k, v in _flatten(params).items()}
        mlflow.log_params(flat_params)
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in numeric_metrics.items()})
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        if model_doc and Path(model_doc).exists():
            mlflow.log_artifact(str(model_doc), artifact_path="docs")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DefaultWatchPyfunc(),
            artifacts={"bundle": str(bundle_path)},
            code_paths=[str(Path(__file__).parent)],
            signature=signature,
            input_example=example,
            pip_requirements=_pins(),
            registered_model_name=t["model_name"],
        )
        version = getattr(info, "registered_model_version", None)
        if version is not None:
            MlflowClient().set_registered_model_alias(t["model_name"], cfg["mlflow"]["alias"], str(version))
        result = {
            "run_id": run.info.run_id,
            "experiment": t["experiment"],
            "tracking_uri": t["tracking_uri"],
            "model_uri": info.model_uri,
            "registered_model": t["model_name"],
            "registered_version": str(version) if version is not None else None,
            "alias": cfg["mlflow"]["alias"],
        }
    log.info("MLflow run %s logged; registered %s version %s", result["run_id"], t["model_name"], version)
    return result
