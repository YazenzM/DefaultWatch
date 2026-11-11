# Databricks notebook source
# MAGIC %md
# MAGIC # DefaultWatch on Databricks Free Edition
# MAGIC
# MAGIC Trains the scorecard and gradient-boosting models, logs the run to the workspace MLflow experiment
# MAGIC `/Shared/defaultwatch`, and registers the champion in Unity Catalog as
# MAGIC `workspace.default.defaultwatch_pd` with the alias `champion`.
# MAGIC
# MAGIC **Before running**
# MAGIC 1. Clone this repository as a Git folder (*Workspace → Create → Git folder*).
# MAGIC 2. Run the first cells once to create the volume, then upload the Freddie Mac `sample_YYYY.zip`
# MAGIC    files to `/Volumes/workspace/default/freddie/raw/` (*Catalog → workspace → default → freddie → Upload*).
# MAGIC 3. Run all.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0 pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "freddie")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
root = f"/Volumes/{catalog}/{schema}/{volume}"
dbutils.fs.mkdirs(f"{root}/raw")
display(dbutils.fs.ls(f"{root}/raw"))

# COMMAND ----------

import logging
import os
import sys

repo_root = os.path.dirname(os.getcwd())          # the notebook runs from <repo>/notebooks
sys.path.insert(0, os.path.join(repo_root, "src"))
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

from defaultwatch.config import load_config

cfg = load_config(overrides={
    "paths": {
        "raw_dir": f"{root}/raw",
        "processed_dir": f"{root}/processed",
        "reports_dir": f"{root}/reports",
        "model_dir": f"{root}/models",
        "docs_dir": f"{root}/docs",
    },
    "mlflow": {"registered_model_uc": f"{catalog}.{schema}.defaultwatch_pd"},
})

# COMMAND ----------

# MAGIC %md ## 1. Build the loan-level table (24-month target, relief split)

# COMMAND ----------

from defaultwatch.ingest import prepare

written = prepare(cfg)
assert written, f"No Freddie Mac files found in {root}/raw"
written

# COMMAND ----------

# MAGIC %md ## 2. Train, validate out-of-time, stress-test 2019, register the champion

# COMMAND ----------

import json

from defaultwatch.pipeline import train

metrics = train(cfg, use_mlflow=True)
print(json.dumps({"champion": metrics["champion"], **metrics["headline"], "mlflow": metrics.get("mlflow")}, indent=2))

# COMMAND ----------

# MAGIC %md ## 3. Monitoring: PSI of the score and top features for each new vintage

# COMMAND ----------

import pandas as pd

display(pd.read_csv(f"{root}/reports/monitoring/psi_summary.csv"))

# COMMAND ----------

# MAGIC %md ## 4. Load the registered champion and score two applications

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.load_model(f"models:/{cfg['mlflow']['registered_model_uc']}@champion")
example = pd.DataFrame([
    {"credit_score": 640.0, "orig_dti": 47.0, "orig_ltv": 95.0, "orig_cltv": 95.0, "orig_loan_term": 360.0,
     "num_units": 1.0, "loan_purpose": "C", "occupancy_status": "I", "property_type": "CO"},
    {"credit_score": 790.0, "orig_dti": 25.0, "orig_ltv": 60.0, "orig_cltv": 60.0, "orig_loan_term": 180.0,
     "num_units": 1.0, "loan_purpose": "P", "occupancy_status": "P", "property_type": "SF"},
]).astype({"loan_purpose": object, "occupancy_status": object, "property_type": object})
display(model.predict(example))
