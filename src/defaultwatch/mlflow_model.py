"""MLflow pyfunc wrapper so the registered model returns PD, score, decision and reasons."""

from __future__ import annotations

import json

import mlflow.pyfunc
import pandas as pd


class DefaultWatchPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from defaultwatch.bundle import ModelBundle

        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input).reset_index(drop=True)
        out = self.bundle.score_frame(frame)
        out["reasons"] = out["reasons"].map(json.dumps)
        return out.reset_index(drop=True)
