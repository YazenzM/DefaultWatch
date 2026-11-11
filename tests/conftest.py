from pathlib import Path

import pytest
import yaml

from defaultwatch.config import load_config


def make_config(base: Path) -> dict:
    paths = {k: str(base / k) for k in ["raw_dir", "processed_dir", "reports_dir", "model_dir", "docs_dir"]}
    return load_config(overrides={"paths": paths})


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Synthetic data -> prepare -> train once for the whole test session (no MLflow)."""
    from defaultwatch.ingest import prepare
    from defaultwatch.pipeline import train
    from defaultwatch.synth import generate

    base = tmp_path_factory.mktemp("dw")
    cfg = make_config(base)
    generate(cfg["paths"]["raw_dir"], loans_per_vintage=3000, seed=11)
    prepare(cfg)
    metrics = train(cfg, use_mlflow=False, make_plots=True)
    cfg_path = base / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg, "metrics": metrics, "base": base, "cfg_path": cfg_path}
