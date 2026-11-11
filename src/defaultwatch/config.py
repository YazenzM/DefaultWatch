"""Configuration loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the YAML config, resolve relative paths against the repo root, apply overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _deep_update(cfg, copy.deepcopy(overrides or {}))
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else (REPO_ROOT / p).resolve())
    return cfg


def feature_list(cfg: dict) -> list[str]:
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])
