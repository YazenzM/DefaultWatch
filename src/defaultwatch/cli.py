"""Command line: defaultwatch {synth,prepare,train,monitor,report,publish,serve,all,dashboard}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "urllib3", "alembic", "matplotlib", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def free_port(preferred: int, attempts: int = 50) -> int:
    """The preferred port if nothing is listening on it, otherwise the next free one."""
    import socket

    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port between {preferred} and {preferred + attempts - 1}")


def build_demo(rebuild: bool = False, loans: int = 8000, base: Path | None = None,
               make_plots: bool = True) -> Path:
    """Synthetic data -> prepare -> train under demo/ (or `base`), so the dashboard has something to show."""
    import yaml

    from .config import REPO_ROOT
    from .ingest import prepare
    from .pipeline import train
    from .synth import generate

    base = Path(base) if base else REPO_ROOT / "demo"
    cfg_path = base / "config.yaml"
    names = {"raw_dir": "raw", "processed_dir": "processed", "reports_dir": "reports", "model_dir": "models",
             "docs_dir": "docs"}
    cfg = load_config(overrides={"paths": {k: str(base / v) for k, v in names.items()}})
    if rebuild or not (Path(cfg["paths"]["reports_dir"]) / "metrics.json").exists():
        logging.getLogger(__name__).info("building SYNTHETIC demo outputs under %s", base)
        generate(cfg["paths"]["raw_dir"], loans_per_vintage=loans)
        prepare(cfg)
        train(cfg, use_mlflow=False, make_plots=make_plots)
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


# Aggregate outputs only: deciles, curve points, score bins, PSI values. No loan-level rows, no fitted model.
PUBLISHED_REPORTS = ["metrics.json", "results.md", "roc_oot.csv", "calibration_oot_*.csv", "approval_curve_oot.csv",
                     "outcome_breakdown.csv", "reason_frequency_oot.csv", "scorecard_points.csv",
                     "monitoring/psi_summary.csv", "monitoring/psi_*.json", "figures/*.png"]


def publish_results(cfg: dict, dest: str | Path | None = None, allow_synthetic: bool = False) -> Path:
    """Copy the aggregate results of a Freddie Mac run to results/, which the hosted demo shows.

    The Freddie Mac terms allow publishing results, not the data. Only the files in PUBLISHED_REPORTS are
    copied; raw files, loan-level parquet and the fitted model stay local (see .gitignore).
    """
    import shutil

    import yaml

    from .config import REPO_ROOT

    src = Path(cfg["paths"]["reports_dir"])
    metrics = src / "metrics.json"
    if not metrics.exists():
        raise FileNotFoundError(f"No metrics.json in {src}. Run `defaultwatch train` first.")
    if json.loads(metrics.read_text()).get("synthetic_data") and not allow_synthetic:
        raise ValueError("These outputs were built from SYNTHETIC data. Train on the Freddie Mac files first.")

    dest = Path(dest).resolve() if dest else REPO_ROOT / "results"
    reports = dest / "reports"
    shutil.rmtree(reports, ignore_errors=True)
    for pattern in PUBLISHED_REPORTS:
        for f in sorted(src.glob(pattern)):
            target = reports / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    def portable(p: Path) -> str:   # repo-relative where possible, so the config works on any machine
        p = Path(p).resolve()
        return p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else str(p)

    out_cfg = {**cfg, "paths": {k: portable(v) for k, v in cfg["paths"].items()}}
    out_cfg["paths"].update(reports_dir=portable(reports), model_dir=portable(dest / "models"))
    (dest / "config.yaml").write_text(yaml.safe_dump(out_cfg))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defaultwatch", description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write SYNTHETIC files in the Freddie Mac layout (for a dry run)")
    s.add_argument("--out", default=None, help="output folder (default: paths.raw_dir)")
    s.add_argument("--loans", type=int, default=5000, help="loans per vintage")
    s.add_argument("--seed", type=int, default=7)

    sub.add_parser("prepare", help="build loan-level parquet files with the 24-month target")

    t = sub.add_parser("train", help="fit scorecard + GBM, validate out-of-time, stress test, save and register")
    t.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking and registration")
    t.add_argument("--no-plots", action="store_true")

    m = sub.add_parser("monitor", help="PSI of score and top features for new vintages")
    m.add_argument("--vintage", type=int, nargs="*", help="vintages to check (default: config monitor list)")
    m.add_argument("--fail-on-alert", action="store_true", help="exit with code 2 if any PSI alert fires")

    sub.add_parser("report", help="re-render docs/model_document.md from reports/metrics.json")

    p = sub.add_parser("publish", help="copy aggregate results (no loan-level data, no model) to results/ "
                                       "for the hosted demo")
    p.add_argument("--out", default=None, help="destination folder (default: results/)")

    v = sub.add_parser("serve", help="run the FastAPI scoring service")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    a = sub.add_parser("all", help="prepare + train")
    a.add_argument("--no-mlflow", action="store_true")

    d = sub.add_parser("dashboard", help="open the Streamlit dashboard on the latest outputs")
    d.add_argument("--demo", action="store_true", help="build SYNTHETIC outputs under demo/ (once) and show those")
    d.add_argument("--rebuild", action="store_true", help="with --demo: regenerate the demo outputs")
    d.add_argument("--port", type=int, default=8650, help="preferred port; the next free one is used if taken")
    d.add_argument("--host", default="localhost",
                   help="address to listen on (default: this machine only; 0.0.0.0 exposes it to the network)")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)

    if args.command == "synth":
        from .synth import generate

        out = generate(args.out or cfg["paths"]["raw_dir"], loans_per_vintage=args.loans, seed=args.seed)
        print(f"SYNTHETIC files written to {out}")
        return 0

    if args.command in ("prepare", "all"):
        from .ingest import prepare

        written = prepare(cfg)
        if not written:
            print(f"No Freddie Mac files found under {cfg['paths']['raw_dir']}. See README > Get the data.",
                  file=sys.stderr)
            return 1

    if args.command in ("train", "all"):
        from .pipeline import train

        metrics = train(cfg, use_mlflow=not args.no_mlflow, make_plots=not getattr(args, "no_plots", False))
        print(json.dumps({"champion": metrics["champion"], **metrics["headline"]}, indent=2))
        return 0

    if args.command == "monitor":
        from .pipeline import monitor

        summary = monitor(cfg, vintages=args.vintage)
        if summary.empty:
            print("No monitorable vintages found.")
            return 0
        print(summary[["vintage", "n", "score_psi", "max_watched_psi", "status", "alerts"]].to_string(index=False))
        if args.fail_on_alert and (summary["status"] == "alert").any():
            return 2
        return 0

    if args.command == "report":
        from .report import write_model_document

        print(write_model_document(cfg))
        return 0

    if args.command == "publish":
        try:
            out = publish_results(cfg, args.out)
        except (FileNotFoundError, ValueError) as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Aggregate results copied to {out}. Review them, then commit results/ and push; "
              "the hosted demo shows them instead of synthetic data.")
        return 0

    if args.command == "dashboard":
        import os
        import subprocess

        from .config import DEFAULT_CONFIG

        cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG
        if args.demo:
            cfg_path = build_demo(rebuild=args.rebuild)
        env = {**os.environ, "DEFAULTWATCH_CONFIG": str(Path(cfg_path).resolve())}
        script = Path(__file__).with_name("dashboard.py")
        port = free_port(args.port)
        if port != args.port:
            print(f"Port {args.port} is in use; using {port}.")
        print(f"DefaultWatch dashboard: http://localhost:{port}")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script),
                                "--server.port", str(port), "--server.address", args.host,
                                "--browser.gatherUsageStats", "false"], env=env)

    if args.command == "serve":
        import os

        import uvicorn

        os.environ.setdefault("MODEL_PATH", str(Path(cfg["paths"]["model_dir"]) / "defaultwatch_bundle.joblib"))
        uvicorn.run("defaultwatch.api:app", host=args.host, port=args.port)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
