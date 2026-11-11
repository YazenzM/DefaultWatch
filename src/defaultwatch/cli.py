"""Command line: defaultwatch {synth,prepare,train,monitor,report,serve,all}."""

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

    v = sub.add_parser("serve", help="run the FastAPI scoring service")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    a = sub.add_parser("all", help="prepare + train")
    a.add_argument("--no-mlflow", action="store_true")

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

    if args.command == "serve":
        import os

        import uvicorn

        os.environ.setdefault("MODEL_PATH", str(Path(cfg["paths"]["model_dir"]) / "defaultwatch_bundle.joblib"))
        uvicorn.run("defaultwatch.api:app", host=args.host, port=args.port)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
