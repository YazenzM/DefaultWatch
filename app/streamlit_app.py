"""Entry point for the hosted demo on Streamlit Community Cloud.

The Freddie Mac files cannot be redistributed and fitted models are kept out of git, so the hosted app
starts empty. On its first start it generates SYNTHETIC loans in the Freddie Mac layout, trains both
models on them and then shows the normal dashboard. On your own machine, use `defaultwatch dashboard`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # the package is used from source; no pip install needed

import streamlit as st  # noqa: E402

DEMO_DIR = Path(os.getenv("DEFAULTWATCH_DEMO_DIR") or Path(tempfile.gettempdir()) / "defaultwatch_demo")
LOANS_PER_VINTAGE = int(os.getenv("DEFAULTWATCH_DEMO_LOANS", "8000"))

st.set_page_config(page_title="DefaultWatch", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="First start: generating synthetic loans and training both models. "
                                "This takes about a minute and only happens once.")
def demo_config() -> str:
    from defaultwatch.cli import build_demo

    return str(build_demo(loans=LOANS_PER_VINTAGE, base=DEMO_DIR, make_plots=False))


os.environ["DEFAULTWATCH_CONFIG"] = demo_config()

from defaultwatch.dashboard import main  # noqa: E402

main()
