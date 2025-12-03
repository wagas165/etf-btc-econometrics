#!/usr/bin/env python3
"""Orchestrate the analysis phases for the ETF-BTC econometrics project."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from analysis.common import OUT, ensure_outdir, load_clean, validate, write_json


PHASES = [
    ("Exploratory data analysis", "analysis.eda"),
    ("Baseline predictive models", "analysis.baseline_models"),
    ("Instrumental variable models", "analysis.iv_models"),
    ("Robustness checks", "analysis.robustness"),
]


def run_phase(description: str, module: str) -> None:
    logging.info("Starting phase: %s", description)
    cmd = [sys.executable, "-m", module]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_PATH), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    subprocess.run(cmd, check=True, env=env)
    logging.info("Completed phase: %s", description)


def write_report(validation: dict) -> None:
    lines = []
    lines.append("# Analysis report\n\n")
    lines.append("## Data validation\n")
    lines.append("```json\n" + json.dumps(validation, indent=2) + "\n```\n\n")

    baseline_summary = OUT / "baseline_static_ols_summary.csv"
    if baseline_summary.exists():
        lines.append("## Baseline: static OLS (compact)\n\n")
        comp_df = pd.read_csv(baseline_summary)
        lines.append(comp_df.to_markdown(index=False))
        lines.append("\n\n")

    eventstudy = OUT / "baseline_eventstudy_car_h125_thresholds.csv"
    if eventstudy.exists():
        lines.append("## Baseline: event study CAR (thresholds)\n\n")
        event_df = pd.read_csv(eventstudy)
        lines.append(event_df.to_markdown(index=False))
        lines.append("\n\n")

    instruments = OUT / "iv_instruments_used.json"
    if instruments.exists():
        lines.append("## IV: instruments used\n\n")
        lines.append("```json\n" + instruments.read_text(encoding="utf-8") + "\n```\n\n")

    lines.append("## Robustness\n\nSee robustness_report.md and the robustness CSV outputs for details.\n")

    (OUT / "analysis_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(message)s")
    ensure_outdir()

    df = load_clean()
    validation = validate(df)
    write_json(OUT / "data_validation.json", validation)

    for description, module in PHASES:
        run_phase(description, module)

    write_report(validation)
    logging.info("Analysis phases completed. Outputs in: %s", OUT)


if __name__ == "__main__":
    main()
