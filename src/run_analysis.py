#!/usr/bin/env python3
"""Orchestrate the ETF/BTC data refresh and econometric analysis."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_TASKS: list[tuple[str, Sequence[str]]] = [
    ("Download ETF flows", ["python", str(REPO_ROOT / "src" / "get_etf_flows.py")]),
    ("Download ETF prices", ["python", str(REPO_ROOT / "src" / "get_etf_prices.py")]),
    ("Normalize ETF NAV inputs", ["python", str(REPO_ROOT / "src" / "get_etf_nav.py")]),
    ("Assemble ETF panel", ["python", str(REPO_ROOT / "src" / "build_etf_panel.py")]),
    ("Download BTC intraday klines", ["python", str(REPO_ROOT / "src" / "get_btc_1m_binance.py")]),
    ("Build BTC windows", ["python", str(REPO_ROOT / "src" / "build_btc_windows.py")]),
    ("Add macro event dummies", ["python", str(REPO_ROOT / "src" / "build_macro.py")]),
    ("Combine into master panel", ["python", str(REPO_ROOT / "src" / "build_master_panel.py")]),
]

ANALYSIS_TASK = (
    "Generate econometric outputs",
    ["python", str(REPO_ROOT / "run_steps_3_6.py")],
)


def run_task(label: str, command: Sequence[str]) -> None:
    print(f"→ {label}")
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full data pipeline, the econometric analysis, or both. "
            "Defaults to running everything in order."
        )
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Refresh datasets without running the econometric analysis.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Assume data already exist and only produce analysis outputs.",
    )
    args = parser.parse_args()

    if args.data_only and args.analysis_only:
        raise SystemExit("Use at most one of --data-only or --analysis-only.")

    run_data = not args.analysis_only
    run_analysis = not args.data_only

    if run_data:
        print("Running data refresh...")
        for label, command in DATA_TASKS:
            run_task(label, command)

    if run_analysis:
        label, command = ANALYSIS_TASK
        print("Running analysis outputs...")
        run_task(label, command)


if __name__ == "__main__":
    main()
