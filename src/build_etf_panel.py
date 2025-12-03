"""Combine ETF flows, prices, and NAV into a daily panel."""

from __future__ import annotations

import _project_paths  # noqa: F401  # adds repo root to sys.path
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import CLEAN_DIR, RAW_DIR


def main() -> None:
    flows_path = CLEAN_DIR / "etf_flows_panel.csv"
    prices_path = RAW_DIR / "etf_prices" / "etf_prices_yahoo.csv"
    nav_path = CLEAN_DIR / "etf_nav_panel.csv"

    for path in [flows_path, prices_path]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = (
                pd.to_datetime(df["date"], utc=True, errors="coerce")
                .dt.tz_localize(None)
                .dt.normalize()
            )
        return df

    flows = _normalize_dates(pd.read_csv(flows_path))
    prices = _normalize_dates(pd.read_csv(prices_path))
    nav = _normalize_dates(pd.read_csv(nav_path)) if nav_path.exists() else pd.DataFrame()

    panel = flows.merge(prices, on=["date", "ticker"], how="left")
    if not nav.empty:
        panel = panel.merge(nav, on=["date", "ticker"], how="left")

    if "volume_shares" in panel.columns and "close_price" in panel.columns:
        panel["dollar_volume"] = panel["close_price"] * panel["volume_shares"]

    if "nav_per_share" in panel.columns and "close_price" in panel.columns:
        panel["premium"] = (panel["close_price"] - panel["nav_per_share"]) / panel["nav_per_share"]

    if "nav_per_share" in panel.columns and "shares_outstanding" in panel.columns:
        panel["aum_usd_check"] = panel["nav_per_share"] * panel["shares_outstanding"]

    if "aum_usd" not in panel.columns and {"close_price", "shares_outstanding"}.issubset(panel.columns):
        panel["aum_usd"] = panel["close_price"] * panel["shares_outstanding"]

    out_path = CLEAN_DIR / "etf_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_values(["date", "ticker"], inplace=True)
    panel.to_csv(out_path, index=False)
    print(f"Saved ETF panel with {len(panel):,} rows to {out_path}")


if __name__ == "__main__":
    main()
