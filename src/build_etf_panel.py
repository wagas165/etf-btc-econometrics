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

    # Compute premium whenever NAV and close are available (issuer or proxy).
    if {"nav_per_share", "close_price"}.issubset(panel.columns):
        nav_mask = panel["nav_per_share"].notna() & (panel["nav_per_share"] != 0)
        panel.loc[nav_mask, "premium"] = (
            panel.loc[nav_mask, "close_price"] - panel.loc[nav_mask, "nav_per_share"]
        ) / panel.loc[nav_mask, "nav_per_share"]

        # If NAV proxies equal closes (common with yfinance), fall back to a
        # lagged-NAV premium so the aggregate is not degenerate.
        if panel["premium"].dropna().nunique() <= 1:
            nav_lag = panel.groupby("ticker")["nav_per_share"].shift(1)
            prem_lag_mask = nav_lag.notna() & (nav_lag != 0)
            panel.loc[prem_lag_mask, "premium"] = (
                panel.loc[prem_lag_mask, "close_price"] - nav_lag[prem_lag_mask]
            ) / nav_lag[prem_lag_mask]

    # Recover missing shares from AUM/NAV when possible, then forward/backward-fill
    # to avoid zero-denominator FlowShock calculations upstream.
    if {"nav_per_share", "aum_usd"}.issubset(panel.columns):
        mask = (
            panel["shares_outstanding"].isna()
            & panel["aum_usd"].notna()
            & panel["nav_per_share"].notna()
            & (panel["nav_per_share"] != 0)
        )
        panel.loc[mask, "shares_outstanding"] = panel.loc[mask, "aum_usd"] / panel.loc[mask, "nav_per_share"]

    # Estimate share counts from cumulative creations/redemptions when direct
    # share data are missing (most ETFs).
    if {"net_flow_usd", "nav_per_share"}.issubset(panel.columns):
        panel["shares_created_est"] = (panel["net_flow_usd"] * 1_000_000) / panel["nav_per_share"]
        flow_cum = panel.groupby("ticker")["shares_created_est"].cumsum().fillna(0)
        baseline = panel.groupby("ticker")["shares_outstanding"].transform(
            lambda s: s.dropna().iloc[0] if s.notna().any() else 0
        )

        # Offset the cumulative flow so the first observed shares point anchors
        # the creation/redemption total, avoiding double-counting earlier flows.
        first_share_flow = panel.groupby("ticker").apply(
            lambda g: flow_cum.loc[g.index[g["shares_outstanding"].notna()][0]]
            if g["shares_outstanding"].notna().any()
            else 0
        )
        flow_cum_offset = panel["ticker"].map(first_share_flow)

        first_share_date = panel.groupby("ticker").apply(
            lambda g: g.loc[g["shares_outstanding"].notna(), "date"].iloc[0]
            if g["shares_outstanding"].notna().any()
            else pd.NaT
        )
        first_share_date_map = panel["ticker"].map(first_share_date)

        flow_cum_adj = flow_cum - flow_cum_offset
        flow_cum_adj = flow_cum_adj.where(
            panel["date"].ge(first_share_date_map) | first_share_date_map.isna(), 0
        )

        panel["shares_outstanding_est"] = panel["shares_outstanding"].fillna(
            baseline + flow_cum_adj
        )

    if "shares_outstanding_est" in panel.columns:
        panel["shares_outstanding"] = panel["shares_outstanding"].fillna(panel["shares_outstanding_est"])

    if "shares_outstanding" in panel.columns:
        panel["shares_outstanding"] = (
            panel.groupby("ticker")["shares_outstanding"].transform(lambda s: s.ffill().bfill())
        )

    if "nav_per_share" in panel.columns and "shares_outstanding" in panel.columns:
        panel["aum_usd_check"] = panel["nav_per_share"] * panel["shares_outstanding"]

    if "aum_usd" not in panel.columns:
        panel["aum_usd"] = np.nan

    if {"nav_per_share", "shares_outstanding"}.issubset(panel.columns):
        mask = panel["aum_usd"].isna() & panel["nav_per_share"].notna()
        panel.loc[mask, "aum_usd"] = panel.loc[mask, "nav_per_share"] * panel.loc[mask, "shares_outstanding"]

    if "aum_usd" in panel.columns:
        panel["aum_usd"] = panel.groupby("ticker")["aum_usd"].transform(lambda s: s.ffill().bfill())

    out_path = CLEAN_DIR / "etf_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_values(["date", "ticker"], inplace=True)
    panel.to_csv(out_path, index=False)
    print(f"Saved ETF panel with {len(panel):,} rows to {out_path}")


if __name__ == "__main__":
    main()
