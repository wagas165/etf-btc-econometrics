"""Combine ETF aggregates, BTC windows, and macro events into one panel."""

from __future__ import annotations

import _project_paths  # noqa: F401  # adds repo root to sys.path
import pandas as pd

from config.config import CLEAN_DIR


def main() -> None:
    flows_path = CLEAN_DIR / "etf_flows_panel.csv"
    etf_panel_path = CLEAN_DIR / "etf_panel.csv"
    btc_path = CLEAN_DIR / "btc_windows.csv"
    macro_path = CLEAN_DIR / "macro_events.csv"

    for path in [flows_path, etf_panel_path, btc_path, macro_path]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    flows = pd.read_csv(flows_path, parse_dates=["date"])
    etf_panel = pd.read_csv(etf_panel_path, parse_dates=["date"])
    btc = pd.read_csv(btc_path, parse_dates=["date"])
    macro = pd.read_csv(macro_path, parse_dates=["date"])

    # ``etf_panel`` can already include ``net_flow_usd`` from prior steps; keep the
    # version from ``flows`` so downstream aggregations can reference the column
    # directly without suffixes.
    daily_funds = flows.merge(
        etf_panel.drop(columns=["net_flow_usd"], errors="ignore"),
        on=["date", "ticker"],
        how="left",
    )

    agg = (
        daily_funds.groupby("date").agg(
            flow_agg_usd_mn=("net_flow_usd", "sum"),
            aum_total=("aum_usd", "sum") if "aum_usd" in daily_funds.columns else ("net_flow_usd", "sum"),
            dollar_volume_total=("dollar_volume", "sum"),
        )
    ).reset_index()

    # Flows are reported in millions of USD; scale to raw dollars for ratios
    agg["flow_agg_usd"] = agg["flow_agg_usd_mn"] * 1_000_000

    agg["aum_total_lag"] = agg["aum_total"].shift(1)
    valid_denom = agg["aum_total_lag"].where(agg["aum_total_lag"] > 0)
    agg["FlowShock"] = agg["flow_agg_usd"] / valid_denom
    agg["Turnover"] = agg["dollar_volume_total"] / valid_denom

    if "premium" in daily_funds.columns and "aum_usd" in daily_funds.columns:
        prem = (
            daily_funds.dropna(subset=["premium", "aum_usd"])
            .assign(weight=lambda d: d["aum_usd"] / d.groupby("date")["aum_usd"].transform("sum"))
            .assign(weighted_prem=lambda d: d["premium"] * d["weight"])
            .groupby("date")["weighted_prem"]
            .sum()
            .rename("Premium_agg")
            .reset_index()
        )
        agg = agg.merge(prem, on="date", how="left")

    master = agg.merge(btc, on="date", how="left").merge(macro, on="date", how="left")

    out_path = CLEAN_DIR / "master_panel.csv"
    master.to_csv(out_path, index=False)
    print(f"Saved master panel with {len(master):,} rows to {out_path}")


if __name__ == "__main__":
    main()
