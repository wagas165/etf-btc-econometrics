from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import _project_paths  # noqa: F401  # adds repo root to sys.path
from config.config import CLEAN_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs"
H = 10


def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_calendar() -> pd.DataFrame:
    calendar_path = CLEAN_DIR / "etf_panel.csv"
    calendar = (
        pd.read_csv(calendar_path, parse_dates=["date"])[["date"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return calendar


def load_master() -> pd.DataFrame:
    master_path = CLEAN_DIR / "master_panel.csv"
    return pd.read_csv(master_path, parse_dates=["date"])


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["FlowShock", "Turnover", "ret_close_close", "ret_intraday", "ret_overnight"]:
        out[f"missing_{col}"] = out[col].isna()
    out["missing_ret_close_close_t1"] = out["ret_close_close"].shift(-1).isna()
    out["missing_flowshock_t1"] = out["FlowShock"].shift(-1).isna()
    return out


def build_step_table(df: pd.DataFrame) -> pd.DataFrame:
    steps: list[dict[str, int | str]] = []
    mask = pd.Series(True, index=df.index)

    def record(step: str) -> None:
        steps.append(
            {
                "step": step,
                "remaining_obs": int(mask.sum()),
            }
        )

    record("ETF calendar (all trading days)")

    # Step 1: restrict to rows with master panel content
    mask &= df["flow_agg_usd_mn"].notna() | df["FlowShock"].notna()
    record("Merge master panel")

    # Step 2: require flow-based instruments
    mask &= df["FlowShock"].notna() & df["Turnover"].notna()
    record("FlowShock and Turnover observed")

    # Step 3: BTC returns for windows used in LP-IV controls
    mask &= df[["ret_close_close", "ret_intraday", "ret_overnight"]].notna().all(axis=1)
    record("BTC return windows observed")

    # Step 4: realized volatility controls
    mask &= df[["rv_intraday", "rv_overnight"]].notna().all(axis=1)
    record("Realized volatility controls observed")

    # Step 5: instrumentation weight (premium) present
    if "Premium_agg" in df:
        mask &= df["Premium_agg"].notna()
        record("Premium aggregation available")

    # Step 6: require t+1 FlowShock for lead instrument and t+1+H return for LP-IV horizons
    mask &= df["FlowShock"].shift(-1).notna()
    mask &= df["ret_close_close"].shift(-(1 + H)).notna()
    record("Lead FlowShock and return available for LP-IV (h<=10)")

    table = pd.DataFrame(steps)
    table["dropped_this_step"] = table["remaining_obs"].shift(1, fill_value=len(df)) - table["remaining_obs"]
    return table


def used_vs_dropped_summary(df: pd.DataFrame, sample_mask: pd.Series) -> pd.DataFrame:
    groups = []
    vars_to_compare = {
        "flows_usd_mn": "flow_agg_usd_mn",
        "aum_total": "aum_total",
        "dollar_volume_total": "dollar_volume_total",
        "ret_close_close": "ret_close_close",
        "ret_intraday": "ret_intraday",
        "ret_overnight": "ret_overnight",
        "rv_intraday": "rv_intraday",
        "rv_overnight": "rv_overnight",
    }

    for label, mask in [("used", sample_mask), ("dropped", ~sample_mask)]:
        subset = df.loc[mask]
        row: dict[str, int | float | str] = {"group": label, "n_obs": int(len(subset))}
        for pretty, col in vars_to_compare.items():
            nonmissing = subset[col].dropna()
            row[f"{pretty}_nonmissing"] = int(nonmissing.shape[0])
            row[f"{pretty}_mean"] = float(nonmissing.mean()) if not nonmissing.empty else np.nan
        groups.append(row)

    return pd.DataFrame(groups)


def main() -> None:
    ensure_outdir()
    calendar = load_calendar()
    master = load_master()
    merged = add_missing_flags(calendar.merge(master, on="date", how="left"))

    step_table = build_step_table(merged)
    sample_mask = (
        merged["FlowShock"].notna()
        & merged["Turnover"].notna()
        & merged[["ret_close_close", "ret_intraday", "ret_overnight", "rv_intraday", "rv_overnight", "Premium_agg"]].notna().all(axis=1)
        & merged["FlowShock"].shift(-1).notna()
        & merged["ret_close_close"].shift(-(1 + H)).notna()
    )

    comparison = used_vs_dropped_summary(merged, sample_mask)

    step_table.to_csv(OUT_DIR / "sample_construction_table.csv", index=False)
    comparison.to_csv(OUT_DIR / "missingness_comparison.csv", index=False)

    print("Saved sample construction table to", OUT_DIR / "sample_construction_table.csv")
    print("Saved missingness comparison to", OUT_DIR / "missingness_comparison.csv")
    print("Final LP-IV sample size:", int(sample_mask.sum()))


if __name__ == "__main__":
    main()
