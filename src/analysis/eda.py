from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .common import OUT, ensure_outdir, load_clean


STATS_VARS = [
    "flow_agg_usd",
    "aum_total",
    "aum_total_lag",
    "dollar_volume_total",
    "FlowShock_frac",
    "Turnover",
    "Premium_agg",
    "ret_close_close",
    "ret_intraday",
    "ret_overnight",
    "rv_intraday",
    "rv_overnight",
    "is_fomc",
    "is_cpi",
]


def run_eda(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["FlowShock_frac", "btc_close_et"]).copy().reset_index(drop=True)

    d[STATS_VARS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(OUT / "eda_summary_stats.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], d["FlowShock_frac"])
    plt.title("FlowShock (fraction)")
    plt.xlabel("Date")
    plt.ylabel("FlowShock")
    plt.tight_layout()
    plt.savefig(OUT / "flowshock_fraction_timeseries.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], d["btc_close_et"])
    plt.title("BTC Close Price (ET close proxy)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(OUT / "btc_price_timeseries.png", dpi=160)
    plt.close()

    tmp = d.copy()
    tmp["ret_cc_t1"] = tmp["ret_close_close"].shift(-1)
    tmp = tmp.dropna(subset=["ret_cc_t1"])
    plt.figure(figsize=(6, 6))
    plt.scatter(tmp["FlowShock_frac"], tmp["ret_cc_t1"])
    plt.title("FlowShock_t vs next-day close-close return")
    plt.xlabel("FlowShock_t")
    plt.ylabel("Return_{t+1} (close-close)")
    plt.tight_layout()
    plt.savefig(OUT / "flowshock_vs_nextday_return.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_outdir()
    df = load_clean()
    run_eda(df)
    print("EDA outputs saved to", OUT)


if __name__ == "__main__":
    main()
