from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    HAC_LAGS,
    OUT,
    ensure_outdir,
    event_study_car,
    load_clean,
    lp_ols,
    predictive_ols,
    save_irf,
    save_summary,
)


def run_baseline_models(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["FlowShock_frac", "ret_close_close", "ret_overnight", "ret_intraday"]).copy()
    d = d.sort_values("date").reset_index(drop=True)

    rows = []
    for dep in ["ret_close_close", "ret_overnight", "ret_intraday"]:
        m_simple = predictive_ols(d, dep, "FlowShock_frac", with_controls=False, maxlags=HAC_LAGS)
        m_ctrl = predictive_ols(d, dep, "FlowShock_frac", with_controls=True, maxlags=HAC_LAGS)
        save_summary(m_simple, f"baseline_static_ols_{dep}_simple.txt")
        save_summary(m_ctrl, f"baseline_static_ols_{dep}_with_controls.txt")
        for spec, m in [("simple", m_simple), ("with_controls", m_ctrl)]:
            rows.append(
                {
                    "window": dep,
                    "spec": spec,
                    "beta": float(m.params["FlowShock_frac"]),
                    "se": float(m.bse["FlowShock_frac"]),
                    "t": float(m.tvalues["FlowShock_frac"]),
                    "p": float(m.pvalues["FlowShock_frac"]),
                    "n": int(m.nobs),
                    "r2": float(m.rsquared),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "baseline_static_ols_summary.csv", index=False)

    save_irf(lp_ols(d, "ret_close_close", HAC_LAGS), "baseline_irf_closeclose", "LP-OLS IRF (close-close)")
    save_irf(lp_ols(d, "ret_overnight", HAC_LAGS), "baseline_irf_overnight", "LP-OLS IRF (overnight)")
    save_irf(lp_ols(d, "ret_intraday", HAC_LAGS), "baseline_irf_intraday", "LP-OLS IRF (intraday)")

    d_es = d.dropna(subset=["FlowShock_frac", "ret_close_close"]).copy().reset_index(drop=True)
    car95, thr95, n95 = event_study_car(d_es, threshold=0.95)
    car90, thr90, n90 = event_study_car(d_es, threshold=0.90)
    out = pd.concat([car95, car90], ignore_index=True)
    out["thr_value"] = pd.NA
    out.loc[out["threshold"] == 0.95, "thr_value"] = thr95
    out.loc[out["threshold"] == 0.90, "thr_value"] = thr90
    out["n_events_total"] = pd.NA
    out.loc[out["threshold"] == 0.95, "n_events_total"] = n95
    out.loc[out["threshold"] == 0.90, "n_events_total"] = n90
    out.to_csv(OUT / "baseline_eventstudy_car_h125_thresholds.csv", index=False)

    baseline95 = car95[car95["label"] == "baseline"].sort_values("h")
    plt.figure(figsize=(6, 4))
    plt.plot(baseline95["h"], baseline95["avg_CAR"], marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("Event study CAR (|FlowShock|>95th)")
    plt.xlabel("Horizon h (days)")
    plt.ylabel("Avg CAR (log)")
    plt.tight_layout()
    plt.savefig(OUT / "baseline_eventstudy_car_95.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_outdir()
    df = load_clean()
    run_baseline_models(df)
    print("Baseline model outputs saved to", OUT)


if __name__ == "__main__":
    main()
