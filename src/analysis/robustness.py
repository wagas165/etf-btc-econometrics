from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import (
    HAC_LAGS,
    OUT,
    CONTROLS_IV,
    ensure_outdir,
    choose_instruments,
    event_study_car,
    load_clean,
    predictive_ols,
    write_json,
)
from .iv_models import lp_iv


def run_robustness_checks(df: pd.DataFrame) -> None:
    df6 = df.copy().sort_values("date").reset_index(drop=True)

    df6["ma60"] = df6["btc_close_et"].rolling(60, min_periods=60).mean()
    df6["regime_bull"] = np.where(df6["btc_close_et"] > df6["ma60"], 1, 0)

    t1_date = pd.Timestamp("2024-05-28")
    df6["post_t1"] = (df6["date"] >= t1_date).astype(int)
    df6["year"] = df6["date"].dt.year

    q1 = df6["FlowShock_frac"].quantile(0.01)
    q99 = df6["FlowShock_frac"].quantile(0.99)
    df6["FlowShock_w"] = df6["FlowShock_frac"].clip(lower=q1, upper=q99)

    variants = {
        "base": df6,
        "exclude_macro": df6[(df6["is_fomc"] == 0) & (df6["is_cpi"] == 0)],
        "bull": df6[df6["regime_bull"] == 1],
        "bear": df6[df6["regime_bull"] == 0],
        "pre_t1": df6[df6["date"] < t1_date],
        "post_t1": df6[df6["date"] >= t1_date],
        "year_2024": df6[df6["year"] == 2024],
        "year_2025": df6[df6["year"] == 2025],
    }
    write_json(OUT / "robustness_variants.json", {"variants": list(variants.keys())})

    rows = []
    for name, vd in variants.items():
        for dep in ["ret_close_close", "ret_overnight", "ret_intraday"]:
            try:
                m = predictive_ols(vd.dropna(subset=["FlowShock_frac", dep]), dep, "FlowShock_frac", with_controls=True, maxlags=HAC_LAGS)
                rows.append(
                    {
                        "variant": name,
                        "window": dep,
                        "shock": "FlowShock_frac",
                        "beta": float(m.params["FlowShock_frac"]),
                        "se": float(m.bse["FlowShock_frac"]),
                        "t": float(m.tvalues["FlowShock_frac"]),
                        "p": float(m.pvalues["FlowShock_frac"]),
                        "n": int(m.nobs),
                        "r2": float(m.rsquared),
                    }
                )
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "variant": name,
                    "window": dep,
                    "shock": "FlowShock_frac",
                    "beta": np.nan,
                    "se": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n": 0,
                    "r2": np.nan,
                    "error": str(e),
                })

    for dep in ["ret_close_close", "ret_overnight", "ret_intraday"]:
        try:
            m = predictive_ols(df6.dropna(subset=["FlowShock_w", dep]), dep, "FlowShock_w", with_controls=True, maxlags=HAC_LAGS)
            rows.append(
                {
                    "variant": "winsor_1_99",
                    "window": dep,
                    "shock": "FlowShock_w",
                    "beta": float(m.params["FlowShock_w"]),
                    "se": float(m.bse["FlowShock_w"]),
                    "t": float(m.tvalues["FlowShock_w"]),
                    "p": float(m.pvalues["FlowShock_w"]),
                    "n": int(m.nobs),
                    "r2": float(m.rsquared),
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append({
                "variant": "winsor_1_99",
                "window": dep,
                "shock": "FlowShock_w",
                "beta": np.nan,
                "se": np.nan,
                "t": np.nan,
                "p": np.nan,
                "n": 0,
                "r2": np.nan,
                "error": str(e),
            })

    pd.DataFrame(rows).to_csv(OUT / "robustness_static_ols.csv", index=False)

    lags_list = [3, 5, 10]
    sens = []
    for L in lags_list:
        m = predictive_ols(df6.dropna(subset=["FlowShock_frac", "ret_close_close"]), "ret_close_close", "FlowShock_frac", with_controls=True, maxlags=L)
        sens.append(
            {
                "model": "static_ols_closeclose",
                "hac_lags": L,
                "beta": float(m.params["FlowShock_frac"]),
                "se": float(m.bse["FlowShock_frac"]),
                "t": float(m.tvalues["FlowShock_frac"]),
                "p": float(m.pvalues["FlowShock_frac"]),
                "n": int(m.nobs),
            }
        )

    instruments = choose_instruments(df6)
    if len(instruments) > 0:
        tmp = df6.copy()
        tmp["FlowShock_t1"] = tmp["FlowShock_frac"].shift(-1)
        for L in lags_list:
            samp = tmp.dropna(subset=["FlowShock_t1"] + instruments + CONTROLS_IV).copy()
            X = sm.add_constant(samp[instruments + CONTROLS_IV])
            y = samp["FlowShock_t1"]
            m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            for ins in instruments:
                sens.append(
                    {
                        "model": "first_stage",
                        "hac_lags": L,
                        "instrument": ins,
                        "coef": float(m.params[ins]),
                        "se": float(m.bse[ins]),
                        "t": float(m.tvalues[ins]),
                        "F": float(m.tvalues[ins] ** 2),
                        "n": int(m.nobs),
                        "r2": float(m.rsquared),
                    }
                )

    pd.DataFrame(sens).to_csv(OUT / "robustness_hac_lags_sensitivity.csv", index=False)

    lpiv_rows = []
    if len(instruments) > 0:
        for name, vd in variants.items():
            for win in ["ret_close_close", "ret_overnight", "ret_intraday"]:
                try:
                    beta, se, t, p, n = lp_iv(vd, instruments, win, h=0, lags=HAC_LAGS)
                    lpiv_rows.append({"variant": name, "window": win, "h": 0, "beta": beta, "se": se, "t": t, "p": p, "n": n})
                except Exception as e:  # noqa: BLE001
                    lpiv_rows.append({
                        "variant": name,
                        "window": win,
                        "h": 0,
                        "beta": np.nan,
                        "se": np.nan,
                        "t": np.nan,
                        "p": np.nan,
                        "n": 0,
                        "error": str(e),
                    })
    pd.DataFrame(lpiv_rows).to_csv(OUT / "robustness_lpiv_h0.csv", index=False)

    es_rows = []
    for name in ["base", "exclude_macro"]:
        vd = variants[name].dropna(subset=["FlowShock_frac", "ret_close_close"]).copy().sort_values("date").reset_index(drop=True)
        for thr in [0.90, 0.95]:
            car_df, thr_val, n_events = event_study_car(vd, threshold=thr)
            car_df = car_df.copy()
            car_df["variant"] = name
            car_df["thr_value"] = thr_val
            car_df["n_events_total"] = n_events
            es_rows.append(car_df)
    pd.concat(es_rows, ignore_index=True).to_csv(OUT / "robustness_eventstudy_thresholds.csv", index=False)

    lines = []
    lines.append("# Robustness checks\n\n")
    lines.append("- Variants: " + ", ".join(variants.keys()) + "\n")
    lines.append("- Files:\n")
    lines.append("  - robustness_static_ols.csv\n")
    lines.append("  - robustness_lpiv_h0.csv\n")
    lines.append("  - robustness_eventstudy_thresholds.csv\n")
    lines.append("  - robustness_hac_lags_sensitivity.csv\n")
    (OUT / "robustness_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ensure_outdir()
    df = load_clean()
    run_robustness_checks(df)
    print("Robustness outputs saved to", OUT)


if __name__ == "__main__":
    main()
