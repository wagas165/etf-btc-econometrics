#!/usr/bin/env python3
"""
Run Steps 3–6 on prepared daily panel data.

Implements the proposal sections:
- Step 3: EDA / sanity checks
- Step 4: Baseline (predictive OLS, LP-OLS IRFs, high-threshold event study with CAR h={1,2,5})
- Step 5: LP–IV (2SLS with HAC): instrument next-day flows using same-day ETF microstructure
- Step 6: Robustness checks (macro exclusions, bull/bear regimes, pre/post T+1 date, subperiods, winsorization, HAC lag sensitivity)

Input expected:
- data/clean/master_panel.csv

Outputs:
- outputs/ (PNGs, CSVs, regression summaries, markdown reports)

Notes:
- If Premium_agg is constant in the provided dataset, it is automatically excluded from the instrument set.
- If FlowShock was computed using flow_agg_usd in US$ millions and AUM in US$, FlowShock is rescaled by 1e6 to become a fraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.stats.sandwich_covariance import cov_hac

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "clean" / "master_panel.csv"
OUT = HERE / "outputs"

H = 10
HAC_LAGS = 5

CONTROLS_OLS = ["ret_close_close", "rv_intraday", "rv_overnight", "is_fomc", "is_cpi"]
CONTROLS_IV = ["ret_close_close", "rv_intraday", "rv_overnight"]


def ensure_outdir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # Macro dummies: fill NA as 0
    for col in ["is_fomc", "is_cpi"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Replace inf with NaN
    for col in ["FlowShock", "Turnover", "Premium_agg", "aum_total_lag"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Unit fix: if FlowShock used flow_agg_usd in US$ millions / AUM in US$ -> multiply by 1e6 to get fraction.
    df["FlowShock_frac"] = df["FlowShock"] * 1e6

    return df


def validate(df: pd.DataFrame) -> dict:
    return {
        "rows_total": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_dates": bool(df["date"].duplicated().any()),
        "rows_finite_flowshock_turnover": int(df.dropna(subset=["FlowShock_frac", "Turnover"]).shape[0]),
        "premium_unique_nonmissing": int(df["Premium_agg"].dropna().nunique()) if "Premium_agg" in df else None,
        "missing_by_col": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }


def save_summary(model, filename: str) -> None:
    (OUT / filename).write_text(model.summary().as_text(), encoding="utf-8")


def save_irf(irf_df: pd.DataFrame, base_name: str, title: str) -> None:
    irf_df.to_csv(OUT / f"{base_name}.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.errorbar(irf_df["h"], irf_df["beta"], yerr=1.96 * irf_df["se"], fmt="o-")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Horizon h (days)")
    plt.ylabel("beta_h")
    plt.tight_layout()
    plt.savefig(OUT / f"{base_name}.png", dpi=160)
    plt.close()


# ---------- Step 3 ----------
def step3_eda(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["FlowShock_frac", "btc_close_et"]).copy().reset_index(drop=True)

    stats_vars = [
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
    d[stats_vars].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(OUT / "step3_summary_stats_valid.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], d["FlowShock_frac"])
    plt.title("FlowShock (fraction)")
    plt.xlabel("Date")
    plt.ylabel("FlowShock")
    plt.tight_layout()
    plt.savefig(OUT / "step3_flowshock_timeseries.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], d["btc_close_et"])
    plt.title("BTC Close Price (ET close proxy)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(OUT / "step3_btc_price_timeseries.png", dpi=160)
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
    plt.savefig(OUT / "step3_scatter_flowshock_nextday_ret.png", dpi=160)
    plt.close()


# ---------- Step 4 ----------
def predictive_ols(d: pd.DataFrame, dep_col: str, shock_col: str, with_controls: bool, maxlags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    tmp = d.copy()
    tmp["y"] = tmp[dep_col].shift(-1)
    cols = [shock_col] + (CONTROLS_OLS if with_controls else [])
    tmp = tmp.dropna(subset=["y"] + cols).copy()
    X = sm.add_constant(tmp[cols])
    y = tmp["y"]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def lp_ols(d: pd.DataFrame, win_col: str, maxlags: int) -> pd.DataFrame:
    base = d.copy().sort_values("date").reset_index(drop=True)
    out = []
    for h in range(H + 1):
        base["y"] = base[win_col].shift(-(1 + h))
        cols = ["FlowShock_frac"] + CONTROLS_OLS
        tmp = base.dropna(subset=["y"] + cols).copy()
        X = sm.add_constant(tmp[cols])
        y = tmp["y"]
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        out.append(
            {
                "h": h,
                "beta": float(m.params["FlowShock_frac"]),
                "se": float(m.bse["FlowShock_frac"]),
                "t": float(m.tvalues["FlowShock_frac"]),
                "p": float(m.pvalues["FlowShock_frac"]),
                "n": int(tmp.shape[0]),
                "r2": float(m.rsquared),
            }
        )
    return pd.DataFrame(out)


def event_study_car(d_es: pd.DataFrame, threshold: float, placebo_shift: int = 10, hs=(1, 2, 5)) -> tuple[pd.DataFrame, float, int]:
    thr = float(d_es["FlowShock_frac"].abs().quantile(threshold))
    events = list(d_es.index[d_es["FlowShock_frac"].abs() > thr])

    mask = np.ones(len(d_es), dtype=bool)
    mask[events] = False
    mu = float(d_es.loc[mask, "ret_close_close"].mean())

    def car_rows(event_idx: list[int], label: str) -> list[dict]:
        rows = []
        for h in hs:
            cars = []
            for idx in event_idx:
                if idx + h < len(d_es):
                    ar = d_es.loc[idx + 1 : idx + h, "ret_close_close"].values - mu
                    cars.append(float(ar.sum()))
            cars_arr = np.array(cars, dtype=float)
            se = cars_arr.std(ddof=1) / np.sqrt(len(cars_arr)) if len(cars_arr) > 1 else np.nan
            tval = cars_arr.mean() / se if (se == se and se != 0) else np.nan
            rows.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "h": h,
                    "n_events_used": int(len(cars_arr)),
                    "avg_CAR": float(cars_arr.mean()) if len(cars_arr) > 0 else np.nan,
                    "se_CAR": float(se) if se == se else np.nan,
                    "t": float(tval) if tval == tval else np.nan,
                }
            )
        return rows

    rows = []
    rows += car_rows(events, "baseline")
    placebo = [i + placebo_shift for i in events if i + placebo_shift < len(d_es)]
    rows += car_rows(placebo, f"placebo_shift{placebo_shift}")
    return pd.DataFrame(rows), thr, len(events)


def step4_baseline(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["FlowShock_frac", "ret_close_close", "ret_overnight", "ret_intraday"]).copy()
    d = d.sort_values("date").reset_index(drop=True)

    # Static predictive OLS
    rows = []
    for dep in ["ret_close_close", "ret_overnight", "ret_intraday"]:
        m_simple = predictive_ols(d, dep, "FlowShock_frac", with_controls=False, maxlags=HAC_LAGS)
        m_ctrl = predictive_ols(d, dep, "FlowShock_frac", with_controls=True, maxlags=HAC_LAGS)
        save_summary(m_simple, f"step4_static_ols_{dep}_simple.txt")
        save_summary(m_ctrl, f"step4_static_ols_{dep}_with_controls.txt")
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
    pd.DataFrame(rows).to_csv(OUT / "step4_static_ols_compact.csv", index=False)

    # LP-OLS IRFs
    save_irf(lp_ols(d, "ret_close_close", HAC_LAGS), "step4_irf_ols_closeclose", "LP-OLS IRF (close-close)")
    save_irf(lp_ols(d, "ret_overnight", HAC_LAGS), "step4_irf_ols_overnight", "LP-OLS IRF (overnight)")
    save_irf(lp_ols(d, "ret_intraday", HAC_LAGS), "step4_irf_ols_intraday", "LP-OLS IRF (intraday)")

    # Event study thresholds (90% and 95%)
    d_es = d.dropna(subset=["FlowShock_frac", "ret_close_close"]).copy().reset_index(drop=True)
    car95, thr95, n95 = event_study_car(d_es, threshold=0.95)
    car90, thr90, n90 = event_study_car(d_es, threshold=0.90)
    out = pd.concat([car95, car90], ignore_index=True)
    out["thr_value"] = np.nan
    out.loc[out["threshold"] == 0.95, "thr_value"] = thr95
    out.loc[out["threshold"] == 0.90, "thr_value"] = thr90
    out["n_events_total"] = np.nan
    out.loc[out["threshold"] == 0.95, "n_events_total"] = n95
    out.loc[out["threshold"] == 0.90, "n_events_total"] = n90
    out.to_csv(OUT / "step4_eventstudy_car_h125_thresholds.csv", index=False)

    baseline95 = car95[car95["label"] == "baseline"].sort_values("h")
    plt.figure(figsize=(6, 4))
    plt.plot(baseline95["h"], baseline95["avg_CAR"], marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("Event study CAR (|FlowShock|>95th)")
    plt.xlabel("Horizon h (days)")
    plt.ylabel("Avg CAR (log)")
    plt.tight_layout()
    plt.savefig(OUT / "step4_eventstudy_car_h125_95.png", dpi=160)
    plt.close()


# ---------- Step 5 ----------
def choose_instruments(df: pd.DataFrame) -> list[str]:
    candidates = ["Turnover", "Premium_agg"]
    used = []
    for c in candidates:
        s = df[c].dropna()
        if s.nunique() > 1:
            used.append(c)
    write_json(OUT / "step5_instruments_used.json", {"candidates": candidates, "used": used})
    return used


def first_stage(df: pd.DataFrame, instruments: list[str], lags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    tmp = df.copy().sort_values("date").reset_index(drop=True)
    tmp["FlowShock_t1"] = tmp["FlowShock_frac"].shift(-1)
    sample = tmp.dropna(subset=["FlowShock_t1"] + instruments + CONTROLS_IV).copy()
    X = sm.add_constant(sample[instruments + CONTROLS_IV])
    y = sample["FlowShock_t1"]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def fit_iv2sls_hac(y: np.ndarray, X_df: pd.DataFrame, Z_df: pd.DataFrame, nlags: int) -> tuple[IV2SLS, np.ndarray]:
    res = IV2SLS(y, X_df.values, Z_df.values).fit()
    cov = cov_hac(res, nlags=nlags)
    se = np.sqrt(np.diag(cov))
    return res, se


def lp_iv(df: pd.DataFrame, instruments: list[str], win_col: str, h: int, lags: int) -> tuple[float, float, float, float, int]:
    tmp = df.copy().sort_values("date").reset_index(drop=True)
    tmp["FlowShock_t1"] = tmp["FlowShock_frac"].shift(-1)
    tmp["y"] = tmp[win_col].shift(-(1 + h))
    sample = tmp.dropna(subset=["y", "FlowShock_t1"] + instruments + CONTROLS_IV).copy()

    y = sample["y"].values
    X = sm.add_constant(sample[["FlowShock_t1"] + CONTROLS_IV], has_constant="add")
    Z = sm.add_constant(sample[instruments + CONTROLS_IV], has_constant="add")

    res, se = fit_iv2sls_hac(y, X, Z, nlags=lags)
    beta = float(res.params[1])
    se_b = float(se[1])
    t = beta / se_b if se_b != 0 else np.nan
    dof = max(int(sample.shape[0] - X.shape[1]), 1)
    p = float(2 * (1 - st.t.cdf(abs(t), df=dof)))
    return beta, se_b, float(t), p, int(sample.shape[0])


def lp_iv_irf(df: pd.DataFrame, instruments: list[str], win_col: str, lags: int) -> pd.DataFrame:
    rows = []
    for h in range(H + 1):
        beta, se, t, p, n = lp_iv(df, instruments, win_col, h=h, lags=lags)
        rows.append({"h": h, "beta": beta, "se": se, "t": t, "p": p, "n": n})
    return pd.DataFrame(rows)


def step5_lpiv(df: pd.DataFrame) -> None:
    instruments = choose_instruments(df)
    if len(instruments) == 0:
        (OUT / "step5_lpiv_skipped.txt").write_text("No usable instruments with variation in provided data.", encoding="utf-8")
        return

    fs = first_stage(df, instruments, lags=HAC_LAGS)
    save_summary(fs, "step5_first_stage.txt")

    save_irf(lp_iv_irf(df, instruments, "ret_close_close", HAC_LAGS), "step5_irf_lpiv_closeclose", "LP-IV IRF (close-close)")
    save_irf(lp_iv_irf(df, instruments, "ret_overnight", HAC_LAGS), "step5_irf_lpiv_overnight", "LP-IV IRF (overnight)")
    save_irf(lp_iv_irf(df, instruments, "ret_intraday", HAC_LAGS), "step5_irf_lpiv_intraday", "LP-IV IRF (intraday)")


# ---------- Step 6 ----------
def step6_robustness(df: pd.DataFrame) -> None:
    df6 = df.copy().sort_values("date").reset_index(drop=True)

    # Regimes (bull/bear by 60d moving average)
    df6["ma60"] = df6["btc_close_et"].rolling(60, min_periods=60).mean()
    df6["regime_bull"] = np.where(df6["btc_close_et"] > df6["ma60"], 1, 0)

    # Pre/post T+1 (SEC compliance date: 2024-05-28)
    t1_date = pd.Timestamp("2024-05-28")
    df6["post_t1"] = (df6["date"] >= t1_date).astype(int)
    df6["year"] = df6["date"].dt.year

    # Winsorize FlowShock (1–99%)
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
    write_json(OUT / "step6_variants.json", {"variants": list(variants.keys())})

    # Static OLS across variants (with controls)
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
            except Exception as e:
                rows.append({"variant": name, "window": dep, "shock": "FlowShock_frac", "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": 0, "r2": np.nan, "error": str(e)})

    # Winsorized OLS
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
        except Exception as e:
            rows.append({"variant": "winsor_1_99", "window": dep, "shock": "FlowShock_w", "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": 0, "r2": np.nan, "error": str(e)})

    pd.DataFrame(rows).to_csv(OUT / "step6_static_ols_robustness.csv", index=False)

    # HAC lag sensitivity (static OLS and first stage)
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

    pd.DataFrame(sens).to_csv(OUT / "step6_hac_lags_sensitivity.csv", index=False)

    # LP-IV robustness: report h=0 across variants
    lpiv_rows = []
    if len(instruments) > 0:
        for name, vd in variants.items():
            for win in ["ret_close_close", "ret_overnight", "ret_intraday"]:
                try:
                    beta, se, t, p, n = lp_iv(vd, instruments, win, h=0, lags=HAC_LAGS)
                    lpiv_rows.append({"variant": name, "window": win, "h": 0, "beta": beta, "se": se, "t": t, "p": p, "n": n})
                except Exception as e:
                    lpiv_rows.append({"variant": name, "window": win, "h": 0, "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": 0, "error": str(e)})
    pd.DataFrame(lpiv_rows).to_csv(OUT / "step6_lpiv_h0_robustness.csv", index=False)

    # Event study in base + macro excluded, thresholds 90/95
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
    pd.concat(es_rows, ignore_index=True).to_csv(OUT / "step6_eventstudy_thresholds.csv", index=False)

    # A short markdown report
    lines = []
    lines.append("# Step 6 Robustness\n\n")
    lines.append("- Variants: " + ", ".join(variants.keys()) + "\n")
    lines.append("- Files:\n")
    lines.append("  - step6_static_ols_robustness.csv\n")
    lines.append("  - step6_lpiv_h0_robustness.csv\n")
    lines.append("  - step6_eventstudy_thresholds.csv\n")
    lines.append("  - step6_hac_lags_sensitivity.csv\n")
    (OUT / "step6_report.md").write_text("".join(lines), encoding="utf-8")


def write_report(validation: dict) -> None:
    lines = []
    lines.append("# Steps 3–6 Report\n\n")
    lines.append("## Validation\n")
    lines.append("```json\n" + json.dumps(validation, indent=2) + "\n```\n\n")

    if (OUT / "step4_static_ols_compact.csv").exists():
        comp = pd.read_csv(OUT / "step4_static_ols_compact.csv")
        lines.append("## Step 4: Static OLS (compact)\n\n")
        lines.append(comp.to_markdown(index=False) + "\n\n")

    if (OUT / "step4_eventstudy_car_h125_thresholds.csv").exists():
        car = pd.read_csv(OUT / "step4_eventstudy_car_h125_thresholds.csv")
        lines.append("## Step 4: Event study CAR (thresholds)\n\n")
        lines.append(car.to_markdown(index=False) + "\n\n")

    if (OUT / "step5_instruments_used.json").exists():
        lines.append("## Step 5: Instruments used\n\n")
        lines.append("```json\n" + (OUT / "step5_instruments_used.json").read_text(encoding="utf-8") + "\n```\n\n")

    lines.append("## Step 6: Robustness\n\nSee step6_report.md and the Step 6 CSVs.\n")

    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ensure_outdir()
    df = load_clean()
    validation = validate(df)
    write_json(OUT / "step2_validation.json", validation)

    step3_eda(df)
    step4_baseline(df)
    step5_lpiv(df)
    step6_robustness(df)
    write_report(validation)

    print("Done. Outputs in:", OUT)


if __name__ == "__main__":
    main()
