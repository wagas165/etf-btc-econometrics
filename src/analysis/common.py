from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "master_panel.csv"
OUT = PROJECT_ROOT / "outputs"

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

    for col in ["is_fomc", "is_cpi"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in ["FlowShock", "Turnover", "Premium_agg", "aum_total_lag"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

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


def standardize_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        std = out[col].std(ddof=0)
        if std and not np.isclose(std, 0):
            out[col] = (out[col] - out[col].mean()) / std
    return out


def orthogonalize_instruments(design: pd.DataFrame, instruments: list[str], controls: list[str]) -> pd.DataFrame:
    controls_present = [c for c in controls if c in design.columns]
    if not controls_present:
        return design

    ctrl = sm.add_constant(design[controls_present], has_constant="add")
    for inst in instruments:
        if inst not in design.columns:
            continue
        fit = sm.OLS(design[inst], ctrl).fit()
        design[inst] = fit.resid
    return design


def predictive_ols(
    d: pd.DataFrame, dep_col: str, shock_col: str, with_controls: bool, maxlags: int
) -> sm.regression.linear_model.RegressionResultsWrapper:
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


def choose_instruments(df: pd.DataFrame, instrument_path: Path | None = None) -> list[str]:
    candidates = ["Turnover", "Premium_agg"]
    used = []
    for c in candidates:
        s = df[c].dropna()
        if s.nunique() > 1:
            used.append(c)
    if instrument_path:
        write_json(instrument_path, {"candidates": candidates, "used": used})
    return used
