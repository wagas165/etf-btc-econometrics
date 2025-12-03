from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.stats.sandwich_covariance import cov_hac

from .common import (
    HAC_LAGS,
    H,
    OUT,
    CONTROLS_IV,
    ensure_outdir,
    choose_instruments,
    load_clean,
    orthogonalize_instruments,
    save_irf,
    save_summary,
    standardize_columns,
)


def first_stage(df: pd.DataFrame, instruments: list[str], lags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    tmp = df.copy().sort_values("date").reset_index(drop=True)
    tmp["FlowShock_t1"] = tmp["FlowShock_frac"].shift(-1)
    sample = tmp.dropna(subset=["FlowShock_t1"] + instruments + CONTROLS_IV).copy()

    design = sample[instruments + CONTROLS_IV].copy()
    design = standardize_columns(design, design.columns.tolist())
    design = orthogonalize_instruments(design, instruments, CONTROLS_IV)

    X = sm.add_constant(design[instruments + CONTROLS_IV])
    y = sample["FlowShock_t1"]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def fit_iv2sls_hac(y: np.ndarray, X_df: pd.DataFrame, Z_df: pd.DataFrame, nlags: int) -> tuple[IV2SLS, np.ndarray]:
    res = IV2SLS(y, X_df.values, Z_df.values).fit()
    cov = cov_hac(res, nlags=nlags)
    se = np.sqrt(np.diag(cov))
    return res, se


def lp_iv(
    df: pd.DataFrame, instruments: list[str], win_col: str, h: int, lags: int
) -> tuple[float, float, float, float, int]:
    tmp = df.copy().sort_values("date").reset_index(drop=True)
    tmp["FlowShock_t1"] = tmp["FlowShock_frac"].shift(-1)
    tmp["y"] = tmp[win_col].shift(-(1 + h))
    sample = tmp.dropna(subset=["y", "FlowShock_t1"] + instruments + CONTROLS_IV).copy()

    design = sample[instruments + CONTROLS_IV].copy()
    design = standardize_columns(design, design.columns.tolist())
    design = orthogonalize_instruments(design, instruments, CONTROLS_IV)

    y = sample["y"].values
    X = sm.add_constant(pd.concat([sample[["FlowShock_t1"]], design[CONTROLS_IV]], axis=1), has_constant="add")
    Z = sm.add_constant(pd.concat([design[instruments], design[CONTROLS_IV]], axis=1), has_constant="add")

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


def run_iv_models(df: pd.DataFrame) -> None:
    instruments = choose_instruments(df, OUT / "iv_instruments_used.json")
    if len(instruments) == 0:
        (OUT / "iv_lpiv_skipped.txt").write_text(
            "No usable instruments with variation in provided data.", encoding="utf-8"
        )
        return

    fs = first_stage(df, instruments, lags=HAC_LAGS)
    save_summary(fs, "iv_first_stage.txt")

    save_irf(lp_iv_irf(df, instruments, "ret_close_close", HAC_LAGS), "iv_irf_closeclose", "LP-IV IRF (close-close)")
    save_irf(lp_iv_irf(df, instruments, "ret_overnight", HAC_LAGS), "iv_irf_overnight", "LP-IV IRF (overnight)")
    save_irf(lp_iv_irf(df, instruments, "ret_intraday", HAC_LAGS), "iv_irf_intraday", "LP-IV IRF (intraday)")


def main() -> None:
    ensure_outdir()
    df = load_clean()
    run_iv_models(df)
    print("IV model outputs saved to", OUT)


if __name__ == "__main__":
    main()
