"""Normalize issuer-provided NAV files into a single panel."""

from __future__ import annotations

import _project_paths  # noqa: F401  # adds repo root to sys.path
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from config.config import CLEAN_DIR, ETF_END_DATE, ETF_START_DATE, RAW_DIR, US_BTC_ETFS


NAV_FILENAME_TEMPLATE = "nav_{ticker}.csv"


def load_single_nav_file(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {
        "Date": "date",
        "date": "date",
        "NAV": "nav_per_share",
        "nav": "nav_per_share",
        "nav_per_share": "nav_per_share",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" not in df.columns or "nav_per_share" not in df.columns:
        raise ValueError(f"NAV file {path} missing required columns: date/nav_per_share")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = ticker
    return df[["date", "ticker", "nav_per_share"]]


def iter_nav_files(tickers: Iterable[str]):
    nav_dir = RAW_DIR / "etf_nav"
    for tkr in tickers:
        fpath = nav_dir / NAV_FILENAME_TEMPLATE.format(ticker=tkr)
        if fpath.exists():
            yield tkr, fpath
        else:
            print(f"NAV file missing for {tkr}: {fpath}")


def fetch_nav_from_yfinance(ticker: str) -> pd.DataFrame:
    """Fallback NAV estimate using Yahoo! Finance closes.

    Issuers often lag publishing NAV files. To keep the pipeline usable, we
    fall back to Yahoo! Finance daily closes as a proxy for NAV per share. This
    at least restores downstream ``premium`` calculations even if issuer CSVs
    are not available.
    """

    hist = yf.Ticker(ticker).history(start=ETF_START_DATE, end=ETF_END_DATE, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No NAV proxy data returned for {ticker} from yfinance")

    hist = hist.reset_index().rename(columns={"Date": "date", "Close": "nav_per_share"})
    hist["ticker"] = ticker
    return hist[["date", "ticker", "nav_per_share"]]


def main() -> None:
    out_path = CLEAN_DIR / "etf_nav_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for ticker in US_BTC_ETFS:
        issuer_path = RAW_DIR / "etf_nav" / NAV_FILENAME_TEMPLATE.format(ticker=ticker)
        if issuer_path.exists():
            try:
                frames.append(load_single_nav_file(issuer_path, ticker))
                print(f"Loaded issuer NAV for {ticker} from {issuer_path}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping issuer NAV for {ticker} ({issuer_path}): {exc}")

        try:
            frames.append(fetch_nav_from_yfinance(ticker))
            print(f"Fetched NAV proxy for {ticker} from yfinance")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to fetch NAV proxy for {ticker}: {exc}")

    frames = [f for f in frames if not f.empty]
    if not frames:
        print("No NAV data available from issuers or yfinance proxies.")
        return

    nav_panel = pd.concat(frames, ignore_index=True)
    nav_panel.to_csv(out_path, index=False)
    print(f"Saved NAV panel with {len(nav_panel):,} rows to {out_path}")


if __name__ == "__main__":
    main()
