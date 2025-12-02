"""Normalize issuer-provided NAV files into a single panel."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config.config import CLEAN_DIR, RAW_DIR, US_BTC_ETFS


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


def main() -> None:
    out_path = CLEAN_DIR / "etf_nav_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for ticker, path in iter_nav_files(US_BTC_ETFS):
        try:
            frames.append(load_single_nav_file(path, ticker))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {ticker} NAV ({path}): {exc}")

    if not frames:
        raise SystemExit("No NAV files found. Place issuer CSVs under data/raw/etf_nav")

    nav_panel = pd.concat(frames, ignore_index=True)
    nav_panel.to_csv(out_path, index=False)
    print(f"Saved NAV panel with {len(nav_panel):,} rows to {out_path}")


if __name__ == "__main__":
    main()
