"""Download ETF prices and volumes from Yahoo Finance using ``yfinance``.

The script saves a combined CSV in ``data/raw/etf_prices/etf_prices_yahoo.csv``
with columns ``date``, ``ticker``, ``close_price``, and ``volume_shares``.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from config.config import ETF_END_DATE, ETF_START_DATE, RAW_DIR, US_BTC_ETFS


def fetch_etf_price(ticker: str) -> pd.DataFrame:
    yf_ticker = yf.Ticker(ticker)
    hist = yf_ticker.history(start=ETF_START_DATE, end=ETF_END_DATE, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")

    hist = hist.reset_index().rename(
        columns={
            "Date": "date",
            "Close": "close_price",
            "Volume": "volume_shares",
        }
    )
    hist["ticker"] = ticker
    return hist[["date", "ticker", "close_price", "volume_shares"]]


def main() -> None:
    out_dir = RAW_DIR / "etf_prices"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for tkr in US_BTC_ETFS:
        try:
            df = fetch_etf_price(tkr)
            frames.append(df)
            print(f"Fetched {len(df):,} rows for {tkr}")
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {tkr}: {exc}")

    if not frames:
        raise SystemExit("No ETF price data downloaded")

    all_prices = pd.concat(frames, ignore_index=True)
    all_prices.to_csv(out_dir / "etf_prices_yahoo.csv", index=False)
    print(f"Saved {len(all_prices):,} rows to {out_dir / 'etf_prices_yahoo.csv'}")


if __name__ == "__main__":
    main()
