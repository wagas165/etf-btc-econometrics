"""Download BTCUSDT 1m klines from Binance REST API or CryptoDataDownload.

The script saves raw candles to ``data/raw/btc_1m/btc_1m_raw.csv``. It first
attempts a bulk download from CryptoDataDownload (CDD), which hosts a
full-history CSV without API limits or regional blocks. If that fails, it falls
back to chunked REST requests against Binance (with an optional mirror URL).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Iterable

import _project_paths  # noqa: F401  # adds repo root to sys.path
import pandas as pd
import requests

from config.config import (
    BINANCE_BASE_URL,
    BINANCE_FALLBACK_BASE_URL,
    BINANCE_INTERVAL,
    BINANCE_SYMBOL,
    CDD_1M_URL,
    ETF_END_DATE,
    ETF_START_DATE,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    RAW_DIR,
)


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=HTTP_MAX_RETRIES)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_klines(
    session: requests.Session,
    base_url: str,
    start_ts_ms: int,
    end_ts_ms: int,
    limit: int = 1000,
):
    params = {
        "symbol": BINANCE_SYMBOL,
        "interval": BINANCE_INTERVAL,
        "startTime": start_ts_ms,
        "endTime": end_ts_ms,
        "limit": limit,
    }
    response = session.get(base_url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def daterange(start: datetime, end: datetime, step: timedelta) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    while cur < end:
        chunk_end = min(cur + step, end)
        yield cur, chunk_end
        cur = chunk_end


def download_from_cdd(start: datetime, end: datetime, out_path: Path) -> bool:
    if not CDD_1M_URL:
        return False

    try:
        df = pd.read_csv(CDD_1M_URL, skiprows=1)
    except Exception as exc:  # noqa: BLE001 - want to log and fallback
        print(f"CDD download failed ({exc}); falling back to Binance API.")
        return False

    if df.empty:
        print("CDD download returned no rows; falling back to Binance API.")
        return False

    df["timestamp_utc"] = pd.to_datetime(df["date"], utc=True)
    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
    if df.empty:
        print(
            "CDD download did not contain the requested date window; "
            "falling back to Binance API."
        )
        return False

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df = df.rename(columns={"Volume BTC": "volume"})
    df["symbol"] = BINANCE_SYMBOL
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume", "symbol"]]
    df.to_csv(out_path, index=False)
    print(
        "Saved "
        f"{len(df):,} rows to {out_path} using CryptoDataDownload (descending order fixed)."
    )
    return True


def download_from_binance(start: datetime, end: datetime, out_path: Path) -> None:
    step = timedelta(days=3)
    rows = []
    session = _session_with_retries()
    for idx, (chunk_start, chunk_end) in enumerate(daterange(start, end, step), start=1):
        cur_start = chunk_start
        chunk_rows = 0
        while cur_start < chunk_end:
            errors: list[str] = []
            data = None
            for base_url in [BINANCE_BASE_URL, BINANCE_FALLBACK_BASE_URL]:
                if not base_url:
                    continue
                try:
                    data = fetch_klines(
                        session,
                        base_url,
                        int(cur_start.timestamp() * 1000),
                        int(chunk_end.timestamp() * 1000),
                    )
                    break
                except requests.HTTPError as exc:
                    errors.append(f"{base_url}: {exc}")
                    print(
                        f"Binance request failed for {base_url} ({exc}); "
                        "checking fallback endpoint."
                    )

            if data is None:
                raise RuntimeError(
                    "All Binance endpoints failed. Set BINANCE_BASE_URL or "
                    "BINANCE_FALLBACK_BASE_URL to a reachable mirror.\n" + "\n".join(errors)
                )

            if not data:
                break

            for k in data:
                open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                open_, high, low, close = map(float, k[1:5])
                volume = float(k[5])
                rows.append(
                    {
                        "timestamp_utc": open_time,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "symbol": BINANCE_SYMBOL,
                    }
                )

            chunk_rows += len(data)
            last_open_time = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
            cur_start = last_open_time + timedelta(minutes=1)
            sleep(0.25)  # polite pause between API calls

        print(
            f"Chunk {idx}: {chunk_start.date()} to {chunk_end.date()} -> {chunk_rows} rows"
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


def main() -> None:
    out_path = RAW_DIR / "btc_1m" / "btc_1m_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = ETF_START_DATE.replace(tzinfo=timezone.utc)
    end = ETF_END_DATE.replace(tzinfo=timezone.utc)

    if download_from_cdd(start, end, out_path):
        return

    download_from_binance(start, end, out_path)


if __name__ == "__main__":
    main()
