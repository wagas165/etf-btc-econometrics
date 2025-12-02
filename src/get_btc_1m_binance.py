"""Download BTCUSDT 1m klines from Binance REST API.

The script saves raw candles to ``data/raw/btc_1m/btc_1m_raw.csv``.
API limits require chunked requests; adjust the step size if needed. If the
primary endpoint is blocked in your region, set ``BINANCE_BASE_URL`` or
``BINANCE_FALLBACK_BASE_URL`` to an accessible mirror and the downloader will
fail over automatically.
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


def main() -> None:
    out_path = RAW_DIR / "btc_1m" / "btc_1m_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = ETF_START_DATE.replace(tzinfo=timezone.utc)
    end = ETF_END_DATE.replace(tzinfo=timezone.utc)
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


if __name__ == "__main__":
    main()
