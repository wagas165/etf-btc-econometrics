"""Download BTCUSDT 1m klines from public archives without hitting the API.

The script pulls monthly CSV archives from the Binance data portal
(`https://data.binance.vision`), which is served as static files and is less
likely to be blocked than the REST API. If the archives are missing, we fall
back to the CryptoDataDownload mirror and, if that also fails, ask the user to
supply a CSV manually.

The combined, time-ordered candles are written to
``data/raw/btc_1m/btc_1m_raw.csv``.
"""

from __future__ import annotations

import zipfile
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import _project_paths  # noqa: F401  # adds repo root to sys.path
import pandas as pd
import requests

from config.config import (
    BINANCE_DATA_BASE_URL,
    BINANCE_INTERVAL,
    BINANCE_SYMBOL,
    CDD_1M_URL,
    ETF_END_DATE,
    ETF_START_DATE,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    RAW_DIR,
)

ARCHIVE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "num_trades",
    "taker_base_volume",
    "taker_quote_volume",
    "ignore",
]


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=HTTP_MAX_RETRIES)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def month_range(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    cur = datetime(start.year, start.month, 1, tzinfo=start.tzinfo)
    last = datetime(end.year, end.month, 1, tzinfo=end.tzinfo)
    while cur <= last:
        yield cur.year, cur.month
        days = monthrange(cur.year, cur.month)[1]
        cur = cur + timedelta(days=days)


def _download_monthly_archive(session: requests.Session, year: int, month: int) -> pd.DataFrame:
    url = (
        f"{BINANCE_DATA_BASE_URL}/{BINANCE_SYMBOL}/{BINANCE_INTERVAL}/"
        f"{BINANCE_SYMBOL}-{BINANCE_INTERVAL}-{year}-{month:02d}.zip"
    )
    response = session.get(url, timeout=HTTP_TIMEOUT)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"Archive {url} contained no files")
        with zf.open(names[0]) as fp:
            df = pd.read_csv(fp, header=None, names=ARCHIVE_COLUMNS)

    max_ts = df["open_time_ms"].astype("int64", errors="ignore").max()
    unit = "ms" if pd.notna(max_ts) and max_ts < 1_000_000_000_000_000 else "us"

    df["timestamp_utc"] = pd.to_datetime(df["open_time_ms"], unit=unit, utc=True)
    df = df.rename(columns={"volume": "volume_base"})
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume_base"]]
    df["symbol"] = BINANCE_SYMBOL
    return df


def download_from_archives(start: datetime, end: datetime, out_path: Path) -> bool:
    session = _session_with_retries()
    frames: list[pd.DataFrame] = []
    expected_months = list(month_range(start, end))
    downloaded_months: set[tuple[int, int]] = set()

    for year, month in expected_months:
        try:
            df_month = _download_monthly_archive(session, year, month)
            frames.append(df_month)
            downloaded_months.add((year, month))
            print(f"Downloaded {year}-{month:02d} archive with {len(df_month):,} rows")
        except requests.HTTPError as exc:
            print(f"Archive missing for {year}-{month:02d} ({exc}); continuing")
            continue

    if not frames:
        return False

    missing_months = [
        f"{year}-{month:02d}" for year, month in expected_months if (year, month) not in downloaded_months
    ]
    if missing_months:
        print(
            "Archive downloads incomplete; missing months: "
            + ", ".join(missing_months)
            + ". Saving partial dataset from available archives."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path} from Binance archives")
    return True


def download_from_cdd(start: datetime, end: datetime, out_path: Path) -> bool:
    if not CDD_1M_URL:
        return False

    try:
        df = pd.read_csv(CDD_1M_URL, skiprows=1)
    except Exception as exc:  # noqa: BLE001 - want to log and fallback
        print(f"CDD download failed ({exc}); please download manually and re-run.")
        return False

    df["timestamp_utc"] = pd.to_datetime(df["date"], utc=True)
    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
    if df.empty:
        print("CDD download did not contain the requested date window.")
        return False

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df = df.rename(columns={"Volume BTC": "volume_base"})
    df["symbol"] = BINANCE_SYMBOL
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume_base", "symbol"]]
    df.to_csv(out_path, index=False)
    print(
        "Saved "
        f"{len(df):,} rows to {out_path} using CryptoDataDownload (descending order fixed)."
    )
    return True


def main() -> None:
    out_path = RAW_DIR / "btc_1m" / "btc_1m_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    requested_start = ETF_START_DATE.replace(tzinfo=timezone.utc)
    requested_end = ETF_END_DATE.replace(tzinfo=timezone.utc)

    # Binance publishes full-month archives; avoid requesting future months that
    # do not exist yet by clipping the end date to the last completed month.
    now_utc = datetime.now(timezone.utc)
    last_full_month_start = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
    last_full_month_end = last_full_month_start - timedelta(microseconds=1)
    archive_end = min(requested_end, last_full_month_end)

    if archive_end < requested_start:
        print(
            "ETF_START_DATE falls in the current partial month; skipping archive downloads "
            "and relying on available CDD data instead."
        )
    elif requested_end > last_full_month_end:
        print(
            "ETF_END_DATE extends beyond available Binance archives; "
            "clipping to last complete month."
        )
    
    if archive_end >= requested_start and download_from_archives(requested_start, archive_end, out_path):
        return

    if download_from_cdd(requested_start, requested_end, out_path):
        return

    raise SystemExit(
        "Could not download BTC 1m data from archives or CryptoDataDownload. "
        "Please provide the CSV manually under data/raw/btc_1m/btc_1m_raw.csv"
    )


if __name__ == "__main__":
    main()
