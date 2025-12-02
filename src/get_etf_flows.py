"""Download and harmonize ETF daily flow data.

Two sources are supported out of the box:
1. Farside: scraped via ``pandas.read_html`` from the public flow page.
2. SoSoValue: either fetched from the documented API endpoint (if provided
   via the ``SOSOVALUE_URL`` environment variable) or read from a cached JSON
   file under ``data/raw/etf_flows/sosovalue_flows_raw.json``.

The script produces two artifacts:
* ``data/raw/etf_flows/farside_flows_raw.csv`` — wide table (date × ticker).
* ``data/clean/etf_flows_panel.csv`` — long-format panel with consistent
  column names and source attribution.

Usage:
    python src/get_etf_flows.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from config.config import (
    CLEAN_DIR,
    ETF_END_DATE,
    ETF_START_DATE,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    RAW_DIR,
    US_BTC_ETFS,
)

FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
SOSOVALUE_FALLBACK = RAW_DIR / "etf_flows" / "sosovalue_flows_raw.json"


@dataclass
class FlowSource:
    name: str
    wide_table: pd.DataFrame


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=HTTP_MAX_RETRIES)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_farside_flows() -> FlowSource:
    """Fetch Farside flow table and return as a wide DataFrame."""
    session = _session_with_retries()
    response = session.get(FARSIDE_URL, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    if not tables:
        raise ValueError("No tables found on Farside page")

    df = tables[0].copy()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= ETF_START_DATE) & (df["date"] <= ETF_END_DATE)]

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return FlowSource(name="farside", wide_table=df)


def _load_sosovalue_from_json(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict) and "data" in payload:
        df = pd.DataFrame(payload["data"])
    else:
        raise ValueError("Unrecognized SoSoValue JSON structure")
    return df


def fetch_sosovalue_flows() -> Optional[FlowSource]:
    """Fetch SoSoValue flows if an endpoint or cached file is provided.

    Returns ``None`` when neither source is available, allowing the pipeline
    to proceed with only Farside data.
    """

    api_url = os.getenv("SOSOVALUE_URL")
    if api_url:
        session = _session_with_retries()
        response = session.get(api_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
    else:
        df = _load_sosovalue_from_json(SOSOVALUE_FALLBACK)
        if df is None:
            return None

    # Attempt common column name normalizations
    col_map = {
        "date": "date",
        "Date": "date",
        "ticker": "ticker",
        "symbol": "ticker",
        "net_flow_usd": "net_flow_usd",
        "netflow": "net_flow_usd",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" not in df.columns or "net_flow_usd" not in df.columns:
        raise ValueError("SoSoValue data missing required columns: date and net_flow_usd")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if "ticker" not in df.columns and "fund" in df.columns:
        df = df.rename(columns={"fund": "ticker"})
    if "ticker" not in df.columns:
        raise ValueError("SoSoValue data missing ticker column")

    df = df[(df["date"] >= ETF_START_DATE) & (df["date"] <= ETF_END_DATE)]
    df["net_flow_usd"] = pd.to_numeric(df["net_flow_usd"], errors="coerce")

    wide = df.pivot_table(index="date", columns="ticker", values="net_flow_usd")
    wide = wide.reset_index()
    return FlowSource(name="sosovalue", wide_table=wide)


def _wide_to_long(source: FlowSource, tickers: Iterable[str]) -> pd.DataFrame:
    long_df = source.wide_table.melt(id_vars="date", var_name="ticker", value_name="net_flow_usd")
    long_df["source"] = source.name
    long_df = long_df[long_df["ticker"].isin(tickers)]
    long_df = long_df.dropna(subset=["net_flow_usd"])
    long_df.sort_values(["date", "ticker"], inplace=True)
    return long_df


def harmonize_flows(sources: list[FlowSource]) -> pd.DataFrame:
    panels = [_wide_to_long(src, US_BTC_ETFS) for src in sources]
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def main() -> None:
    raw_dir = RAW_DIR / "etf_flows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)

    sources: list[FlowSource] = []

    farside = fetch_farside_flows()
    farside.wide_table.to_csv(raw_dir / "farside_flows_raw.csv", index=False)
    sources.append(farside)

    try:
        sosovalue = fetch_sosovalue_flows()
    except Exception as exc:  # noqa: BLE001 - we want to log and continue
        print(f"SoSoValue fetch failed: {exc}")
        sosovalue = None
    if sosovalue is not None:
        sosovalue.wide_table.to_csv(raw_dir / "sosovalue_flows_raw.csv", index=False)
        sources.append(sosovalue)

    panel = harmonize_flows(sources)
    panel.to_csv(clean_dir / "etf_flows_panel.csv", index=False)
    print(f"Saved {len(panel):,} flow rows to {clean_dir / 'etf_flows_panel.csv'}")


if __name__ == "__main__":
    main()
