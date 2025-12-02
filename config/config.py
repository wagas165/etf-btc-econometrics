"""Project-wide configuration for ETF/BTC econometrics pipeline.

Update these values to reflect the time range and funds you want to pull.
"""

import os
from datetime import datetime
from pathlib import Path

# Base directory for all data artifacts (raw downloads and cleaned panels)
DATA_DIR = Path("data")

# Date range for ETF/BTC history
ETF_START_DATE = datetime(2024, 1, 11)
ETF_END_DATE = datetime(2025, 12, 31)

# List of US spot Bitcoin ETFs to process
US_BTC_ETFS = [
    "IBIT",
    "FBTC",
    "ARKB",
    "BITB",
    "HODL",
    "BTCO",
    "BRRR",
    "EZBC",
    "DEFI",
    "BTCW",
    "GBTC",
]

# BTC minute data download settings
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
BINANCE_DATA_BASE_URL = os.getenv(
    "BINANCE_DATA_BASE_URL",
    "https://data.binance.vision/data/spot/monthly/klines",
)
CDD_1M_URL = os.getenv(
    "CDD_1M_URL", "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv"
)

# US equity regular trading session (Eastern Time)
US_EQUITY_OPEN_ET = "09:30"
US_EQUITY_CLOSE_ET = "16:00"

# Convenience paths derived from DATA_DIR
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

# Network access can be flaky; tweak retry limits here if needed
HTTP_TIMEOUT = 15
HTTP_MAX_RETRIES = 3
