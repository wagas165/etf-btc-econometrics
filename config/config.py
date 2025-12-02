"""Project-wide configuration for ETF/BTC econometrics pipeline.

Update these values to reflect the time range and funds you want to pull.
"""

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

# Binance download settings
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"

# US equity regular trading session (Eastern Time)
US_EQUITY_OPEN_ET = "09:30"
US_EQUITY_CLOSE_ET = "16:00"

# Convenience paths derived from DATA_DIR
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

# Network access can be flaky; tweak retry limits here if needed
HTTP_TIMEOUT = 15
HTTP_MAX_RETRIES = 3
