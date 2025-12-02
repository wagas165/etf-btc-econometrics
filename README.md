# ETF BTC Econometrics

Lightweight pipeline to download Bitcoin ETF flows, prices, NAVs, BTC intraday data, and macro events into reproducible CSV panels. Each script can be run independently to refresh its portion of the dataset.

## Setup
1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Optionally set an environment variable for SoSoValue if you have the API endpoint:
   ```bash
   export SOSOVALUE_URL="https://example.com/api"
   ```

## Pipeline steps
Run the scripts in this order after placing issuer NAV CSVs under `data/raw/etf_nav/`:

1. `python src/get_etf_flows.py`
2. `python src/get_etf_prices.py`
3. `python src/get_etf_nav.py`
4. `python src/build_etf_panel.py`
5. `python src/get_btc_1m_binance.py`
6. `python src/build_btc_windows.py`
7. `python src/build_macro.py`
8. `python src/build_master_panel.py`

Outputs are written under `data/clean/` and are ready for analysis/notebooks.

## Troubleshooting downloads

* **Farside ETF flows returning 403**: the scraper now caches the page HTML to
  `data/raw/etf_flows/farside_flows_raw.html`. If the live request is blocked,
  save the page manually in your browser and re-run the script, or point the
  `FARSIDE_HTML_FALLBACK` environment variable to your saved HTML copy.
* **Binance klines blocked in your region**: set `BINANCE_BASE_URL` or
  `BINANCE_FALLBACK_BASE_URL` to an accessible mirror (for example,
  `https://data-api.binance.vision/api/v3/klines`). The downloader will try the
  primary endpoint first and automatically fall back if it fails.
