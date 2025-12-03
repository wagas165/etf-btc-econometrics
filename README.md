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

## Pipeline
Run everything from a single entry point:

```bash
python src/run_analysis.py
```

This orchestrates data collection, feature engineering, and the econometric study. Use `--data-only` to skip the analysis or `--analysis-only` to rerun charts/regressions after data updates.

### What happens

| Stage | Description | Entrypoint | Key outputs |
| --- | --- | --- | --- |
| Download ETF data | Pulls flows, prices, and NAV inputs for listed U.S. BTC ETFs. | `src/get_etf_flows.py`, `src/get_etf_prices.py`, `src/get_etf_nav.py` | Raw HTML/JSON caches under `data/raw/etf_flows/` plus harmonized flows and pricing panels in `data/clean/` | 
| Build BTC windows | Downloads Binance 1-minute klines and aggregates them into intraday/overnight windows. | `src/get_btc_1m_binance.py`, `src/build_btc_windows.py` | BTC feature windows in `data/clean/btc_windows_*.csv` |
| Construct macro & master panel | Flags macro announcement dates, merges all data into `data/clean/master_panel.csv`, and keeps intermediate ETF panels. | `src/build_macro.py`, `src/build_master_panel.py` | Final analysis dataset under `data/clean/master_panel.csv` |
| Run econometric analysis | Generates descriptive plots, impulse responses, and robustness checks. | `run_steps_3_6.py` (called by `src/run_analysis.py`) | Charts, CSVs, and reports in `outputs/` with stage-prefixed filenames |

If you prefer a Bash wrapper, use `./run_data_pipeline.sh` to refresh data only via the same Python entrypoint.

## Troubleshooting downloads

* **Farside ETF flows returning 403**: the scraper now caches the page HTML to
  `data/raw/etf_flows/farside_flows_raw.html`. If the live request is blocked,
  save the page manually in your browser and re-run the script, or point the
  `FARSIDE_HTML_FALLBACK` environment variable to your saved HTML copy.
* **Binance klines blocked in your region**: set `BINANCE_BASE_URL` or
  `BINANCE_FALLBACK_BASE_URL` to an accessible mirror (for example,
  `https://data-api.binance.vision/api/v3/klines`). The downloader will try the
  primary endpoint first and automatically fall back if it fails.
