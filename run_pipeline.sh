#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

python src/get_etf_flows.py
python src/get_etf_prices.py
python src/get_etf_nav.py
python src/build_etf_panel.py
python src/get_btc_1m_binance.py
python src/build_btc_windows.py
python src/build_macro.py
python src/build_master_panel.py
