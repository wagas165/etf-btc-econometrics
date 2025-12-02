#!/usr/bin/env bash
set -uo pipefail

run_step() {
    echo "Running: $1"
    bash -c "$1" || echo "[ERROR] Step failed: $1"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

run_step "python src/get_etf_flows.py"
run_step "python src/get_etf_prices.py"
run_step "python src/get_etf_nav.py"
run_step "python src/build_etf_panel.py"
run_step "python src/get_btc_1m_binance.py"
run_step "python src/build_btc_windows.py"
run_step "python src/build_macro.py"
run_step "python src/build_master_panel.py"
