#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Starting ETF/BTC data pipeline via run_analysis.py"
python src/run_analysis.py --data-only
