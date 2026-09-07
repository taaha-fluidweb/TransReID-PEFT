#!/usr/bin/env bash
# Wrapper for run.py — unzip, setup, and train in one command.
# Usage (from directory containing uploaded zips, or from repo root):
#   bash run.sh
#   bash run.sh --config configs/Market/ssf_0_11_case1.yml
set -euo pipefail
cd "$(dirname "$0")"
python run.py "$@"
