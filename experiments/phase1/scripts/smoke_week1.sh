#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../venv/bin/python tests/smoke_strategy_start.py
