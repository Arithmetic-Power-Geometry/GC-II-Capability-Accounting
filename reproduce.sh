#!/usr/bin/env bash
set -euo pipefail
python scripts/reproduce_all.py
pytest -q
