#!/usr/bin/env bash
set -euo pipefail

python -m src.jobs.export_ml_features "$@"
