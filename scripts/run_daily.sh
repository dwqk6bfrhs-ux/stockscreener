#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: log to file (also prints to stdout)
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_daily_$(date +%F).log"

echo "[$(date -Is)] Starting daily run in $REPO_DIR" | tee -a "$LOG_FILE"

run_step () {
  local name="$1"
  shift
  echo "[$(date -Is)] >>> $name" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  echo "[$(date -Is)] <<< $name done" | tee -a "$LOG_FILE"
}

run_step "EOD fetch"  docker compose run --rm eod_fetch
run_step "Report"     docker compose run --rm report
run_step "Email"      docker compose run --rm email

echo "[$(date -Is)] Daily run completed successfully." | tee -a "$LOG_FILE"
