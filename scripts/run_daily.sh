#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Log setup
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

# Determine ET trade date (last completed trading day) inside container env
# IMPORTANT: because report service has an ENTRYPOINT now, we must override it to run ad-hoc python.
TRADE_DATE="$(
  docker compose run --rm --entrypoint python report -c \
    "from src.common.timeutil import last_completed_trading_day_et; print(last_completed_trading_day_et())" \
    2>/dev/null | tr -d '\r' | grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' | tail -n 1
)"

if [[ ! "$TRADE_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[$(date -Is)] ERROR: Failed to resolve TRADE_DATE. Got: '$TRADE_DATE'" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date -Is)] Trade date (ET) = $TRADE_DATE" | tee -a "$LOG_FILE"

# Tunables (canonical data config)
UNIVERSE_LIMIT="${UNIVERSE_LIMIT:-200}"
FEED="${FEED:-sip}"
ADJ="${ADJ:-raw}"

# 1) Universe snapshot pinned to trade date (replace to avoid stale leftovers)
run_step "Universe fetch" docker compose run --rm universe_fetch \
  --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT" --replace

# 2) Daily bars for that trade date (range mode for deterministic pinning)
run_step "EOD fetch (trade_date)" docker compose run --rm eod_fetch \
  --use-universe --universe-date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT" \
  --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
  --feed "$FEED" --adjustment "$ADJ"

# 3) Hourly bars for that trade date
run_step "Hourly fetch (trade_date)" docker compose run --rm hourly_fetch \
  --use-universe --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT" \
  --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
  --feed "$FEED" --adjustment "$ADJ"

# 4) Generate signals for that trade date
run_step "Generate signals" docker compose run --rm generate_signals \
  --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT"

# 5) Report + Email pinned to report date
run_step "Report" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" report
run_step "Email"  docker compose run --rm -e REPORT_DATE="$TRADE_DATE" email

echo "[$(date -Is)] Daily run completed successfully." | tee -a "$LOG_FILE"
