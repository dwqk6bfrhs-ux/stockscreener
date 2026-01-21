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

# If REPORT_DATE is provided (e.g. 01-16-2026), normalize it to YYYY-MM-DD and use it.
# Otherwise, compute last completed trading day ET.
if [[ -n "${REPORT_DATE:-}" ]]; then
  TRADE_DATE="$(
    docker compose run --rm --entrypoint python report -c \
      "from datetime import datetime; s='${REPORT_DATE}'; \
try: print(datetime.strptime(s,'%Y-%m-%d').strftime('%Y-%m-%d')); \
except: print(datetime.strptime(s,'%m-%d-%Y').strftime('%Y-%m-%d'))" \
      2>/dev/null | tr -d '\r' | grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' | tail -n 1
  )"
else
  TRADE_DATE="$(
    docker compose run --rm --entrypoint python report -c \
      "from src.common.timeutil import last_completed_trading_day_et; print(last_completed_trading_day_et())" \
      2>/dev/null | tr -d '\r' | grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' | tail -n 1
  )"
fi

if [[ ! "$TRADE_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[$(date -Is)] ERROR: Failed to resolve TRADE_DATE. Got: '$TRADE_DATE' (REPORT_DATE='${REPORT_DATE:-}') " | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date -Is)] Trade date (ET) = $TRADE_DATE" | tee -a "$LOG_FILE"

# Tunables (canonical data config)
UNIVERSE_LIMIT="${UNIVERSE_LIMIT:-200}"
FEED="${FEED:-sip}"
ADJ="${ADJ:-raw}"
ENABLE_HOURLY="${ENABLE_HOURLY:-1}"

# Optional --limit args
LIMIT_ARG=()
if [[ -n "${UNIVERSE_LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$UNIVERSE_LIMIT")
fi

# 1) Universe snapshot pinned to trade date
run_step "Universe fetch" docker compose run --rm universe_fetch \
  --date "$TRADE_DATE" --replace \
  "${LIMIT_ARG[@]}"

# 2) Daily bars for that trade date
run_step "EOD fetch (trade_date)" docker compose run --rm eod_fetch \
  --use-universe --universe-date "$TRADE_DATE" \
  "${LIMIT_ARG[@]}" \
  --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
  --feed "$FEED" --adjustment "$ADJ"

# 3) Hourly bars for that trade date (optional)
if [[ "$ENABLE_HOURLY" == "1" ]]; then
  run_step "Hourly fetch (trade_date)" docker compose run --rm hourly_fetch \
    --use-universe --date "$TRADE_DATE" \
    "${LIMIT_ARG[@]}" \
    --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
    --feed "$FEED" --adjustment "$ADJ"
else
  echo "[$(date -Is)] Skipping hourly fetch (ENABLE_HOURLY=$ENABLE_HOURLY)" | tee -a "$LOG_FILE"
fi

# 4) Generate signals
run_step "Generate signals" docker compose run --rm generate_signals \
  --date "$TRADE_DATE" \
  "${LIMIT_ARG[@]}"

# 5) Report (this writes rank_scores_daily in your current setup)
run_step "Report" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" report

# 6) Export dossier (LLM-ready JSONL)
run_step "Export dossier" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" export_dossier

# 7) Generate orders (consumes rank_scores_daily)
run_step "Generate orders" docker compose run --rm generate_orders --date "$TRADE_DATE" \
  --top-x 20 --max-entries 20 --overlap-bonus 0.25

# 8) Email
run_step "Email" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" email

echo "[$(date -Is)] Daily run completed successfully." | tee -a "$LOG_FILE"
