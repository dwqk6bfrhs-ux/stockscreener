#!/usr/bin/env bash
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

# Determine ET trade date (last completed trading day) inside container env
TRADE_DATE="$(docker compose run --rm report python - <<'PY'
from src.common.timeutil import last_completed_trading_day_et
print(last_completed_trading_day_et())
PY
)"

echo "[$(date -Is)] Trade date (ET) = $TRADE_DATE" | tee -a "$LOG_FILE"

# Tunables
UNIVERSE_LIMIT="${UNIVERSE_LIMIT:-200}"
FEED="${FEED:-sip}"

run_step "Universe fetch" docker compose run --rm universe_fetch --limit "$UNIVERSE_LIMIT"

run_step "EOD fetch (daily)" docker compose run --rm eod_fetch \
  --use-universe --limit "$UNIVERSE_LIMIT" \
  --mode daily --feed "$FEED"

run_step "Generate signals" docker compose run --rm generate_signals \
  --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT"

run_step "Report" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" report

run_step "Email" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" email

echo "[$(date -Is)] Daily run completed successfully." | tee -a "$LOG_FILE"
