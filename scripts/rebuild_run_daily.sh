#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p scripts logs

# Backup existing run_daily.sh (if present)
if [ -f scripts/run_daily.sh ]; then
  ts="$(date +%F_%H%M%S)"
  cp scripts/run_daily.sh "scripts/run_daily.sh.bak.${ts}"
  echo "[OK] Backed up scripts/run_daily.sh -> scripts/run_daily.sh.bak.${ts}"
fi

cat > scripts/run_daily.sh <<'SH'
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
TRADE_DATE="$(docker compose run --rm report python - <<'PY'
from src.common.timeutil import last_completed_trading_day_et
print(last_completed_trading_day_et())
PY
)"
TRADE_DATE="$(echo "$TRADE_DATE" | tr -d '\r' | tail -n 1)"

echo "[$(date -Is)] Trade date (ET) = $TRADE_DATE" | tee -a "$LOG_FILE"

# Tunables
UNIVERSE_LIMIT="${UNIVERSE_LIMIT:-200}"
FEED="${FEED:-iex}"

# 1) Universe snapshot (must align to trade date)
run_step "Universe fetch" docker compose run --rm universe_fetch \
  --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT"

# 2) EOD bars for the same trade date (range mode avoids 'today' issues)
run_step "EOD fetch (trade_date)" docker compose run --rm eod_fetch \
  --use-universe --limit "$UNIVERSE_LIMIT" \
  --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
  --feed "$FEED"

# 3) Hourly bars for the same trade date
run_step "Hourly fetch (trade_date)" docker compose run --rm hourly_fetch \
  --use-universe --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT" \
  --mode range --start "$TRADE_DATE" --end "$TRADE_DATE" \
  --feed "$FEED"

# 4) Generate signals for that trade date
run_step "Generate signals" docker compose run --rm generate_signals \
  --date "$TRADE_DATE" --limit "$UNIVERSE_LIMIT"

# 5) Report + Email pinned to report date
run_step "Report" docker compose run --rm -e REPORT_DATE="$TRADE_DATE" report
run_step "Email"  docker compose run --rm -e REPORT_DATE="$TRADE_DATE" email

echo "[$(date -Is)] Daily run completed successfully." | tee -a "$LOG_FILE"
SH

chmod +x scripts/run_daily.sh
echo "[OK] Wrote scripts/run_daily.sh"
