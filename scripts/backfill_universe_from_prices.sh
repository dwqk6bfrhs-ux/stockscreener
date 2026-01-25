#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <start-date> <end-date> [source] [universe_fetch_args...]" >&2
  echo "Example: $0 2024-01-01 2024-03-31 alpaca --limit 200" >&2
  echo "Override fetch command with UNIVERSE_FETCH_CMD (default: docker compose run --rm universe_fetch)." >&2
  exit 1
fi

start_date="$1"
end_date="$2"
source="${3:-alpaca}"
shift 3 || shift 2

db_path="${DB_PATH:-data/app.db}"
fetch_cmd="${UNIVERSE_FETCH_CMD:-docker compose run --rm universe_fetch}"

sqlite3 "$db_path" \
  "SELECT DISTINCT date FROM prices_daily WHERE source='${source}' AND date BETWEEN '${start_date}' AND '${end_date}' ORDER BY date;" \
| while read -r d; do
    if [[ -n "$d" ]]; then
      ${fetch_cmd} --date "$d" --replace "$@"
    fi
  done
