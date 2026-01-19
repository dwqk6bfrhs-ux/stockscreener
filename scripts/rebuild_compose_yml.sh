#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Backup existing compose file (if present)
if [ -f docker-compose.yml ]; then
  ts="$(date +%F_%H%M%S)"
  cp docker-compose.yml "docker-compose.yml.bak.${ts}"
  echo "[OK] Backed up docker-compose.yml -> docker-compose.yml.bak.${ts}"
fi

cat > docker-compose.yml <<'YAML'
services:
  universe_fetch:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    entrypoint: ["python", "-m", "src.jobs.universe_fetch"]

  eod_fetch:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      - ./tickers.txt:/app/tickers.txt:ro
    entrypoint: ["python", "-m", "src.jobs.eod_fetch"]

  hourly_fetch:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      - ./tickers.txt:/app/tickers.txt:ro
    entrypoint: ["python", "-m", "src.jobs.hourly_fetch"]

  generate_signals:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      - ./tickers.txt:/app/tickers.txt:ro
    entrypoint: ["python", "-m", "src.jobs.generate_signals"]

  report:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      - ./tickers.txt:/app/tickers.txt:ro
    # Keep this if your report code expects it; harmless otherwise
    environment:
      - STRATEGY_CONFIG=/app/configs/retest_shrink.yaml
    entrypoint: ["python", "-m", "src.jobs.report"]

  email:
    build: .
    env_file: .env
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    entrypoint: ["python", "-m", "src.jobs.send_email"]

  backtest:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      - ./tickers.txt:/app/tickers.txt:ro
    environment:
      - STRATEGY_CONFIG=/app/configs/retest_shrink.yaml
    entrypoint: ["python", "-m", "src.jobs.backtest"]

  fred_fetch:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      # Optional: OpenBB creds mount (only if you configure it)
      # - ./.openbb_platform:/root/.openbb_platform:ro
    entrypoint: ["python", "-m", "src.jobs.fred_fetch"]

  finra_fetch:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs:ro
      # Optional: OpenBB creds mount (only if you configure it)
      # - ./.openbb_platform:/root/.openbb_platform:ro
    entrypoint: ["python", "-m", "src.jobs.finra_fetch"]
YAML

echo "[OK] Wrote docker-compose.yml"

# Validate
docker compose config >/dev/null
echo "[OK] docker compose config validation passed"
