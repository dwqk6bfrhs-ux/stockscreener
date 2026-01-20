Stock Screener (Universe → Data → Signals → Report → Email)

A personal stock screening and signal generation pipeline driven by daily universe snapshots, Alpaca OHLCV data (daily + hourly), YAML-defined strategies, and a report/email output suitable for daily use and future LLM-assisted review.

Goals

Build a repeatable pipeline you can run daily (and later multiple times/day).

Store data in a local DB (SQLite) with deterministic, reproducible snapshots:

Universe snapshot by ET trade date

Daily OHLCV bars

Hourly OHLCV bars

Signals & strategy metadata

Produce ranking-oriented outputs (action/watch lists) that are both:

Human-readable

LLM-ready for future strategy review and refinement

High-level flow

Daily run (pinned to ET trade date)

universe_fetch

eod_fetch (daily bars)

hourly_fetch (hourly bars)

generate_signals (strategies from YAML)

report (rank + coverage gating + CSVs)

email (send action/watch lists)

Outputs are written to:

outputs/<REPORT_DATE>/
  action_list.csv
  watch_list.csv
  summary.txt

Repository layout (typical)
configs/
  strategies.yaml
scripts/
  run_daily.sh
src/
  common/
    db.py
    timeutil.py
  jobs/
    universe_fetch.py
    eod_fetch.py
    hourly_fetch.py
    generate_signals.py
    report.py
    send_email.py
  strategy/
    ma_cross.py
    retest_shrink.py
data/
  app.db
outputs/
logs/
docker-compose.yml
.env

Prerequisites

Docker + Docker Compose v2

Alpaca account + API keys

SMTP credentials (for sending emails)

Configuration
.env (example)

Create a .env in the repo root:

# --- Alpaca ---
ALPACA_API_KEY=YOUR_KEY
ALPACA_SECRET_KEY=YOUR_SECRET
ALPACA_PAPER=true

# Canonical data config (recommended)
ALPACA_DATA_FEED=sip          # prefer sip if entitled; fallback: iex
ALPACA_ADJUSTMENT=raw         # keep raw for consistent analytics

# --- App paths (defaults shown) ---
DB_PATH=/app/data/app.db
TICKERS_PATH=/app/tickers.txt

# --- Strategy config ---
STRATEGIES_CONFIG=/app/configs/strategies.yaml

# --- Email ---
EMAIL_FROM=you@example.com
EMAIL_TO=you@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@example.com
SMTP_PASS=your_app_password


Notes:

If feed=sip returns 403, your account may not be entitled for SIP on the data endpoint. Set ALPACA_DATA_FEED=iex (or implement a SIP→IEX fallback).

ALPACA_ADJUSTMENT=raw is recommended for deterministic “what the market printed” behavior.

Docker Compose services

Your docker-compose.yml should expose these jobs as services:

universe_fetch

eod_fetch

hourly_fetch

generate_signals

report

email

(backtest_runner)

(optional) backtest, fred_fetch, finra_fetch

Example pattern:

Use entrypoint: ["python","-m","src.jobs.<job>"] for job services.

Mount ./data to /app/data, ./outputs to /app/outputs, ./logs to /app/logs.

Daily run script

scripts/run_daily.sh drives the whole pipeline. Key behaviors:

Resolves TRADE_DATE in America/New_York time.

Runs all jobs with REPORT_DATE pinned to that same date.

Uses consistent feed + adjustment for both daily & hourly pulls.
Honors ENABLE_HOURLY=0 to skip hourly fetches on full-universe runs.

Typical run:

chmod +x scripts/run_daily.sh
./scripts/run_daily.sh

To skip hourly bars for full-universe runs:

ENABLE_HOURLY=0 ./scripts/run_daily.sh

Important: trade date correctness (holidays)

Your “last completed trading day” must be holiday-aware.
If your helper function only skips weekends, it will incorrectly treat holidays as trade dates and can lead to failed data pulls (or empty/incomplete data).

Recommended approach:

Implement last_completed_trading_day_et() using Alpaca market calendar (holiday-aware), and only return “today” after market close + a grace window.

Universe snapshots
What is universe_daily?

universe_daily stores a snapshot of tradable symbols for a given ET date and source.

It can change over time due to listings, delistings, symbol changes, eligibility, and exchange/tradability changes.

You should treat it as a daily snapshot rather than “one permanent list.”

Limiting the universe

For development, you can cap with --limit (e.g., 200).
For full-market scans, remove the limit (or increase to several thousand) once your compute and data plan support it.

Replacing a snapshot

If you already have rows for (date, source) and want a clean refresh, use:

docker compose run --rm universe_fetch --date 2026-01-16 --limit 200 --replace


This deletes existing rows for that (date, source) before inserting the new set.

Daily data (prices_daily)

eod_fetch pulls daily OHLCV bars from Alpaca and writes to prices_daily.

Typical invocation (pinned range for one trade date):

docker compose run --rm eod_fetch \
  --use-universe --limit 200 \
  --mode range --start 2026-01-16 --end 2026-01-16 \
  --feed sip --adjustment raw


What is pulled:

Daily bars: open/high/low/close/volume (OHLCV)

Only for tickers sourced from universe (if --use-universe) or from tickers.txt

What is not pulled:

Fundamentals (market cap, shares outstanding, etc.) — those require a separate provider.

Corporate actions beyond what --adjustment applies.

Hourly data (prices_hourly)

hourly_fetch pulls 1-hour OHLCV from Alpaca and writes to prices_hourly.

Pinned range example:

docker compose run --rm hourly_fetch \
  --use-universe --date 2026-01-16 --limit 200 \
  --mode range --start 2026-01-16 --end 2026-01-16 \
  --feed sip --adjustment raw


Notes:

Hourly bars are useful for intraday confirmation and richer ranking features.

Coverage can be incomplete depending on symbol eligibility, entitlement, and market activity.

Expect many symbols to have 0 hourly bars if they are non-standard tickers or unsupported by the endpoint; a coverage filter in report is recommended.

Strategies and signals (signals_daily)
Strategy contract

Each strategy module must export:

def evaluate(df: pd.DataFrame, params: dict) -> dict:
    return {
      "state": "ENTRY|WATCH|PASS|ERROR",
      "score": float | None,
      "stop": float | None,
      "meta": dict,
    }

Why normalize to ENTRY/WATCH/PASS/ERROR?

To avoid breaking downstream report/email, the adapter normalizes internal strategy states into a stable set:

ENTRY: actionable today

WATCH: keep an eye on it

PASS: nothing

ERROR: strategy failed for this ticker

Raw/internal states should be preserved in meta, e.g. meta["raw_state"].

YAML strategy config

configs/strategies.yaml defines which strategies run:

universe_source: alpaca
lookback_days: 240
strategies:
  - name: retest_shrink_v1
    module: src.strategy.retest_shrink
    params:
      atr_window: 14
      # ...
  - name: ma_cross_5_10
    module: src.strategy.ma_cross
    params:
      fast: 5
      slow: 10

Reporting and ranking

The report step should:

Load signals for REPORT_DATE

Apply data coverage gating so rankings are meaningful

Compute rank features (trend/liquidity/risk, etc.)

Output action_list.csv, watch_list.csv, summary.txt

Coverage gating (recommended)

Example policy:

Require dollar_vol_20 present for liquidity-aware ranking

Require ≥ N bars of history for the strategy’s indicators (e.g., at least 60–240 daily bars depending on features)

For hourly features, require ≥ N hourly bars for the day (e.g., ≥ 4)

This prevents false rankings on symbols with insufficient or missing data.

Backfilling

To seed enough history for:

ATR windows

MA(200)

rolling dollar volume

robust scoring

You will generally want to backfill daily bars over a longer range (e.g., 1–2 years).

Example:

docker compose run --rm eod_fetch \
  --use-universe --limit 200 \
  --mode range --start 2024-01-01 --end 2026-01-16 \
  --feed sip --adjustment raw

Backtest runner (deterministic replay)

The backtest runner replays daily signals -> report -> orders, then simulates fills at close:

docker compose run --rm backtest_runner \
  --start 2024-01-01 --end 2024-12-31 \
  --book combined --reset-book

Outputs:

outputs/backtests/<run_id>/
  backtest_equity.csv
  backtest_summary.csv
  backtest_trades.csv


If you plan to “always backfill when data is not enough,” implement a simple check:

If a ticker has fewer than required rows for the strategy window, fetch more history automatically (or run a scheduled weekly backfill job).

Roadmap (next milestones)

Stabilize trade date + canonical data config

Holiday-aware trade date function

Consistent feed/adjustment across daily/hourly

Ranking feature engineering (report-side first)

Coverage gating

Strategy-specific rank columns (trend/liquidity/risk)

Output top-N per strategy and combined

Strategy meta improvements (without changing logic)

Keep D0/retest/confirm logic fixed

Expand meta and scoring only

Ensure adapter normalization remains stable

Hourly-informed ranking

Add intraday features for MA strategies

Improve scoring/ranking for MA cross + retest shrink

LLM-ready datasets

Build a compact “ticker dossier” payload (signals + rank features + key windows)

Store in DB or export JSON for RAG later

Scaling for full-market scans

Increase universe size

Schedule runs (cron / Cloud Scheduler)

Consider migrating DB to Postgres or BigQuery for analytics/ML
