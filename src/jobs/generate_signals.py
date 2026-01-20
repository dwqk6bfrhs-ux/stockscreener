from __future__ import annotations

import argparse
import importlib
import json
import os
import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("generate_signals")


def load_yaml(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def read_universe(date_str: str, source: str) -> list[str]:
  with connect() as conn:
    rows = conn.execute(
      "SELECT ticker FROM universe_daily WHERE date=? AND source=? ORDER BY ticker",
      (date_str, source),
    ).fetchall()
  return [r[0] for r in rows]


def read_tickers_file(path: str) -> list[str]:
  tickers = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      t = line.strip()
      if not t or t.startswith("#"):
        continue
      tickers.append(t.upper())
  return tickers


def read_prices_tickers(start_date: str, end_date: str) -> list[str]:
  with connect() as conn:
    rows = conn.execute(
      "SELECT DISTINCT ticker FROM prices_daily WHERE date BETWEEN ? AND ? ORDER BY ticker",
      (start_date, end_date),
    ).fetchall()
  return [r[0] for r in rows]


def read_prices_daily_window(end_date: str, lookback_days: int) -> pd.DataFrame:
  end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - dt.timedelta(days=lookback_days)).isoformat()
  with connect() as conn:
    q = """
      SELECT ticker, date, open, high, low, close, volume
      FROM prices_daily
      WHERE date BETWEEN ? AND ?
    """
    df = pd.read_sql_query(q, conn, params=(start, end_date))
  if df.empty:
    return df
  df["date"] = pd.to_datetime(df["date"])
  return df


def read_prices_hourly_window(end_date_et: str, lookback_days: int) -> pd.DataFrame:
  """
  Reads hourly bars by ET date bucket: prices_hourly(date_et).
  lookback_days is calendar days.
  """
  end = dt.datetime.strptime(end_date_et, "%Y-%m-%d").date()
  start = (end - dt.timedelta(days=lookback_days)).isoformat()
  with connect() as conn:
    q = """
      SELECT ticker, ts, date_et, open, high, low, close, volume
      FROM prices_hourly
      WHERE date_et BETWEEN ? AND ?
    """
    df = pd.read_sql_query(q, conn, params=(start, end_date_et))
  if df.empty:
    return df
  df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
  return df


def upsert_signals(rows: list[tuple]) -> int:
  if not rows:
    return 0
  sql = """
    INSERT INTO signals_daily (date, ticker, strategy, state, score, stop, meta_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, ticker, strategy) DO UPDATE SET
      state=excluded.state,
      score=excluded.score,
      stop=excluded.stop,
      meta_json=excluded.meta_json
  """
  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()
  return len(rows)


def load_strategy(module_path: str):
  return importlib.import_module(module_path)


def call_evaluate(mod, dft: pd.DataFrame, params: dict, ctx: dict):
  """
  Supports:
    evaluate(df, params=params)
    evaluate(df, params=params, ctx=ctx)
  """
  if not hasattr(mod, "evaluate"):
    raise RuntimeError(f"{mod.__name__} missing evaluate()")

  fn = getattr(mod, "evaluate")
  try:
    return fn(dft, params=params, ctx=ctx)
  except TypeError:
    return fn(dft, params=params)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default last completed trading day)")
  ap.add_argument("--config", default=None, help="Strategies YAML path (default env STRATEGIES_CONFIG)")
  ap.add_argument("--limit", type=int, default=None, help="Limit tickers for testing")
  ap.add_argument("--only", action="append", default=None, help="Run only these strategy names (repeatable)")
  ap.add_argument(
    "--tickers-source",
    choices=["universe", "tickers", "prices"],
    default=os.environ.get("SIGNALS_TICKERS_SOURCE", "universe"),
    help="Where to source tickers: universe | tickers | prices (default: universe)",
  )
  ap.add_argument(
    "--tickers-path",
    default=os.environ.get("TICKERS_PATH", "/app/tickers.txt"),
    help="Path to tickers.txt (used when tickers-source=tickers)",
  )
  args = ap.parse_args()

  date_str = args.date or last_completed_trading_day_et()
  config_path = args.config or os.environ.get("STRATEGIES_CONFIG", "/app/configs/strategies.yaml")

  init_db()

  cfg = load_yaml(config_path)
  universe_source = cfg.get("universe_source", "alpaca")

  daily_lookback_days = int(cfg.get("daily_lookback_days", cfg.get("lookback_days", 240)))
  hourly_lookback_days = int(cfg.get("hourly_lookback_days", 90))

  strategies = cfg.get("strategies", []) or []
  if not strategies:
    raise RuntimeError(f"No strategies found in {config_path}")

  df_daily_all = read_prices_daily_window(end_date=date_str, lookback_days=daily_lookback_days)
  if df_daily_all.empty:
    raise RuntimeError("No daily prices found in DB for requested window. Run eod_fetch/backfill first.")

  if args.tickers_source == "universe":
    tickers = read_universe(date_str, universe_source)
    if not tickers:
      raise RuntimeError(f"No universe tickers for date={date_str} source={universe_source}. Run universe_fetch first.")
  elif args.tickers_source == "tickers":
    tickers = read_tickers_file(args.tickers_path)
  else:
    window_start = (dt.datetime.strptime(date_str, "%Y-%m-%d").date() - dt.timedelta(days=daily_lookback_days)).isoformat()
    tickers = read_prices_tickers(window_start, date_str)

  if not tickers:
    raise RuntimeError("No tickers resolved for signal generation. Check tickers source or data availability.")
  if args.limit:
    tickers = tickers[:args.limit]

  df_daily_all = df_daily_all[df_daily_all["ticker"].isin(tickers)].copy()

  df_hourly_all = read_prices_hourly_window(end_date_et=date_str, lookback_days=hourly_lookback_days)
  if not df_hourly_all.empty:
    df_hourly_all = df_hourly_all[df_hourly_all["ticker"].isin(tickers)].copy()

  wanted = set(args.only or [])
  rows: list[tuple] = []

  for s in strategies:
    name = s["name"]
    if wanted and name not in wanted:
      continue

    module_path = s["module"]
    params = s.get("params", {}) or {}

    mod = load_strategy(module_path)

    ok = 0
    fail = 0

    for t in tickers:
      dft = df_daily_all[df_daily_all["ticker"] == t].sort_values("date")
      if dft.empty:
        continue

      dft_h = None
      if not df_hourly_all.empty:
        dft_h = df_hourly_all[df_hourly_all["ticker"] == t].sort_values("ts")

      ctx = {
        "trade_date": date_str,
        "ticker": t,
        "hourly": dft_h,  # can be None
      }

      try:
        res = call_evaluate(mod, dft, params=params, ctx=ctx) or {}
        state = str(res.get("state", "PASS"))
        score = res.get("score", None)
        stop = res.get("stop", None)
        meta = res.get("meta", {}) or {}

        rows.append((
          date_str,
          t,
          name,
          state,
          float(score) if score is not None else None,
          float(stop) if stop is not None else None,
          json.dumps(meta, ensure_ascii=False),
        ))
        ok += 1
      except Exception as e:
        rows.append((date_str, t, name, "ERROR", None, None, json.dumps({"error": str(e)}, ensure_ascii=False)))
        fail += 1

    log.info(f"Strategy done: {name} module={module_path} tickers={len(tickers)} ok={ok} fail={fail}")

  n = upsert_signals(rows)
  log.info(f"Signals upserted: date={date_str} rows={n} (tickers={len(tickers)} strategies={len(strategies)})")


if __name__ == "__main__":
  main()
