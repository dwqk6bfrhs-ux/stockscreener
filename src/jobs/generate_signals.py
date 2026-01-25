from __future__ import annotations

import argparse
import importlib
import json
import os
import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from src.common.db import init_db, connect, get_prices_daily_source
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


def read_prices_tickers(start_date: str, end_date: str, source: str) -> list[str]:
  with connect() as conn:
    rows = conn.execute(
      "SELECT DISTINCT ticker FROM prices_daily WHERE source=? AND date BETWEEN ? AND ? ORDER BY ticker",
      (source, start_date, end_date),
    ).fetchall()
  return [r[0] for r in rows]


def read_prices_daily_window(end_date: str, lookback_days: int, source: str) -> pd.DataFrame:
  end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - dt.timedelta(days=lookback_days)).isoformat()
  with connect() as conn:
    q = """
      SELECT ticker, date, open, high, low, close, volume
      FROM prices_daily
      WHERE source=? AND date BETWEEN ? AND ?
    """
    df = pd.read_sql_query(q, conn, params=(source, start, end_date))
  if df.empty:
    return df
  df["date"] = pd.to_datetime(df["date"])
  return df


def previous_trading_date(date_str: str, source: str) -> str:
  with connect() as conn:
    row = conn.execute(
      "SELECT MAX(date) FROM prices_daily WHERE source=? AND date < ?",
      (source, date_str),
    ).fetchone()
  if not row or row[0] is None:
    raise RuntimeError(f"No prior trading date found before {date_str} for source={source}.")
  return str(row[0])


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


def _liquidity_filter(
  df_daily: pd.DataFrame,
  min_avg_volume: Optional[float],
  min_adv20_dollars: Optional[float],
) -> list[str]:
  if df_daily.empty:
    return []

  vol_floor = float(min_avg_volume) if min_avg_volume is not None else None
  dv20_floor = float(min_adv20_dollars) if min_adv20_dollars is not None else None
  if vol_floor is None and dv20_floor is None:
    return sorted(df_daily["ticker"].unique().tolist())

  df = df_daily.copy()
  df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
  df["close"] = pd.to_numeric(df["close"], errors="coerce")
  df["dollar_vol"] = df["close"].abs() * df["volume"].abs()

  grouped = df.sort_values("date").groupby("ticker", sort=False)
  rows = []
  for ticker, g in grouped:
    g_tail = g.tail(20)
    if len(g_tail) < 20:
      avg_vol = None
      avg_dv = None
    else:
      avg_vol = g_tail["volume"].mean()
      avg_dv = g_tail["dollar_vol"].mean()
    rows.append({"ticker": ticker, "avg_vol": avg_vol, "avg_dv": avg_dv})

  df_liq = pd.DataFrame(rows)
  if df_liq.empty:
    return []

  mask = pd.Series(True, index=df_liq.index)
  if vol_floor is not None:
    mask &= df_liq["avg_vol"].fillna(0.0) >= vol_floor
  if dv20_floor is not None:
    mask &= df_liq["avg_dv"].fillna(0.0) >= dv20_floor
  return df_liq.loc[mask, "ticker"].astype(str).tolist()


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
  ap.add_argument(
    "--no-lookahead",
    action="store_true",
    help="Compute signals using data up to the prior trading day (no same-day lookahead).",
  )
  ap.add_argument(
    "--min-avg-volume",
    type=float,
    default=None,
    help="Minimum 20-day average share volume required to keep a ticker.",
  )
  ap.add_argument(
    "--min-adv20-dollars",
    type=float,
    default=None,
    help="Minimum 20-day average dollar volume (close*volume) required to keep a ticker.",
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

  price_source = get_prices_daily_source()
  data_end_date = previous_trading_date(date_str, price_source) if args.no_lookahead else date_str
  df_daily_all = read_prices_daily_window(end_date=data_end_date, lookback_days=daily_lookback_days, source=price_source)
  if df_daily_all.empty:
    raise RuntimeError("No daily prices found in DB for requested window. Run eod_fetch/backfill first.")

  if args.tickers_source == "universe":
    tickers = read_universe(date_str, universe_source)
    if not tickers:
      raise RuntimeError(
        f"No universe tickers for date={date_str} source={universe_source}. Run universe_fetch first."
      )
  elif args.tickers_source == "tickers":
    tickers = read_tickers_file(args.tickers_path)
  else:
    window_start = (dt.datetime.strptime(data_end_date, "%Y-%m-%d").date() - dt.timedelta(days=daily_lookback_days)).isoformat()
    tickers = read_prices_tickers(window_start, data_end_date, price_source)

  if not tickers:
    raise RuntimeError("No tickers resolved for signal generation. Check tickers source or data availability.")
  if args.limit:
    tickers = tickers[:args.limit]

  df_daily_all = df_daily_all[df_daily_all["ticker"].isin(tickers)].copy()
  if args.min_avg_volume is not None or args.min_adv20_dollars is not None:
    allowed = set(_liquidity_filter(
      df_daily_all,
      min_avg_volume=args.min_avg_volume,
      min_adv20_dollars=args.min_adv20_dollars,
    ))
    tickers = [t for t in tickers if t in allowed]
    df_daily_all = df_daily_all[df_daily_all["ticker"].isin(tickers)].copy()
    if not tickers:
      raise RuntimeError("No tickers left after applying liquidity filters.")

  df_hourly_all = read_prices_hourly_window(end_date_et=data_end_date, lookback_days=hourly_lookback_days)
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
