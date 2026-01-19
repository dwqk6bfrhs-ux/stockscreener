import os
import json
import argparse
import importlib
import pandas as pd
import yaml

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("generate_signals")


def load_yaml(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def read_universe(date_str: str, source: str) -> list[str]:
  with connect() as conn:
    rows = conn.execute(
      "SELECT ticker FROM universe_daily WHERE date=? AND source=? ORDER BY ticker",
      (date_str, source),
    ).fetchall()
  return [r[0] for r in rows]


def read_prices_window(end_date: str, lookback_days: int) -> pd.DataFrame:
  import datetime
  end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - datetime.timedelta(days=lookback_days)).isoformat()

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
  mod = importlib.import_module(module_path)
  if not hasattr(mod, "evaluate"):
    raise RuntimeError(f"{module_path} missing evaluate(df, params)")
  return mod


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default last completed trading day)")
  ap.add_argument("--config", default=None, help="Strategies YAML path (default env STRATEGIES_CONFIG)")
  ap.add_argument("--limit", type=int, default=None, help="Limit tickers for testing")
  ap.add_argument("--only", action="append", default=None, help="Run only these strategy names (repeatable)")
  args = ap.parse_args()

  date_str = args.date or last_completed_trading_day_et()
  config_path = args.config or os.environ.get("STRATEGIES_CONFIG", "/app/configs/strategies.yaml")

  init_db()

  cfg = load_yaml(config_path)
  universe_source = cfg.get("universe_source", "alpaca")
  lookback_days = int(cfg.get("lookback_days", 240))
  strategies = cfg.get("strategies", [])

  if not strategies:
    raise RuntimeError(f"No strategies found in {config_path}")

  tickers = read_universe(date_str, universe_source)
  if not tickers:
    raise RuntimeError(f"No universe tickers for date={date_str} source={universe_source}. Run universe_fetch first.")
  if args.limit:
    tickers = tickers[:args.limit]

  df_all = read_prices_window(end_date=date_str, lookback_days=lookback_days)
  if df_all.empty:
    raise RuntimeError("No prices found in DB for requested window. Run eod_fetch/backfill first.")

  # Filter to tickers in universe for speed
  df_all = df_all[df_all["ticker"].isin(tickers)].copy()

  wanted = set([s for s in (args.only or [])])
  rows: list[tuple] = []

  for s in strategies:
    name = s["name"]
    if wanted and name not in wanted:
      continue

    module = s["module"]
    params = s.get("params", {}) or {}

    mod = load_strategy(module)

    ok = 0
    fail = 0

    for t in tickers:
      dft = df_all[df_all["ticker"] == t].sort_values("date")
      if dft.empty:
        continue
      try:
        res = mod.evaluate(dft, params=params)
        state = str(res.get("state", "PASS"))
        score = res.get("score", None)
        stop = res.get("stop", None)
        meta = res.get("meta", {})

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

    log.info(f"Strategy done: {name} module={module} tickers={len(tickers)} ok={ok} fail={fail}")

  n = upsert_signals(rows)
  log.info(f"Signals upserted: date={date_str} rows={n} (tickers={len(tickers)} strategies={len(strategies)})")


if __name__ == "__main__":
  main()
