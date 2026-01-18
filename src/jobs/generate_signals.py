import os
import json
import argparse
import yaml
import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.strategy.features import add_basic_features
from src.strategy.retest_shrink import Params, evaluate_ticker

log = setup_logger("generate_signals")


def read_prices() -> pd.DataFrame:
  with connect() as conn:
    df = pd.read_sql_query(
      "SELECT ticker, date, open, high, low, close, volume FROM prices_daily",
      conn,
    )
  df["date"] = pd.to_datetime(df["date"])
  return df


def load_cfg(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def build_params(cfg: dict) -> Params:
  return Params(
    min_close=float(cfg["universe"]["min_close"]),
    min_adv20_dollars=float(cfg["universe"]["min_adv20_dollars"]),
    min_history_days=int(cfg["universe"]["min_history_days"]),
    exclude_tickers=set(cfg["universe"].get("exclude_tickers", [])),
    vol_pct_min=float(cfg["pressure_test"]["vol_pct_min"]),
    down_atr_min=float(cfg["pressure_test"]["down_atr_min"]),
    range_atr_min=float(cfg["pressure_test"]["range_atr_min"]),
    nft_window_days=int(cfg["no_follow_through"]["window_days"]),
    nft_undercut_atr_max=float(cfg["no_follow_through"]["undercut_atr_max"]),
    nft_vol_max_mult=float(cfg["no_follow_through"]["vol_max_mult"]),
    retest_window_days=int(cfg["retest"]["window_days"]),
    retest_zone_atr=float(cfg["retest"]["zone_atr"]),
    retest_shrink_max=float(cfg["retest"]["shrink_max"]),
    retest_undercut_atr_max=float(cfg["retest"]["undercut_atr_max"]),
    confirm_window_days=int(cfg["confirm"]["window_days"]),
    confirm_close_strength=bool(cfg["confirm"]["close_strength"]),
    confirm_vol_max_mult=float(cfg["confirm"]["vol_max_mult"]),
    stop_atr=float(cfg["risk"]["stop_atr"]),
  )


def upsert_signals(rows: list[tuple]):
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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", default=os.environ.get("STRATEGY_CONFIG", "/app/configs/retest_shrink.yaml"))
  ap.add_argument("--strategy", default="retest_shrink")
  ap.add_argument("--date", default=None, help="Single date YYYY-MM-DD (ET date in your DB)")
  ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
  ap.add_argument("--end", default=None, help="End date YYYY-MM-DD")
  args = ap.parse_args()

  if not args.date and not (args.start and args.end):
    raise SystemExit("Provide either --date YYYY-MM-DD or --start YYYY-MM-DD --end YYYY-MM-DD")

  init_db()
  cfg = load_cfg(args.config)
  params = build_params(cfg)

  df = read_prices()
  if df.empty:
    raise RuntimeError("No prices in DB. Run eod_fetch first.")

  # Filter date range
  if args.date:
    start = pd.to_datetime(args.date)
    end = pd.to_datetime(args.date)
  else:
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

  df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
  if df.empty:
    raise RuntimeError("No prices in selected range.")

  # Add required features
  df = add_basic_features(
    df,
    atr_n=int(cfg["lookbacks"]["atr"]),
    pct_window=int(cfg["lookbacks"]["pct"]),
    adv_n=int(cfg["lookbacks"]["adv"]),
  )

  # Evaluate per day, per ticker (using history up to that day)
  dates = sorted(df["date"].unique())
  all_prices = read_prices()  # full history for "history up to date"
  all_prices = add_basic_features(
    all_prices,
    atr_n=int(cfg["lookbacks"]["atr"]),
    pct_window=int(cfg["lookbacks"]["pct"]),
    adv_n=int(cfg["lookbacks"]["adv"]),
  )
  all_prices = all_prices.sort_values(["ticker", "date"])

  rows = []
  total = 0
  for d in dates:
    day_rows = []
    for t, g in all_prices.groupby("ticker"):
      sub = g[g["date"] <= d]
      if sub.empty:
        continue
      r = evaluate_ticker(sub, params)
      if not r:
        continue

      state = str(r.get("state", "NEUTRAL"))
      score = r.get("score", None)
      stop = r.get("stop", None)
      meta = json.dumps(r, ensure_ascii=False)

      day_rows.append((
        pd.Timestamp(d).date().isoformat(),
        t,
        args.strategy,
        state,
        float(score) if score is not None else None,
        float(stop) if stop is not None else None,
        meta
      ))

    n = upsert_signals(day_rows)
    total += n
    log.info(f"{pd.Timestamp(d).date().isoformat()} upserted signals: {n}")

  log.info(f"Done. Total upserted rows: {total}")


if __name__ == "__main__":
  main()
