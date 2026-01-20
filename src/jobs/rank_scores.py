from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et
from src.common.ranking import (
  build_daily_features,
  build_hourly_coverage,
  score_ma_cross_row,
  score_retest_shrink_row,
)

log = setup_logger("rank_scores")


def ensure_rank_tables() -> None:
  sql = """
  CREATE TABLE IF NOT EXISTS rank_scores_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    rank_score REAL,
    meta_json TEXT,
    updated_at TEXT,
    PRIMARY KEY (date, ticker, strategy)
  );
  """
  with connect() as conn:
    conn.execute(sql)
    conn.commit()


def upsert_rank_scores(df: pd.DataFrame) -> int:
  if df.empty:
    return 0

  rows = []
  now = datetime.utcnow().isoformat()
  for _, r in df.iterrows():
    rows.append((
      str(r["date"]),
      str(r["ticker"]),
      str(r["strategy"]),
      (float(r["rank_score"]) if pd.notna(r["rank_score"]) else None),
      json.dumps(r["meta"], ensure_ascii=False),
      now,
    ))

  sql = """
  INSERT INTO rank_scores_daily (date, ticker, strategy, rank_score, meta_json, updated_at)
  VALUES (?, ?, ?, ?, ?, ?)
  ON CONFLICT(date, ticker, strategy) DO UPDATE SET
    rank_score=excluded.rank_score,
    meta_json=excluded.meta_json,
    updated_at=excluded.updated_at
  """
  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()
  return len(rows)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default last completed trading day)")
  ap.add_argument("--min-hourly-bars", type=int, default=4, help="Coverage gate: require >=N hourly bars")
  ap.add_argument("--min-daily-bars-60", type=int, default=30, help="Coverage gate: require >=N daily bars in last 60d")
  args = ap.parse_args()

  date_str = args.date or last_completed_trading_day_et()

  init_db()
  ensure_rank_tables()

  # Load signals for the date
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT date, ticker, strategy, state, score, stop, meta_json
      FROM signals_daily
      WHERE date=?
      """,
      conn,
      params=(date_str,),
    )

  if df.empty:
    raise RuntimeError(f"No signals_daily rows for date={date_str}. Run generate_signals first.")

  tickers = sorted(df["ticker"].dropna().unique().tolist())

  # Build external features + coverage
  daily_feat = build_daily_features(end_date=date_str, lookback_days=260, tickers=tickers)
  hourly_cov = build_hourly_coverage(date_et=date_str, tickers=tickers)

  out = df.merge(daily_feat, on="ticker", how="left").merge(hourly_cov, on="ticker", how="left")
  out["hourly_bars"] = out["hourly_bars"].fillna(0).astype(int)
  out["daily_bars_60"] = out["daily_bars_60"].fillna(0).astype(int)

  ranks = []
  for _, r in out.iterrows():
    ticker = str(r["ticker"])
    strategy = str(r["strategy"])
    state = str(r["state"])

    # Coverage gating (keeps delisted / no-data names from polluting top lists)
    coverage_ok = (int(r["hourly_bars"]) >= args.min_hourly_bars) and (int(r["daily_bars_60"]) >= args.min_daily_bars_60)

    # Keep rank_score nullable when failing coverage; report/orders can filter easily
    meta: Dict[str, Any] = {
      "coverage": {
        "hourly_bars": int(r["hourly_bars"]),
        "daily_bars_60": int(r["daily_bars_60"]),
        "coverage_ok": bool(coverage_ok),
      },
      "state": state,
    }

    rank_score = None
    extra = {}

    # Strategy-specific ranking
    if strategy.startswith("ma_cross"):
      if coverage_ok:
        rank_score, extra = score_ma_cross_row(r)
    elif strategy.startswith("retest_shrink"):
      if coverage_ok:
        rank_score, extra = score_retest_shrink_row(r, r.get("score"))
    else:
      # fallback: just use signal score if present
      if coverage_ok and r.get("score") is not None:
        try:
          rank_score = float(r["score"])
          extra = {"fallback": "signal_score"}
        except Exception:
          rank_score = None

    meta["rank_terms"] = extra
    meta["price_features"] = {
      "close": (float(r["close"]) if pd.notna(r.get("close")) else None),
      "atr_pct": (float(r["atr_pct"]) if pd.notna(r.get("atr_pct")) else None),
      "dollar_vol_20": (float(r["dollar_vol_20"]) if pd.notna(r.get("dollar_vol_20")) else None),
      "ma5": (float(r["ma5"]) if pd.notna(r.get("ma5")) else None),
      "ma10": (float(r["ma10"]) if pd.notna(r.get("ma10")) else None),
      "ma20": (float(r["ma20"]) if pd.notna(r.get("ma20")) else None),
      "ma50": (float(r["ma50"]) if pd.notna(r.get("ma50")) else None),
      "ma200": (float(r["ma200"]) if pd.notna(r.get("ma200")) else None),
    }

    ranks.append({
      "date": date_str,
      "ticker": ticker,
      "strategy": strategy,
      "rank_score": rank_score,
      "meta": meta,
    })

  df_rank = pd.DataFrame(ranks)
  n = upsert_rank_scores(df_rank)
  log.info(f"Rank scores upserted: date={date_str} rows={n}")

  # quick summary
  usable = df_rank[df_rank["rank_score"].notna()]
  log.info(f"Rankable rows: {len(usable)} / {len(df_rank)}")
  if not usable.empty:
    top = usable.sort_values(["strategy", "rank_score"], ascending=[True, False]).groupby("strategy").head(3)
    for _, rr in top.iterrows():
      log.info(f"TOP {rr['strategy']}: {rr['ticker']} rank={rr['rank_score']:.4f}")


if __name__ == "__main__":
  main()
