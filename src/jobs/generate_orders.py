from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et
from src.common.ranking import build_daily_features

log = setup_logger("generate_orders")


def ensure_tables() -> None:
  sql_positions = """
  CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    status TEXT NOT NULL,              -- OPEN/CLOSED
    opened_date TEXT NOT NULL,
    closed_date TEXT,
    entry_price REAL,
    qty REAL,
    stop REAL,
    trailing_stop REAL,
    max_close REAL,
    meta_json TEXT,
    PRIMARY KEY (ticker, strategy)
  );
  """
  sql_orders = """
  CREATE TABLE IF NOT EXISTS orders_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,                -- BUY/SELL
    qty REAL,
    reason TEXT,
    meta_json TEXT,
    created_at TEXT,
    PRIMARY KEY (date, ticker, strategy, side)
  );
  """
  with connect() as conn:
    conn.execute(sql_positions)
    conn.execute(sql_orders)
    conn.commit()


def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
  if not s:
    return {}
  try:
    return json.loads(s)
  except Exception:
    return {}


def get_close_price(date_str: str, ticker: str) -> Optional[float]:
  with connect() as conn:
    row = conn.execute(
      "SELECT close FROM prices_daily WHERE date=? AND ticker=?",
      (date_str, ticker),
    ).fetchone()
  if not row or row[0] is None:
    return None
  try:
    return float(row[0])
  except Exception:
    return None


def upsert_order(date_str: str, ticker: str, strategy: str, side: str, qty: float, reason: str, meta: Dict[str, Any]) -> None:
  sql = """
  INSERT INTO orders_daily (date, ticker, strategy, side, qty, reason, meta_json, created_at)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(date, ticker, strategy, side) DO UPDATE SET
    qty=excluded.qty,
    reason=excluded.reason,
    meta_json=excluded.meta_json,
    created_at=excluded.created_at
  """
  now = datetime.utcnow().isoformat()
  with connect() as conn:
    conn.execute(sql, (
      date_str, ticker, strategy, side, float(qty), reason, json.dumps(meta, ensure_ascii=False), now
    ))
    conn.commit()


def get_open_positions() -> pd.DataFrame:
  with connect() as conn:
    df = pd.read_sql_query(
      "SELECT * FROM positions WHERE status='OPEN'",
      conn,
    )
  return df


def upsert_position(
  ticker: str,
  strategy: str,
  status: str,
  opened_date: str,
  entry_price: Optional[float],
  qty: float,
  stop: Optional[float],
  trailing_stop: Optional[float],
  max_close: Optional[float],
  meta: Dict[str, Any],
  closed_date: Optional[str] = None,
) -> None:
  sql = """
  INSERT INTO positions (ticker, strategy, status, opened_date, closed_date, entry_price, qty, stop, trailing_stop, max_close, meta_json)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(ticker, strategy) DO UPDATE SET
    status=excluded.status,
    opened_date=excluded.opened_date,
    closed_date=excluded.closed_date,
    entry_price=excluded.entry_price,
    qty=excluded.qty,
    stop=excluded.stop,
    trailing_stop=excluded.trailing_stop,
    max_close=excluded.max_close,
    meta_json=excluded.meta_json
  """
  with connect() as conn:
    conn.execute(sql, (
      ticker, strategy, status, opened_date, closed_date,
      (float(entry_price) if entry_price is not None else None),
      float(qty),
      (float(stop) if stop is not None else None),
      (float(trailing_stop) if trailing_stop is not None else None),
      (float(max_close) if max_close is not None else None),
      json.dumps(meta, ensure_ascii=False),
    ))
    conn.commit()


def select_entries(
  date_str: str,
  top_x: int,
  max_entries: int,
  overlap_bonus: float,
  require_overlap: bool,
) -> List[Dict[str, Any]]:
  """
  Uses rank_scores_daily + signals_daily ENTRY to pick candidates.
  """
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT r.date, r.ticker, r.strategy, r.rank_score, s.state, s.stop, s.meta_json AS signal_meta
      FROM rank_scores_daily r
      JOIN signals_daily s
        ON s.date=r.date AND s.ticker=r.ticker AND s.strategy=r.strategy
      WHERE r.date=?
        AND s.state='ENTRY'
        AND r.rank_score IS NOT NULL
      """,
      conn,
      params=(date_str,),
    )

  if df.empty:
    return []

  # Top X per strategy
  per = (
    df.sort_values(["strategy", "rank_score"], ascending=[True, False])
      .groupby("strategy", group_keys=False)
      .head(top_x)
      .copy()
  )

  # Combine across strategies (union with overlap bonus)
  agg = {}
  for _, r in per.iterrows():
    t = str(r["ticker"])
    agg.setdefault(t, {"ticker": t, "legs": []})
    agg[t]["legs"].append({
      "strategy": str(r["strategy"]),
      "rank_score": float(r["rank_score"]),
      "stop": (float(r["stop"]) if r["stop"] is not None else None),
      "signal_meta": _safe_json_loads(r.get("signal_meta")),
    })

  candidates = []
  for t, v in agg.items():
    nlegs = len(v["legs"])
    if require_overlap and nlegs < 2:
      continue
    combined = sum(x["rank_score"] for x in v["legs"]) + (overlap_bonus if nlegs >= 2 else 0.0)
    candidates.append({
      "ticker": t,
      "combined_score": float(combined),
      "n_strategies": int(nlegs),
      "legs": v["legs"],
    })

  candidates.sort(key=lambda x: x["combined_score"], reverse=True)
  return candidates[:max_entries]


def ma_cross_exit_signal(daily_feat_row: pd.Series) -> bool:
  ma5 = daily_feat_row.get("ma5")
  ma10 = daily_feat_row.get("ma10")
  if pd.isna(ma5) or pd.isna(ma10):
    return False
  return float(ma5) < float(ma10)


def update_trailing_stop_atr(
  prev_trail: Optional[float],
  close: float,
  atr14: Optional[float],
  k: float,
) -> Optional[float]:
  if atr14 is None or pd.isna(atr14) or float(atr14) <= 0:
    return prev_trail
  cand = float(close) - k * float(atr14)
  if prev_trail is None:
    return cand
  return max(float(prev_trail), cand)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default last completed trading day)")
  ap.add_argument("--top-x", type=int, default=20, help="Top X per strategy (rank_score desc)")
  ap.add_argument("--max-entries", type=int, default=20, help="Max total entries for the day")
  ap.add_argument("--overlap-bonus", type=float, default=0.25, help="Extra bonus if ticker appears in >=2 strategies")
  ap.add_argument("--require-overlap", action="store_true", help="If set, only enter tickers that appear in >=2 strategies")

  ap.add_argument("--default-qty", type=float, default=1.0, help="Placeholder qty for paper orders")
  ap.add_argument("--atr-k-ma", type=float, default=2.0, help="ATR trailing k for MA cross strategy")
  ap.add_argument("--atr-k-retest", type=float, default=3.0, help="ATR trailing k for retest_shrink after +1R")

  ap.add_argument("--outputs-dir", default=None, help="Override outputs dir (default /app/outputs or OUTPUT_DIR env)")
  args = ap.parse_args()

  date_str = args.date or last_completed_trading_day_et()
  init_db()
  ensure_tables()

  # For exit logic we need daily features (MA/ATR etc.)
  open_pos = get_open_positions()
  tickers_for_feat = sorted(open_pos["ticker"].unique().tolist()) if not open_pos.empty else []
  daily_feat = build_daily_features(end_date=date_str, lookback_days=260, tickers=tickers_for_feat) if tickers_for_feat else pd.DataFrame(columns=["ticker"])
  daily_feat = daily_feat.set_index("ticker", drop=False) if not daily_feat.empty else daily_feat

  # 1) EXITS
  if not open_pos.empty:
    for _, pos in open_pos.iterrows():
      ticker = str(pos["ticker"])
      strategy = str(pos["strategy"])
      entry_price = float(pos["entry_price"]) if pos["entry_price"] is not None else None
      qty = float(pos["qty"]) if pos["qty"] is not None else float(args.default_qty)
      stop = float(pos["stop"]) if pos["stop"] is not None else None
      trailing = float(pos["trailing_stop"]) if pos["trailing_stop"] is not None else None
      max_close = float(pos["max_close"]) if pos["max_close"] is not None else None

      close = get_close_price(date_str, ticker)
      if close is None:
        continue

      # update max_close
      max_close_new = close if max_close is None else max(max_close, close)

      # fetch indicators if available
      feat_row = daily_feat.loc[ticker] if (not daily_feat.empty and ticker in daily_feat.index) else None
      atr14 = None
      if feat_row is not None:
        atrv = feat_row.get("atr14")
        if atrv is not None and (not pd.isna(atrv)):
          atr14 = float(atrv)

      exit_reason = None

      # MA cross strategy exits
      if strategy.startswith("ma_cross"):
        if feat_row is not None and ma_cross_exit_signal(feat_row):
          exit_reason = "MA5_cross_below_MA10"
        # risk: ATR trailing always on (simple baseline)
        trailing_new = update_trailing_stop_atr(trailing, close, atr14, args.atr_k_ma)
        trailing = trailing_new
        if trailing is not None and close < trailing:
          exit_reason = exit_reason or "ATR_trailing_stop"

      # retest_shrink strategy exits
      if strategy.startswith("retest_shrink"):
        # structural stop always
        if stop is not None and close < stop:
          exit_reason = "STRUCTURAL_stop_break"

        # after +1R, start ATR trailing
        if entry_price is not None and stop is not None and atr14 is not None:
          R = entry_price - stop
          if R > 0 and max_close_new >= entry_price + 1.0 * R:
            trailing_new = update_trailing_stop_atr(trailing, close, atr14, args.atr_k_retest)
            trailing = trailing_new
            if trailing is not None and close < trailing:
              exit_reason = exit_reason or "ATR_trailing_after_1R"

      # persist position updates (even if no exit)
      meta = _safe_json_loads(pos.get("meta_json"))
      meta.setdefault("last_update", {})
      meta["last_update"][date_str] = {"close": close}

      upsert_position(
        ticker=ticker,
        strategy=strategy,
        status="OPEN",
        opened_date=str(pos["opened_date"]),
        entry_price=entry_price,
        qty=qty,
        stop=stop,
        trailing_stop=trailing,
        max_close=max_close_new,
        meta=meta,
        closed_date=None,
      )

      if exit_reason:
        upsert_order(
          date_str=date_str,
          ticker=ticker,
          strategy=strategy,
          side="SELL",
          qty=qty,
          reason=exit_reason,
          meta={"close": close, "stop": stop, "trailing_stop": trailing},
        )
        # mark CLOSED (paper assumption)
        upsert_position(
          ticker=ticker,
          strategy=strategy,
          status="CLOSED",
          opened_date=str(pos["opened_date"]),
          entry_price=entry_price,
          qty=qty,
          stop=stop,
          trailing_stop=trailing,
          max_close=max_close_new,
          meta=meta,
          closed_date=date_str,
        )

  # 2) ENTRIES
  entries = select_entries(
    date_str=date_str,
    top_x=args.top_x,
    max_entries=args.max_entries,
    overlap_bonus=args.overlap_bonus,
    require_overlap=args.require_overlap,
  )

  # Don’t re-enter already open positions for any strategy leg
  open_pos = get_open_positions()
  open_keys = set()
  if not open_pos.empty:
    for _, p in open_pos.iterrows():
      open_keys.add((str(p["ticker"]), str(p["strategy"])))

  for c in entries:
    ticker = c["ticker"]
    close = get_close_price(date_str, ticker)
    if close is None:
      continue

    for leg in c["legs"]:
      strategy = leg["strategy"]
      if (ticker, strategy) in open_keys:
        continue

      qty = float(args.default_qty)
      stop = leg.get("stop")

      # create BUY order intent
      upsert_order(
        date_str=date_str,
        ticker=ticker,
        strategy=strategy,
        side="BUY",
        qty=qty,
        reason="RANKED_ENTRY",
        meta={
          "combined_score": c["combined_score"],
          "n_strategies": c["n_strategies"],
          "leg_rank_score": leg["rank_score"],
        },
      )

      # open paper position immediately at close (for now)
      upsert_position(
        ticker=ticker,
        strategy=strategy,
        status="OPEN",
        opened_date=date_str,
        entry_price=close,
        qty=qty,
        stop=stop,
        trailing_stop=None,
        max_close=close,
        meta={
          "entry": {
            "date": date_str,
            "assumed_fill": "close",
            "combined_score": c["combined_score"],
            "n_strategies": c["n_strategies"],
            "leg_rank_score": leg["rank_score"],
          }
        },
        closed_date=None,
      )

  # 3) Write orders CSV for convenience
  out_base = args.outputs_dir or (str(Path("/app/outputs")))

  out_dir = Path(out_base) / date_str
  out_dir.mkdir(parents=True, exist_ok=True)

  with connect() as conn:
    df_orders = pd.read_sql_query(
      "SELECT date, ticker, strategy, side, qty, reason, meta_json FROM orders_daily WHERE date=? ORDER BY side, strategy, ticker",
      conn,
      params=(date_str,),
    )

  out_path = out_dir / "orders.csv"
  df_orders.to_csv(out_path, index=False)
  log.info(f"Wrote orders to {out_path} (rows={len(df_orders)})")


if __name__ == "__main__":
  main()
