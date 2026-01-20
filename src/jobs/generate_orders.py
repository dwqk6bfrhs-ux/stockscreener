from __future__ import annotations
import os

import argparse
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


@dataclass
class EntryPolicy:
  top_x_per_strategy: int = 20
  max_entries_total: int = 20
  overlap_bonus: float = 0.25
  require_overlap: bool = False


@dataclass
class ExitPolicy:
  # MA strategy exits
  ma_fast: int = 5
  ma_slow: int = 10
  ma_exit_confirm_days: int = 1  # 1 = cross-down today; 2 = require two days
  ma_trail_atr_k: float = 2.0

  # retest_shrink exits
  retest_trail_start_r: float = 1.0     # start trailing after +1R
  retest_trail_atr_k: float = 3.0

  # universal
  default_qty: float = 1.0


def ensure_tables() -> None:
  """
  Minimal persistence so:
  - we don't re-enter the same ticker repeatedly
  - we can generate exits deterministically
  """
  sql = """
  CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT NOT NULL,
    book TEXT NOT NULL,                 -- e.g. "combined"
    status TEXT NOT NULL,               -- OPEN / CLOSED
    opened_date TEXT NOT NULL,
    closed_date TEXT,
    entry_price REAL,
    qty REAL,
    stop REAL,
    trailing_stop REAL,
    max_close REAL,
    meta_json TEXT,
    PRIMARY KEY (ticker, book)
  );

  CREATE TABLE IF NOT EXISTS orders_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    book TEXT NOT NULL,
    side TEXT NOT NULL,                 -- BUY / SELL
    qty REAL,
    reason TEXT,
    meta_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, ticker, book, side)
  );

  CREATE INDEX IF NOT EXISTS idx_orders_date ON orders_daily(date);
  """
  with connect() as conn:
    conn.executescript(sql)
    conn.commit()


def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
  if not s:
    return {}
  try:
    return json.loads(s)
  except Exception:
    return {}


def _safe_json_dumps(d: Dict[str, Any]) -> str:
  return json.dumps(d, ensure_ascii=False)


def get_close(date_str: str, ticker: str) -> Optional[float]:
  with connect() as conn:
    row = conn.execute("SELECT close FROM prices_daily WHERE date=? AND ticker=?", (date_str, ticker)).fetchone()
  if not row or row[0] is None:
    return None
  try:
    return float(row[0])
  except Exception:
    return None


def get_open_positions(book: str) -> pd.DataFrame:
  with connect() as conn:
    df = pd.read_sql_query("SELECT * FROM positions WHERE status='OPEN' AND book=?", conn, params=(book,))
  return df


def upsert_position(
  *,
  ticker: str,
  book: str,
  status: str,
  opened_date: str,
  closed_date: Optional[str],
  entry_price: Optional[float],
  qty: float,
  stop: Optional[float],
  trailing_stop: Optional[float],
  max_close: Optional[float],
  meta: Dict[str, Any],
) -> None:
  sql = """
  INSERT INTO positions (ticker, book, status, opened_date, closed_date, entry_price, qty, stop, trailing_stop, max_close, meta_json)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(ticker, book) DO UPDATE SET
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
      ticker, book, status, opened_date, closed_date,
      (float(entry_price) if entry_price is not None else None),
      float(qty),
      (float(stop) if stop is not None else None),
      (float(trailing_stop) if trailing_stop is not None else None),
      (float(max_close) if max_close is not None else None),
      _safe_json_dumps(meta),
    ))
    conn.commit()


def upsert_order(date_str: str, ticker: str, book: str, side: str, qty: float, reason: str, meta: Dict[str, Any]) -> None:
  sql = """
  INSERT INTO orders_daily (date, ticker, book, side, qty, reason, meta_json)
  VALUES (?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(date, ticker, book, side) DO UPDATE SET
    qty=excluded.qty,
    reason=excluded.reason,
    meta_json=excluded.meta_json
  """
  with connect() as conn:
    conn.execute(sql, (date_str, ticker, book, side, float(qty), reason, _safe_json_dumps(meta)))
    conn.commit()


def read_rank_entries(date_str: str) -> pd.DataFrame:
  """
  Pull only rankable ENTRY rows (rank_score NOT NULL and coverage_ok True in meta).
  """
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT r.date, r.ticker, r.strategy, r.rank_score, r.meta_json,
             s.state, s.stop, s.meta_json AS signal_meta
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
    return df

  def cov_ok(mj: str) -> bool:
    meta = _safe_json_loads(mj)
    cov = meta.get("coverage") or {}
    return bool(cov.get("coverage_ok", True))

  df["coverage_ok"] = df["meta_json"].apply(cov_ok)
  df = df[df["coverage_ok"]].copy()
  return df


def pick_entries_union_with_bonus(df: pd.DataFrame, policy: EntryPolicy) -> List[Dict[str, Any]]:
  if df.empty:
    return []

  # Top X per strategy
  per = (
    df.sort_values(["strategy", "rank_score"], ascending=[True, False])
      .groupby("strategy", group_keys=False)
      .head(policy.top_x_per_strategy)
      .copy()
  )

  # Union with overlap bonus
  agg: Dict[str, Dict[str, Any]] = {}
  for _, r in per.iterrows():
    t = str(r["ticker"])
    agg.setdefault(t, {"ticker": t, "legs": []})
    agg[t]["legs"].append({
      "strategy": str(r["strategy"]),
      "rank_score": float(r["rank_score"]),
      "stop": (float(r["stop"]) if r["stop"] is not None else None),
      "rank_meta": _safe_json_loads(r.get("meta_json")),
      "signal_meta": _safe_json_loads(r.get("signal_meta")),
    })

  out: List[Dict[str, Any]] = []
  for t, v in agg.items():
    nlegs = len(v["legs"])
    if policy.require_overlap and nlegs < 2:
      continue
    combined = sum(x["rank_score"] for x in v["legs"]) + (policy.overlap_bonus if nlegs >= 2 else 0.0)
    out.append({
      "ticker": t,
      "combined_score": float(combined),
      "n_strategies": int(nlegs),
      "legs": v["legs"],
    })

  out.sort(key=lambda x: x["combined_score"], reverse=True)
  return out[:policy.max_entries_total]


def ma_cross_down(daily_row: pd.Series, fast: int, slow: int) -> bool:
  """
  exit if ma_fast < ma_slow today
  """
  ma_fast = daily_row.get(f"ma{fast}") or daily_row.get("ma5")
  ma_slow = daily_row.get(f"ma{slow}") or daily_row.get("ma10")
  if pd.isna(ma_fast) or pd.isna(ma_slow):
    return False
  return float(ma_fast) < float(ma_slow)


def update_trailing_stop(trail: Optional[float], close: float, atr14: Optional[float], k: float) -> Optional[float]:
  if atr14 is None or pd.isna(atr14) or float(atr14) <= 0:
    return trail
  cand = float(close) - float(k) * float(atr14)
  if trail is None:
    return cand
  return max(float(trail), cand)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None)
  ap.add_argument("--book", default="combined")

  # Entry policy
  ap.add_argument("--top-x", type=int, default=20)
  ap.add_argument("--max-entries", type=int, default=20)
  ap.add_argument("--overlap-bonus", type=float, default=0.25)
  ap.add_argument("--require-overlap", action="store_true")

  # Exit policy
  ap.add_argument("--ma-fast", type=int, default=5)
  ap.add_argument("--ma-slow", type=int, default=10)
  ap.add_argument("--ma-exit-confirm-days", type=int, default=1)
  ap.add_argument("--ma-trail-atr-k", type=float, default=2.0)

  ap.add_argument("--retest-trail-start-r", type=float, default=1.0)
  ap.add_argument("--retest-trail-atr-k", type=float, default=3.0)

  ap.add_argument("--default-qty", type=float, default=1.0)
  args = ap.parse_args()

  init_db()
  ensure_tables()

  date_str = args.date or last_completed_trading_day_et()
  book = args.book

  entry_policy = EntryPolicy(
    top_x_per_strategy=args.top_x,
    max_entries_total=args.max_entries,
    overlap_bonus=args.overlap_bonus,
    require_overlap=args.require_overlap,
  )
  exit_policy = ExitPolicy(
    ma_fast=args.ma_fast,
    ma_slow=args.ma_slow,
    ma_exit_confirm_days=args.ma_exit_confirm_days,
    ma_trail_atr_k=args.ma_trail_atr_k,
    retest_trail_start_r=args.retest_trail_start_r,
    retest_trail_atr_k=args.retest_trail_atr_k,
    default_qty=args.default_qty,
  )

  log.info(f"Generate orders: date={date_str} book={book} entry=Top{entry_policy.top_x_per_strategy}/strategy max={entry_policy.max_entries_total} overlap_bonus={entry_policy.overlap_bonus}")

  # ---- EXITS ----
  open_pos = get_open_positions(book)
  tickers_open = sorted(open_pos["ticker"].unique().tolist()) if not open_pos.empty else []
  daily_feat = build_daily_features(end_date=date_str, lookback_days=260, tickers=tickers_open) if tickers_open else pd.DataFrame()
  daily_feat = daily_feat.set_index("ticker", drop=False) if not daily_feat.empty else daily_feat

  for _, pos in open_pos.iterrows():
    ticker = str(pos["ticker"])
    close = get_close(date_str, ticker)
    if close is None:
      continue

    entry_price = float(pos["entry_price"]) if pos["entry_price"] is not None else None
    qty = float(pos["qty"]) if pos["qty"] is not None else float(exit_policy.default_qty)
    stop = float(pos["stop"]) if pos["stop"] is not None else None
    trail = float(pos["trailing_stop"]) if pos["trailing_stop"] is not None else None
    max_close = float(pos["max_close"]) if pos["max_close"] is not None else None
    max_close_new = close if max_close is None else max(max_close, close)

    meta = _safe_json_loads(pos.get("meta_json"))
    meta.setdefault("updates", {})
    meta["updates"][date_str] = {"close": close}

    feat_row = daily_feat.loc[ticker] if (not daily_feat.empty and ticker in daily_feat.index) else None
    atr14 = None
    if feat_row is not None and pd.notna(feat_row.get("atr14")):
      atr14 = float(feat_row["atr14"])

    exit_reason = None

    # Strategy tags for exits:
    # if it was entered with overlap, it will have multiple legs in meta["entry"]["legs"]
    # We'll treat "MA exit" and "Retest exit" as universal for the combined book.
    # MA exit:
    if feat_row is not None:
      # optional: require 2-day confirm by checking yesterday too (future enhancement)
      if ma_cross_down(feat_row, exit_policy.ma_fast, exit_policy.ma_slow):
        exit_reason = "MA5_below_MA10"

    # universal ATR trailing (helps gaps/fast reversals)
    trail = update_trailing_stop(trail, close, atr14, exit_policy.ma_trail_atr_k)
    if trail is not None and close < trail:
      exit_reason = exit_reason or "ATR_trailing_stop"

    # retest structural stop (if set)
    if stop is not None and close < stop:
      exit_reason = "STRUCTURAL_stop_break"

    # retest trailing after +1R if entry_price/stop available
    if entry_price is not None and stop is not None and atr14 is not None:
      R = entry_price - stop
      if R > 0 and max_close_new >= entry_price + exit_policy.retest_trail_start_r * R:
        trail = update_trailing_stop(trail, close, atr14, exit_policy.retest_trail_atr_k)
        if trail is not None and close < trail:
          exit_reason = exit_reason or "ATR_trailing_after_1R"

    # persist updated trailing/max_close even if not exiting
    upsert_position(
      ticker=ticker, book=book, status="OPEN",
      opened_date=str(pos["opened_date"]),
      closed_date=None,
      entry_price=entry_price,
      qty=qty,
      stop=stop,
      trailing_stop=trail,
      max_close=max_close_new,
      meta=meta,
    )

    if exit_reason:
      upsert_order(
        date_str=date_str, ticker=ticker, book=book, side="SELL", qty=qty,
        reason=exit_reason,
        meta={"close": close, "stop": stop, "trailing_stop": trail, "max_close": max_close_new},
      )
      upsert_position(
        ticker=ticker, book=book, status="CLOSED",
        opened_date=str(pos["opened_date"]),
        closed_date=date_str,
        entry_price=entry_price,
        qty=qty,
        stop=stop,
        trailing_stop=trail,
        max_close=max_close_new,
        meta=meta,
      )

  # ---- ENTRIES ----
  df_rank = read_rank_entries(date_str)
  picks = pick_entries_union_with_bonus(df_rank, entry_policy)

  open_pos = get_open_positions(book)
  open_tickers = set(open_pos["ticker"].astype(str).tolist()) if not open_pos.empty else set()

  for c in picks:
    ticker = c["ticker"]
    if ticker in open_tickers:
      continue

    close = get_close(date_str, ticker)
    if close is None:
      continue

    # choose a conservative stop: if any leg provides a stop, pick the tightest (highest stop) for longs
    stops = [leg.get("stop") for leg in c["legs"] if leg.get("stop") is not None]
    stop = max(stops) if stops else None

    meta = {
      "entry": {
        "date": date_str,
        "assumed_fill": "close",
        "combined_score": c["combined_score"],
        "n_strategies": c["n_strategies"],
        "legs": [{"strategy": l["strategy"], "rank_score": l["rank_score"]} for l in c["legs"]],
      }
    }

    qty = float(exit_policy.default_qty)

    upsert_order(
      date_str=date_str, ticker=ticker, book=book, side="BUY", qty=qty,
      reason="RANKED_ENTRY_UNION",
      meta=meta,
    )

    upsert_position(
      ticker=ticker, book=book, status="OPEN",
      opened_date=date_str, closed_date=None,
      entry_price=close, qty=qty,
      stop=stop, trailing_stop=None, max_close=close,
      meta=meta,
    )

  # ---- Write orders CSV ----
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / date_str
  out_dir.mkdir(parents=True, exist_ok=True)

  with connect() as conn:
    df_orders = pd.read_sql_query(
      "SELECT date, ticker, book, side, qty, reason, meta_json FROM orders_daily WHERE date=? ORDER BY side, ticker",
      conn,
      params=(date_str,),
    )
  out_path = out_dir / "orders.csv"
  df_orders.to_csv(out_path, index=False)

  log.info(f"Wrote orders.csv: {out_path} rows={len(df_orders)}")


if __name__ == "__main__":
  main()
