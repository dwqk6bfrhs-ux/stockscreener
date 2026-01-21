from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.db import connect, init_db, get_prices_daily_source
from src.common.logging import setup_logger
from src.common.ranking import build_daily_features, build_hourly_coverage
from src.common.timeutil import last_completed_trading_day_et
from src.common.datefmt import normalize_date_str

log = setup_logger("export_dossier")


def _out_dir_for_date(date: str) -> Path:
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
  day_dir = out_dir / date
  day_dir.mkdir(parents=True, exist_ok=True)
  return day_dir


def _safe_json_loads(value: Optional[str]) -> Dict[str, Any]:
  if not value:
    return {}
  try:
    return json.loads(value)
  except Exception:
    return {}


def _read_signals(date: str) -> pd.DataFrame:
  with connect() as conn:
    return pd.read_sql_query(
      """
      SELECT date, ticker, strategy, state, score, stop, meta_json
      FROM signals_daily
      WHERE date = ?
      """,
      conn,
      params=(date,),
    )


def _read_rank_scores(date: str) -> pd.DataFrame:
  with connect() as conn:
    return pd.read_sql_query(
      """
      SELECT date, ticker, strategy, rank_score, meta_json
      FROM rank_scores_daily
      WHERE date = ?
      """,
      conn,
      params=(date,),
    )


def _read_daily_snapshot(date: str, tickers: List[str]) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()
  ph = ",".join(["?"] * len(tickers))
  source = get_prices_daily_source()
  with connect() as conn:
    df = pd.read_sql_query(
      f"""
      SELECT ticker, date, open, high, low, close, volume
      FROM prices_daily
      WHERE source = ?
        AND date = ?
        AND ticker IN ({ph})
      """,
      conn,
      params=[source, date] + tickers,
    )
  if df.empty:
    return df
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


def _serialize_signal_row(row: pd.Series) -> Dict[str, Any]:
  meta = _safe_json_loads(row.get("meta_json"))
  return {
    "strategy": row.get("strategy"),
    "state": row.get("state"),
    "score": (float(row["score"]) if pd.notna(row.get("score")) else None),
    "stop": (float(row["stop"]) if pd.notna(row.get("stop")) else None),
    "raw_state": meta.get("raw_state"),
    "features": meta.get("features", {}),
    "meta": meta,
  }


def _serialize_rank_row(row: pd.Series) -> Dict[str, Any]:
  meta = _safe_json_loads(row.get("meta_json"))
  return {
    "strategy": row.get("strategy"),
    "rank_score": (float(row["rank_score"]) if pd.notna(row.get("rank_score")) else None),
    "meta": meta,
  }


def build_dossiers(date: str) -> List[Dict[str, Any]]:
  df_signals = _read_signals(date)
  if df_signals.empty:
    raise RuntimeError(f"No signals_daily rows for date={date}. Run generate_signals first.")

  tickers = sorted(df_signals["ticker"].dropna().astype(str).unique().tolist())

  df_rank = _read_rank_scores(date)
  df_daily_features = build_daily_features(end_date=date, lookback_days=260, tickers=tickers)
  df_hourly_cov = build_hourly_coverage(date_et=date, tickers=tickers)
  df_snapshot = _read_daily_snapshot(date, tickers)

  signals_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
  for _, row in df_signals.iterrows():
    ticker = str(row.get("ticker"))
    signals_by_ticker.setdefault(ticker, []).append(_serialize_signal_row(row))

  ranks_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
  if not df_rank.empty:
    for _, row in df_rank.iterrows():
      ticker = str(row.get("ticker"))
      ranks_by_ticker.setdefault(ticker, []).append(_serialize_rank_row(row))

  daily_by_ticker = {}
  if not df_daily_features.empty:
    for _, row in df_daily_features.iterrows():
      daily_by_ticker[str(row.get("ticker"))] = row.to_dict()

  hourly_by_ticker = {}
  if not df_hourly_cov.empty:
    for _, row in df_hourly_cov.iterrows():
      hourly_by_ticker[str(row.get("ticker"))] = {
        "hourly_bars": int(row.get("hourly_bars") or 0),
      }

  snapshot_by_ticker = {}
  if not df_snapshot.empty:
    for _, row in df_snapshot.iterrows():
      snapshot_by_ticker[str(row.get("ticker"))] = {
        "open": (float(row["open"]) if pd.notna(row.get("open")) else None),
        "high": (float(row["high"]) if pd.notna(row.get("high")) else None),
        "low": (float(row["low"]) if pd.notna(row.get("low")) else None),
        "close": (float(row["close"]) if pd.notna(row.get("close")) else None),
        "volume": (float(row["volume"]) if pd.notna(row.get("volume")) else None),
      }

  dossiers = []
  for ticker in tickers:
    dossiers.append({
      "date": date,
      "ticker": ticker,
      "signals": signals_by_ticker.get(ticker, []),
      "rank_scores": ranks_by_ticker.get(ticker, []),
      "daily_features": daily_by_ticker.get(ticker, {}),
      "hourly_coverage": hourly_by_ticker.get(ticker, {"hourly_bars": 0}),
      "daily_snapshot": snapshot_by_ticker.get(ticker, {}),
    })

  return dossiers


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (default last completed trading day)")
  ap.add_argument("--output", default=None, help="Output JSONL path (default outputs/<date>/dossier.jsonl)")
  args = ap.parse_args()

  init_db()
  raw_date = args.date
  date = normalize_date_str(raw_date) if raw_date else last_completed_trading_day_et()

  dossiers = build_dossiers(date)
  out_path = Path(args.output) if args.output else (_out_dir_for_date(date) / "dossier.jsonl")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as handle:
    for row in dossiers:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")

  log.info(f"Wrote {len(dossiers)} dossiers to {out_path}")


if __name__ == "__main__":
  main()
