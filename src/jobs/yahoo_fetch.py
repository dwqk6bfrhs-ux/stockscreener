from __future__ import annotations

import argparse
from datetime import datetime
import os
from typing import List

import pandas as pd
import yfinance as yf

from src.common.db import init_db, connect
from src.common.logging import setup_logger

log = setup_logger("yahoo_fetch")


def read_tickers(path: str) -> List[str]:
  tickers = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      t = line.strip()
      if not t or t.startswith("#"):
        continue
      tickers.append(t.upper())
  return tickers


def upsert_daily(rows: List[tuple]) -> int:
  if not rows:
    return 0
  sql = """
  INSERT INTO prices_daily (ticker, date, open, high, low, close, volume)
  VALUES (?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(ticker, date) DO UPDATE SET
    open=excluded.open,
    high=excluded.high,
    low=excluded.low,
    close=excluded.close,
    volume=excluded.volume
  """
  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()
  return len(rows)


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
  df = yf.download(
    tickers=ticker,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=False,
    progress=False,
  )
  if df.empty:
    return df
  df = df.reset_index()
  df.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
  }, inplace=True)
  df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
  df["ticker"] = ticker
  return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def main() -> None:
  ap = argparse.ArgumentParser(description="Fetch daily OHLCV from Yahoo Finance into prices_daily.")
  ap.add_argument("--start", required=True, help="YYYY-MM-DD")
  ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive, like Yahoo)")
  ap.add_argument("--tickers-path", default=os.environ.get("TICKERS_PATH", "/app/tickers.txt"))
  ap.add_argument("--limit", type=int, default=None)
  args = ap.parse_args()

  datetime.strptime(args.start, "%Y-%m-%d")
  datetime.strptime(args.end, "%Y-%m-%d")

  init_db()
  tickers = read_tickers(args.tickers_path)
  if args.limit:
    tickers = tickers[:args.limit]
  if not tickers:
    raise RuntimeError("No tickers found to fetch.")

  total = 0
  for t in tickers:
    df = fetch_ticker(t, args.start, args.end)
    if df.empty:
      log.warning(f"No Yahoo data for {t}")
      continue
    rows = list(df.itertuples(index=False, name=None))
    total += upsert_daily(rows)
    log.info(f"Fetched {t}: rows={len(rows)} total_upserted={total}")

  log.info(f"Yahoo fetch complete. Total rows upserted: {total}")


if __name__ == "__main__":
  main()
