from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
import time
from typing import List

import pandas as pd
import requests
import requests_cache
import yfinance as yf
from yfinance.exceptions import YFTzMissingError

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
  INSERT INTO prices_daily (ticker, date, source, open, high, low, close, volume)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(ticker, date, source) DO UPDATE SET
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


def _build_session(cache_path: str, expire_after: int) -> requests.Session:
  session = requests_cache.CachedSession(
    cache_path,
    expire_after=expire_after,
    allowable_methods=("GET", "POST"),
    stale_if_error=True,
  )
  session.headers.update({
    "User-Agent": (
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
  })
  return session


def fetch_ticker(
  ticker: str,
  start: str,
  end: str,
  *,
  session: requests.Session,
  retries: int,
  backoff: float,
) -> pd.DataFrame:
  last_err: Exception | None = None
  for attempt in range(1, retries + 1):
    try:
      df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        session=session,
      )
      if not df.empty:
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
      return df
    except (json.JSONDecodeError, ValueError, requests.RequestException, YFTzMissingError) as e:
      last_err = e
      sleep_s = backoff ** (attempt - 1)
      log.warning(f"Yahoo fetch error for {ticker}. Retry {attempt}/{retries} after {sleep_s:.1f}s: {e}")
      time.sleep(sleep_s)
      continue

  raise RuntimeError(f"Failed to fetch Yahoo data for {ticker}. Last error: {last_err}")


def main() -> None:
  ap = argparse.ArgumentParser(description="Fetch daily OHLCV from Yahoo Finance into prices_daily.")
  ap.add_argument("--start", required=True, help="YYYY-MM-DD")
  ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive, like Yahoo)")
  ap.add_argument("--tickers-path", default=os.environ.get("TICKERS_PATH", "/app/tickers.txt"))
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--source", default=os.environ.get("YAHOO_SOURCE", "yahoo"), help="Source label stored in DB")
  ap.add_argument("--cache-path", default=os.environ.get("YAHOO_CACHE_PATH", "/app/data/yahoo_cache"))
  ap.add_argument("--cache-expire-seconds", type=int, default=int(os.environ.get("YAHOO_CACHE_EXPIRE_SECS", "3600")))
  ap.add_argument("--retries", type=int, default=int(os.environ.get("YAHOO_RETRIES", "4")))
  ap.add_argument("--backoff", type=float, default=float(os.environ.get("YAHOO_BACKOFF", "1.7")))
  args = ap.parse_args()

  datetime.strptime(args.start, "%Y-%m-%d")
  datetime.strptime(args.end, "%Y-%m-%d")

  init_db()
  tickers = read_tickers(args.tickers_path)
  if args.limit:
    tickers = tickers[:args.limit]
  if not tickers:
    raise RuntimeError("No tickers found to fetch.")

  session = _build_session(args.cache_path, args.cache_expire_seconds)
  total = 0
  for t in tickers:
    df = fetch_ticker(
      t,
      args.start,
      args.end,
      session=session,
      retries=args.retries,
      backoff=args.backoff,
    )
    if df.empty:
      log.warning(f"No Yahoo data for {t}")
      continue
    df["source"] = args.source
    rows = list(df[["ticker", "date", "source", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None))
    total += upsert_daily(rows)
    log.info(f"Fetched {t}: rows={len(rows)} total_upserted={total}")

  log.info(f"Yahoo fetch complete. Total rows upserted: {total}")


if __name__ == "__main__":
  main()
