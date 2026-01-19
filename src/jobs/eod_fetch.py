from __future__ import annotations

import os
import json
import time
import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Iterable, List, Tuple, Dict, Any
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import today_et, last_completed_trading_day_et

log = setup_logger("eod_fetch")


ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _env(name: str, default: Optional[str] = None) -> str:
  v = os.environ.get(name, default)
  if v is None or v == "":
    raise RuntimeError(f"Missing required env var: {name}")
  return v


def _read_tickers_file(path: str) -> list[str]:
  out: list[str] = []
  with open(path, "r", encoding="utf-8") as f:
    for ln in f:
      s = ln.strip().upper()
      if not s or s.startswith("#"):
        continue
      out.append(s)
  return out


def _read_tickers_from_universe(universe_date: Optional[str], source: str = "alpaca", limit: Optional[int] = None) -> list[str]:
  with connect() as conn:
    if universe_date:
      q = "SELECT ticker FROM universe_daily WHERE date=? AND source=? ORDER BY ticker"
      rows = conn.execute(q, (universe_date, source)).fetchall()
    else:
      q = "SELECT ticker FROM universe_daily WHERE date=(SELECT MAX(date) FROM universe_daily) AND source=? ORDER BY ticker"
      rows = conn.execute(q, (source,)).fetchall()
  tickers = [r[0] for r in rows]
  if limit:
    tickers = tickers[:limit]
  return tickers


def _latest_price_date_in_db() -> Optional[str]:
  with connect() as conn:
    row = conn.execute("SELECT MAX(date) FROM prices_daily").fetchone()
  return row[0] if row and row[0] else None


def _parse_yyyy_mm_dd(s: str) -> date:
  return datetime.strptime(s, "%Y-%m-%d").date()


def _date_to_utc_range_inclusive(start_d: date, end_d: date) -> tuple[datetime, datetime]:
  """
  Alpaca expects datetimes; we make the interval inclusive by setting:
  start = start_d 00:00 ET
  end   = (end_d + 1 day) 00:00 ET
  """
  start_et = datetime(start_d.year, start_d.month, start_d.day, 0, 0, 0, tzinfo=ET)
  end_et = datetime(end_d.year, end_d.month, end_d.day, 0, 0, 0, tzinfo=ET) + timedelta(days=1)
  return start_et.astimezone(UTC), end_et.astimezone(UTC)


def _iso_z(dt: datetime) -> str:
  return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _chunk_date_ranges(start_d: date, end_d: date, chunk_days: int) -> list[tuple[date, date]]:
  """
  Split inclusive date range [start_d, end_d] into chunks of at most chunk_days days.
  """
  if chunk_days <= 0:
    return [(start_d, end_d)]
  out = []
  cur = start_d
  while cur <= end_d:
    nxt = min(end_d, cur + timedelta(days=chunk_days - 1))
    out.append((cur, nxt))
    cur = nxt + timedelta(days=1)
  return out


def _batches(xs: list[str], batch_size: int) -> Iterable[list[str]]:
  if batch_size <= 0:
    yield xs
    return
  for i in range(0, len(xs), batch_size):
    yield xs[i:i + batch_size]


def _alpaca_http_get(url: str, headers: dict[str, str], retries: int = 5, backoff: float = 1.5) -> dict[str, Any]:
  """
  Simple resilient GET with retry on 429/5xx.
  """
  last_err = None
  for attempt in range(1, retries + 1):
    try:
      req = Request(url, headers=headers, method="GET")
      with urlopen(req, timeout=60) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)
    except HTTPError as e:
      last_err = e
      code = getattr(e, "code", None)
      if code in (429, 500, 502, 503, 504):
        sleep_s = (backoff ** (attempt - 1))
        log.warning(f"HTTP {code} from Alpaca data API. Retry {attempt}/{retries} after {sleep_s:.1f}s")
        time.sleep(sleep_s)
        continue
      raise
    except URLError as e:
      last_err = e
      sleep_s = (backoff ** (attempt - 1))
      log.warning(f"Network error calling Alpaca data API. Retry {attempt}/{retries} after {sleep_s:.1f}s: {e}")
      time.sleep(sleep_s)
      continue

  raise RuntimeError(f"Failed after retries. Last error: {last_err}")


def _bars_to_rows(bars: list[dict[str, Any]]) -> list[tuple]:
  """
  Convert Alpaca bars JSON -> sqlite rows (ticker,date,open,high,low,close,volume)
  Alpaca fields typically: t,o,h,l,c,v,n,vw (t is RFC3339).
  """
  out = []
  for b in bars:
    sym = b.get("S") or b.get("symbol") or b.get("sym")  # sometimes present depending on endpoint
    if not sym:
      # In /v2/stocks/bars response, symbol is not repeated inside each bar; it's embedded in grouping.
      # But Alpaca's multi-symbol /bars response returns a "bars" dict per symbol in older formats.
      # We only handle the current documented list format where each bar includes "S" in some SDKs.
      # If your response doesn't include it, we will handle in _normalize_response().
      raise RuntimeError("Bar record missing symbol. Response normalization failed.")
    ts = b.get("t")
    if not ts:
      continue

    # parse timestamp
    dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
    d_et = dt_utc.astimezone(ET).date().isoformat()

    out.append((
      sym,
      d_et,
      float(b.get("o")) if b.get("o") is not None else None,
      float(b.get("h")) if b.get("h") is not None else None,
      float(b.get("l")) if b.get("l") is not None else None,
      float(b.get("c")) if b.get("c") is not None else None,
      float(b.get("v")) if b.get("v") is not None else None,
    ))
  return out


def _normalize_response(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
  """
  Alpaca /v2/stocks/bars can return:
  - { "bars": [ {t,o,h,l,c,v,S}, ... ], "next_page_token": "..." }  (common)
  - or a dict keyed by symbol in older patterns (rare)
  This function returns a flat list of bars with symbol embedded + next token.
  """
  token = payload.get("next_page_token") or payload.get("next_page_token".upper())

  bars = payload.get("bars")
  if isinstance(bars, list):
    # If symbol not embedded, we cannot use it; but most modern responses include "S".
    return bars, token

  # Older alternative: bars is dict like {"AAPL":[{...}], "MSFT":[{...}]}
  if isinstance(bars, dict):
    flat: list[dict[str, Any]] = []
    for sym, arr in bars.items():
      if not isinstance(arr, list):
        continue
      for b in arr:
        if isinstance(b, dict):
          b = dict(b)
          b.setdefault("S", sym)
          flat.append(b)
    return flat, token

  return [], token


def upsert_prices(rows: list[tuple]) -> int:
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


def fetch_bars_batch(
  symbols: list[str],
  start_dt_utc: datetime,
  end_dt_utc: datetime,
  timeframe: str,
  feed: str,
  adjustment: str,
  page_limit: int,
) -> int:
  """
  Fetch bars for one symbol batch and one time chunk, handling pagination.
  Returns number of rows upserted.
  """
  key = _env("ALPACA_API_KEY")
  sec = _env("ALPACA_SECRET_KEY")

  headers = {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": sec,
    "Accept": "application/json",
  }

  base_url = "https://data.alpaca.markets/v2/stocks/bars"

  total_rows = 0
  page_token = None

  # Alpaca expects comma-separated symbols parameter: symbols=AAPL,MSFT
  symbols_csv = ",".join(symbols)

  while True:
    params = {
      "symbols": symbols_csv,
      "timeframe": timeframe,
      "start": _iso_z(start_dt_utc),
      "end": _iso_z(end_dt_utc),
      "limit": str(page_limit),
      "sort": "asc",
      "feed": feed,
      "adjustment": adjustment,
    }
    if page_token:
      params["page_token"] = page_token

    url = f"{base_url}?{urlencode(params)}"
    payload = _alpaca_http_get(url, headers=headers)

    bars, next_token = _normalize_response(payload)
    if not bars:
      break

    # Ensure symbol exists in each bar (some formats include "S"; our normalizer adds it for dict format)
    # For list format without symbol, this will fail loudly (better than silent bad DB writes).
    rows = _bars_to_rows(bars)
    n = upsert_prices(rows)
    total_rows += n

    if not next_token:
      break
    page_token = next_token

  return total_rows


def resolve_date_range(args) -> tuple[date, date, str]:
  """
  Returns (start_date, end_date, mode_used)
  mode_used is one of: daily/backfill/range/auto-daily/auto-backfill
  """
  # explicit mode
  mode = args.mode

  # If no explicit mode, choose based on DB content
  if mode == "auto":
    latest = _latest_price_date_in_db()
    if latest:
      # Incremental: fetch from latest -> last trading day
      start_d = _parse_yyyy_mm_dd(latest)
      end_d = _parse_yyyy_mm_dd(last_completed_trading_day_et())
      return start_d, end_d, "auto-daily"
    else:
      # Seed: backfill last LOOKBACK_DAYS
      lookback = int(os.environ.get("LOOKBACK_DAYS", "180"))
      end_d = _parse_yyyy_mm_dd(last_completed_trading_day_et())
      start_d = end_d - timedelta(days=lookback)
      return start_d, end_d, "auto-backfill"

  # daily: default to last completed trading day; start=end (or end-(days-1))
  if mode == "daily":
    end_s = args.end or last_completed_trading_day_et()
    end_d = _parse_yyyy_mm_dd(end_s)

    days = args.days or 1
    start_s = args.start
    start_d = _parse_yyyy_mm_dd(start_s) if start_s else (end_d - timedelta(days=days - 1))

    return start_d, end_d, "daily"

  # backfill: requires start; end default today
  if mode == "backfill":
    if not args.start:
      raise RuntimeError("--mode backfill requires --start YYYY-MM-DD")
    start_d = _parse_yyyy_mm_dd(args.start)
    end_s = args.end or today_et()
    end_d = _parse_yyyy_mm_dd(end_s)
    return start_d, end_d, "backfill"

  # range: flexible (start optional via lookback)
  if mode == "range":
    end_s = args.end or today_et()
    end_d = _parse_yyyy_mm_dd(end_s)
    if args.start:
      start_d = _parse_yyyy_mm_dd(args.start)
    else:
      lookback = args.lookback_days or int(os.environ.get("LOOKBACK_DAYS", "180"))
      start_d = end_d - timedelta(days=lookback)
    return start_d, end_d, "range"

  raise RuntimeError(f"Unknown mode: {mode}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", default="auto", choices=["auto", "daily", "backfill", "range"],
                  help="auto: incremental if DB has data else seed; daily/backfill/range explicit")
  ap.add_argument("--start", default=None, help="YYYY-MM-DD (required for backfill)")
  ap.add_argument("--end", default=None, help="YYYY-MM-DD (default today_et)")
  ap.add_argument("--days", type=int, default=None, help="daily mode: number of days (default 1)")
  ap.add_argument("--lookback-days", type=int, default=None, help="range mode: lookback if --start not set")

  ap.add_argument("--use-universe", action="store_true", help="Use universe_daily as ticker source")
  ap.add_argument("--universe-date", default=None, help="Universe date YYYY-MM-DD (default latest)")
  ap.add_argument("--universe-source", default="alpaca", help="Universe source label (default alpaca)")
  ap.add_argument("--tickers-path", default=None, help="Fallback tickers file path (default env TICKERS_PATH)")
  ap.add_argument("--limit", type=int, default=None, help="Limit number of tickers (testing)")

  ap.add_argument("--batch-size", type=int, default=300, help="Number of symbols per request")
  ap.add_argument("--chunk-days", type=int, default=30, help="Split date range into chunks (days)")
  ap.add_argument("--page-limit", type=int, default=10000, help="Alpaca API page limit (max 10000)")

  ap.add_argument("--feed", default=None, choices=["iex", "sip"], help="Data feed (default env ALPACA_DATA_FEED or sip)")
  ap.add_argument("--adjustment", default=None, choices=["raw", "split", "dividend", "all"],
                  help="Corporate action adjustment (default env ALPACA_ADJUSTMENT or raw)")
  ap.add_argument("--timeframe", default="1Day", help="Alpaca timeframe string (default 1Day)")

  args = ap.parse_args()

  init_db()

  feed = (args.feed or os.environ.get("ALPACA_DATA_FEED", "sip")).lower()
  adjustment = (args.adjustment or os.environ.get("ALPACA_ADJUSTMENT", "raw")).lower()

  tickers_path = args.tickers_path or os.environ.get("TICKERS_PATH", "/app/tickers.txt")

  if args.use_universe or os.environ.get("USE_UNIVERSE", "0") in ("1", "true", "yes", "y"):
    tickers = _read_tickers_from_universe(args.universe_date, args.universe_source, args.limit)
    if not tickers:
      raise RuntimeError("No tickers found in universe_daily. Run universe_fetch first.")
  else:
    tickers = _read_tickers_file(tickers_path)
    if args.limit:
      tickers = tickers[:args.limit]

  if not tickers:
    raise RuntimeError("Ticker list is empty.")

  start_d, end_d, mode_used = resolve_date_range(args)

  log.info(f"Fetching EOD bars: {len(tickers)} tickers | {start_d.isoformat()} -> {end_d.isoformat()} | mode={mode_used} feed={feed} adj={adjustment}")
  date_chunks = _chunk_date_ranges(start_d, end_d, args.chunk_days)

  total_upserted = 0
  batches = list(_batches(tickers, args.batch_size))

  for ci, (cs, ce) in enumerate(date_chunks, start=1):
    start_dt, end_dt = _date_to_utc_range_inclusive(cs, ce)
    log.info(f"Chunk {ci}/{len(date_chunks)}: {cs.isoformat()} -> {ce.isoformat()} (UTC {start_dt.isoformat()} -> {end_dt.isoformat()})")

    for bi, syms in enumerate(batches, start=1):
      try:
        n = fetch_bars_batch(
          symbols=syms,
          start_dt_utc=start_dt,
          end_dt_utc=end_dt,
          timeframe=args.timeframe,
          feed=feed,
          adjustment=adjustment,
          page_limit=args.page_limit,
        )
        total_upserted += n
        if bi % 10 == 0 or bi == len(batches):
          log.info(f"  batch {bi}/{len(batches)} upserted={n} (total_upserted={total_upserted})")
      except Exception as e:
        log.exception(f"Failed batch {bi}/{len(batches)} for chunk {ci}/{len(date_chunks)}: {e}")
        raise

  log.info(f"Done. Upserted rows: {total_upserted}")


if __name__ == "__main__":
  main()
