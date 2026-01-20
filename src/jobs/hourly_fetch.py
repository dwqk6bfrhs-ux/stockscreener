import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("hourly_fetch")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
DEFAULT_TIMEFRAME = "1Hour"

# Keep URL sizes reasonable; Alpaca supports many symbols, but too many makes huge URLs.
SYMBOLS_PER_REQUEST = 200


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


def _require_env(name: str) -> str:
  v = os.environ.get(name)
  if not v:
    raise RuntimeError(f"Missing required env: {name}")
  return v


def alpaca_headers() -> dict[str, str]:
  key = _require_env("ALPACA_API_KEY")
  sec = _require_env("ALPACA_SECRET_KEY")
  return {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": sec,
    "Accept": "application/json",
  }


def parse_yyyy_mm_dd(s: str) -> date:
  return datetime.strptime(s, "%Y-%m-%d").date()


def iso_z(dt: datetime) -> str:
  # Alpaca accepts RFC3339. Use 'Z' for UTC.
  dt_utc = dt.astimezone(UTC)
  return dt_utc.isoformat().replace("+00:00", "Z")


def et_date_to_utc_range(d: date) -> Tuple[str, str]:
  """
  Convert an ET calendar day to a [start,end) UTC range that covers that ET day.
  Works across DST transitions.
  """
  start_et = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=ET)
  end_et = start_et + timedelta(days=1)
  return iso_z(start_et), iso_z(end_et)


def chunk_list(xs: List[str], n: int) -> Iterable[List[str]]:
  for i in range(0, len(xs), n):
    yield xs[i:i+n]


def read_tickers_from_file(path: str) -> List[str]:
  out: List[str] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      t = line.strip()
      if not t or t.startswith("#"):
        continue
      out.append(t.upper())
  return out


def read_universe_tickers(date_et: str, limit: int | None) -> List[str]:
  with connect() as conn:
    rows = conn.execute(
      "SELECT ticker FROM universe_daily WHERE date=? ORDER BY ticker",
      (date_et,),
    ).fetchall()
  tickers = [r[0].upper() for r in rows]
  if limit is not None:
    tickers = tickers[:limit]
  return tickers


def get_tickers(args, effective_date_et: str) -> List[str]:
  if args.use_universe:
    tickers = read_universe_tickers(effective_date_et, args.limit)
    if not tickers:
      raise RuntimeError(f"No universe_daily rows for date={effective_date_et}. Run universe_fetch first.")
  else:
    path = os.environ.get("TICKERS_PATH", "/app/tickers.txt")
    tickers = read_tickers_from_file(path)
    if args.limit is not None:
      tickers = tickers[:args.limit]

  # Ensure benchmarks exist (harmless if duplicated)
  for b in ("SPY", "IWM"):
    if b not in tickers:
      tickers.append(b)

  return tickers


def build_bars_url(
  symbols: List[str],
  start_utc: str,
  end_utc: str,
  timeframe: str,
  feed: str,
  adjustment: str,
  page_token: str | None,
  limit: int = 10000,
) -> str:
  params = {
    "symbols": ",".join(symbols),
    "timeframe": timeframe,
    "start": start_utc,
    "end": end_utc,
    "adjustment": adjustment,
    "feed": feed,
    "limit": str(limit),
  }
  if page_token:
    params["page_token"] = page_token
  return f"{ALPACA_BARS_URL}?{urlencode(params)}"


def fetch_hourly_bars_batch(
  symbols: List[str],
  start_utc: str,
  end_utc: str,
  feed: str,
  adjustment: str,
  headers: dict[str, str],
) -> List[Tuple[str, str, str, float, float, float, float, float]]:
  """
  Returns rows: (ticker, ts, date_et, open, high, low, close, volume)
  """
  rows: List[Tuple[str, str, str, float, float, float, float, float]] = []
  page_token: str | None = None

  while True:
    url = build_bars_url(
      symbols=symbols,
      start_utc=start_utc,
      end_utc=end_utc,
      timeframe=DEFAULT_TIMEFRAME,
      feed=feed,
      adjustment=adjustment,
      page_token=page_token,
    )
    payload = _alpaca_http_get(url, headers=headers)

    bars_by_symbol = payload.get("bars") or {}
    # bars_by_symbol: { "AAPL": [ {t,o,h,l,c,v,...}, ... ], ... }
    for ticker, bars in bars_by_symbol.items():
      for b in bars:
        ts = b.get("t")
        if not ts:
          continue
        # Parse to ET date
        dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
        dt_et = dt_utc.astimezone(ET)
        date_et = dt_et.date().isoformat()

        o = float(b.get("o")) if b.get("o") is not None else None
        h = float(b.get("h")) if b.get("h") is not None else None
        l = float(b.get("l")) if b.get("l") is not None else None
        c = float(b.get("c")) if b.get("c") is not None else None
        v = float(b.get("v")) if b.get("v") is not None else None

        # store None as NULL; SQLite is fine with that
        rows.append((ticker, ts, date_et, o, h, l, c, v))

    page_token = payload.get("next_page_token")
    if not page_token:
      break

  return rows


def upsert_hourly_rows(rows: List[Tuple[str, str, str, float, float, float, float, float]]) -> int:
  if not rows:
    return 0
  sql = """
  INSERT INTO prices_hourly (ticker, ts, date_et, open, high, low, close, volume)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(ticker, ts) DO UPDATE SET
    date_et=excluded.date_et,
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


def iter_date_chunks(start_d: date, end_d: date, chunk_days: int = 30) -> Iterable[Tuple[date, date]]:
  cur = start_d
  while cur <= end_d:
    nxt = min(end_d, cur + timedelta(days=chunk_days - 1))
    yield cur, nxt
    cur = nxt + timedelta(days=1)


def resolve_mode_and_dates(args) -> Tuple[date, date, str]:
  mode = args.mode

  if mode == "daily":
    # last completed trading day (ET)
    end_s = args.end or last_completed_trading_day_et()
    end_d = parse_yyyy_mm_dd(end_s)

    days = args.days or 1
    if args.start:
      start_d = parse_yyyy_mm_dd(args.start)
    else:
      start_d = end_d - timedelta(days=days - 1)

    return start_d, end_d, "daily"

  if mode == "range":
    if not args.start or not args.end:
      raise RuntimeError("range mode requires --start and --end")
    start_d = parse_yyyy_mm_dd(args.start)
    end_d = parse_yyyy_mm_dd(args.end)
    return start_d, end_d, "range"

  raise RuntimeError(f"Unknown mode: {mode}")


def parse_args():
  p = argparse.ArgumentParser(description="Fetch hourly OHLCV bars from Alpaca into prices_hourly.")
  p.add_argument("--use-universe", action="store_true", help="Use tickers from universe_daily for the effective date.")
  p.add_argument("--date", default=None, help="Universe date (ET) to read from universe_daily (default=end date).")
  p.add_argument("--limit", type=int, default=None, help="Limit number of tickers (dev throttle).")

  p.add_argument("--mode", choices=["daily", "range"], default="daily")
  p.add_argument("--start", default=None)
  p.add_argument("--end", default=None)
  p.add_argument("--days", type=int, default=None, help="In daily mode: fetch last N calendar days ending at end date.")

  p.add_argument("--feed", choices=["iex", "sip"], default=os.environ.get("ALPACA_FEED", "iex"))
  p.add_argument("--adjustment", choices=["raw", "split", "dividend", "all"], default=os.environ.get("ALPACA_ADJ", "raw"))

  p.add_argument("--chunk-days", type=int, default=30, help="Range chunk size in calendar days.")
  return p.parse_args()


def main():
  init_db()
  args = parse_args()
  enable_hourly = os.environ.get("ENABLE_HOURLY", "1")
  if str(enable_hourly).lower() in ("0", "false", "no", "off"):
    log.info(f"Hourly fetch disabled via ENABLE_HOURLY={enable_hourly}. Skipping.")
    return

  start_d, end_d, mode = resolve_mode_and_dates(args)
  log.info(f"Hourly fetch: {start_d} -> {end_d} | mode={mode} feed={args.feed} adj={args.adjustment}")

  # Determine which universe snapshot date to use when --use-universe
  effective_date_et = args.date or end_d.isoformat()

  tickers = get_tickers(args, effective_date_et)
  log.info(f"Tickers: {len(tickers)} (use_universe={args.use_universe}, universe_date={effective_date_et}, limit={args.limit})")

  headers = alpaca_headers()
  total_upserted = 0

  for (cstart, cend) in iter_date_chunks(start_d, end_d, chunk_days=args.chunk_days):
    start_utc, _ = et_date_to_utc_range(cstart)
    _, end_utc = et_date_to_utc_range(cend + timedelta(days=1))  # end is exclusive
    # Simpler: use end as (cend+1 day at 00:00 ET)
    # But we already have helper; compute explicitly:
    start_utc, _ = et_date_to_utc_range(cstart)
    end_utc = et_date_to_utc_range(cend + timedelta(days=1))[0]

    log.info(f"Chunk: {cstart} -> {cend} (UTC {start_utc} -> {end_utc})")

    for sym_chunk in chunk_list(tickers, SYMBOLS_PER_REQUEST):
      rows = fetch_hourly_bars_batch(
        symbols=sym_chunk,
        start_utc=start_utc,
        end_utc=end_utc,
        feed=args.feed,
        adjustment=args.adjustment,
        headers=headers,
      )
      n = upsert_hourly_rows(rows)
      total_upserted += n
      log.info(f"  symbols={len(sym_chunk)} fetched_rows={len(rows)} upserted={n} total_upserted={total_upserted}")

  log.info(f"Done. Upserted rows: {total_upserted}")


if __name__ == "__main__":
  main()
