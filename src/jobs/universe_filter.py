import argparse
import json
from typing import Dict, List, Tuple

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("universe_filter")


def _chunk(xs: List[str], n: int) -> List[List[str]]:
  return [xs[i:i+n] for i in range(0, len(xs), n)]


def read_universe(date_et: str, source: str, limit: int | None) -> List[Tuple[str, str]]:
  """
  Returns [(ticker, meta_json), ...]
  """
  sql = "SELECT ticker, COALESCE(meta_json,'{}') FROM universe_daily WHERE date=? AND source=? ORDER BY ticker"
  params = [date_et, source]
  if limit is not None:
    sql += " LIMIT ?"
    params.append(limit)

  with connect() as conn:
    rows = conn.execute(sql, params).fetchall()
  return [(r[0], r[1]) for r in rows]


def hourly_counts(date_et: str, tickers: List[str]) -> Dict[str, int]:
  """
  COUNT(*) from prices_hourly for each ticker on date_et.
  Uses IN chunks to avoid SQLite param limits.
  """
  out: Dict[str, int] = {t: 0 for t in tickers}
  if not tickers:
    return out

  with connect() as conn:
    for chunk in _chunk(tickers, 900):
      placeholders = ",".join(["?"] * len(chunk))
      sql = f"""
        SELECT ticker, COUNT(*) AS cnt
        FROM prices_hourly
        WHERE date_et=? AND ticker IN ({placeholders})
        GROUP BY ticker
      """
      rows = conn.execute(sql, [date_et] + chunk).fetchall()
      for t, c in rows:
        out[str(t).upper()] = int(c)
  return out


def write_filtered_universe(
  date_et: str,
  output_source: str,
  rows: List[Tuple[str, str]],
  counts: Dict[str, int],
  min_hourly_bars: int,
  prune: bool,
) -> int:
  """
  rows: [(ticker, meta_json)]
  """
  if prune:
    with connect() as conn:
      conn.execute("DELETE FROM universe_daily WHERE date=? AND source=?", (date_et, output_source))
      conn.commit()

  ins = """
    INSERT INTO universe_daily (date, ticker, source, meta_json)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(date, ticker, source) DO UPDATE SET
      meta_json=excluded.meta_json
  """

  out_rows = []
  for ticker, meta_json in rows:
    t = ticker.upper()
    cnt = int(counts.get(t, 0))

    # Merge original meta_json + our annotations
    try:
      meta = json.loads(meta_json) if meta_json else {}
      if not isinstance(meta, dict):
        meta = {}
    except Exception:
      meta = {}

    meta["hourly_cnt"] = cnt
    meta["min_hourly_bars"] = min_hourly_bars
    meta["filtered_intraday"] = True

    out_rows.append((date_et, t, output_source, json.dumps(meta, ensure_ascii=False)))

  if not out_rows:
    return 0

  with connect() as conn:
    conn.executemany(ins, out_rows)
    conn.commit()

  return len(out_rows)


def main():
  ap = argparse.ArgumentParser(description="Filter universe_daily into an intraday-usable universe based on prices_hourly coverage.")
  ap.add_argument("--date", default=None, help="ET date (YYYY-MM-DD). Default: last_completed_trading_day_et()")
  ap.add_argument("--input-source", default="alpaca", help="Universe source to read (default: alpaca)")
  ap.add_argument("--output-source", default="alpaca_intraday", help="Universe source to write (default: alpaca_intraday)")
  ap.add_argument("--min-hourly-bars", type=int, default=4, help="Keep tickers with >= this many hourly bars on date_et")
  ap.add_argument("--limit", type=int, default=None, help="Optional limit (dev throttle)")
  ap.add_argument("--no-prune", action="store_true", help="Do not delete existing output-source rows for the date")
  args = ap.parse_args()

  date_et = args.date or last_completed_trading_day_et()
  init_db()

  src_rows = read_universe(date_et=date_et, source=args.input_source, limit=args.limit)
  tickers = [t.upper() for (t, _) in src_rows]
  log.info(f"Read universe: date={date_et} source={args.input_source} tickers={len(tickers)} limit={args.limit}")

  counts = hourly_counts(date_et=date_et, tickers=tickers)

  # Filter
  kept = [(t, mj) for (t, mj) in src_rows if counts.get(t.upper(), 0) >= args.min_hourly_bars]
  zero = sum(1 for t in tickers if counts.get(t.upper(), 0) == 0)
  log.info(f"Hourly coverage: zero={zero} kept={len(kept)} min_hourly_bars={args.min_hourly_bars}")

  n = write_filtered_universe(
    date_et=date_et,
    output_source=args.output_source,
    rows=kept,
    counts=counts,
    min_hourly_bars=args.min_hourly_bars,
    prune=(not args.no_prune),
  )

  log.info(f"Wrote filtered universe: date={date_et} source={args.output_source} rows={n}")


if __name__ == "__main__":
  main()
