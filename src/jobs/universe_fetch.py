import os
import json
import argparse
from typing import Optional, List

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("universe_fetch")


def _env(name: str, default: Optional[str] = None) -> str:
  v = os.environ.get(name, default)
  if v is None or v == "":
    raise RuntimeError(f"Missing required env var: {name}")
  return v


def fetch_alpaca_assets(
  include_otc: bool,
  exchanges: Optional[List[str]],
  limit: Optional[int],
) -> List[dict]:
  """
  Returns list of dict assets with at least: symbol, exchange, status, tradable, asset_class, name, ...
  """
  try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetStatus, AssetClass
  except Exception as e:
    raise RuntimeError(
      "alpaca-py is not available in the image. Ensure it exists in requirements.txt (alpaca-py)."
    ) from e

  api_key = _env("ALPACA_API_KEY")
  secret_key = _env("ALPACA_SECRET_KEY")

  # paper flag only affects trading endpoint; assets list is fine either way.
  paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

  client = TradingClient(api_key, secret_key, paper=paper)
  req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
  assets = client.get_all_assets(req)

  out = []
  for a in assets:
    symbol = getattr(a, "symbol", None)
    if not symbol:
      continue

    tradable = bool(getattr(a, "tradable", False))
    if not tradable:
      continue

    exchange = (getattr(a, "exchange", None) or "").upper()
    if (not include_otc) and exchange == "OTC":
      continue

    if exchanges and exchange and exchange not in exchanges:
      continue

    # Keep meta small but useful; you can add more later.
    meta = {
      "symbol": symbol,
      "name": getattr(a, "name", None),
      "exchange": exchange,
      "asset_class": str(getattr(a, "asset_class", "")),
      "status": str(getattr(a, "status", "")),
      "tradable": tradable,
      "marginable": bool(getattr(a, "marginable", False)),
      "shortable": bool(getattr(a, "shortable", False)),
      "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
      "fractionable": bool(getattr(a, "fractionable", False)),
    }
    out.append(meta)

    if limit and len(out) >= limit:
      break

  return out


def upsert_universe_rows(date_str: str, source: str, assets: List[dict]) -> int:
  if not assets:
    return 0

  rows = []
  for a in assets:
    symbol = a["symbol"]
    rows.append((
      date_str,
      symbol,
      source,
      json.dumps(a, ensure_ascii=False),
    ))

  sql = """
  INSERT INTO universe_daily (date, ticker, source, meta_json)
  VALUES (?, ?, ?, ?)
  ON CONFLICT(date, ticker, source) DO UPDATE SET
    meta_json = excluded.meta_json
  """

  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()

  return len(rows)


def ensure_benchmarks(date_str: str, source: str, tickers: List[str]) -> int:
  if not tickers:
    return 0

  rows = []
  for t in tickers:
    meta = {"symbol": t, "forced": True}
    rows.append((date_str, t, source, json.dumps(meta)))

  sql = """
  INSERT INTO universe_daily (date, ticker, source, meta_json)
  VALUES (?, ?, ?, ?)
  ON CONFLICT(date, ticker, source) DO UPDATE SET
    meta_json = excluded.meta_json
  """

  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()

  return len(rows)


def delete_universe_snapshot(date_str: str, source: str) -> int:
  """
  Delete all universe_daily rows for a given (date, source).
  Useful to "refresh" a snapshot if you want the DB to match the latest upstream list exactly.
  """
  with connect() as conn:
    cur = conn.execute(
      "DELETE FROM universe_daily WHERE date=? AND source=?",
      (date_str, source),
    )
    conn.commit()
  return cur.rowcount


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Universe date, YYYY-MM-DD. Default: last_completed_trading_day_et()")
  ap.add_argument("--source", default="alpaca", help="Source label stored in DB")
  ap.add_argument("--include-otc", action="store_true", help="Include OTC exchange assets")
  ap.add_argument(
    "--exchange",
    action="append",
    default=None,
    help="Exchange filter, repeatable. Example: --exchange NYSE --exchange NASDAQ",
  )
  ap.add_argument("--limit", type=int, default=None, help="Limit count for testing")
  ap.add_argument(
    "--force-benchmarks",
    default="SPY,IWM",
    help="Comma-separated tickers to force into universe_daily (default SPY,IWM)",
  )
  ap.add_argument(
    "--replace",
    action="store_true",
    help="Delete existing universe_daily rows for (date, source) before insert",
  )
  args = ap.parse_args()

  date_str = args.date or last_completed_trading_day_et()
  exchanges = [x.upper() for x in args.exchange] if args.exchange else None
  force_bm = [x.strip().upper() for x in (args.force_benchmarks or "").split(",") if x.strip()]

  init_db()  # IMPORTANT: ensure tables exist before delete/insert

  if args.replace:
    log.info(f"Replacing universe snapshot: date={date_str} source={args.source}")
    deleted = delete_universe_snapshot(date_str, args.source)
    log.info(f"Deleted universe_daily rows: {deleted}")

  log.info(
    f"Fetching universe from Alpaca: date={date_str} include_otc={args.include_otc} exchanges={exchanges} limit={args.limit}"
  )
  assets = fetch_alpaca_assets(include_otc=args.include_otc, exchanges=exchanges, limit=args.limit)

  n = upsert_universe_rows(date_str=date_str, source=args.source, assets=assets)
  nbm = ensure_benchmarks(date_str=date_str, source=args.source, tickers=force_bm)

  log.info(f"Upserted universe_daily rows: {n} (+forced benchmarks={nbm})")


if __name__ == "__main__":
  main()
