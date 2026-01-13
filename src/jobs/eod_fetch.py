import os
from datetime import datetime, timedelta
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.common.db import init_db, connect
from src.common.logging import setup_logger

log = setup_logger("eod_fetch")


def load_tickers() -> list[str]:
  path = os.environ.get("TICKERS_PATH", "/app/tickers.txt")
  with open(path, "r", encoding="utf-8") as f:
    out = []
    for line in f:
      t = line.strip().upper()
      if t and not t.startswith("#"):
        out.append(t)
  return sorted(set(out))


def upsert_prices(df: pd.DataFrame):
  if df.empty:
    return
  rows = []
  for _, r in df.iterrows():
    rows.append((r["ticker"], r["date"], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])))
  with connect() as conn:
    conn.executemany(
      """
      INSERT INTO prices_daily(ticker,date,open,high,low,close,volume)
      VALUES(?,?,?,?,?,?,?)
      ON CONFLICT(ticker,date) DO UPDATE SET
        open=excluded.open,
        high=excluded.high,
        low=excluded.low,
        close=excluded.close,
        volume=excluded.volume
      """,
      rows,
    )
    conn.commit()


def main():
  init_db()
  key = os.environ["ALPACA_API_KEY"]
  secret = os.environ["ALPACA_SECRET_KEY"]
  client = StockHistoricalDataClient(key, secret)

  tickers = load_tickers()
  if not tickers:
    raise RuntimeError("tickers.txt is empty")

  end = datetime.utcnow().date()
  start = end - timedelta(days=160)

  log.info(f"Fetching EOD bars: {len(tickers)} tickers | {start} -> {end}")

  batch_size = 200
  all_df = []

  for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    req = StockBarsRequest(
      symbol_or_symbols=batch,
      timeframe=TimeFrame.Day,
      start=datetime.combine(start, datetime.min.time()),
      end=datetime.combine(end, datetime.min.time()),
      adjustment=None,
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    if df.empty:
      continue
    df.rename(columns={"symbol": "ticker"}, inplace=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    df = df[["ticker","date","open","high","low","close","volume"]]
    all_df.append(df)

  if all_df:
    final = pd.concat(all_df, ignore_index=True)
    upsert_prices(final)
    log.info(f"Upserted rows: {len(final)}")
  else:
    log.warning("No bars returned.")


if __name__ == "__main__":
  main()
