import os
import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import today_et

log = setup_logger("report")


def read_prices() -> pd.DataFrame:
  with connect() as conn:
    return pd.read_sql_query("SELECT ticker, date, open, high, low, close, volume FROM prices_daily", conn)


def main():
  init_db()
  out_dir = os.environ.get("OUTPUT_DIR", "/app/outputs")
  date = today_et()
  day_dir = os.path.join(out_dir, date)
  os.makedirs(day_dir, exist_ok=True)

  df = read_prices()
  if df.empty:
    raise RuntimeError("No prices in DB. Run eod_fetch first.")

  df["date"] = pd.to_datetime(df["date"])
  last_date = df["date"].max()
  dlast = df[df["date"] == last_date].copy()

  # MVP 占位：按 “成交量 * 当日振幅” 排序
  dlast["range_pct"] = (dlast["high"] - dlast["low"]) / dlast["close"].replace(0, pd.NA)
  dlast["score"] = dlast["volume"].fillna(0) * dlast["range_pct"].fillna(0)

  watch = dlast.sort_values("score", ascending=False).head(30)[["ticker","close","volume","range_pct","score"]]
  action = watch.head(5).copy()

  watch.to_csv(os.path.join(day_dir, "watch_list.csv"), index=False)
  action.to_csv(os.path.join(day_dir, "action_list.csv"), index=False)

  log.info(f"Generated reports in {day_dir} (action={len(action)}, watch={len(watch)})")


if __name__ == "__main__":
  main()
