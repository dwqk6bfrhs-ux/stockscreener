import os
import json
import sqlite3
import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("report")


def _out_dir_for_date(date: str) -> str:
  out_dir = os.environ.get("OUTPUT_DIR", "/app/outputs")
  day_dir = os.path.join(out_dir, date)
  os.makedirs(day_dir, exist_ok=True)
  return day_dir


def read_signals(date: str) -> pd.DataFrame:
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT date, ticker, strategy, state, score, stop, meta_json, created_at
      FROM signals_daily
      WHERE date = ?
      """,
      conn,
      params=(date,),
    )
  return df


def read_last_close(date: str, tickers: list[str]) -> pd.DataFrame:
  """
  Pull close/volume/range_pct for the specific date, for ranking and display.
  """
  if not tickers:
    return pd.DataFrame(columns=["ticker","close","volume","range_pct"])
  ph = ",".join(["?"] * len(tickers))
  with connect() as conn:
    df = pd.read_sql_query(
      f"""
      SELECT ticker, close, volume, high, low
      FROM prices_daily
      WHERE date = ? AND ticker IN ({ph})
      """,
      conn,
      params=[date] + tickers,
    )
  if df.empty:
    return df
  df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
  return df[["ticker","close","volume","range_pct"]]


def main():
  init_db()

  date = os.environ.get("REPORT_DATE") or last_completed_trading_day_et()
  day_dir = _out_dir_for_date(date)

  df = read_signals(date)
  if df.empty:
    raise RuntimeError(f"No signals found for {date}. Run generate_signals first.")

  # Normalize types
  df["score"] = pd.to_numeric(df["score"], errors="coerce")

  # Prepare strategy outputs
  strategies = sorted(df["strategy"].unique().tolist())
  summary_lines = []
  summary_lines.append(f"Report date: {date}")
  summary_lines.append(f"Strategies: {', '.join(strategies)}")
  summary_lines.append("")

  for strat in strategies:
    d = df[df["strategy"] == strat].copy()

    # Pull pricing info for tickers present in this strategy
    tickers = d["ticker"].dropna().astype(str).unique().tolist()
    px = read_last_close(date, tickers)

    if not px.empty:
      d = d.merge(px, on="ticker", how="left")

    # Sort by score desc (nulls last)
    d = d.sort_values(["state", "score"], ascending=[True, False])

    # Write full CSV per strategy
    out_csv = os.path.join(day_dir, f"signals_{strat}.csv")
    d.to_csv(out_csv, index=False)

    # Summarize states
    counts = d["state"].value_counts(dropna=False).to_dict()
    summary_lines.append(f"[{strat}] states: " + ", ".join([f"{k}={v}" for k,v in sorted(counts.items())]))

    # Optional: show top entries/exits/watch
    def top_state(state: str, n: int = 10) -> pd.DataFrame:
      x = d[d["state"] == state].copy()
      if x.empty:
        return x
      return x.sort_values("score", ascending=False).head(n)

    for state in ["ENTRY", "EXIT", "WATCH"]:
      top = top_state(state, n=10)
      if top.empty:
        continue
      summary_lines.append(f"  Top {state} (up to 10): " + ", ".join(top["ticker"].astype(str).tolist()))

    summary_lines.append("")

  # Write summary.txt
  summary_path = os.path.join(day_dir, "summary.txt")
  with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

  log.info(f"Wrote reports to {day_dir} (files={len(strategies) + 1})")


if __name__ == "__main__":
  main()
