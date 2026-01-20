from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger

log = setup_logger("backtest_runner")


@dataclass
class Position:
  ticker: str
  qty: float
  entry_px: float
  entry_date: str


def _run_job(args: List[str], extra_env: Dict[str, str] | None = None) -> None:
  env = os.environ.copy()
  if extra_env:
    env.update(extra_env)
  log.info(f"Running job: {' '.join(args)}")
  subprocess.run(args, check=True, env=env)


def _trade_dates(start: str, end: str) -> List[str]:
  with connect() as conn:
    df = pd.read_sql_query(
      "SELECT DISTINCT date FROM prices_daily WHERE date BETWEEN ? AND ? ORDER BY date",
      conn,
      params=(start, end),
    )
  return df["date"].astype(str).tolist()


def _close_price(date_str: str, ticker: str) -> float | None:
  with connect() as conn:
    row = conn.execute(
      "SELECT close FROM prices_daily WHERE date=? AND ticker=?",
      (date_str, ticker),
    ).fetchone()
  if not row or row[0] is None:
    return None
  try:
    return float(row[0])
  except Exception:
    return None


def _read_orders(date_str: str, book: str) -> pd.DataFrame:
  with connect() as conn:
    return pd.read_sql_query(
      "SELECT date, ticker, side, qty FROM orders_daily WHERE date=? AND book=? ORDER BY side, ticker",
      conn,
      params=(date_str, book),
    )


def _reset_book(book: str, start: str, end: str) -> None:
  with connect() as conn:
    conn.execute("DELETE FROM positions WHERE book=?", (book,))
    conn.execute("DELETE FROM orders_daily WHERE book=? AND date BETWEEN ? AND ?", (book, start, end))
    conn.commit()


def _simulate(
  dates: List[str],
  book: str,
  initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  cash = float(initial_cash)
  positions: Dict[str, Position] = {}
  trades: List[dict] = []
  equity_rows: List[dict] = []

  for date_str in dates:
    orders = _read_orders(date_str, book)

    # process sells first
    for _, row in orders[orders["side"] == "SELL"].iterrows():
      ticker = str(row["ticker"])
      qty = float(row["qty"] or 0.0)
      if qty <= 0:
        continue
      pos = positions.get(ticker)
      if not pos:
        continue
      close_px = _close_price(date_str, ticker)
      if close_px is None:
        continue

      sell_qty = min(qty, pos.qty)
      pnl = (close_px - pos.entry_px) * sell_qty
      cash += close_px * sell_qty
      pos.qty -= sell_qty

      trades.append({
        "ticker": ticker,
        "entry_date": pos.entry_date,
        "entry_px": pos.entry_px,
        "exit_date": date_str,
        "exit_px": close_px,
        "qty": sell_qty,
        "pnl": pnl,
        "hold_days": (pd.to_datetime(date_str) - pd.to_datetime(pos.entry_date)).days,
      })

      if pos.qty <= 0:
        positions.pop(ticker, None)

    # process buys
    for _, row in orders[orders["side"] == "BUY"].iterrows():
      ticker = str(row["ticker"])
      qty = float(row["qty"] or 0.0)
      if qty <= 0:
        continue
      close_px = _close_price(date_str, ticker)
      if close_px is None:
        continue

      cost = close_px * qty
      cash -= cost
      if ticker in positions:
        pos = positions[ticker]
        new_qty = pos.qty + qty
        if new_qty <= 0:
          continue
        pos.entry_px = (pos.entry_px * pos.qty + close_px * qty) / new_qty
        pos.qty = new_qty
      else:
        positions[ticker] = Position(
          ticker=ticker,
          qty=qty,
          entry_px=close_px,
          entry_date=date_str,
        )

    # mark-to-market
    positions_value = 0.0
    for pos in positions.values():
      close_px = _close_price(date_str, pos.ticker)
      if close_px is None:
        continue
      positions_value += close_px * pos.qty

    equity = cash + positions_value
    equity_rows.append({
      "date": date_str,
      "cash": cash,
      "positions_value": positions_value,
      "equity": equity,
      "n_positions": len(positions),
    })

  eq = pd.DataFrame(equity_rows)
  if not eq.empty:
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
  trades_df = pd.DataFrame(trades)
  return eq, trades_df


def _summary(eq: pd.DataFrame, trades: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
  if eq.empty:
    return pd.DataFrame([{
      "total_return": 0.0,
      "cagr": 0.0,
      "max_drawdown": 0.0,
      "win_rate": 0.0,
      "total_trades": 0,
      "avg_trade_pnl": 0.0,
      "spy_return": 0.0,
      "iwm_return": 0.0,
    }])

  start = float(eq["equity"].iloc[0])
  end = float(eq["equity"].iloc[-1])
  total_return = (end / start) - 1.0 if start else 0.0
  dates = pd.to_datetime(eq["date"])
  years = max((dates.max() - dates.min()).days, 1) / 365.25
  cagr = (end / start) ** (1 / years) - 1.0 if start > 0 else 0.0

  peak = eq["equity"].cummax()
  max_dd = float(((eq["equity"] / peak) - 1.0).min()) if len(eq) else 0.0

  win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
  avg_trade_pnl = float(trades["pnl"].mean()) if not trades.empty else 0.0

  spy_return = float(bench.get("SPY", pd.Series([0.0])).iloc[-1]) if not bench.empty else 0.0
  iwm_return = float(bench.get("IWM", pd.Series([0.0])).iloc[-1]) if not bench.empty else 0.0

  return pd.DataFrame([{
    "total_return": total_return,
    "cagr": cagr,
    "max_drawdown": max_dd,
    "win_rate": win_rate,
    "total_trades": int(len(trades)),
    "avg_trade_pnl": avg_trade_pnl,
    "spy_return": spy_return,
    "iwm_return": iwm_return,
  }])


def _bench_returns(dates: List[str]) -> pd.DataFrame:
  if not dates:
    return pd.DataFrame()
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT date, ticker, close
      FROM prices_daily
      WHERE date IN ({ph})
        AND ticker IN ('SPY', 'IWM')
      """.format(ph=",".join(["?"] * len(dates))),
      conn,
      params=dates,
    )
  if df.empty:
    return pd.DataFrame()
  df["date"] = pd.to_datetime(df["date"])
  date_index = pd.to_datetime(pd.Series(dates))
  out = {"date": [d.date().isoformat() for d in date_index]}
  for t in ["SPY", "IWM"]:
    g = df[df["ticker"] == t].sort_values("date")
    if g.empty:
      continue
    ret = g.set_index("date")["close"].pct_change().fillna(0.0)
    ret = ret.reindex(date_index, fill_value=0.0)
    out[t] = (1.0 + ret).cumprod() - 1.0
  return pd.DataFrame(out)


def main() -> None:
  ap = argparse.ArgumentParser(description="Deterministic backtest runner (signals -> report -> orders -> replay).")
  ap.add_argument("--start", required=True)
  ap.add_argument("--end", required=True)
  ap.add_argument("--book", default="combined")
  ap.add_argument("--run-id", default=None)
  ap.add_argument("--initial-cash", type=float, default=100000.0)
  ap.add_argument("--reset-book", action="store_true", help="Clear orders/positions for this book before running.")
  ap.add_argument(
    "--tickers-source",
    choices=["universe", "tickers", "prices"],
    default="universe",
    help="Ticker source for signal generation (default: universe).",
  )
  ap.add_argument("--tickers-path", default=os.environ.get("TICKERS_PATH", "/app/tickers.txt"))

  ap.add_argument("--top-x", type=int, default=20)
  ap.add_argument("--max-entries", type=int, default=20)
  ap.add_argument("--overlap-bonus", type=float, default=0.25)
  ap.add_argument("--require-overlap", action="store_true")
  args = ap.parse_args()

  init_db()
  dates = _trade_dates(args.start, args.end)
  if not dates:
    raise RuntimeError("No trading dates found in prices_daily for requested range.")

  if args.reset_book:
    _reset_book(args.book, dates[0], dates[-1])

  for d in dates:
    _run_job([
      "python", "-m", "src.jobs.generate_signals",
      "--date", d,
      "--tickers-source", args.tickers_source,
      "--tickers-path", args.tickers_path,
    ])
    _run_job(["python", "-m", "src.jobs.report"], extra_env={"REPORT_DATE": d})
    _run_job([
      "python", "-m", "src.jobs.generate_orders",
      "--date", d,
      "--book", args.book,
      "--top-x", str(args.top_x),
      "--max-entries", str(args.max_entries),
      "--overlap-bonus", str(args.overlap_bonus),
    ] + (["--require-overlap"] if args.require_overlap else []))

  eq, trades = _simulate(dates, args.book, args.initial_cash)
  bench = _bench_returns(dates)
  if not bench.empty:
    eq = eq.merge(bench, on="date", how="left")

  summary = _summary(eq, trades, bench)

  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / "backtests"
  run_id = args.run_id or f"runner_{dates[0]}_{dates[-1]}_{args.book}"
  run_path = out_dir / run_id
  run_path.mkdir(parents=True, exist_ok=True)

  eq.to_csv(run_path / "backtest_equity.csv", index=False)
  summary.to_csv(run_path / "backtest_summary.csv", index=False)
  trades.to_csv(run_path / "backtest_trades.csv", index=False)

  log.info(f"Backtest runner complete: {run_path}")


if __name__ == "__main__":
  main()
