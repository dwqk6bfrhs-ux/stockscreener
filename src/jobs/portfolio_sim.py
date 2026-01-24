import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.common.db import connect, get_prices_daily_source, init_db
from src.common.logging import setup_logger

log = setup_logger("portfolio_sim")


@dataclass
class Position:
  ticker: str
  strategy: str
  entry_date: str
  entry_px: float
  qty: float
  stop: Optional[float]
  hold_days: int = 0


def _read_signals(start: str, end: str, strategy: Optional[str]) -> pd.DataFrame:
  params: list[str] = [start, end]
  strategy_clause = ""
  if strategy:
    strategy_clause = "AND r.strategy = ?"
    params.append(strategy)

  sql = f"""
    SELECT r.date, r.ticker, r.strategy, r.rank_score, r.meta_json AS rank_meta,
           s.state, s.stop
    FROM rank_scores_daily r
    JOIN signals_daily s
      ON s.date = r.date AND s.ticker = r.ticker AND s.strategy = r.strategy
    WHERE r.date BETWEEN ? AND ?
      AND r.rank_score IS NOT NULL
      {strategy_clause}
    ORDER BY r.date, r.ticker
  """
  with connect() as conn:
    df = pd.read_sql_query(sql, conn, params=params)
  if df.empty:
    return df
  df["date"] = pd.to_datetime(df["date"])
  df["rank_score"] = pd.to_numeric(df["rank_score"], errors="coerce")
  return df


def _parse_meta_features(meta_json: str | None) -> Dict[str, float | None]:
  if not meta_json:
    return {"dv20": None, "atr_pct": None}
  try:
    obj = json.loads(meta_json)
  except Exception:
    return {"dv20": None, "atr_pct": None}
  features = obj.get("features", {})
  return {
    "dv20": features.get("dv20"),
    "atr_pct": features.get("atr_pct"),
  }


def _read_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()
  source = get_prices_daily_source()
  ph = ",".join(["?"] * len(tickers))
  sql = f"""
    SELECT ticker, date, open, high, low, close
    FROM prices_daily
    WHERE source=?
      AND date BETWEEN ? AND ?
      AND ticker IN ({ph})
    ORDER BY ticker, date
  """
  params = [source, start, end] + tickers
  with connect() as conn:
    df = pd.read_sql_query(sql, conn, params=params)
  if df.empty:
    return df
  df["date"] = pd.to_datetime(df["date"])
  for c in ["open", "high", "low", "close"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


def _build_price_lookup(prices: pd.DataFrame) -> Dict[tuple[str, str], Dict[str, float]]:
  lookup: Dict[tuple[str, str], Dict[str, float]] = {}
  for _, row in prices.iterrows():
    lookup[(row["ticker"], row["date"].date().isoformat())] = {
      "open": float(row["open"]) if pd.notna(row["open"]) else None,
      "high": float(row["high"]) if pd.notna(row["high"]) else None,
      "low": float(row["low"]) if pd.notna(row["low"]) else None,
      "close": float(row["close"]) if pd.notna(row["close"]) else None,
    }
  return lookup


def _trade_dates(prices: pd.DataFrame) -> List[str]:
  return sorted(prices["date"].dt.date.astype(str).unique().tolist())


def _entry_price(
  price_row: Dict[str, float],
  execution: str,
) -> Optional[float]:
  if execution == "next_open":
    return price_row.get("open")
  return price_row.get("close")


def _exit_price(
  price_row: Dict[str, float],
  execution: str,
) -> Optional[float]:
  if execution == "next_open":
    return price_row.get("open")
  return price_row.get("close")


def main() -> None:
  ap = argparse.ArgumentParser(description="Stage 2 portfolio simulator (cash/slots constraints).")
  ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
  ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
  ap.add_argument("--strategy", default=None, help="Strategy filter (optional)")
  ap.add_argument("--initial-cash", type=float, default=100.0)
  ap.add_argument("--max-positions", type=int, default=None)
  ap.add_argument("--ticket-size-pct", type=float, default=None)
  ap.add_argument("--min-ticket-pct", type=float, default=0.05)
  ap.add_argument("--hold-days", type=int, default=5)
  ap.add_argument("--exit-on-invalid", action="store_true")
  ap.add_argument("--entry-exec", choices=["close", "next_open"], default="close")
  ap.add_argument("--exit-exec", choices=["close", "next_open"], default="close")
  ap.add_argument("--stop-on-open", action="store_true")
  ap.add_argument("--allow-fractional", action="store_true")
  ap.add_argument("--orders-per-day-limit", type=int, default=None)
  ap.add_argument("--run-id", default=None)
  args = ap.parse_args()

  init_db()
  signals = _read_signals(args.start, args.end, args.strategy)
  if signals.empty:
    raise RuntimeError("No rank_scores_daily entries found for the requested range.")

  signals["features"] = signals["rank_meta"].apply(_parse_meta_features)
  signals["dv20"] = signals["features"].apply(lambda x: x.get("dv20"))
  signals["atr_pct"] = signals["features"].apply(lambda x: x.get("atr_pct"))

  signals = signals[signals["state"] == "ENTRY"].copy()
  if signals.empty:
    raise RuntimeError("No ENTRY signals found for the requested range.")

  tickers = sorted(signals["ticker"].unique().tolist())
  max_horizon = args.hold_days + 2
  start_dt = pd.to_datetime(args.start)
  end_dt = pd.to_datetime(args.end) + pd.Timedelta(days=max_horizon * 2)
  prices = _read_prices(tickers, start_dt.date().isoformat(), end_dt.date().isoformat())
  if prices.empty:
    raise RuntimeError("No price data found for signal tickers.")

  price_lookup = _build_price_lookup(prices)
  dates = _trade_dates(prices)
  dates = [d for d in dates if args.start <= d <= args.end]
  if not dates:
    raise RuntimeError("No trading dates found for the requested range.")

  if args.ticket_size_pct is None and args.max_positions:
    args.ticket_size_pct = 1.0 / float(args.max_positions)
  if args.ticket_size_pct is None:
    raise RuntimeError("Provide --ticket-size-pct or --max-positions.")

  cash = float(args.initial_cash)
  positions: Dict[str, Position] = {}
  trades: List[dict] = []
  equity_rows: List[dict] = []
  pending_entries: Dict[str, List[dict]] = {}
  pending_exits: Dict[str, List[dict]] = {}

  signals_by_date = {d: g.copy() for d, g in signals.groupby(signals["date"].dt.date.astype(str))}

  for idx, date_str in enumerate(dates):
    todays_entries = pending_entries.pop(date_str, [])
    todays_exits = pending_exits.pop(date_str, [])

    def _execute_exit(ticker: str, reason: str, forced_px: Optional[float] = None) -> None:
      nonlocal cash
      pos = positions.get(ticker)
      if not pos:
        return
      price_row = price_lookup.get((ticker, date_str))
      if not price_row:
        return
      exit_px = forced_px if forced_px is not None else _exit_price(price_row, args.exit_exec)
      if exit_px is None:
        return
      pnl = (exit_px - pos.entry_px) * pos.qty
      cash += exit_px * pos.qty
      trades.append({
        "ticker": ticker,
        "strategy": pos.strategy,
        "entry_date": pos.entry_date,
        "exit_date": date_str,
        "entry_px": pos.entry_px,
        "exit_px": exit_px,
        "qty": pos.qty,
        "pnl": pnl,
        "hold_days": pos.hold_days,
        "reason": reason,
      })
      positions.pop(ticker, None)

    for order in todays_exits:
      _execute_exit(order["ticker"], order.get("reason", "exit"), order.get("exit_px"))

    for order in todays_entries:
      ticker = order["ticker"]
      if ticker in positions:
        continue
      if args.max_positions is not None and len(positions) >= args.max_positions:
        continue
      price_row = price_lookup.get((ticker, date_str))
      if not price_row:
        continue
      entry_px = _entry_price(price_row, args.entry_exec)
      if entry_px is None or entry_px <= 0:
        continue

      equity = cash + sum(
        (price_lookup.get((p.ticker, date_str), {}).get("close", p.entry_px) * p.qty)
        for p in positions.values()
      )
      if cash < args.min_ticket_pct * equity:
        continue

      ticket_value = args.ticket_size_pct * equity
      if ticket_value > cash:
        ticket_value = cash
      qty = ticket_value / entry_px if entry_px > 0 else 0.0
      if not args.allow_fractional:
        qty = float(int(qty))
      if qty <= 0:
        continue
      cash -= qty * entry_px
      positions[ticker] = Position(
        ticker=ticker,
        strategy=order["strategy"],
        entry_date=date_str,
        entry_px=entry_px,
        qty=qty,
        stop=order.get("stop"),
        hold_days=0,
      )

    todays_signals = signals_by_date.get(date_str, pd.DataFrame())
    if not todays_signals.empty:
      candidates = todays_signals.copy()
      candidates = candidates.sort_values(
        by=["rank_score", "dv20", "atr_pct"],
        ascending=[False, False, True],
        na_position="last",
      )
      if args.orders_per_day_limit is not None:
        candidates = candidates.head(args.orders_per_day_limit)

      entry_date = date_str
      if args.entry_exec == "next_open" and idx + 1 < len(dates):
        entry_date = dates[idx + 1]
      pending_entries.setdefault(entry_date, [])
      for _, row in candidates.iterrows():
        pending_entries[entry_date].append({
          "ticker": row["ticker"],
          "strategy": row["strategy"],
          "stop": float(row["stop"]) if pd.notna(row["stop"]) else None,
        })

    for pos in positions.values():
      pos.hold_days += 1

    for ticker, pos in list(positions.items()):
      price_row = price_lookup.get((ticker, date_str))
      if not price_row:
        continue
      low = price_row.get("low")
      open_px = price_row.get("open")
      close_px = price_row.get("close")
      if pos.stop is not None and low is not None and low <= pos.stop:
        exit_px = pos.stop
        if args.stop_on_open and open_px is not None and open_px <= pos.stop:
          exit_px = open_px
        _execute_exit(ticker, "stop", exit_px)
        continue

      if pos.hold_days >= args.hold_days:
        if args.exit_exec == "next_open" and idx + 1 < len(dates):
          exit_date = dates[idx + 1]
          pending_exits.setdefault(exit_date, []).append({"ticker": ticker, "reason": "time_exit"})
        else:
          _execute_exit(ticker, "time_exit")
        continue

      if args.exit_on_invalid:
        today_signals = signals_by_date.get(date_str, pd.DataFrame())
        if today_signals.empty:
          continue
        match = today_signals[
          (today_signals["ticker"] == ticker) & (today_signals["strategy"] == pos.strategy)
        ]
        if match.empty or str(match.iloc[0]["state"]) != "ENTRY":
          if args.exit_exec == "next_open" and idx + 1 < len(dates):
            exit_date = dates[idx + 1]
            pending_exits.setdefault(exit_date, []).append({"ticker": ticker, "reason": "invalid_signal"})
          else:
            _execute_exit(ticker, "invalid_signal")

    positions_value = 0.0
    for pos in positions.values():
      price_row = price_lookup.get((pos.ticker, date_str))
      close_px = price_row.get("close") if price_row else pos.entry_px
      if close_px is None:
        close_px = pos.entry_px
      positions_value += close_px * pos.qty

    equity_rows.append({
      "date": date_str,
      "cash": cash,
      "positions_value": positions_value,
      "equity": cash + positions_value,
      "n_positions": len(positions),
    })

  equity = pd.DataFrame(equity_rows)
  if not equity.empty:
    equity["ret"] = equity["equity"].pct_change().fillna(0.0)
  trades_df = pd.DataFrame(trades)

  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / "stage2_portfolio"
  run_id = args.run_id or f"{args.start}_{args.end}"
  run_path = out_dir / run_id
  run_path.mkdir(parents=True, exist_ok=True)

  equity.to_csv(run_path / "equity_curve.csv", index=False)
  trades_df.to_csv(run_path / "trades.csv", index=False)

  log.info(f"Stage 2 portfolio simulation complete: {run_path}")


if __name__ == "__main__":
  main()
