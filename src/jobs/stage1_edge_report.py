import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.backtest.regimes import add_spy_regime
from src.common.db import connect, get_prices_daily_source, init_db
from src.common.logging import setup_logger

log = setup_logger("stage1_edge_report")


@dataclass
class EdgeConfig:
  start: str
  end: str
  horizons: List[int]
  strategy: str | None


def _parse_horizons(raw: str) -> List[int]:
  out = []
  for part in (raw or "").split(","):
    part = part.strip()
    if not part:
      continue
    out.append(int(part))
  return sorted(set(out))


def _read_signal_scores(cfg: EdgeConfig) -> pd.DataFrame:
  params: list[str] = [cfg.start, cfg.end]
  strategy_clause = ""
  if cfg.strategy:
    strategy_clause = "AND r.strategy = ?"
    params.append(cfg.strategy)

  sql = f"""
    SELECT r.date, r.ticker, r.strategy, r.rank_score, r.meta_json AS rank_meta,
           s.state, s.stop
    FROM rank_scores_daily r
    JOIN signals_daily s
      ON s.date = r.date AND s.ticker = r.ticker AND s.strategy = r.strategy
    WHERE r.date BETWEEN ? AND ?
      AND s.state = 'ENTRY'
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


def _forward_metrics(prices: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
  rows = []
  for ticker, g in prices.groupby("ticker", sort=False):
    g = g.sort_values("date").reset_index(drop=True)
    base = g[["date", "ticker", "close", "high", "low"]].copy()
    for h in horizons:
      base[f"fwd_close_{h}"] = g["close"].shift(-h)
      future_high = g["high"].shift(-1)
      future_low = g["low"].shift(-1)
      base[f"fwd_high_{h}"] = future_high.rolling(window=h, min_periods=h).max().shift(-(h - 1))
      base[f"fwd_low_{h}"] = future_low.rolling(window=h, min_periods=h).min().shift(-(h - 1))
      base[f"ret_{h}"] = base[f"fwd_close_{h}"] / g["close"] - 1.0
      base[f"mfe_{h}"] = base[f"fwd_high_{h}"] / g["close"] - 1.0
      base[f"mae_{h}"] = base[f"fwd_low_{h}"] / g["close"] - 1.0
    rows.append(base)
  if not rows:
    return pd.DataFrame()
  return pd.concat(rows, ignore_index=True)


def _score_deciles(df: pd.DataFrame) -> pd.Series:
  scores = df["rank_score"].astype(float)
  try:
    return pd.qcut(scores, 10, labels=False, duplicates="drop") + 1
  except ValueError:
    return pd.Series([1] * len(df), index=df.index)


def _extract_meta_fields(df: pd.DataFrame) -> pd.DataFrame:
  def _parse(meta: str) -> Dict[str, float | None]:
    if not meta:
      return {"dv20": None, "atr_pct": None}
    try:
      obj = json.loads(meta)
    except Exception:
      return {"dv20": None, "atr_pct": None}
    features = obj.get("features", {})
    return {
      "dv20": features.get("dv20"),
      "atr_pct": features.get("atr_pct"),
    }

  meta = df["rank_meta"].apply(_parse)
  df["dv20"] = meta.apply(lambda x: x.get("dv20"))
  df["atr_pct"] = meta.apply(lambda x: x.get("atr_pct"))
  return df


def _bench_regime(start: str, end: str) -> pd.DataFrame:
  source = get_prices_daily_source()
  with connect() as conn:
    spy = pd.read_sql_query(
      """
      SELECT date, close
      FROM prices_daily
      WHERE source=? AND ticker='SPY' AND date BETWEEN ? AND ?
      ORDER BY date
      """,
      conn,
      params=(source, start, end),
    )
  if spy.empty:
    return spy
  spy["date"] = pd.to_datetime(spy["date"])
  spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
  return add_spy_regime(spy)


def _decile_report(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
  ret_col = f"ret_{horizon}"
  mae_col = f"mae_{horizon}"
  mfe_col = f"mfe_{horizon}"
  out = df.dropna(subset=[ret_col]).copy()
  if out.empty:
    return pd.DataFrame()
  out["decile"] = _score_deciles(out)
  grouped = out.groupby("decile")
  report = grouped[ret_col].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "avg_return"})
  report["hit_rate"] = grouped[ret_col].apply(lambda s: (s > 0).mean())
  report["avg_mae"] = grouped[mae_col].mean()
  report["avg_mfe"] = grouped[mfe_col].mean()
  report = report.reset_index()
  report["horizon_days"] = horizon
  return report


def _regime_report(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
  ret_col = f"ret_{horizon}"
  out = df.dropna(subset=[ret_col]).copy()
  if out.empty:
    return pd.DataFrame()
  grouped = out.groupby("spy_regime")
  report = grouped[ret_col].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "avg_return"})
  report["hit_rate"] = grouped[ret_col].apply(lambda s: (s > 0).mean())
  report = report.reset_index()
  report["horizon_days"] = horizon
  return report


def _calibration_report(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
  ret_col = f"ret_{horizon}"
  out = df.dropna(subset=[ret_col]).copy()
  if out.empty:
    return pd.DataFrame()
  out["score_pctile"] = out["rank_score"].rank(pct=True)
  bins = pd.interval_range(start=0.0, end=1.0, freq=0.05, closed="right")
  out["pctile_bin"] = pd.cut(out["score_pctile"], bins=bins)
  grouped = out.groupby("pctile_bin")[ret_col]
  report = grouped.agg(["count", "mean"]).rename(columns={"count": "n", "mean": "avg_return"}).reset_index()
  report["horizon_days"] = horizon
  return report


def main() -> None:
  ap = argparse.ArgumentParser(description="Stage 1 edge report: forward returns by score decile.")
  ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
  ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
  ap.add_argument("--horizons", default="5,10,20", help="Comma-separated forward horizons in trading days.")
  ap.add_argument("--strategy", default=None, help="Filter by strategy name (optional).")
  ap.add_argument("--run-id", default=None, help="Output run id (defaults to date range).")
  args = ap.parse_args()

  init_db()
  horizons = _parse_horizons(args.horizons)
  if not horizons:
    raise RuntimeError("No horizons provided.")

  cfg = EdgeConfig(start=args.start, end=args.end, horizons=horizons, strategy=args.strategy)
  signals = _read_signal_scores(cfg)
  if signals.empty:
    raise RuntimeError("No ENTRY signals found in rank_scores_daily for the requested range.")

  signals = _extract_meta_fields(signals)

  max_h = max(horizons)
  start_dt = pd.to_datetime(args.start)
  end_dt = pd.to_datetime(args.end) + pd.Timedelta(days=max_h * 2)
  tickers = sorted(signals["ticker"].unique().tolist())
  prices = _read_prices(tickers, start_dt.date().isoformat(), end_dt.date().isoformat())
  if prices.empty:
    raise RuntimeError("No price data found for signal tickers in the requested window.")

  forward = _forward_metrics(prices, horizons)
  merged = signals.merge(forward, on=["date", "ticker"], how="left")

  regime = _bench_regime(args.start, end_dt.date().isoformat())
  if not regime.empty:
    regime["date"] = pd.to_datetime(regime["date"])
    merged = merged.merge(regime, on="date", how="left")

  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / "stage1_edge"
  run_id = args.run_id or f"{args.start}_{args.end}"
  run_path = out_dir / run_id
  run_path.mkdir(parents=True, exist_ok=True)

  merged.to_csv(run_path / "signals_with_forwards.csv", index=False)

  for h in horizons:
    decile = _decile_report(merged, h)
    if not decile.empty:
      decile.to_csv(run_path / f"decile_report_{h}d.csv", index=False)

    regime_report = _regime_report(merged, h)
    if not regime_report.empty:
      regime_report.to_csv(run_path / f"regime_report_{h}d.csv", index=False)

    calibration = _calibration_report(merged, h)
    if not calibration.empty:
      calibration.to_csv(run_path / f"calibration_{h}d.csv", index=False)

  log.info(f"Stage 1 edge report written to {run_path}")


if __name__ == "__main__":
  main()
