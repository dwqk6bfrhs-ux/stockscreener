from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.db import connect

EPS = 1e-12


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
  if x < lo:
    return lo
  if x > hi:
    return hi
  return x


def _safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
  if not s:
    return {}
  try:
    return json.loads(s)
  except Exception:
    return {}


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
  prev_close = close.shift(1)
  tr1 = (high - low).abs()
  tr2 = (high - prev_close).abs()
  tr3 = (low - prev_close).abs()
  return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def build_daily_features(
  end_date: str,
  lookback_days: int = 260,
  tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
  """
  Returns one row per ticker for end_date with:
    close, ma5, ma10, ma20, ma50, ma200,
    atr14, atr_pct, dollar_vol_20,
    daily_bars_60
  """
  end = datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - timedelta(days=lookback_days)).isoformat()
  start60 = (end - timedelta(days=60)).isoformat()

  params: list[Any] = [start, end_date]
  where_tickers = ""
  if tickers:
    where_tickers = " AND ticker IN (%s)" % ",".join(["?"] * len(tickers))
    params.extend(tickers)

  q = f"""
    SELECT ticker, date, open, high, low, close, volume
    FROM prices_daily
    WHERE date BETWEEN ? AND ?
    {where_tickers}
    ORDER BY ticker, date
  """

  with connect() as conn:
    df = pd.read_sql_query(q, conn, params=params)

    # coverage last 60 days
    params2: list[Any] = [start60, end_date]
    where_tickers2 = ""
    if tickers:
      where_tickers2 = " AND ticker IN (%s)" % ",".join(["?"] * len(tickers))
      params2.extend(tickers)

    q2 = f"""
      SELECT ticker, COUNT(*) AS daily_bars_60
      FROM prices_daily
      WHERE date BETWEEN ? AND ?
      {where_tickers2}
      GROUP BY ticker
    """
    df_cov = pd.read_sql_query(q2, conn, params=params2)

  if df.empty:
    return pd.DataFrame(columns=["ticker"])

  df["date"] = pd.to_datetime(df["date"])
  df["close"] = df["close"].astype(float)
  df["high"] = df["high"].astype(float)
  df["low"] = df["low"].astype(float)
  df["volume"] = df["volume"].astype(float)

  df["dollar_vol"] = df["close"].abs() * df["volume"].abs()

  def _calc(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["ma5"] = g["close"].rolling(5, min_periods=5).mean()
    g["ma10"] = g["close"].rolling(10, min_periods=10).mean()
    g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
    g["ma50"] = g["close"].rolling(50, min_periods=50).mean()
    g["ma200"] = g["close"].rolling(200, min_periods=200).mean()

    tr = _true_range(g["high"], g["low"], g["close"])
    g["atr14"] = tr.rolling(14, min_periods=14).mean()
    g["atr_pct"] = g["atr14"] / (g["close"].abs() + EPS)

    g["dollar_vol_20"] = g["dollar_vol"].rolling(20, min_periods=20).mean()
    return g

  df = df.groupby("ticker", group_keys=False).apply(_calc)

  # keep end_date rows only
  end_ts = pd.to_datetime(end_date)
  out = df[df["date"] == end_ts].copy()
  out = out[["ticker", "close", "ma5", "ma10", "ma20", "ma50", "ma200", "atr14", "atr_pct", "dollar_vol_20"]]

  out = out.merge(df_cov, on="ticker", how="left")
  out["daily_bars_60"] = out["daily_bars_60"].fillna(0).astype(int)

  return out


def build_hourly_coverage(date_et: str, tickers: Optional[list[str]] = None) -> pd.DataFrame:
  params: list[Any] = [date_et]
  where_tickers = ""
  if tickers:
    where_tickers = " AND ticker IN (%s)" % ",".join(["?"] * len(tickers))
    params.extend(tickers)

  q = f"""
    SELECT ticker, COUNT(*) AS hourly_bars
    FROM prices_hourly
    WHERE date_et=?
    {where_tickers}
    GROUP BY ticker
  """

  with connect() as conn:
    df = pd.read_sql_query(q, conn, params=params)

  if df.empty:
    return pd.DataFrame(columns=["ticker", "hourly_bars"])

  df["hourly_bars"] = df["hourly_bars"].fillna(0).astype(int)
  return df


def score_ma_cross_row(row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
  """
  Ranking score for MA cross entries:
    - spread (ma5-ma10)/close (primary)
    - alignment ma20>ma50>ma200 (trend)
    - liquidity (log10 dollar_vol_20)
    - risk (low atr_pct)
  """
  close = row.get("close")
  ma5 = row.get("ma5")
  ma10 = row.get("ma10")
  ma20 = row.get("ma20")
  ma50 = row.get("ma50")
  ma200 = row.get("ma200")

  if pd.isna(close) or close is None or float(close) <= 0:
    return None, {"reason": "missing_close"}

  # spread pct
  if pd.isna(ma5) or pd.isna(ma10):
    spread_pct = None
  else:
    spread_pct = float(ma5 - ma10) / (float(close) + EPS)

  # normalize spread: 0%..2% -> 0..1
  spread_score = 0.0
  if spread_pct is not None:
    spread_score = _clamp(spread_pct / 0.02)

  # trend alignment
  trend = 0.0
  if (not pd.isna(ma20)) and (not pd.isna(ma50)) and (not pd.isna(ma200)):
    if float(ma20) > float(ma50):
      trend += 0.5
    if float(ma50) > float(ma200):
      trend += 0.5

  dv20 = row.get("dollar_vol_20")
  liq = 0.0
  if dv20 is not None and (not pd.isna(dv20)) and float(dv20) > 0:
    # log scale then clamp
    liq = _clamp((float(np.log10(float(dv20) + 1.0)) - 6.0) / (9.0 - 6.0))  # 1e6..1e9
  atr_pct = row.get("atr_pct")
  risk = 0.0
  if atr_pct is not None and (not pd.isna(atr_pct)) and float(atr_pct) > 0:
    risk = _clamp(1.0 - float(atr_pct) / 0.12)

  rank = 0.50 * spread_score + 0.20 * trend + 0.20 * liq + 0.10 * risk
  meta = {
    "spread_pct": float(spread_pct) if spread_pct is not None else None,
    "spread_score": float(spread_score),
    "trend_score": float(trend),
    "liq_score": float(liq),
    "risk_score": float(risk),
  }
  return float(rank), meta


def score_retest_shrink_row(row: pd.Series, signal_score: Optional[float]) -> Tuple[Optional[float], Dict[str, Any]]:
  """
  Ranking score for retest_shrink:
    - base = strategy's own score (primary, expected ~0..1)
    - trend/liquidity/risk from daily features
  """
  base = None
  if signal_score is not None:
    try:
      base = float(signal_score)
    except Exception:
      base = None

  if base is None or np.isnan(base):
    return None, {"reason": "missing_signal_score"}

  # clamp base to 0..1
  base_c = _clamp(base)

  # trend
  ma20, ma50, ma200 = row.get("ma20"), row.get("ma50"), row.get("ma200")
  trend = 0.0
  if (not pd.isna(ma20)) and (not pd.isna(ma50)) and (not pd.isna(ma200)):
    if float(ma20) > float(ma50):
      trend += 0.5
    if float(ma50) > float(ma200):
      trend += 0.5

  # liquidity
  dv20 = row.get("dollar_vol_20")
  liq = 0.0
  if dv20 is not None and (not pd.isna(dv20)) and float(dv20) > 0:
    liq = _clamp((float(np.log10(float(dv20) + 1.0)) - 6.0) / (9.0 - 6.0))

  # risk
  atr_pct = row.get("atr_pct")
  risk = 0.0
  if atr_pct is not None and (not pd.isna(atr_pct)) and float(atr_pct) > 0:
    risk = _clamp(1.0 - float(atr_pct) / 0.12)

  rank = 0.60 * base_c + 0.15 * trend + 0.15 * liq + 0.10 * risk
  meta = {
    "base_score": float(base_c),
    "trend_score": float(trend),
    "liq_score": float(liq),
    "risk_score": float(risk),
  }
  return float(rank), meta
