from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _safe_float(x) -> Optional[float]:
  try:
    if x is None:
      return None
    v = float(x)
    if pd.isna(v):
      return None
    return v
  except Exception:
    return None


def sma(series: pd.Series, window: int) -> pd.Series:
  return series.rolling(window=window, min_periods=window).mean()


def last_value(series: pd.Series) -> Optional[float]:
  if series is None or len(series) == 0:
    return None
  v = series.iloc[-1]
  return _safe_float(v)


def ma_snapshot(df: pd.DataFrame, windows: List[int], price_col: str = "close") -> Dict[int, Optional[float]]:
  out: Dict[int, Optional[float]] = {}
  if df is None or df.empty or price_col not in df.columns:
    for w in windows:
      out[w] = None
    return out

  s = df[price_col].astype(float)
  for w in windows:
    out[w] = last_value(sma(s, w))
  return out


def ma_order_score(mas: Dict[int, Optional[float]], order: List[int], higher_is_better: bool = True) -> float:
  """
  Returns 0..1 based on how many adjacent inequalities are satisfied.
  Example bullish order: close > MA14 > MA30 > MA60 > MA120
  Here we only score the MA chain portion: MA14 > MA30 > MA60 > MA120.
  """
  vals = [mas.get(w) for w in order]
  if any(v is None for v in vals):
    return 0.0

  ok = 0
  total = max(1, len(vals) - 1)
  for i in range(total):
    a = vals[i]
    b = vals[i + 1]
    if a is None or b is None:
      continue
    if higher_is_better:
      ok += 1 if a >= b else 0
    else:
      ok += 1 if a <= b else 0
  return ok / total


def pct_above(close: Optional[float], level: Optional[float]) -> Optional[float]:
  if close is None or level is None or close == 0:
    return None
  return (close - level) / close


def clamp01(x: float) -> float:
  if x < 0:
    return 0.0
  if x > 1:
    return 1.0
  return x


def rel_volume(df: pd.DataFrame, vol_col: str = "volume", window: int = 20) -> Optional[float]:
  if df is None or df.empty or vol_col not in df.columns:
    return None
  v = df[vol_col].astype(float)
  if len(v) < window:
    return None
  avg = v.rolling(window=window, min_periods=window).mean().iloc[-1]
  if avg is None or pd.isna(avg) or avg == 0:
    return None
  return float(v.iloc[-1] / avg)


def vol_contraction_ratio(df: pd.DataFrame, vol_col: str = "volume", short: int = 5, long: int = 20) -> Optional[float]:
  """
  short_avg / long_avg. <1 means recent contraction (usually good for breakout setups).
  """
  if df is None or df.empty or vol_col not in df.columns:
    return None
  v = df[vol_col].astype(float)
  if len(v) < long:
    return None
  short_avg = v.tail(short).mean()
  long_avg = v.tail(long).mean()
  if long_avg == 0 or pd.isna(long_avg):
    return None
  return float(short_avg / long_avg)


def atr14(df: pd.DataFrame) -> Optional[float]:
  """
  Basic ATR(14) from daily OHLC.
  """
  req = {"high", "low", "close"}
  if df is None or df.empty or not req.issubset(df.columns):
    return None
  if len(df) < 15:
    return None

  high = df["high"].astype(float)
  low = df["low"].astype(float)
  close = df["close"].astype(float)
  prev_close = close.shift(1)

  tr = pd.concat(
    [
      (high - low).abs(),
      (high - prev_close).abs(),
      (low - prev_close).abs(),
    ],
    axis=1,
  ).max(axis=1)

  atr = tr.rolling(window=14, min_periods=14).mean().iloc[-1]
  return _safe_float(atr)


def atr_pct(df: pd.DataFrame) -> Optional[float]:
  a = atr14(df)
  if a is None:
    return None
  close = _safe_float(df["close"].astype(float).iloc[-1]) if "close" in df.columns else None
  if close is None or close == 0:
    return None
  return float(a / close)


def liquidity_dollar_vol(df: pd.DataFrame, window: int = 20) -> Optional[float]:
  """
  Avg (close * volume) over window. Works for daily or hourly.
  """
  req = {"close", "volume"}
  if df is None or df.empty or not req.issubset(df.columns):
    return None
  if len(df) < window:
    return None
  dv = (df["close"].astype(float) * df["volume"].astype(float)).rolling(window=window, min_periods=window).mean().iloc[-1]
  return _safe_float(dv)


def score_from_components(components: Dict[str, float], weights: Dict[str, float]) -> float:
  s = 0.0
  wsum = 0.0
  for k, w in weights.items():
    if k in components:
      s += float(components[k]) * float(w)
      wsum += float(w)
  if wsum <= 0:
    return 0.0
  return 100.0 * (s / wsum)
