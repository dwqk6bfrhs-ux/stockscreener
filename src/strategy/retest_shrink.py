# src/strategy/retest_shrink.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

EPS = 1e-12

STRATEGY_NAME = "retest_shrink"

# -----------------------------
# Parameters
# -----------------------------
@dataclass
class Params:
  # Indicator windows
  atr_window: int = 14

  # D0 detection
  d0_lookback_days: int = 80            # scan this many most-recent bars to find the most recent D0
  d0_vol_lookback: int = 20             # baseline vol window
  d0_vol_mult: float = 2.0              # D0 if volume >= baseline * mult
  d0_range_pct_min: float = 0.04        # D0 if (high-low)/close >= this
  d0_require_red: bool = False          # optional: require close < open

  # Retest search window after D0
  retest_window_days: int = 15
  retest_zone_atr: float = 1.0          # low_j <= l0 + zone_atr * atr0
  retest_shrink_max: float = 0.35       # vol_j <= shrink_max * v0
  retest_undercut_atr_max: float = 0.50 # low_j >= l0 - undercut_atr_max * atr0

  # Confirm after retest
  confirm_window_days: int = 3
  confirm_vol_max_mult: float = 0.80    # allow confirm if vol <= mult * v0
  confirm_close_strength: bool = True   # allow confirm if close > prev close

  # Stop
  stop_atr: float = 1.0

  # Optional "pressure test" heuristic:
  # If latest bar shows heavy selling near l0 but no breakdown, label PRESSURE_TEST.
  pressure_zone_atr: float = 1.0
  pressure_vol_min_mult: float = 0.90   # volume >= 0.9*v0
  pressure_require_no_new_low: bool = True


# A dict version for YAML convenience (optional)
DEFAULT_PARAMS: Dict[str, Any] = Params().__dict__.copy()


# -----------------------------
# Helpers: ATR + feature prep
# -----------------------------
def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
  if "date" not in df.columns:
    raise ValueError("Input df must have a 'date' column.")
  out = df.copy()
  # tolerate str dates
  out["date"] = pd.to_datetime(out["date"])
  return out


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
  """
  ATR (simple moving average of True Range).
  Requires columns: high, low, close
  """
  high = df["high"].astype(float)
  low = df["low"].astype(float)
  close = df["close"].astype(float)
  prev_close = close.shift(1)

  tr1 = (high - low).abs()
  tr2 = (high - prev_close).abs()
  tr3 = (low - prev_close).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

  atr = tr.rolling(window=window, min_periods=window).mean()
  return atr


def _prep(df: pd.DataFrame, p: Params) -> pd.DataFrame:
  """
  Returns df sorted by date ascending with extra columns:
  - range: (high-low)/close
  - atr: ATR(window)
  """
  gg = _ensure_datetime(df)
  gg = gg.sort_values("date").reset_index(drop=True)

  for col in ["open", "high", "low", "close", "volume"]:
    if col not in gg.columns:
      raise ValueError(f"Input df missing required column: {col}")

  gg["open"] = gg["open"].astype(float)
  gg["high"] = gg["high"].astype(float)
  gg["low"] = gg["low"].astype(float)
  gg["close"] = gg["close"].astype(float)
  gg["volume"] = gg["volume"].astype(float)

  gg["range"] = (gg["high"] - gg["low"]) / (gg["close"].abs() + EPS)
  gg["atr"] = _compute_atr(gg, window=p.atr_window)

  return gg


# -----------------------------
# Core engine
# -----------------------------
def evaluate_ticker(df: pd.DataFrame, params: Params, ticker: Optional[str] = None) -> Dict[str, Any]:
  """
  Returns a rich dict. Your generate_signals pipeline will store:
  - state in signals_daily.state (via adapter)
  - score in signals_daily.score
  - stop in signals_daily.stop
  - the entire dict into meta_json (via adapter)
  """
  p = params
  gg = _prep(df, p)
  last = gg.iloc[-1]
  ticker = ticker or (str(last.get("ticker")) if "ticker" in gg.columns else "UNKNOWN")

  # Need enough data for ATR + D0 detection
  if len(gg) < max(p.atr_window + 5, p.d0_vol_lookback + 5):
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "INSUFFICIENT_HISTORY",
      "score": None,
      "watch": False,
      "action": False,
      "stop": None,
    }

  # Baseline volume (rolling median is more robust than mean)
  gg["vol_base"] = gg["volume"].rolling(window=p.d0_vol_lookback, min_periods=p.d0_vol_lookback).median()

  # D0 candidate definition
  red_ok = (gg["close"] < gg["open"]) if p.d0_require_red else pd.Series([True] * len(gg))
  vol_ok = gg["volume"] >= (gg["vol_base"].fillna(0) * p.d0_vol_mult)
  range_ok = gg["range"] >= p.d0_range_pct_min
  d0_mask = red_ok & vol_ok & range_ok & gg["atr"].notna()

  # Only scan recent region for the most recent D0
  start_i = max(0, len(gg) - p.d0_lookback_days)
  d0_idx_candidates = gg.index[start_i:][d0_mask.iloc[start_i:]].tolist()

  if not d0_idx_candidates:
    # No D0 -> no setup
    score = float(last["volume"]) * float(last["range"])
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "NO_D0",
      "score": score,
      "watch": False,
      "action": False,
      "stop": None,
    }

  d0_i = d0_idx_candidates[-1]
  d0_row = gg.loc[d0_i]
  d0_date = d0_row["date"]
  l0 = float(d0_row["low"])
  v0 = float(d0_row["volume"])
  atr0 = float(d0_row["atr"])

  # If ATR is bad, abort gracefully
  if not (atr0 > 0):
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "BAD_ATR",
      "score": None,
      "watch": False,
      "action": False,
      "stop": None,
      "d0_date": str(d0_date.date()),
      "l0": l0,
      "v0": v0,
      "atr0": atr0,
    }

  # Optional: detect a "pressure test" on the latest bar (large sell pressure but no breakdown)
  # This matches the spirit of your theory: large selling occurs but price does not continue down.
  l0_floor = l0 - p.retest_undercut_atr_max * atr0
  pressure_in_zone = float(last["low"]) <= (l0 + p.pressure_zone_atr * atr0)
  pressure_vol_ok = float(last["volume"]) >= (p.pressure_vol_min_mult * v0)
  pressure_no_new_low = float(last["low"]) >= l0_floor if p.pressure_require_no_new_low else True

  if pressure_in_zone and pressure_vol_ok and pressure_no_new_low:
    score = float(last["volume"]) * float(last["range"])
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "PRESSURE_TEST",
      "d0_date": str(d0_date.date()),
      "l0": l0,
      "v0": v0,
      "atr0": atr0,
      "score": score,
      "watch": True,
      "action": False,
      "stop": l0 - p.stop_atr * atr0,
    }

  # Retest search
  retest_end = min(len(gg) - 1, d0_i + p.retest_window_days)
  retest_i = None
  shrink_ratio = None

  for j in range(d0_i + 2, retest_end + 1):
    row = gg.loc[j]
    low_j = float(row["low"])
    vol_j = float(row["volume"])

    in_zone = low_j <= l0 + p.retest_zone_atr * atr0
    shrunk = vol_j <= p.retest_shrink_max * v0
    no_new_low = low_j >= l0 - p.retest_undercut_atr_max * atr0

    if in_zone and shrunk and no_new_low:
      retest_i = j
      shrink_ratio = vol_j / (v0 + EPS)
      break

  if retest_i is None:
    score = float(last["volume"]) * float(last["range"])
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "NO_FOLLOW_THROUGH",
      "d0_date": str(d0_date.date()),
      "l0": l0,
      "v0": v0,
      "atr0": atr0,
      "score": score,
      "watch": True,
      "action": False,
      "stop": l0 - p.stop_atr * atr0,
    }

  # Confirm in next 1..confirm_window
  confirm_end = min(len(gg) - 1, retest_i + p.confirm_window_days)
  l0_floor = l0 - p.retest_undercut_atr_max * atr0
  min_low = float(gg.loc[retest_i:confirm_end, "low"].min())

  confirmed = False
  confirm_date = None

  if min_low >= l0_floor:
    for j in range(retest_i + 1, confirm_end + 1):
      prev = gg.loc[j - 1]
      cur = gg.loc[j]
      close_strength = float(cur["close"]) > float(prev["close"])
      vol_ok = float(cur["volume"]) <= p.confirm_vol_max_mult * v0
      if (p.confirm_close_strength and close_strength) or vol_ok:
        confirmed = True
        confirm_date = cur["date"]
        break

  # Scoring
  last_close = float(gg.iloc[-1]["close"])
  dist = abs(last_close - l0)
  s1 = 1.0 - float(shrink_ratio)
  s2 = max(0.0, 1.0 - dist / (1.0 * atr0 + EPS))
  s3 = float(gg.iloc[-1]["atr"]) / (last_close + EPS)
  score = 0.45 * s1 + 0.35 * s2 + 0.20 * min(0.20, s3)

  state = "RETEST_CONFIRMED" if confirmed else "RETEST_CANDIDATE"

  return {
    "ticker": ticker,
    "date": str(last["date"].date()),
    "state": state,
    "d0_date": str(d0_date.date()),
    "l0": l0,
    "v0": v0,
    "atr0": atr0,
    "retest_date": str(gg.loc[retest_i, "date"].date()),
    "confirm_date": (str(confirm_date.date()) if confirm_date is not None else None),
    "shrink_ratio": float(shrink_ratio),
    "stop": l0 - p.stop_atr * atr0,
    "score": float(score),
    "watch": True,
    "action": bool(confirmed),
  }


# -----------------------------
# Adapter: required by generate_signals (YAML-driven)
# -----------------------------
def _build_params(params_dict: Optional[dict]) -> Params:
  """
  Robustly construct Params from a dict:
  - if Params(**dict) works, use it
  - otherwise create Params() and setattr known fields
  """
  params_dict = params_dict or {}
  try:
    return Params(**params_dict)
  except Exception:
    p = Params()
    for k, v in params_dict.items():
      if hasattr(p, k):
        try:
          setattr(p, k, v)
        except Exception:
          pass
    return p


def evaluate(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
  """
  Standard interface for the multi-strategy runner.
  Returns shape:
    {state, score, stop, meta}
  Where 'state' is normalized to ENTRY/WATCH/PASS/ERROR so downstream is consistent,
  while the original rich state is preserved in meta["raw_state"] and meta payload.
  """
  p = _build_params(params)

  try:
    raw = evaluate_ticker(df, params=p)
    # Normalize to pipeline-friendly states
    # - action True => ENTRY
    # - watch True => WATCH
    # - else PASS
    raw_state = str(raw.get("state", "PASS"))
    action = bool(raw.get("action", False))
    watch = bool(raw.get("watch", False))

    if raw_state in ("ERROR", "BAD_ATR"):
      state = "ERROR"
    elif action:
      state = "ENTRY"
    elif watch:
      state = "WATCH"
    else:
      state = "PASS"

    return {
      "state": state,
      "score": raw.get("score", None),
      "stop": raw.get("stop", None),
      "meta": {
        "raw_state": raw_state,
        "params": p.__dict__,
        "raw": raw,
      },
    }
  except Exception as e:
    return {
      "state": "ERROR",
      "score": None,
      "stop": None,
      "meta": {"error": str(e), "params": p.__dict__},
    }

