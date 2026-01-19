# src/strategy/retest_shrink.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

EPS = 1e-12
STRATEGY_NAME = "retest_shrink"

# -----------------------------
# Parameters (D0/retest/confirm logic unchanged)
# -----------------------------
@dataclass
class Params:
  # Indicator windows
  atr_window: int = 14

  # D0 detection (unchanged)
  d0_lookback_days: int = 80
  d0_vol_lookback: int = 20
  d0_vol_mult: float = 2.0
  d0_range_pct_min: float = 0.04
  d0_require_red: bool = False

  # Retest search window after D0 (unchanged)
  retest_window_days: int = 15
  retest_zone_atr: float = 1.0
  retest_shrink_max: float = 0.35
  retest_undercut_atr_max: float = 0.50

  # Confirm after retest (unchanged)
  confirm_window_days: int = 3
  confirm_vol_max_mult: float = 0.80
  confirm_close_strength: bool = True

  # Stop (unchanged)
  stop_atr: float = 1.0

  # Optional "pressure test" heuristic (unchanged)
  pressure_zone_atr: float = 1.0
  pressure_vol_min_mult: float = 0.90
  pressure_require_no_new_low: bool = True

  # -----------------------------
  # NEW: Ranking / feature engineering params
  # -----------------------------
  # Trend features (do NOT change state logic; score/meta only)
  ma_fast: int = 20
  ma_mid: int = 50
  ma_slow: int = 200

  # Liquidity features
  dollar_vol_window: int = 20
  liq_dv20_min: float = 1.0e7     # $10M average dollar volume => starts getting credit
  liq_dv20_max: float = 2.0e8     # $200M => full credit

  # Risk features (ATR percent)
  risk_atr_pct_max: float = 0.12  # ATR/close >= 12% -> risk score goes to ~0

  # Score weights (sum to 1.0)
  w_base: float = 0.60
  w_trend: float = 0.15
  w_liq: float = 0.15
  w_risk: float = 0.10

  # Base score weights (inside base term)
  w_shrink: float = 0.55
  w_prox: float = 0.45


DEFAULT_PARAMS: Dict[str, Any] = Params().__dict__.copy()


# -----------------------------
# Helpers: ATR + feature prep
# -----------------------------
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
  if x < lo:
    return lo
  if x > hi:
    return hi
  return x


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
  if "date" not in df.columns:
    raise ValueError("Input df must have a 'date' column.")
  out = df.copy()
  out["date"] = pd.to_datetime(out["date"])
  return out


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
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

  # NEW: dollar volume
  gg["dollar_vol"] = gg["close"].abs() * gg["volume"].abs()

  # NEW: moving averages (for meta/scoring only)
  gg["ma_fast"] = gg["close"].rolling(p.ma_fast, min_periods=p.ma_fast).mean()
  gg["ma_mid"] = gg["close"].rolling(p.ma_mid, min_periods=p.ma_mid).mean()
  gg["ma_slow"] = gg["close"].rolling(p.ma_slow, min_periods=p.ma_slow).mean()

  return gg


def _compute_dollar_vol_20(gg: pd.DataFrame, p: Params) -> Optional[float]:
  # Uses last N rows in the dataframe (assumes daily bars)
  if len(gg) < p.dollar_vol_window:
    return None
  tail = gg.tail(p.dollar_vol_window)
  dv = float(tail["dollar_vol"].mean())
  return dv


def _trend_score(last_row: pd.Series) -> float:
  # Simple alignment score: MA_fast > MA_mid > MA_slow (0..1)
  mf = last_row.get("ma_fast")
  mm = last_row.get("ma_mid")
  ms = last_row.get("ma_slow")

  if pd.isna(mf) or pd.isna(mm) or pd.isna(ms):
    return 0.0

  # give partial credit if some order holds
  score = 0.0
  if float(mf) > float(mm):
    score += 0.5
  if float(mm) > float(ms):
    score += 0.5
  return score


def _liquidity_score(dv20: Optional[float], p: Params) -> float:
  if dv20 is None or dv20 <= 0:
    return 0.0
  # linear normalization between min and max
  return _clamp((dv20 - p.liq_dv20_min) / (p.liq_dv20_max - p.liq_dv20_min + EPS))


def _risk_score(atr_pct: Optional[float], p: Params) -> float:
  if atr_pct is None or atr_pct <= 0:
    return 0.0
  # Lower ATR% is better; goes to 0 by risk_atr_pct_max
  return _clamp(1.0 - (atr_pct / (p.risk_atr_pct_max + EPS)))


# -----------------------------
# Core engine (logic unchanged; score/meta upgraded)
# -----------------------------
def evaluate_ticker(df: pd.DataFrame, params: Params, ticker: Optional[str] = None) -> Dict[str, Any]:
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
      "features": {
        "n_bars": int(len(gg)),
      },
    }

  # Baseline volume (unchanged)
  gg["vol_base"] = gg["volume"].rolling(window=p.d0_vol_lookback, min_periods=p.d0_vol_lookback).median()

  # D0 candidate definition (unchanged)
  red_ok = (gg["close"] < gg["open"]) if p.d0_require_red else pd.Series([True] * len(gg))
  vol_ok = gg["volume"] >= (gg["vol_base"].fillna(0) * p.d0_vol_mult)
  range_ok = gg["range"] >= p.d0_range_pct_min
  d0_mask = red_ok & vol_ok & range_ok & gg["atr"].notna()

  start_i = max(0, len(gg) - p.d0_lookback_days)
  d0_idx_candidates = gg.index[start_i:][d0_mask.iloc[start_i:]].tolist()

  # Compute ranking features (meta-only; does not affect state logic)
  dv20 = _compute_dollar_vol_20(gg, p)
  atr_last = float(last["atr"]) if pd.notna(last["atr"]) else None
  last_close = float(last["close"])
  atr_pct = (float(atr_last) / (last_close + EPS)) if atr_last is not None else None
  tscore = _trend_score(last)
  lscore = _liquidity_score(dv20, p)
  rscore = _risk_score(atr_pct, p)

  # Helper for packaging features
  def pack_features(extra: Dict[str, Any]) -> Dict[str, Any]:
    base = {
      "n_bars": int(len(gg)),
      "last_close": float(last_close),
      "last_volume": float(last["volume"]),
      "dollar_vol_20": float(dv20) if dv20 is not None else None,
      "atr_last": float(atr_last) if atr_last is not None else None,
      "atr_pct": float(atr_pct) if atr_pct is not None else None,
      "ma_fast": (float(last["ma_fast"]) if pd.notna(last.get("ma_fast")) else None),
      "ma_mid": (float(last["ma_mid"]) if pd.notna(last.get("ma_mid")) else None),
      "ma_slow": (float(last["ma_slow"]) if pd.notna(last.get("ma_slow")) else None),
      "trend_score": float(tscore),
      "liquidity_score": float(lscore),
      "risk_score": float(rscore),
    }
    base.update(extra)
    return base

  if not d0_idx_candidates:
    score = float(last["volume"]) * float(last["range"])
    return {
      "ticker": ticker,
      "date": str(last["date"].date()),
      "state": "NO_D0",
      "score": score,
      "watch": False,
      "action": False,
      "stop": None,
      "features": pack_features({}),
    }

  d0_i = d0_idx_candidates[-1]
  d0_row = gg.loc[d0_i]
  d0_date = d0_row["date"]
  l0 = float(d0_row["low"])
  v0 = float(d0_row["volume"])
  atr0 = float(d0_row["atr"])

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
      "features": pack_features({"d0_i": int(d0_i)}),
    }

  # Pressure test (unchanged)
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
      "features": pack_features({"d0_i": int(d0_i)}),
    }

  # Retest search (unchanged)
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
      "features": pack_features({"d0_i": int(d0_i)}),
    }

  # Confirm (unchanged)
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
      vol_ok2 = float(cur["volume"]) <= p.confirm_vol_max_mult * v0
      if (p.confirm_close_strength and close_strength) or vol_ok2:
        confirmed = True
        confirm_date = cur["date"]
        break

  # -----------------------------
  # NEW: Scoring (base + trend + liquidity + risk)
  # -----------------------------
  # Base terms (similar intent as before)
  last_close = float(gg.iloc[-1]["close"])
  dist = abs(last_close - l0)

  # shrink: smaller ratio is better
  s_shrink = _clamp(1.0 - float(shrink_ratio))

  # proximity: within ~1 ATR0 is good
  s_prox = _clamp(1.0 - (dist / (1.0 * atr0 + EPS)))

  base = (p.w_shrink * s_shrink) + (p.w_prox * s_prox)

  # Trend/liquidity/risk scores already computed above from latest row
  final_score = (
    p.w_base * base +
    p.w_trend * tscore +
    p.w_liq * lscore +
    p.w_risk * rscore
  )

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
    "score": float(final_score),
    "watch": True,
    "action": bool(confirmed),
    "features": pack_features({
      "d0_i": int(d0_i),
      "retest_i": int(retest_i),
      "state_raw": state,
      "dist_to_l0_atr": float(dist / (atr0 + EPS)),
      "s_shrink": float(s_shrink),
      "s_prox": float(s_prox),
      "score_base": float(base),
    }),
  }


# -----------------------------
# Adapter: required by generate_signals (UNCHANGED normalization)
# -----------------------------
def _build_params(params_dict: Optional[dict]) -> Params:
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
  Returns shape: {state, score, stop, meta}
  Normalized state: ENTRY/WATCH/PASS/ERROR
  Raw state preserved in meta["raw_state"].
  """
  p = _build_params(params)

  try:
    raw = evaluate_ticker(df, params=p)
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

    # Keep meta relatively LLM/report-friendly: flatten key features
    features = raw.get("features", {}) or {}
    meta = {
      "raw_state": raw_state,
      "features": features,
      "params": p.__dict__,
    }

    return {
      "state": state,
      "score": raw.get("score", None),
      "stop": raw.get("stop", None),
      "meta": meta,
    }
  except Exception as e:
    return {
      "state": "ERROR",
      "score": None,
      "stop": None,
      "meta": {"error": str(e), "params": p.__dict__},
    }
