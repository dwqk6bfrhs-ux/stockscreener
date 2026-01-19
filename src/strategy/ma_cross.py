from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.strategy.ranklib import (
  ma_snapshot,
  ma_order_score,
  rel_volume,
  vol_contraction_ratio,
  atr14,
  atr_pct,
  liquidity_dollar_vol,
  clamp01,
  pct_above,
  score_from_components,
)


def evaluate(df: pd.DataFrame, params: dict, ctx: Optional[dict] = None) -> dict:
  """
  Buy when MA(fast) crosses above MA(slow). Rank BUY signals using:
    - Daily MA(14/30/60/120) bullish alignment
    - Hourly MA(14/30/60/120) bullish alignment
    - Volume patterns (rel vol + contraction)
    - Risk/liquidity (ATR%, dollar volume)
  """
  fast = int(params.get("fast", 5))
  slow = int(params.get("slow", 10))

  daily_ctx_windows = params.get("daily_ctx_windows", [14, 30, 60, 120])
  hourly_ctx_windows = params.get("hourly_ctx_windows", [14, 30, 60, 120])

  min_daily = max(slow + 2, max(daily_ctx_windows) + 2)
  if df is None or df.empty or len(df) < min_daily:
    return {"state": "PASS", "score": None, "stop": None, "meta": {"reason": "insufficient_daily_history"}}

  df = df.copy()
  df = df.sort_values("date")
  close = float(df["close"].astype(float).iloc[-1])

  # Fast/slow MAs (daily)
  ma_fast = df["close"].astype(float).rolling(window=fast, min_periods=fast).mean()
  ma_slow = df["close"].astype(float).rolling(window=slow, min_periods=slow).mean()

  if ma_fast.isna().iloc[-1] or ma_slow.isna().iloc[-1] or ma_fast.isna().iloc[-2] or ma_slow.isna().iloc[-2]:
    return {"state": "PASS", "score": None, "stop": None, "meta": {"reason": "insufficient_ma_values"}}

  fast_y, slow_y = float(ma_fast.iloc[-2]), float(ma_slow.iloc[-2])
  fast_t, slow_t = float(ma_fast.iloc[-1]), float(ma_slow.iloc[-1])

  cross_up = (fast_y <= slow_y) and (fast_t > slow_t)
  cross_down = (fast_y >= slow_y) and (fast_t < slow_t)

  if cross_up:
    state = "BUY"
  elif cross_down:
    state = "SELL"
  else:
    state = "PASS"

  # Daily context alignment
  daily_mas = ma_snapshot(df, windows=list(daily_ctx_windows), price_col="close")
  daily_align = ma_order_score(daily_mas, order=list(daily_ctx_windows), higher_is_better=True)

  # Prefer price above key MAs (soft score)
  above_ma14 = pct_above(close, daily_mas.get(daily_ctx_windows[0]))
  above_bonus = clamp01((above_ma14 or 0.0) / 0.05)  # 0..1 when 0..+5%

  # Volume / risk / liquidity (daily)
  rv20 = rel_volume(df, window=20)  # last vol / 20d avg
  vc = vol_contraction_ratio(df, short=5, long=20)  # short/long
  atr = atr14(df)
  atrp = atr_pct(df)
  dv20 = liquidity_dollar_vol(df, window=20)

  # Hourly context (optional)
  hourly_align = 0.0
  hourly_rv20 = None
  if ctx and isinstance(ctx.get("hourly"), pd.DataFrame) and ctx["hourly"] is not None and not ctx["hourly"].empty:
    hf = ctx["hourly"].copy()
    # ensure numeric
    hf["close"] = hf["close"].astype(float)
    hf["volume"] = hf["volume"].astype(float)

    # Need at least max window
    if len(hf) >= max(hourly_ctx_windows):
      hourly_mas = ma_snapshot(hf, windows=list(hourly_ctx_windows), price_col="close")
      hourly_align = ma_order_score(hourly_mas, order=list(hourly_ctx_windows), higher_is_better=True)
      hourly_rv20 = rel_volume(hf, window=min(20, max(5, len(hf)//2)))  # adapt if limited

  # -------- Ranking normalization --------
  # Volume quality: mild preference for rel vol >=1 and contraction <=1
  vol_score = 0.0
  if rv20 is not None:
    vol_score += clamp01(min(rv20, 3.0) / 3.0)  # 0..1
  if vc is not None:
    # contraction good: vc=0.5 -> ~1.0 ; vc=1.5 -> ~0.4
    vol_score += clamp01(1.0 / (1.0 + max(0.0, vc - 0.5)))
  vol_score = vol_score / 2.0 if vol_score > 0 else 0.0

  # Risk score: prefer moderate ATR% (too high risk, too low can mean dead)
  risk_score = 0.0
  if atrp is not None:
    # Target band ~1%..6%
    if atrp <= 0.01:
      risk_score = 0.4
    elif atrp >= 0.10:
      risk_score = 0.0
    else:
      # linear decay after 6%
      risk_score = clamp01(1.0 - max(0.0, (atrp - 0.06)) / 0.04)
  # Liquidity score: prefer dv20 >= $20M/day
  liq_score = 0.0
  if dv20 is not None:
    liq_score = clamp01(min(dv20, 50_000_000.0) / 50_000_000.0)  # 0..1, saturates at 50M

  components = {
    "daily_align": float(daily_align),
    "hourly_align": float(hourly_align),
    "above_bonus": float(above_bonus),
    "volume": float(vol_score),
    "risk": float(risk_score),
    "liquidity": float(liq_score),
  }
  weights = {
    "daily_align": 0.30,
    "hourly_align": 0.25,
    "above_bonus": 0.10,
    "volume": 0.15,
    "risk": 0.10,
    "liquidity": 0.10,
  }

  raw_score = score_from_components(components, weights)

  # Only rank BUY (or WATCH) signals meaningfully
  if state == "BUY":
    score = raw_score
  elif state == "SELL":
    score = raw_score * 0.25
  else:
    score = None

  stop = None
  if atr is not None:
    stop_atr = float(params.get("stop_atr", 2.0))
    stop = close - stop_atr * float(atr)

  meta: Dict[str, Any] = {
    "signal": {"cross_up": bool(cross_up), "cross_down": bool(cross_down)},
    "daily": {
      "close": close,
      "ma_fast": fast_t,
      "ma_slow": slow_t,
      "ctx_mas": daily_mas,
      "rel_vol_20": rv20,
      "vol_contraction_5_20": vc,
      "atr14": atr,
      "atr_pct": atrp,
      "dollar_vol_20": dv20,
      "align_score": daily_align,
    },
    "hourly": {
      "align_score": hourly_align,
      "rel_vol_like": hourly_rv20,
      "has_hourly": bool(ctx and ctx.get("hourly") is not None and isinstance(ctx.get("hourly"), pd.DataFrame)),
    },
    "ranking": {
      "components": components,
      "weights": weights,
      "raw_score_0_100": raw_score,
    },
    "watch": bool(state in ("BUY", "SELL")),
    "action": bool(state == "BUY"),
  }

  return {"state": state, "score": score, "stop": stop, "meta": meta}
