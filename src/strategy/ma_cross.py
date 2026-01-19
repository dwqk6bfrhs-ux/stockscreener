import pandas as pd

STRATEGY_NAME = "ma_cross"

DEFAULT_PARAMS = {
  "fast_window": 5,
  "slow_window": 10,
  "price_col": "close",
  "min_history": 30,   # calendar rows requirement; you can tune
  "min_cross_pct": 0.0 # optional filter: require spread% > X on cross day
}

def evaluate(df: pd.DataFrame, params: dict) -> dict:
  """
  df columns expected: date (datetime64), close (float), ticker (optional)
  Must be sorted by date ascending.
  """
  p = {**DEFAULT_PARAMS, **(params or {})}

  fast = int(p["fast_window"])
  slow = int(p["slow_window"])
  price_col = p["price_col"]
  min_history = int(p["min_history"])
  min_cross_pct = float(p.get("min_cross_pct", 0.0))

  if df is None or df.empty:
    return {"state": "ERROR", "score": None, "stop": None, "meta": {"reason": "empty_df"}}

  df = df.sort_values("date").copy()

  if price_col not in df.columns:
    return {"state": "ERROR", "score": None, "stop": None, "meta": {"reason": f"missing_col:{price_col}"}}

  if len(df) < max(min_history, slow + 2):
    return {"state": "PASS", "score": None, "stop": None, "meta": {"reason": "insufficient_history", "rows": len(df)}}

  px = df[price_col].astype(float)
  ma_fast = px.rolling(window=fast, min_periods=fast).mean()
  ma_slow = px.rolling(window=slow, min_periods=slow).mean()

  # Need last two points to detect a cross today
  f_prev, f_now = ma_fast.iloc[-2], ma_fast.iloc[-1]
  s_prev, s_now = ma_slow.iloc[-2], ma_slow.iloc[-1]

  if pd.isna(f_prev) or pd.isna(f_now) or pd.isna(s_prev) or pd.isna(s_now):
    return {"state": "PASS", "score": None, "stop": None, "meta": {"reason": "ma_nan"}}

  # Cross conditions
  crossed_up = (f_prev <= s_prev) and (f_now > s_now)
  crossed_down = (f_prev >= s_prev) and (f_now < s_now)

  # Spread as a % of price (useful for scoring / filtering)
  close_now = float(px.iloc[-1])
  spread = float(f_now - s_now)
  spread_pct = (spread / close_now) if close_now else 0.0

  # Optional filter to ignore tiny crosses
  if (crossed_up or crossed_down) and abs(spread_pct) < min_cross_pct:
    return {
      "state": "PASS",
      "score": None,
      "stop": None,
      "meta": {"reason": "cross_too_small", "spread_pct": spread_pct, "min_cross_pct": min_cross_pct}
    }

  if crossed_up:
    return {
      "state": "ENTRY",
      "score": abs(spread_pct),
      "stop": None,
      "meta": {
        "fast_window": fast, "slow_window": slow,
        "ma_fast_prev": float(f_prev), "ma_slow_prev": float(s_prev),
        "ma_fast_now": float(f_now),  "ma_slow_now": float(s_now),
        "spread": spread, "spread_pct": spread_pct
      }
    }

  if crossed_down:
    return {
      "state": "EXIT",
      "score": abs(spread_pct),
      "stop": None,
      "meta": {
        "fast_window": fast, "slow_window": slow,
        "ma_fast_prev": float(f_prev), "ma_slow_prev": float(s_prev),
        "ma_fast_now": float(f_now),  "ma_slow_now": float(s_now),
        "spread": spread, "spread_pct": spread_pct
      }
    }

  # No signal today
  return {
    "state": "PASS",
    "score": float(abs(spread_pct)),  # optional: keep for ranking even without a cross
    "stop": None,
    "meta": {
      "fast_window": fast, "slow_window": slow,
      "ma_fast_now": float(f_now), "ma_slow_now": float(s_now),
      "spread": spread, "spread_pct": spread_pct
    }
  }
