from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et

log = setup_logger("report")

EPS = 1e-12

# -----------------------------
# Output paths
# -----------------------------
def _out_dir_for_date(date: str) -> Path:
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
  day_dir = out_dir / date
  day_dir.mkdir(parents=True, exist_ok=True)
  return day_dir


# -----------------------------
# DB reads
# -----------------------------
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


def _ph(tickers: List[str]) -> str:
  return ",".join(["?"] * len(tickers))


def read_prices_daily_window(end_date: str, tickers: List[str], lookback_days: int) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()

  end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
  start_d = end_d - timedelta(days=lookback_days)

  with connect() as conn:
    q = f"""
      SELECT ticker, date, open, high, low, close, volume
      FROM prices_daily
      WHERE date BETWEEN ? AND ?
        AND ticker IN ({_ph(tickers)})
    """
    df = pd.read_sql_query(q, conn, params=[start_d.isoformat(), end_date] + tickers)

  if df.empty:
    return df

  df["date"] = pd.to_datetime(df["date"])
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


def read_prices_hourly_window(end_date_et: str, tickers: List[str], lookback_days: int) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()

  end_d = datetime.strptime(end_date_et, "%Y-%m-%d").date()
  start_d = end_d - timedelta(days=lookback_days)

  with connect() as conn:
    q = f"""
      SELECT ticker, ts, date_et, open, high, low, close, volume
      FROM prices_hourly
      WHERE date_et BETWEEN ? AND ?
        AND ticker IN ({_ph(tickers)})
    """
    df = pd.read_sql_query(q, conn, params=[start_d.isoformat(), end_date_et] + tickers)

  if df.empty:
    return df

  # ts is RFC3339-like string from Alpaca; pandas can parse it
  df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


# -----------------------------
# Feature engineering
# -----------------------------
def _compute_atr_14(g: pd.DataFrame) -> pd.Series:
  high = g["high"].astype(float)
  low = g["low"].astype(float)
  close = g["close"].astype(float)
  prev_close = close.shift(1)

  tr1 = (high - low).abs()
  tr2 = (high - prev_close).abs()
  tr3 = (low - prev_close).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

  return tr.rolling(window=14, min_periods=14).mean()


def compute_daily_features(px: pd.DataFrame, end_date: str) -> pd.DataFrame:
  """
  Returns per-ticker features as-of end_date (or last available <= end_date):
    daily_bar_count, close, volume, range_pct,
    ma14/30/60/120, trend_daily_score,
    atr14, atr_pct,
    vol20, dollar_vol_20, vol_rel
  """
  if px.empty:
    return pd.DataFrame()

  px = px.sort_values(["ticker", "date"]).copy()
  px["range_pct"] = (px["high"] - px["low"]) / (px["close"].abs() + EPS)
  px["dollar_vol"] = (px["close"].abs() * px["volume"].abs())

  feats = []
  end_ts = pd.to_datetime(end_date)

  for t, g in px.groupby("ticker", sort=False):
    g = g.sort_values("date").reset_index(drop=True)
    g = g[g["date"] <= end_ts].copy()
    if g.empty:
      continue

    g["atr14"] = _compute_atr_14(g)
    for w in (14, 30, 60, 120):
      g[f"ma{w}"] = g["close"].rolling(window=w, min_periods=w).mean()

    g["vol20"] = g["volume"].rolling(window=20, min_periods=20).mean()
    g["dollar_vol_20"] = g["dollar_vol"].rolling(window=20, min_periods=20).mean()

    last = g.iloc[-1]

    ma14 = last.get("ma14")
    ma30 = last.get("ma30")
    ma60 = last.get("ma60")
    ma120 = last.get("ma120")

    # MA stack score: MA14>MA30>MA60>MA120 (0..3)
    td = 0
    if pd.notna(ma14) and pd.notna(ma30) and ma14 > ma30:
      td += 1
    if pd.notna(ma30) and pd.notna(ma60) and ma30 > ma60:
      td += 1
    if pd.notna(ma60) and pd.notna(ma120) and ma60 > ma120:
      td += 1

    close = float(last["close"]) if pd.notna(last["close"]) else None
    atr14 = float(last["atr14"]) if pd.notna(last.get("atr14")) else None
    atr_pct = (atr14 / (close + EPS)) if (atr14 is not None and close is not None) else None

    vol = float(last["volume"]) if pd.notna(last["volume"]) else None
    vol20 = float(last["vol20"]) if pd.notna(last.get("vol20")) else None
    vol_rel = (vol / (vol20 + EPS)) if (vol is not None and vol20 is not None) else None

    feats.append({
      "ticker": t,
      "daily_bar_count": int(len(g)),
      "close": close,
      "volume": vol,
      "range_pct": float(last["range_pct"]) if pd.notna(last["range_pct"]) else None,
      "ma14_d": float(ma14) if pd.notna(ma14) else None,
      "ma30_d": float(ma30) if pd.notna(ma30) else None,
      "ma60_d": float(ma60) if pd.notna(ma60) else None,
      "ma120_d": float(ma120) if pd.notna(ma120) else None,
      "trend_daily_score": td,
      "atr14": atr14,
      "atr_pct": float(atr_pct) if atr_pct is not None else None,
      "vol20": vol20,
      "dollar_vol_20": float(last["dollar_vol_20"]) if pd.notna(last.get("dollar_vol_20")) else None,
      "vol_rel": float(vol_rel) if vol_rel is not None else None,
    })

  return pd.DataFrame(feats)


def compute_hourly_features(hx: pd.DataFrame, trade_date_et: str) -> pd.DataFrame:
  """
  Returns per-ticker hourly features as-of last bar <= trade_date_et end:
    hourly_bar_count_window, hourly_bar_count_trade_date,
    ma14/30/60/120 (hourly), trend_hourly_score
  """
  if hx.empty:
    return pd.DataFrame()

  hx = hx.sort_values(["ticker", "ts"]).copy()

  feats = []
  for t, g in hx.groupby("ticker", sort=False):
    g = g.sort_values("ts").reset_index(drop=True)
    if g.empty:
      continue

    for w in (14, 30, 60, 120):
      g[f"ma{w}"] = g["close"].rolling(window=w, min_periods=w).mean()

    last = g.iloc[-1]
    ma14 = last.get("ma14")
    ma30 = last.get("ma30")
    ma60 = last.get("ma60")
    ma120 = last.get("ma120")

    th = 0
    if pd.notna(ma14) and pd.notna(ma30) and ma14 > ma30:
      th += 1
    if pd.notna(ma30) and pd.notna(ma60) and ma30 > ma60:
      th += 1
    if pd.notna(ma60) and pd.notna(ma120) and ma60 > ma120:
      th += 1

    cnt_trade = int((g["date_et"] == trade_date_et).sum())
    feats.append({
      "ticker": t,
      "hourly_bar_count_window": int(len(g)),
      "hourly_bar_count_trade_date": cnt_trade,
      "ma14_h": float(ma14) if pd.notna(ma14) else None,
      "ma30_h": float(ma30) if pd.notna(ma30) else None,
      "ma60_h": float(ma60) if pd.notna(ma60) else None,
      "ma120_h": float(ma120) if pd.notna(ma120) else None,
      "trend_hourly_score": th,
    })

  return pd.DataFrame(feats)


# -----------------------------
# Meta parsing (retest_shrink needs fields inside meta_json)
# -----------------------------
def parse_meta_fields(df: pd.DataFrame) -> pd.DataFrame:
  """
  Extract a few useful fields without assuming a strict shape.
  generate_signals stores meta_json from strategy adapter: {"raw_state","params","raw":{...}}
  """
  def _get(d: Any, path: List[str]) -> Any:
    cur = d
    for k in path:
      if not isinstance(cur, dict):
        return None
      cur = cur.get(k)
    return cur

  out = df.copy()
  raw_states = []
  shrink_ratios = []
  l0s = []
  atr0s = []
  v0s = []
  d0_dates = []
  retest_dates = []
  confirm_dates = []

  for s in out.get("meta_json", pd.Series([None] * len(out))):
    try:
      meta = json.loads(s) if isinstance(s, str) and s else {}
    except Exception:
      meta = {}

    raw_state = _get(meta, ["raw_state"])
    raw = _get(meta, ["raw"])  # this should be the rich dict
    if not isinstance(raw, dict):
      raw = {}

    raw_states.append(raw_state if raw_state is not None else None)
    shrink_ratios.append(_get(raw, ["shrink_ratio"]))
    l0s.append(_get(raw, ["l0"]))
    atr0s.append(_get(raw, ["atr0"]))
    v0s.append(_get(raw, ["v0"]))
    d0_dates.append(_get(raw, ["d0_date"]))
    retest_dates.append(_get(raw, ["retest_date"]))
    confirm_dates.append(_get(raw, ["confirm_date"]))

  out["raw_state"] = raw_states
  out["shrink_ratio"] = pd.to_numeric(shrink_ratios, errors="coerce")
  out["l0"] = pd.to_numeric(l0s, errors="coerce")
  out["atr0"] = pd.to_numeric(atr0s, errors="coerce")
  out["v0"] = pd.to_numeric(v0s, errors="coerce")
  out["d0_date"] = d0_dates
  out["retest_date"] = retest_dates
  out["confirm_date"] = confirm_dates
  return out


# -----------------------------
# Ranking
# -----------------------------
def _state_rank(s: str) -> int:
  # lower is "more important"
  s = (s or "").upper()
  order = {
    "ENTRY": 0,
    "EXIT": 1,
    "WATCH": 2,
    "PASS": 3,
    "ERROR": 9,
  }
  return order.get(s, 5)


def compute_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()

  # Liquidity scaler: log10(dollar_vol_20+1) => typical range ~ 4..9
  out["liq_log10"] = (out["dollar_vol_20"].fillna(0) + 1.0).apply(lambda x: float(pd.np.log10(x)) if x > 0 else 0.0)  # type: ignore

  # Normalize some terms into rough 0..1 bands
  out["trend_d_n"] = out["trend_daily_score"].fillna(0) / 3.0
  out["trend_h_n"] = out["trend_hourly_score"].fillna(0) / 3.0

  # vol_rel: cap at 2.0, scale to 0..1
  vr = out["vol_rel"].copy()
  out["vol_rel_n"] = (vr.clip(lower=0, upper=2.0) / 2.0).fillna(0)

  # atr_pct: penalty, cap at 0.12 (12% ATR)
  ap = out["atr_pct"].copy()
  out["risk_pen_n"] = (ap.clip(lower=0, upper=0.12) / 0.12).fillna(0)

  # retest proximity: dist to l0 in ATR0 units
  dist_atr = (out["close"] - out["l0"]).abs() / (out["atr0"] + EPS)
  out["dist_to_l0_atr"] = dist_atr
  out["prox_n"] = (1.0 / (1.0 + dist_atr)).fillna(0)

  # shrink quality: 1 - shrink_ratio (clip 0..1)
  out["shrink_n"] = (1.0 - out["shrink_ratio"]).clip(lower=0, upper=1).fillna(0)

  # Liquidity: map log10 to 0..1 using 4..9 typical band
  out["liq_n"] = ((out["liq_log10"] - 4.0) / 5.0).clip(lower=0, upper=1).fillna(0)

  def _rank_row(r: pd.Series) -> float:
    strat = str(r.get("strategy", ""))
    # Default: fall back to strategy score if present
    base = r.get("score", None)
    base_n = 0.0
    try:
      if base is not None and pd.notna(base):
        # light compression
        base_n = float(max(0.0, min(1.0, float(base))))
    except Exception:
      base_n = 0.0

    # MA cross ranking: trend + liquidity + volume + risk
    if strat == "ma_cross_5_10":
      return float(
        0.45 * r.get("trend_d_n", 0.0) +
        0.25 * r.get("trend_h_n", 0.0) +
        0.15 * r.get("vol_rel_n", 0.0) +
        0.15 * r.get("liq_n", 0.0) -
        0.15 * r.get("risk_pen_n", 0.0) +
        0.05 * base_n
      )

    # Retest/shrink ranking: shrink quality + proximity + trend + liquidity - risk
    if strat.startswith("retest_shrink"):
      return float(
        0.45 * r.get("shrink_n", 0.0) +
        0.25 * r.get("prox_n", 0.0) +
        0.15 * r.get("trend_d_n", 0.0) +
        0.15 * r.get("liq_n", 0.0) -
        0.20 * r.get("risk_pen_n", 0.0)
      )

    # Generic fallback
    return float(
      0.40 * r.get("trend_d_n", 0.0) +
      0.20 * r.get("trend_h_n", 0.0) +
      0.20 * r.get("liq_n", 0.0) +
      0.10 * r.get("vol_rel_n", 0.0) -
      0.15 * r.get("risk_pen_n", 0.0) +
      0.05 * base_n
    )

  out["rank_score"] = out.apply(_rank_row, axis=1)
  return out


# -----------------------------
# Main
# -----------------------------
def main():
  init_db()

  date = os.environ.get("REPORT_DATE") or last_completed_trading_day_et()
  day_dir = _out_dir_for_date(date)

  df = read_signals(date)
  if df.empty:
    raise RuntimeError(f"No signals found for {date}. Run generate_signals first.")

  # Normalize numeric
  df["score"] = pd.to_numeric(df["score"], errors="coerce")
  df["stop"] = pd.to_numeric(df["stop"], errors="coerce")
  df["ticker"] = df["ticker"].astype(str)

  # Parse meta fields (for retest_shrink ranking)
  df = parse_meta_fields(df)

  # Build features only for tickers present today
  tickers = df["ticker"].dropna().astype(str).unique().tolist()

  DAILY_LOOKBACK_DAYS = int(os.environ.get("DAILY_LOOKBACK_DAYS", "300"))
  HOURLY_LOOKBACK_DAYS = int(os.environ.get("HOURLY_LOOKBACK_DAYS", "30"))

  px = read_prices_daily_window(end_date=date, tickers=tickers, lookback_days=DAILY_LOOKBACK_DAYS)
  hx = read_prices_hourly_window(end_date_et=date, tickers=tickers, lookback_days=HOURLY_LOOKBACK_DAYS)

  daily_f = compute_daily_features(px, end_date=date)
  hourly_f = compute_hourly_features(hx, trade_date_et=date)

  if not daily_f.empty:
    df = df.merge(daily_f, on="ticker", how="left")
  else:
    # ensure columns exist
    for c in ["daily_bar_count","close","volume","range_pct","ma14_d","ma30_d","ma60_d","ma120_d",
              "trend_daily_score","atr14","atr_pct","vol20","dollar_vol_20","vol_rel"]:
      df[c] = pd.NA

  if not hourly_f.empty:
    df = df.merge(hourly_f, on="ticker", how="left")
  else:
    for c in ["hourly_bar_count_window","hourly_bar_count_trade_date",
              "ma14_h","ma30_h","ma60_h","ma120_h","trend_hourly_score"]:
      df[c] = pd.NA

  # Coverage gating (tunable)
  DAILY_MIN_BARS = int(os.environ.get("DAILY_MIN_BARS", "120"))
  HOURLY_MIN_BARS_TRADE_DATE = int(os.environ.get("HOURLY_MIN_BARS_TRADE_DATE", "5"))

  # By default: require dollar_vol_20 present; optional min threshold
  DOLLAR_VOL20_MIN = float(os.environ.get("DOLLAR_VOL20_MIN", "0"))

  df["coverage_ok"] = (
    (pd.to_numeric(df["daily_bar_count"], errors="coerce").fillna(0) >= DAILY_MIN_BARS) &
    (pd.to_numeric(df["hourly_bar_count_trade_date"], errors="coerce").fillna(0) >= HOURLY_MIN_BARS_TRADE_DATE) &
    (pd.to_numeric(df["dollar_vol_20"], errors="coerce").notna()) &
    (pd.to_numeric(df["dollar_vol_20"], errors="coerce").fillna(-1) >= DOLLAR_VOL20_MIN)
  )

  # Ranking
  df = compute_rank_scores(df)

  # Prepare outputs
  strategies = sorted(df["strategy"].dropna().unique().tolist())

  summary_lines: List[str] = []
  summary_lines.append(f"Report date: {date}")
  summary_lines.append(f"Strategies: {', '.join(strategies)}")
  summary_lines.append("")
  summary_lines.append("Coverage thresholds:")
  summary_lines.append(f"  DAILY_MIN_BARS={DAILY_MIN_BARS}")
  summary_lines.append(f"  HOURLY_MIN_BARS_TRADE_DATE={HOURLY_MIN_BARS_TRADE_DATE}")
  summary_lines.append(f"  DOLLAR_VOL20_MIN={DOLLAR_VOL20_MIN}")
  summary_lines.append("")

  # Per-strategy CSVs
  files_written = 0
  for strat in strategies:
    d = df[df["strategy"] == strat].copy()

    # Sort by state importance then rank_score desc (nulls last)
    d["state_rank"] = d["state"].apply(_state_rank)
    d = d.sort_values(["state_rank", "rank_score"], ascending=[True, False], na_position="last")

    out_csv = day_dir / f"signals_{strat}.csv"
    d.drop(columns=["state_rank"], errors="ignore").to_csv(out_csv, index=False)
    files_written += 1

    counts = d["state"].value_counts(dropna=False).to_dict()
    cov_ok = int(d["coverage_ok"].fillna(False).sum())
    summary_lines.append(f"[{strat}] states: " + ", ".join([f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]))
    summary_lines.append(f"[{strat}] coverage_ok: {cov_ok}/{len(d)}")
    summary_lines.append("")

  # Build action/watch lists (to satisfy email job)
  def _dedup_best(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
      return x
    x = x.sort_values(["rank_score"], ascending=[False], na_position="last")
    return x.drop_duplicates(subset=["ticker"], keep="first")

  action = df[df["state"].isin(["ENTRY", "EXIT"]) & df["coverage_ok"].fillna(False)].copy()
  watch = df[(df["state"] == "WATCH") & df["coverage_ok"].fillna(False)].copy()

  action = _dedup_best(action)
  watch = _dedup_best(watch)

  # Keep columns concise for email
  keep_cols = [
    "ticker","strategy","state","rank_score",
    "close","stop","dollar_vol_20","trend_daily_score","trend_hourly_score","atr_pct",
    "raw_state","shrink_ratio","dist_to_l0_atr",
  ]
  action_out = action[[c for c in keep_cols if c in action.columns]].copy()
  watch_out = watch[[c for c in keep_cols if c in watch.columns]].copy()

  action_path = day_dir / "action_list.csv"
  watch_path = day_dir / "watch_list.csv"
  action_out.to_csv(action_path, index=False)
  watch_out.to_csv(watch_path, index=False)
  files_written += 2

  # Summary extras
  summary_lines.append("Top ACTION (up to 10): " + ", ".join(action_out["ticker"].astype(str).head(10).tolist()) if not action_out.empty else "Top ACTION: (none)")
  summary_lines.append("Top WATCH  (up to 10): " + ", ".join(watch_out["ticker"].astype(str).head(10).tolist()) if not watch_out.empty else "Top WATCH: (none)")
  summary_lines.append("")

  summary_path = day_dir / "summary.txt"
  summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
  files_written += 1

  log.info(f"Wrote reports to {day_dir} (files={files_written})")


if __name__ == "__main__":
  main()
