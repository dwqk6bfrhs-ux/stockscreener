# src/jobs/report.py
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import last_completed_trading_day_et
from datetime import datetime

log = setup_logger("report")


# -----------------------------
# Ranking configuration (override via env if desired)
# -----------------------------
# Coverage gates
MIN_DAILY_BARS = int(os.environ.get("REPORT_MIN_DAILY_BARS", "60"))
MIN_HOURLY_BARS_TODAY = int(os.environ.get("REPORT_MIN_HOURLY_BARS_TODAY", "6"))

# Liquidity normalization for dv20 (mean close*volume over 20 days)
LIQ_DV20_MIN = float(os.environ.get("REPORT_LIQ_DV20_MIN", "1e7"))   # $10M
LIQ_DV20_MAX = float(os.environ.get("REPORT_LIQ_DV20_MAX", "2e8"))   # $200M

# Risk normalization (ATR% cap)
RISK_ATR_PCT_MAX = float(os.environ.get("REPORT_RISK_ATR_PCT_MAX", "0.12"))

# Retest-shrink rank weights
RS_W_BASE = float(os.environ.get("REPORT_RS_W_BASE", "0.60"))
RS_W_TREND = float(os.environ.get("REPORT_RS_W_TREND", "0.15"))
RS_W_LIQ = float(os.environ.get("REPORT_RS_W_LIQ", "0.15"))
RS_W_RISK = float(os.environ.get("REPORT_RS_W_RISK", "0.10"))

# Retest-shrink base weights
RS_W_SHRINK = float(os.environ.get("REPORT_RS_W_SHRINK", "0.55"))
RS_W_PROX = float(os.environ.get("REPORT_RS_W_PROX", "0.45"))


# MA cross ranking (v1 – simple but useful)
MC_W_DAILY_TREND = float(os.environ.get("REPORT_MC_W_DAILY_TREND", "0.50"))
MC_W_HOURLY_TREND = float(os.environ.get("REPORT_MC_W_HOURLY_TREND", "0.30"))
MC_W_VOL = float(os.environ.get("REPORT_MC_W_VOL", "0.20"))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
  if x is None or (isinstance(x, float) and math.isnan(x)):
    return 0.0
  if x < lo:
    return lo
  if x > hi:
    return hi
  return x


def _out_dir_for_date(date: str) -> Path:
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
  day_dir = out_dir / date
  day_dir.mkdir(parents=True, exist_ok=True)
  return day_dir


def _read_signals(date: str) -> pd.DataFrame:
  with connect() as conn:
    df = pd.read_sql_query(
      """
      SELECT date, ticker, strategy, state, score, stop, meta_json
      FROM signals_daily
      WHERE date = ?
      """,
      conn,
      params=(date,),
    )
  return df


def _parse_meta(meta_json: Optional[str]) -> Dict[str, Any]:
  if not meta_json:
    return {}
  try:
    return json.loads(meta_json)
  except Exception:
    return {}


def _sql_in_placeholders(n: int) -> str:
  return ",".join(["?"] * n)


def _read_prices_daily_window(end_date: str, tickers: List[str], lookback_days: int = 260) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()

  import datetime
  end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - datetime.timedelta(days=lookback_days)).isoformat()
  ph = _sql_in_placeholders(len(tickers))

  with connect() as conn:
    df = pd.read_sql_query(
      f"""
      SELECT ticker, date, open, high, low, close, volume
      FROM prices_daily
      WHERE date BETWEEN ? AND ?
        AND ticker IN ({ph})
      ORDER BY ticker, date
      """,
      conn,
      params=[start, end_date] + tickers,
    )

  if df.empty:
    return df
  df["date"] = pd.to_datetime(df["date"])
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


def _read_prices_hourly_window(end_date: str, tickers: List[str], lookback_days: int = 35) -> pd.DataFrame:
  if not tickers:
    return pd.DataFrame()

  import datetime
  end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
  start = (end - datetime.timedelta(days=lookback_days)).isoformat()
  ph = _sql_in_placeholders(len(tickers))

  with connect() as conn:
    df = pd.read_sql_query(
      f"""
      SELECT ticker, ts, date_et, open, high, low, close, volume
      FROM prices_hourly
      WHERE date_et BETWEEN ? AND ?
        AND ticker IN ({ph})
      ORDER BY ticker, ts
      """,
      conn,
      params=[start, end_date] + tickers,
    )

  if df.empty:
    return df
  # ts is RFC3339-ish string; ordering already ensured by SQL
  for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df


def _compute_atr14(g: pd.DataFrame) -> pd.Series:
  high = g["high"]
  low = g["low"]
  close = g["close"]
  prev_close = close.shift(1)
  tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
  return tr.rolling(window=14, min_periods=14).mean()


def _stack_score_3(ma_fast: float, ma_mid: float, ma_slow: float) -> float:
  # 0..1 with partial credit
  if any(pd.isna(x) for x in [ma_fast, ma_mid, ma_slow]):
    return 0.0
  s = 0.0
  if float(ma_fast) > float(ma_mid):
    s += 0.5
  if float(ma_mid) > float(ma_slow):
    s += 0.5
  return s


def _stack_score_4(ma1: float, ma2: float, ma3: float, ma4: float) -> float:
  # MA14>MA30>MA60>MA120 style: 3 comparisons => each 1/3
  if any(pd.isna(x) for x in [ma1, ma2, ma3, ma4]):
    return 0.0
  s = 0.0
  s += (1.0 / 3.0) if float(ma1) > float(ma2) else 0.0
  s += (1.0 / 3.0) if float(ma2) > float(ma3) else 0.0
  s += (1.0 / 3.0) if float(ma3) > float(ma4) else 0.0
  return float(s)


def _liq_score(dv20: Optional[float]) -> float:
  if dv20 is None or dv20 <= 0:
    return 0.0
  return _clamp((dv20 - LIQ_DV20_MIN) / (LIQ_DV20_MAX - LIQ_DV20_MIN + 1e-12))


def _risk_score(atr_pct: Optional[float]) -> float:
  if atr_pct is None or atr_pct <= 0:
    return 0.0
  return _clamp(1.0 - (atr_pct / (RISK_ATR_PCT_MAX + 1e-12)))


def _build_daily_features(daily: pd.DataFrame, report_date: str) -> pd.DataFrame:
  if daily.empty:
    return pd.DataFrame()

  rows = []
  for t, g in daily.groupby("ticker", sort=False):
    g = g.sort_values("date").reset_index(drop=True)

    g["dollar_vol"] = (g["close"].abs() * g["volume"].abs())
    g["dv20"] = g["dollar_vol"].rolling(20, min_periods=20).mean()
    g["atr14"] = _compute_atr14(g)
    g["atr_pct"] = g["atr14"] / (g["close"].abs() + 1e-12)

    # MAs for retest trend context
    g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
    g["ma50"] = g["close"].rolling(50, min_periods=50).mean()
    g["ma200"] = g["close"].rolling(200, min_periods=200).mean()

    # MAs for MA stack scoring
    g["ma14"] = g["close"].rolling(14, min_periods=14).mean()
    g["ma30"] = g["close"].rolling(30, min_periods=30).mean()
    g["ma60"] = g["close"].rolling(60, min_periods=60).mean()
    g["ma120"] = g["close"].rolling(120, min_periods=120).mean()

    g["date_s"] = g["date"].dt.strftime("%Y-%m-%d")
    last = g[g["date_s"] == report_date].tail(1)
    if last.empty:
      continue
    lr = last.iloc[0]

    daily_bars = int(g[g["date_s"] <= report_date].shape[0])

    trend_rs = _stack_score_3(lr.get("ma20"), lr.get("ma50"), lr.get("ma200"))
    trend_stack_4 = _stack_score_4(lr.get("ma14"), lr.get("ma30"), lr.get("ma60"), lr.get("ma120"))

    rows.append({
      "ticker": t,
      "daily_bars": daily_bars,
      "close": float(lr["close"]) if pd.notna(lr["close"]) else None,
      "volume": float(lr["volume"]) if pd.notna(lr["volume"]) else None,
      "range_pct": (float(lr["high"]) - float(lr["low"])) / (float(lr["close"]) + 1e-12) if pd.notna(lr["close"]) else None,
      "dv20": float(lr["dv20"]) if pd.notna(lr["dv20"]) else None,
      "atr14": float(lr["atr14"]) if pd.notna(lr["atr14"]) else None,
      "atr_pct": float(lr["atr_pct"]) if pd.notna(lr["atr_pct"]) else None,
      "trend_rs": float(trend_rs),
      "trend_stack_4_daily": float(trend_stack_4),
      "ma14": float(lr["ma14"]) if pd.notna(lr["ma14"]) else None,
      "ma30": float(lr["ma30"]) if pd.notna(lr["ma30"]) else None,
      "ma60": float(lr["ma60"]) if pd.notna(lr["ma60"]) else None,
      "ma120": float(lr["ma120"]) if pd.notna(lr["ma120"]) else None,
    })

  return pd.DataFrame(rows)


def _build_hourly_features(hourly: pd.DataFrame, report_date: str) -> pd.DataFrame:
  if hourly.empty:
    return pd.DataFrame()

  rows = []
  for t, g in hourly.groupby("ticker", sort=False):
    g = g.sort_values("ts").reset_index(drop=True)

    # Only compute on close series
    g["ma14"] = g["close"].rolling(14, min_periods=14).mean()
    g["ma30"] = g["close"].rolling(30, min_periods=30).mean()
    g["ma60"] = g["close"].rolling(60, min_periods=60).mean()
    g["ma120"] = g["close"].rolling(120, min_periods=120).mean()

    # coverage on the report date
    bars_today = int((g["date_et"] == report_date).sum())

    last = g[g["date_et"] == report_date].tail(1)
    if last.empty:
      # still keep coverage (0) info
      rows.append({
        "ticker": t,
        "hourly_bars_today": bars_today,
        "trend_stack_4_hourly": 0.0,
      })
      continue

    lr = last.iloc[0]
    trend_stack = _stack_score_4(lr.get("ma14"), lr.get("ma30"), lr.get("ma60"), lr.get("ma120"))

    rows.append({
      "ticker": t,
      "hourly_bars_today": bars_today,
      "trend_stack_4_hourly": float(trend_stack),
    })

  return pd.DataFrame(rows)


def _rank_retest_shrink(row: pd.Series) -> Tuple[Optional[float], str]:
  # Requires shrink_ratio + dist_to_l0_atr from strategy meta.features,
  # plus dv20/atr_pct/trend from daily features.
  shrink_ratio = row.get("shrink_ratio")
  dist_to_l0_atr = row.get("dist_to_l0_atr")

  if shrink_ratio is None or dist_to_l0_atr is None:
    return None, "missing_core_features"

  s_shrink = _clamp(1.0 - float(shrink_ratio))
  s_prox = _clamp(1.0 - float(dist_to_l0_atr))  # already in ATR units
  base = RS_W_SHRINK * s_shrink + RS_W_PROX * s_prox

  trend = float(row.get("trend_rs", 0.0))
  liq = _liq_score(row.get("dv20"))
  risk = _risk_score(row.get("atr_pct"))

  score = RS_W_BASE * base + RS_W_TREND * trend + RS_W_LIQ * liq + RS_W_RISK * risk
  reason = f"base={base:.3f} trend={trend:.2f} liq={liq:.2f} risk={risk:.2f}"
  return float(score), reason


def _rank_ma_cross(row: pd.Series) -> Tuple[Optional[float], str]:
  # Minimal v1 ranking: daily trend stack + hourly trend stack + volume quality proxy
  td = float(row.get("trend_stack_4_daily", 0.0))
  th = float(row.get("trend_stack_4_hourly", 0.0))

  # Volume quality proxy: dv20 normalized (works even before you add more volume-pattern logic)
  liq = _liq_score(row.get("dv20"))
  score = MC_W_DAILY_TREND * td + MC_W_HOURLY_TREND * th + MC_W_VOL * liq
  reason = f"daily_stack={td:.2f} hourly_stack={th:.2f} liq={liq:.2f}"
  return float(score), reason

def _ensure_rank_scores_table() -> None:
  """
  Persist report ranking so other jobs (orders/backtest/LLM) can reuse it.
  """
  sql = """
  CREATE TABLE IF NOT EXISTS rank_scores_daily (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    rank_score REAL,
    meta_json TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, ticker, strategy)
  );

  CREATE INDEX IF NOT EXISTS idx_rank_scores_date_strategy_score
    ON rank_scores_daily(date, strategy, rank_score);

  CREATE INDEX IF NOT EXISTS idx_rank_scores_date_strategy
    ON rank_scores_daily(date, strategy);
  """
  with connect() as conn:
    conn.executescript(sql)
    conn.commit()


def _upsert_rank_scores(date: str, df: pd.DataFrame) -> int:
  """
  Write rank_score + key debug fields to rank_scores_daily.
  Uses an UPSERT so reruns are safe (no dup rows).
  """
  if df.empty:
    return 0

  now = datetime.utcnow().isoformat()
  rows = []

  # Ensure we have numeric rank_score
  rank_num = pd.to_numeric(df.get("rank_score"), errors="coerce")

  for i, r in df.iterrows():
    ticker = str(r.get("ticker"))
    strategy = str(r.get("strategy"))
    if not ticker or ticker == "None" or not strategy or strategy == "None":
      continue
    rs = rank_num.loc[i]
    rank_score = float(rs) if pd.notna(rs) else None

    meta = {
      "rank_reason": r.get("rank_reason"),
      "state": r.get("state"),
      "raw_state": r.get("raw_state"),
      "coverage": {
        "coverage_ok": bool(r.get("coverage_ok")) if r.get("coverage_ok") is not None else None,
        "coverage_daily_ok": bool(r.get("coverage_daily_ok")) if r.get("coverage_daily_ok") is not None else None,
        "coverage_hourly_ok": bool(r.get("coverage_hourly_ok")) if r.get("coverage_hourly_ok") is not None else None,
        "daily_bars": int(r.get("daily_bars")) if pd.notna(r.get("daily_bars")) else None,
        "hourly_bars_today": int(r.get("hourly_bars_today")) if pd.notna(r.get("hourly_bars_today")) else None,
      },
      # helpful context for execution/LLM later; small + stable
      "features": {
        "close": float(r.get("close")) if pd.notna(r.get("close")) else None,
        "dv20": float(r.get("dv20")) if pd.notna(r.get("dv20")) else None,
        "atr_pct": float(r.get("atr_pct")) if pd.notna(r.get("atr_pct")) else None,
      },
    }

    rows.append((
      date,
      ticker,
      strategy,
      rank_score,
      json.dumps(meta, ensure_ascii=False),
      now,
    ))

  sql = """
  INSERT INTO rank_scores_daily (date, ticker, strategy, rank_score, meta_json, updated_at)
  VALUES (?, ?, ?, ?, ?, ?)
  ON CONFLICT(date, ticker, strategy) DO UPDATE SET
    rank_score=excluded.rank_score,
    meta_json=excluded.meta_json,
    updated_at=excluded.updated_at
  """

  with connect() as conn:
    conn.executemany(sql, rows)
    conn.commit()

  return len(rows)

def main():
  init_db()

  date = os.environ.get("REPORT_DATE") or last_completed_trading_day_et()
  out_dir = _out_dir_for_date(date)

  df = _read_signals(date)
  if df.empty:
    raise RuntimeError(f"No signals found for {date}. Run generate_signals first.")

  # normalize numeric
  df["score"] = pd.to_numeric(df["score"], errors="coerce")
  df["stop"] = pd.to_numeric(df["stop"], errors="coerce")

  # parse meta/features
  metas = df["meta_json"].apply(_parse_meta)
  df["raw_state"] = metas.apply(lambda m: m.get("raw_state"))
  df["features"] = metas.apply(lambda m: m.get("features") or {})

  # pull core strategy features into columns (safe if absent)
  df["shrink_ratio"] = df["features"].apply(lambda f: f.get("shrink_ratio"))
  df["dist_to_l0_atr"] = df["features"].apply(lambda f: f.get("dist_to_l0_atr"))

  tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())

  # Build daily & hourly feature tables
  daily_hist = _read_prices_daily_window(date, tickers, lookback_days=260)
  daily_feat = _build_daily_features(daily_hist, date)

  hourly_hist = _read_prices_hourly_window(date, tickers, lookback_days=35)
  hourly_feat = _build_hourly_features(hourly_hist, date)

  # Merge pricing/feature info
  if not daily_feat.empty:
    df = df.merge(daily_feat, on="ticker", how="left")
  else:
    df["daily_bars"] = None

  if not hourly_feat.empty:
    df = df.merge(hourly_feat, on="ticker", how="left")
  else:
    df["hourly_bars_today"] = None
    df["trend_stack_4_hourly"] = 0.0

  # Coverage gates
  df["coverage_daily_ok"] = df["daily_bars"].fillna(0).astype(int) >= MIN_DAILY_BARS
  df["coverage_hourly_ok"] = df["hourly_bars_today"].fillna(0).astype(int) >= MIN_HOURLY_BARS_TODAY
  df["coverage_ok"] = df["coverage_daily_ok"]  # hourly optional for now; can tighten later

  # Rank score
  rank_scores = []
  rank_reasons = []
  for _, r in df.iterrows():
    strat = str(r["strategy"])
    if "retest" in strat:
      s, reason = _rank_retest_shrink(r)
    elif "ma_cross" in strat or "ma" in strat:
      s, reason = _rank_ma_cross(r)
    else:
      s, reason = (None, "no_ranker")
    rank_scores.append(s)
    rank_reasons.append(reason)

  df["rank_score"] = rank_scores
  df["rank_reason"] = rank_reasons

  # Sort with rank_score as primary (fallback to score)
  df["rank_score_num"] = pd.to_numeric(df["rank_score"], errors="coerce")
  df["sort_score"] = df["rank_score_num"].fillna(df["score"])

  # Write per-strategy debug outputs
  for strat in sorted(df["strategy"].unique().tolist()):
    d = df[df["strategy"] == strat].copy()
    d = d.sort_values(["state", "sort_score"], ascending=[True, False])
    out_csv = out_dir / f"signals_{strat}.csv"
    d.to_csv(out_csv, index=False)

  # Persist rank scores into DB for reuse (orders/backtest/LLM)
  _ensure_rank_scores_table()
  n_rank = _upsert_rank_scores(date, df)
  log.info(f"Upserted rank_scores_daily: date={date} rows={n_rank}")


  # Build action/watch lists for email
  # Keep email stable: must produce action_list.csv and watch_list.csv
  common_cols = [
    "date", "ticker", "strategy", "state", "raw_state",
    "rank_score", "rank_reason", "score", "stop",
    "close", "volume", "range_pct", "dv20", "atr_pct",
    "trend_rs", "trend_stack_4_daily", "trend_stack_4_hourly",
    "daily_bars", "hourly_bars_today",
    "coverage_ok", "coverage_daily_ok", "coverage_hourly_ok",
  ]
  for c in common_cols:
    if c not in df.columns:
      df[c] = None

  action = df[(df["state"] == "ENTRY") & (df["coverage_ok"])].copy()
  action = action.sort_values(["sort_score"], ascending=[False])
  action.to_csv(out_dir / "action_list.csv", index=False, columns=common_cols)

  watch = df[(df["state"] == "WATCH") & (df["coverage_ok"])].copy()
  watch = watch.sort_values(["sort_score"], ascending=[False])
  watch.to_csv(out_dir / "watch_list.csv", index=False, columns=common_cols)

  # Summary
  summary_lines = []
  summary_lines.append(f"Report date: {date}")
  summary_lines.append(f"Strategies: {', '.join(sorted(df['strategy'].unique().tolist()))}")
  summary_lines.append("")
  summary_lines.append(f"Coverage gate: daily_bars>={MIN_DAILY_BARS}, hourly_bars_today>={MIN_HOURLY_BARS_TODAY} (hourly currently optional)")
  summary_lines.append(f"Universe tickers in signals: {df['ticker'].nunique()}")
  summary_lines.append(f"Coverage OK: {int(df['coverage_ok'].sum())}/{len(df)} rows")
  summary_lines.append("")

  for strat in sorted(df["strategy"].unique().tolist()):
    d = df[df["strategy"] == strat]
    counts = d["state"].value_counts(dropna=False).to_dict()
    summary_lines.append(f"[{strat}] states: " + ", ".join([f"{k}={v}" for k, v in sorted(counts.items())]))
  summary_lines.append("")

  def top_list(state: str, n: int = 10) -> List[str]:
    x = df[(df["state"] == state) & (df["coverage_ok"])].copy()
    x = x.sort_values(["sort_score"], ascending=[False]).head(n)
    return x["ticker"].astype(str).tolist()

  for st in ["ENTRY", "WATCH"]:
    top = top_list(st, 10)
    if top:
      summary_lines.append(f"Top {st} (up to 10): {', '.join(top)}")

  (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

  log.info(f"Wrote reports to {out_dir} (files={len(sorted(df['strategy'].unique().tolist())) + 3})")


if __name__ == "__main__":
  main()
