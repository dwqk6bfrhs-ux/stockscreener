from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.common.db import connect, get_prices_daily_source, init_db
from src.common.datefmt import normalize_date_str
from src.common.logging import setup_logger

log = setup_logger("export_ml_features")


def _sql_literal(value: str) -> str:
  return "'" + value.replace("'", "''") + "'"


def _build_view_sql(label_threshold: float, source: str) -> str:
  source_literal = _sql_literal(source)
  return f"""
DROP VIEW IF EXISTS v_ml_features_daily;
CREATE VIEW v_ml_features_daily AS
WITH base_prices AS (
  SELECT
    ticker,
    date,
    open,
    high,
    low,
    close,
    volume,
    LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
    LAG(close, 3) OVER (PARTITION BY ticker ORDER BY date) AS close_lag_3,
    LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) AS close_lag_5,
    LAG(close, 10) OVER (PARTITION BY ticker ORDER BY date) AS close_lag_10,
    LEAD(high) OVER (PARTITION BY ticker ORDER BY date) AS next_high,
    LEAD(close) OVER (PARTITION BY ticker ORDER BY date) AS next_close,
    close * volume AS dv
  FROM prices_daily
  WHERE source = {source_literal}
),
calc AS (
  SELECT
    *,
    CASE
      WHEN prev_close IS NOT NULL AND prev_close != 0
      THEN close / prev_close - 1
    END AS ret_1d,
    CASE
      WHEN close_lag_3 IS NOT NULL AND close_lag_3 != 0
      THEN close / close_lag_3 - 1
    END AS ret_3d,
    CASE
      WHEN close_lag_5 IS NOT NULL AND close_lag_5 != 0
      THEN close / close_lag_5 - 1
    END AS ret_5d,
    CASE
      WHEN close_lag_10 IS NOT NULL AND close_lag_10 != 0
      THEN close / close_lag_10 - 1
    END AS ret_10d,
    CASE
      WHEN prev_close IS NOT NULL AND prev_close != 0
      THEN open / prev_close - 1
    END AS gap_pct,
    CASE
      WHEN high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL AND close != 0
      THEN (high - low) / close
    END AS range_pct,
    CASE
      WHEN high IS NOT NULL AND low IS NOT NULL AND high != low
      THEN (close - low) / (high - low)
    END AS close_pos,
    CASE
      WHEN prev_close IS NULL THEN (high - low)
      ELSE MAX(
        high - low,
        ABS(high - prev_close),
        ABS(low - prev_close)
      )
    END AS tr
  FROM base_prices
),
windowed AS (
  SELECT
    *,
    AVG(tr) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr_14,
    AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_ma20,
    AVG(dv) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS adv20_dollars,
    AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS vol_ma5,
    AVG(volume * volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS vol_m2_5
  FROM calc
),
signals_pivot AS (
  SELECT
    date,
    ticker,
    MAX(CASE WHEN strategy = 'ma_cross_5_10' AND state = 'ENTRY' THEN 1 ELSE 0 END) AS buy_ma_cross_5_10,
    MAX(CASE WHEN strategy = 'ma_cross_5_10' THEN score END) AS score_ma_cross_5_10,
    MAX(CASE WHEN strategy = 'retest_shrink_v1' AND state = 'ENTRY' THEN 1 ELSE 0 END) AS buy_retest_shrink_v1,
    MAX(CASE WHEN strategy = 'retest_shrink_v1' THEN score END) AS score_retest_shrink_v1,
    SUM(CASE WHEN state = 'ENTRY' THEN 1 ELSE 0 END) AS n_active_buy_signals
  FROM signals_daily
  GROUP BY date, ticker
),
joined AS (
  SELECT
    w.*, 
    s.buy_ma_cross_5_10,
    s.score_ma_cross_5_10,
    s.buy_retest_shrink_v1,
    s.score_retest_shrink_v1,
    s.n_active_buy_signals,
    CASE
      WHEN w.atr_14 IS NOT NULL AND w.close IS NOT NULL AND w.close != 0
      THEN w.atr_14 / w.close
    END AS atr_14_pct,
    CASE
      WHEN w.vol_ma20 IS NOT NULL AND w.vol_ma20 != 0
      THEN w.volume / w.vol_ma20
    END AS vol_ratio_20,
    CASE
      WHEN (w.vol_m2_5 - w.vol_ma5 * w.vol_ma5) > 0
      THEN (w.volume - w.vol_ma5) / sqrt(w.vol_m2_5 - w.vol_ma5 * w.vol_ma5)
    END AS vol_z5,
    CASE
      WHEN w.next_high IS NULL OR w.close IS NULL OR w.close = 0 THEN NULL
      WHEN (w.next_high / w.close - 1) >= {label_threshold}
      THEN 1
      ELSE 0
    END AS label_high_up_x,
    CASE
      WHEN w.next_close IS NULL OR w.close IS NULL OR w.close = 0 THEN NULL
      WHEN (w.next_close / w.close - 1) >= {label_threshold}
      THEN 1
      ELSE 0
    END AS label_close_up_x
  FROM windowed w
  INNER JOIN universe_daily u
    ON u.date = w.date
    AND u.ticker = w.ticker
    AND u.source = {source_literal}
  LEFT JOIN signals_pivot s
    ON s.date = w.date
    AND s.ticker = w.ticker
)
SELECT
  date,
  ticker,
  open,
  high,
  low,
  close,
  volume,
  ret_1d,
  ret_3d,
  ret_5d,
  ret_10d,
  gap_pct,
  range_pct,
  close_pos,
  tr,
  atr_14,
  atr_14_pct,
  dv,
  vol_ma20,
  adv20_dollars,
  vol_ratio_20,
  vol_z5,
  buy_ma_cross_5_10,
  score_ma_cross_5_10,
  buy_retest_shrink_v1,
  score_retest_shrink_v1,
  n_active_buy_signals,
  CASE
    WHEN ret_5d IS NOT NULL
    THEN PERCENT_RANK() OVER (PARTITION BY date ORDER BY ret_5d)
  END AS ret_5d_pct_rank,
  CASE
    WHEN adv20_dollars IS NOT NULL
    THEN PERCENT_RANK() OVER (PARTITION BY date ORDER BY adv20_dollars)
  END AS adv20_dollars_pct_rank,
  CASE
    WHEN n_active_buy_signals IS NOT NULL
    THEN PERCENT_RANK() OVER (PARTITION BY date ORDER BY n_active_buy_signals)
  END AS n_active_buy_signals_pct_rank,
  label_high_up_x,
  label_close_up_x
FROM joined;
"""


def _recreate_view(label_threshold: float, source: str) -> None:
  sql = _build_view_sql(label_threshold, source)
  with connect() as conn:
    conn.executescript(sql)
    conn.commit()


def _resolve_date_bounds(date: Optional[str], start: Optional[str], end: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
  if date:
    norm = normalize_date_str(date)
    return norm, norm
  start_norm = normalize_date_str(start) if start else None
  end_norm = normalize_date_str(end) if end else None
  return start_norm, end_norm


def _default_output_base(start: Optional[str], end: Optional[str]) -> Path:
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / "ml_features"
  out_dir.mkdir(parents=True, exist_ok=True)
  if start and end:
    if start == end:
      return out_dir / f"ml_features_{start}"
    return out_dir / f"ml_features_{start}_{end}"
  if start:
    return out_dir / f"ml_features_from_{start}"
  if end:
    return out_dir / f"ml_features_to_{end}"
  return out_dir / "ml_features_all"


def _resolve_output_paths(output: Optional[str], fmt: str, start: Optional[str], end: Optional[str]) -> Tuple[Path, Optional[Path]]:
  base = Path(output) if output else _default_output_base(start, end)
  suffix = base.suffix.lower()
  if fmt == "csv":
    return (base if suffix == ".csv" else base.with_suffix(".csv")), None
  if fmt == "parquet":
    return (base if suffix == ".parquet" else base.with_suffix(".parquet")), None
  csv_path = base if suffix == ".csv" else base.with_suffix(".csv")
  parquet_path = base if suffix == ".parquet" else base.with_suffix(".parquet")
  return csv_path, parquet_path


def _load_features(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
  where = []
  params = []
  if start:
    where.append("date >= ?")
    params.append(start)
  if end:
    where.append("date <= ?")
    params.append(end)
  where_sql = " WHERE " + " AND ".join(where) if where else ""
  query = f"SELECT * FROM v_ml_features_daily{where_sql} ORDER BY date, ticker"
  with connect() as conn:
    return pd.read_sql_query(query, conn, params=params)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
  try:
    df.to_parquet(path, index=False)
  except Exception as exc:
    raise RuntimeError(
      "Failed to write parquet. Install pyarrow or fastparquet to enable parquet export."
    ) from exc


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--date", default=None, help="Trade date YYYY-MM-DD (single-day export)")
  ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
  ap.add_argument("--end", default=None, help="End date YYYY-MM-DD")
  ap.add_argument("--label-threshold", type=float, default=0.03, help="Label threshold (default 0.03)")
  ap.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
  ap.add_argument("--output", default=None, help="Output path (with or without extension)")
  ap.add_argument("--skip-view", action="store_true", help="Skip recreating the view")
  args = ap.parse_args()

  init_db()
  source = get_prices_daily_source()
  if not args.skip_view:
    _recreate_view(args.label_threshold, source)

  start, end = _resolve_date_bounds(args.date, args.start, args.end)
  df = _load_features(start, end)

  if df.empty:
    log.warning("No rows returned from v_ml_features_daily for requested range.")

  csv_path, parquet_path = _resolve_output_paths(args.output, args.format, start, end)
  if args.format in {"csv", "both"}:
    df.to_csv(csv_path, index=False)
    log.info(f"Wrote CSV: {csv_path} rows={len(df)}")
  if args.format in {"parquet", "both"} and parquet_path is not None:
    _write_parquet(df, parquet_path)
    log.info(f"Wrote parquet: {parquet_path} rows={len(df)}")


if __name__ == "__main__":
  main()
