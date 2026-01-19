import os
import json
import pandas as pd
import yaml

from src.common.db import init_db, connect
from src.common.logging import setup_logger
from src.common.timeutil import today_et

from src.strategy.features import add_basic_features
from src.strategy.retest_shrink import Params, evaluate_ticker

log = setup_logger("report")

def read_prices() -> pd.DataFrame:
  with connect() as conn:
    return pd.read_sql_query(
      "SELECT ticker, date, open, high, low, close, volume FROM prices_daily",
      conn
    )

def load_config() -> dict:
  path = os.environ.get("STRATEGY_CONFIG", "/app/configs/retest_shrink.yaml")
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)

def main():
  init_db()
  out_dir = os.environ.get("OUTPUT_DIR", "/app/outputs")
  date = today_et()
  day_dir = os.path.join(out_dir, date)
  os.makedirs(day_dir, exist_ok=True)

  cfg = load_config()

  df = read_prices()
  if df.empty:
    raise RuntimeError("No prices in DB. Run eod_fetch first.")

  df["date"] = pd.to_datetime(df["date"])
  df = add_basic_features(
    df,
    atr_n=int(cfg["lookbacks"]["atr"]),
    pct_window=int(cfg["lookbacks"]["pct"]),
    adv_n=int(cfg["lookbacks"]["adv"]),
  )

  p = Params(
    min_close=float(cfg["universe"]["min_close"]),
    min_adv20_dollars=float(cfg["universe"]["min_adv20_dollars"]),
    min_history_days=int(cfg["universe"]["min_history_days"]),
    exclude_tickers=set(cfg["universe"].get("exclude_tickers", [])),

    vol_pct_min=float(cfg["pressure_test"]["vol_pct_min"]),
    down_atr_min=float(cfg["pressure_test"]["down_atr_min"]),
    range_atr_min=float(cfg["pressure_test"]["range_atr_min"]),

    nft_window_days=int(cfg["no_follow_through"]["window_days"]),
    nft_undercut_atr_max=float(cfg["no_follow_through"]["undercut_atr_max"]),
    nft_vol_max_mult=float(cfg["no_follow_through"]["vol_max_mult"]),

    retest_window_days=int(cfg["retest"]["window_days"]),
    retest_zone_atr=float(cfg["retest"]["zone_atr"]),
    retest_shrink_max=float(cfg["retest"]["shrink_max"]),
    retest_undercut_atr_max=float(cfg["retest"]["undercut_atr_max"]),

    confirm_window_days=int(cfg["confirm"]["window_days"]),
    confirm_close_strength=bool(cfg["confirm"]["close_strength"]),
    confirm_vol_max_mult=float(cfg["confirm"]["vol_max_mult"]),

    stop_atr=float(cfg["risk"]["stop_atr"]),
  )

  results = []
  for ticker, g in df.groupby("ticker"):
    r = evaluate_ticker(g, p)
    if r is not None:
      results.append(r)

  res = pd.DataFrame(results)
  if res.empty:
    raise RuntimeError("No eligible tickers after filters. Check universe thresholds.")

  watch = res[res["watch"] == True].sort_values("score", ascending=False).head(int(cfg["output"]["watch_size"]))
  action = res[res["action"] == True].sort_values("score", ascending=False).head(int(cfg["output"]["action_size"]))
  cand = res.sort_values("score", ascending=False).head(int(cfg["output"]["candidates_size"]))

  watch_cols = [c for c in ["ticker","state","score","d0_date","retest_date","confirm_date","shrink_ratio","l0","stop"] if c in watch.columns]
  action_cols = [c for c in ["ticker","state","score","d0_date","retest_date","confirm_date","shrink_ratio","l0","stop"] if c in action.columns]

  watch.to_csv(os.path.join(day_dir, "watch_list.csv"), index=False, columns=watch_cols)
  action.to_csv(os.path.join(day_dir, "action_list.csv"), index=False, columns=action_cols)

  with open(os.path.join(day_dir, "candidates_top20.json"), "w", encoding="utf-8") as f:
    json.dump(cand.to_dict(orient="records"), f, indent=2)

  # Run summary for debugging
  summary = {
    "date": date,
    "strategy": cfg.get("name"),
    "counts_by_state": res["state"].value_counts().to_dict(),
    "action_count": int(len(action)),
    "watch_count": int(len(watch)),
  }
  with open(os.path.join(day_dir, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  log.info(f"Generated reports in {day_dir} (action={len(action)}, watch={len(watch)})")

if __name__ == "__main__":
  main()


