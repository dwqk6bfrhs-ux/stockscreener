import argparse
import itertools
import json
import os
import pandas as pd
import yaml
import concurrent.futures
from typing import Any, Dict, List, Tuple
from pathlib import Path

from src.common.db import init_db, connect
from src.common.logging import setup_logger

# Import your existing logic
from src.strategy.features import add_basic_features
from src.strategy.retest_shrink import Params, evaluate_ticker
from src.backtest.engine import simulate, ExecModel, ExitRules
from src.backtest.metrics import summarize

log = setup_logger("tune")

# ---------------------------------------------------------------------------
# 1. Tuning Configuration
# ---------------------------------------------------------------------------
# Define the parameter grid to sweep.
# The keys must correspond to the structure in retest_shrink.yaml (flattened logic used below)
GRID = {
    # Strategy Logic
    "retest_window_days": [10, 15, 20],
    "retest_shrink_max": [0.35, 0.45],
    "confirm_window_days": [3, 5],
    
    # Risk Management
    "stop_atr": [0.5, 1.0, 1.5],
    
    # Execution (Optional)
    "max_hold_days": [10, 15],
}

# ---------------------------------------------------------------------------
# 2. Helpers (Mirrors backtest.py logic)
# ---------------------------------------------------------------------------
def load_base_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_params_to_cfg(base_cfg: dict, params: dict) -> dict:
    """
    Patches the base config with values from the tuning grid.
    """
    cfg = base_cfg.copy() # Shallow copy is likely enough for simple replacements
    
    # Map flat params to nested YAML structure
    # This mapping must align with how build_params reads them
    if "retest_window_days" in params:
        cfg.setdefault("retest", {})["window_days"] = params["retest_window_days"]
    if "retest_shrink_max" in params:
        cfg.setdefault("retest", {})["shrink_max"] = params["retest_shrink_max"]
    if "confirm_window_days" in params:
        cfg.setdefault("confirm", {})["window_days"] = params["confirm_window_days"]
    if "stop_atr" in params:
        cfg.setdefault("risk", {})["stop_atr"] = params["stop_atr"]
    
    # Exit rules are in a different dict in backtest.py, but likely in 'exits' in yaml
    if "max_hold_days" in params:
        cfg.setdefault("exits", {})["max_hold_days"] = params["max_hold_days"]
        
    return cfg

def build_strategy_params(cfg: dict) -> Params:
    # Copied/Adapted from src/jobs/backtest.py to ensure consistency
    return Params(
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

# ---------------------------------------------------------------------------
# 3. Core Backtest Routine (Worker)
# ---------------------------------------------------------------------------
def run_strategy_instance(
    df_features: pd.DataFrame, 
    cfg: dict, 
    tuning_params: dict,
    common_args: dict
) -> dict:
    """
    Runs one pass of Signal Generation -> Simulation -> Metrics
    """
    try:
        # 1. Setup
        p = build_strategy_params(cfg)
        
        # 2. Generate Signals
        # (This logic is copied from backtest.py but runs on the pre-featured DF)
        dates = sorted(df_features["date"].unique())
        # Group by ticker once for speed
        grouped = {t: g.sort_values("date").reset_index(drop=True) for t, g in df_features.groupby("ticker")}
        
        signals = {}
        # NOTE: This loop is the bottleneck. In a production tuner, 
        # we would optimize evaluate_ticker to be vectorized.
        # For now, we accept the overhead.
        for d in dates:
            todays = []
            for t, g in grouped.items():
                # Passing full history up to date d
                # Optimization: evaluate_ticker usually only needs the last N rows.
                # If we trust the strategy to handle the full DF, we pass:
                sub = g[g["date"] <= d]
                if sub.empty: continue
                
                # Only run if we have enough history to matter
                if len(sub) < 50: continue 

                r = evaluate_ticker(sub, p)
                if r and r.get("action") and r.get("stop") is not None:
                    todays.append(r)
            
            if todays:
                signals[pd.Timestamp(d)] = todays
        
        # 3. Simulate Trades
        exec_cfg = cfg.get("execution", {})
        exec_model = ExecModel(
            entry=str(exec_cfg.get("entry", "next_open")), 
            exit=str(exec_cfg.get("exit", "close"))
        )
        
        exits_cfg = cfg.get("exits", {})
        exit_rules = ExitRules(
            min_hold_days=int(exits_cfg.get("min_hold_days", 0)),
            max_hold_days=int(exits_cfg.get("max_hold_days", 10)),
            no_progress_days=int(exits_cfg["no_progress_days"]) if exits_cfg.get("no_progress_days") else None,
            no_progress_min_r=float(exits_cfg["no_progress_min_r"]) if exits_cfg.get("no_progress_min_r") else None,
        )
        
        eq, trades = simulate(
            prices=df_features[["ticker", "date", "open", "high", "low", "close"]],
            signals=signals,
            exec_model=exec_model,
            exit_rules=exit_rules,
            max_new_per_day=common_args["max_new_per_day"],
            max_positions=common_args["max_positions"],
            initial_equity=100000.0,
            risk_per_trade=0.01,
        )
        
        # 4. Calculate Metrics
        if eq.empty:
            return {**tuning_params, "sharpe": 0.0, "cagr": 0.0, "trades": 0}
            
        metrics = summarize(eq, bench_cols=[]) # Skip benchmarks for speed in tuning
        
        return {
            **tuning_params,
            "sharpe": metrics["sharpe"],
            "cagr": metrics["cagr"],
            "max_dd": metrics["max_dd"],
            "win_rate": metrics["win_rate"],
            "trades": metrics["total_trades"],
            "expectancy": metrics["expectancy"]
        }
        
    except Exception as e:
        log.error(f"Run failed for {tuning_params}: {e}")
        return {**tuning_params, "sharpe": -99.0, "error": str(e)}

# ---------------------------------------------------------------------------
# 4. Main Controller
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/configs/retest_shrink.yaml")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=1, help="Parallel jobs (careful with RAM!)")
    args = ap.parse_args()
    
    init_db()
    base_cfg = load_base_cfg(args.config)
    
    # 1. Load Data ONCE
    log.info("Loading price data...")
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM prices_daily", 
            conn
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(args.start)) & (df["date"] <= pd.to_datetime(args.end))].copy()
    
    # 2. Pre-calculate Basic Features ONCE
    # (Assuming basic features like ATR/Volume don't change with the grid params being tuned.
    #  If you tune ATR window, you must move this inside the loop or pre-calc variations.)
    log.info("Adding basic features...")
    df = add_basic_features(
        df,
        atr_n=int(base_cfg["lookbacks"]["atr"]),
        pct_window=int(base_cfg["lookbacks"]["pct"]),
        adv_n=int(base_cfg["lookbacks"]["adv"]),
    )
    
    # 3. Build Grid
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    log.info(f"Generated {len(combinations)} parameter sets to test.")
    
    # Common simulation args
    sim_args = {
        "max_new_per_day": 5,
        "max_positions": 15
    }
    
    results = []
    
    # 4. Run Loop
    # Using ProcessPoolExecutor to bypass GIL, but passing large DF is expensive.
    # If RAM is tight, set workers=1.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for params in combinations:
            # Create specific config for this run
            run_cfg = apply_params_to_cfg(base_cfg, params)
            
            futures.append(
                executor.submit(run_strategy_instance, df, run_cfg, params, sim_args)
            )
            
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            if (i+1) % 5 == 0:
                log.info(f"Finished {i+1}/{len(combinations)} runs.")
                
    # 5. Report
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("sharpe", ascending=False)
    
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"tune_results_{args.start}_{args.end}.csv"
    res_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print("TOP 10 PARAMETER SETS")
    print("="*40)
    print(res_df.head(10).to_string(index=False))
    print(f"\nFull results saved to: {csv_path}")

if __name__ == "__main__":
    main()
