import os
import json
import argparse
import yaml
import pandas as pd

from src.common.db import init_db, connect
from src.common.logging import setup_logger

from src.strategy.features import add_basic_features
from src.strategy.retest_shrink import Params, evaluate_ticker

from src.backtest.engine import simulate, ExecModel, ExitRules
from src.backtest.metrics import summarize
from src.backtest.regimes import add_spy_regime

log = setup_logger("backtest")

def read_prices() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM prices_daily",
            conn,
        )

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_params(cfg: dict) -> Params:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("STRATEGY_CONFIG", "/app/configs/retest_shrink.yaml"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--max_new_per_day", type=int, default=5)
    ap.add_argument("--max_positions", type=int, default=15)
    ap.add_argument("--initial_equity", type=float, default=100000.0)
    ap.add_argument("--risk_per_trade", type=float, default=0.01)
    args = ap.parse_args()

    init_db()
    cfg = load_cfg(args.config)
    p = build_params(cfg)

    df = read_prices()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(args.start)) & (df["date"] <= pd.to_datetime(args.end))].copy()
    if df.empty:
        raise RuntimeError("No price data in requested range. Run eod_fetch or expand history.")

    df = add_basic_features(
        df,
        atr_n=int(cfg["lookbacks"]["atr"]),
        pct_window=int(cfg["lookbacks"]["pct"]),
        adv_n=int(cfg["lookbacks"]["adv"]),
    )

    # Build signals by date: action==True on that date
    dates = sorted(df["date"].unique())
    grouped = {t: g.sort_values("date").reset_index(drop=True) for t, g in df.groupby("ticker")}

    signals = {}
    for d in dates:
        todays = []
        for t, g in grouped.items():
            sub = g[g["date"] <= d]
            if sub.empty:
                continue
            r = evaluate_ticker(sub, p)
            if r and r.get("action") and r.get("stop") is not None:
                todays.append(r)
        if todays:
            signals[pd.Timestamp(d)] = todays

    exec_cfg = cfg.get("execution", {})
    exec_model = ExecModel(entry=str(exec_cfg.get("entry", "next_open")), exit=str(exec_cfg.get("exit", "close")))

    exits_cfg = cfg.get("exits", {})
    exit_rules = ExitRules(
        min_hold_days=int(exits_cfg.get("min_hold_days", 0)),
        max_hold_days=int(exits_cfg.get("max_hold_days", 10)),
        no_progress_days=int(exits_cfg["no_progress_days"]) if exits_cfg.get("no_progress_days") is not None else None,
        no_progress_min_r=float(exits_cfg["no_progress_min_r"]) if exits_cfg.get("no_progress_min_r") is not None else None,
    )

    eq, trades = simulate(
        prices=df[["ticker", "date", "open", "high", "low", "close"]],
        signals=signals,
        exec_model=exec_model,
        exit_rules=exit_rules,
        max_new_per_day=args.max_new_per_day,
        max_positions=args.max_positions,
        initial_equity=args.initial_equity,
        risk_per_trade=args.risk_per_trade,
    )

    # Benchmark returns (SPY + IWM)
    bcfg = cfg.get("benchmarks", {})
    primary = bcfg.get("primary", "SPY")
    secondary = bcfg.get("secondary", "IWM")

    def bench_ret(sym: str) -> pd.Series:
        b = df[df["ticker"] == sym][["date", "close"]].drop_duplicates("date").sort_values("date")
        b = b.set_index("date")["close"].pct_change().fillna(0.0)
        return b

    eq = eq.set_index("date")
    eq[f"{primary}_ret"] = bench_ret(primary).reindex(eq.index).fillna(0.0)
    eq[f"{secondary}_ret"] = bench_ret(secondary).reindex(eq.index).fillna(0.0)
    eq = eq.reset_index()

    # Regime report (SPY)
    spy = df[df["ticker"] == primary][["date", "close"]].drop_duplicates("date").sort_values("date")
    regime = add_spy_regime(spy)
    eq = eq.merge(regime, on="date", how="left")

    metrics = summarize(eq, bench_cols=[f"{primary}_ret", f"{secondary}_ret"])

    out_dir = os.environ.get("OUTPUT_DIR", "/app/outputs")
    run_id = args.run_id or f"{cfg.get('name','strategy')}_{args.start}_{args.end}_{exec_model.entry}_{exec_model.exit}"
    run_path = os.path.join(out_dir, "backtests", run_id)
    os.makedirs(run_path, exist_ok=True)

    eq.to_csv(os.path.join(run_path, "equity_curve.csv"), index=False)
    trades.to_csv(os.path.join(run_path, "trades.csv"), index=False)
    with open(os.path.join(run_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_path, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    eq.groupby("spy_regime")["ret"].agg(["count", "mean", "std"]).reset_index().to_csv(
        os.path.join(run_path, "regime_report.csv"), index=False
    )

    log.info(f"Backtest complete: {run_path}")

if __name__ == "__main__":
    main()
