import math
from typing import Optional

import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

def sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=0)) * math.sqrt(252))

def alpha_beta(strategy_rets: pd.Series, bench_rets: pd.Series) -> dict:
    s = strategy_rets.dropna()
    b = bench_rets.reindex(s.index).dropna()
    s = s.reindex(b.index).dropna()
    if len(s) < 10 or float(np.var(b.values, ddof=0)) == 0.0:
        return {"alpha_ann": 0.0, "beta": 0.0}
    cov = float(np.cov(s.values, b.values, ddof=0)[0, 1])
    var = float(np.var(b.values, ddof=0))
    beta = cov / var
    alpha_daily = float(s.mean() - beta * b.mean())
    return {"alpha_ann": alpha_daily * 252.0, "beta": float(beta)}

def info_ratio(active_rets: pd.Series) -> float:
    a = active_rets.dropna()
    if len(a) < 2 or a.std(ddof=0) == 0:
        return 0.0
    return float((a.mean() / a.std(ddof=0)) * math.sqrt(252))

def _cagr(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    start = float(equity_curve["equity"].iloc[0])
    end = float(equity_curve["equity"].iloc[-1])
    if start <= 0:
        return 0.0
    dates = pd.to_datetime(equity_curve["date"])
    days = max((dates.max() - dates.min()).days, 1)
    years = days / 365.25
    return float((end / start) ** (1 / years) - 1.0)


def _trade_stats(trades: Optional[pd.DataFrame]) -> dict:
    if trades is None or trades.empty:
        return {"total_trades": 0, "win_rate": 0.0, "expectancy": 0.0}
    trades = trades.copy()
    trades["risk"] = (trades["entry_px"] - trades["stop_px"]).abs() * trades["size"]
    trades = trades[trades["risk"] > 0]
    if trades.empty:
        return {"total_trades": 0, "win_rate": 0.0, "expectancy": 0.0}
    trades["r_mult"] = trades["pnl"] / trades["risk"]
    win_rate = float((trades["pnl"] > 0).mean())
    expectancy = float(trades["r_mult"].mean())
    return {"total_trades": int(len(trades)), "win_rate": win_rate, "expectancy": expectancy}


def summarize(equity_curve: pd.DataFrame, bench_cols: list[str], trades: Optional[pd.DataFrame] = None) -> dict:
    ret = equity_curve["ret"].dropna()
    max_dd = max_drawdown(equity_curve["equity"]) if len(equity_curve) else 0.0
    out = {
        "total_return": float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0)
        if len(equity_curve)
        else 0.0,
        "cagr": _cagr(equity_curve),
        "sharpe": sharpe(ret),
        "max_drawdown": max_dd,
        "max_dd": max_dd,
        "avg_daily_ret": float(ret.mean()) if len(ret) else 0.0,
        "vol_daily": float(ret.std(ddof=0)) if len(ret) else 0.0,
    }
    out.update(_trade_stats(trades))
    for bcol in bench_cols:
        if bcol not in equity_curve.columns:
            continue
        bret = equity_curve[bcol].dropna()
        ab = alpha_beta(ret, bret)
        active = (ret - bret.reindex(ret.index)).dropna()
        out[f"{bcol}_alpha_ann"] = ab["alpha_ann"]
        out[f"{bcol}_beta"] = ab["beta"]
        out[f"{bcol}_info_ratio"] = info_ratio(active)
    return out
