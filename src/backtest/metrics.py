import math
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

def summarize(equity_curve: pd.DataFrame, bench_cols: list[str]) -> dict:
    ret = equity_curve["ret"].dropna()
    out = {
        "total_return": float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0) if len(equity_curve) else 0.0,
        "sharpe": sharpe(ret),
        "max_drawdown": max_drawdown(equity_curve["equity"]) if len(equity_curve) else 0.0,
        "avg_daily_ret": float(ret.mean()) if len(ret) else 0.0,
        "vol_daily": float(ret.std(ddof=0)) if len(ret) else 0.0,
    }
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
