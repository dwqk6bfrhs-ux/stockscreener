import numpy as np
import pandas as pd

EPS = 1e-9

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(n, min_periods=n).mean()

def percentile_rank_rolling(s: pd.Series, window: int) -> pd.Series:
    # percentile of last value within rolling window
    def _pct(x):
        last = x.iloc[-1]
        return float((x <= last).mean())
    return s.rolling(window, min_periods=window).apply(_pct, raw=False)

def add_basic_features(df: pd.DataFrame, atr_n: int, pct_window: int, adv_n: int) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()

    out["tr"] = out.groupby("ticker", group_keys=False).apply(true_range).reset_index(level=0, drop=True)
    out["atr"] = out.groupby("ticker", group_keys=False).apply(lambda g: atr(g, atr_n)).reset_index(level=0, drop=True)

    out["range"] = (out["high"] - out["low"]).abs()
    out["vol_pct"] = out.groupby("ticker")["volume"].transform(lambda s: percentile_rank_rolling(s, pct_window))
    out["range_pct"] = out.groupby("ticker")["range"].transform(lambda s: percentile_rank_rolling(s, pct_window))

    out["adv20_dollars"] = out.groupby("ticker").apply(
        lambda g: (g["volume"] * g["close"]).rolling(adv_n, min_periods=adv_n).mean()
    ).reset_index(level=0, drop=True)

    out["prev_close"] = out.groupby("ticker")["close"].shift(1)
    out["down_move"] = (out["prev_close"] - out["close"]).clip(lower=0)
    out["down_atr"] = out["down_move"] / (out["atr"].replace(0, np.nan) + EPS)

    out["range_atr"] = out["range"] / (out["atr"].replace(0, np.nan) + EPS)
    return out

