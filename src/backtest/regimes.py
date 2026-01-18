import pandas as pd

def add_spy_regime(spy: pd.DataFrame) -> pd.DataFrame:
    s = spy.sort_values("date").copy()
    s["ma200"] = s["close"].rolling(200, min_periods=200).mean()
    s["ma200_slope"] = s["ma200"].diff(20)

    def _reg(row):
        if pd.isna(row["ma200"]) or pd.isna(row["ma200_slope"]):
            return "unknown"
        if row["close"] > row["ma200"] and row["ma200_slope"] > 0:
            return "bull"
        if row["close"] < row["ma200"] and row["ma200_slope"] < 0:
            return "bear"
        return "sideways"

    s["spy_regime"] = s.apply(_reg, axis=1)
    return s[["date", "spy_regime"]]
