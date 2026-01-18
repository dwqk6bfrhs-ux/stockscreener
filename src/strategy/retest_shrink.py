from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

EPS = 1e-9

@dataclass(frozen=True)
class Params:
    min_close: float
    min_adv20_dollars: float
    min_history_days: int
    exclude_tickers: set[str]

    vol_pct_min: float
    down_atr_min: float
    range_atr_min: float

    nft_window_days: int
    nft_undercut_atr_max: float
    nft_vol_max_mult: float

    retest_window_days: int
    retest_zone_atr: float
    retest_shrink_max: float
    retest_undercut_atr_max: float

    confirm_window_days: int
    confirm_close_strength: bool
    confirm_vol_max_mult: float

    stop_atr: float

def evaluate_ticker(g: pd.DataFrame, p: Params) -> dict | None:
    g = g.sort_values("date").reset_index(drop=True)
    if g.empty:
        return None

    ticker = str(g.iloc[-1]["ticker"])
    if ticker in p.exclude_tickers:
        return None

    if len(g) < p.min_history_days:
        return None

    last = g.iloc[-1]
    if pd.isna(last["atr"]) or pd.isna(last["adv20_dollars"]) or pd.isna(last["vol_pct"]) or pd.isna(last["down_atr"]):
        return None

    if float(last["close"]) < p.min_close:
        return None
    if float(last["adv20_dollars"]) < p.min_adv20_dollars:
        return None

    # Search for most recent D0 within a bounded lookback
    max_lookback = p.retest_window_days + p.confirm_window_days + p.nft_window_days + 10
    start = max(0, len(g) - max_lookback)
    gg = g.iloc[start:].copy().reset_index(drop=True)

    # D0 = strong sell pressure proxy
    d0_mask = (gg["vol_pct"] >= p.vol_pct_min) & (gg["down_atr"] >= p.down_atr_min)
    d0_candidates = gg[d0_mask]
    if d0_candidates.empty:
        return {"ticker": ticker, "date": str(last["date"].date()), "state": "NEUTRAL", "score": 0.0, "watch": False, "action": False}

    d0_i = int(d0_candidates.index.max())
    d0 = gg.loc[d0_i]

    l0 = float(d0["low"])
    v0 = float(d0["volume"])
    atr0 = float(d0["atr"])
    d0_date = d0["date"]

    days_since_d0 = (len(gg) - 1) - d0_i
    if days_since_d0 > (p.retest_window_days + p.confirm_window_days + 1):
        return {"ticker": ticker, "date": str(last["date"].date()), "state": "NEUTRAL", "score": 0.0, "watch": False, "action": False}

    # No-follow-through in D0+1..D0+nft_window
    nft_end = min(len(gg) - 1, d0_i + p.nft_window_days)
    nft_ok = False
    invalid = False

    for j in range(d0_i + 1, nft_end + 1):
        row = gg.loc[j]
        low_j = float(row["low"])
        vol_j = float(row["volume"])

        # invalidation: meaningful new low with high effort
        if low_j < l0 - 0.25 * atr0 and vol_j > 0.9 * v0:
            invalid = True
            break

        if (low_j >= l0 - p.nft_undercut_atr_max * atr0) and (vol_j <= p.nft_vol_max_mult * v0):
            nft_ok = True

    if invalid:
        return {
            "ticker": ticker, "date": str(last["date"].date()), "state": "INVALIDATED",
            "d0_date": str(d0_date.date()), "l0": l0, "v0": v0, "atr0": atr0,
            "score": 0.0, "watch": False, "action": False,
        }

    # If D0 exists but NFT not seen yet: watch
    if not nft_ok:
        score = float(gg.iloc[-1]["volume"]) * float(gg.iloc[-1]["range"])
        return {
            "ticker": ticker, "date": str(last["date"].date()), "state": "PRESSURE_TEST",
            "d0_date": str(d0_date.date()), "l0": l0, "v0": v0, "atr0": atr0,
            "score": score, "watch": True, "action": False,
            "stop": l0 - p.stop_atr * atr0,
        }

    # Retest search
    retest_end = min(len(gg) - 1, d0_i + p.retest_window_days)
    retest_i = None
    shrink_ratio = None

    for j in range(d0_i + 2, retest_end + 1):
        row = gg.loc[j]
        low_j = float(row["low"])
        vol_j = float(row["volume"])

        in_zone = low_j <= l0 + p.retest_zone_atr * atr0
        shrunk = vol_j <= p.retest_shrink_max * v0
        no_new_low = low_j >= l0 - p.retest_undercut_atr_max * atr0

        if in_zone and shrunk and no_new_low:
            retest_i = j
            shrink_ratio = vol_j / (v0 + EPS)
            break

    if retest_i is None:
        score = float(gg.iloc[-1]["volume"]) * float(gg.iloc[-1]["range"])
        return {
            "ticker": ticker, "date": str(last["date"].date()), "state": "NO_FOLLOW_THROUGH",
            "d0_date": str(d0_date.date()), "l0": l0, "v0": v0, "atr0": atr0,
            "score": score, "watch": True, "action": False,
            "stop": l0 - p.stop_atr * atr0,
        }

    # Confirm in next 1..confirm_window
    confirm_end = min(len(gg) - 1, retest_i + p.confirm_window_days)
    l0_floor = l0 - p.retest_undercut_atr_max * atr0
    min_low = float(gg.loc[retest_i:confirm_end, "low"].min())

    confirmed = False
    confirm_date = None

    if min_low >= l0_floor:
        for j in range(retest_i + 1, confirm_end + 1):
            prev = gg.loc[j - 1]
            cur = gg.loc[j]
            close_strength = float(cur["close"]) > float(prev["close"])
            vol_ok = float(cur["volume"]) <= p.confirm_vol_max_mult * v0
            if (p.confirm_close_strength and close_strength) or vol_ok:
                confirmed = True
                confirm_date = cur["date"]
                break

    # Scoring
    last_close = float(gg.iloc[-1]["close"])
    dist = abs(last_close - l0)
    s1 = 1.0 - float(shrink_ratio)
    s2 = max(0.0, 1.0 - dist / (1.0 * atr0 + EPS))
    s3 = float(gg.iloc[-1]["atr"]) / (last_close + EPS)
    score = 0.45*s1 + 0.35*s2 + 0.20*min(0.20, s3)

    state = "RETEST_CONFIRMED" if confirmed else "RETEST_CANDIDATE"

    return {
        "ticker": ticker,
        "date": str(last["date"].date()),
        "state": state,
        "d0_date": str(d0_date.date()),
        "l0": l0,
        "v0": v0,
        "atr0": atr0,
        "retest_date": str(gg.loc[retest_i, "date"].date()),
        "confirm_date": (str(confirm_date.date()) if confirm_date is not None else None),
        "shrink_ratio": float(shrink_ratio),
        "stop": l0 - p.stop_atr * atr0,
        "score": float(score),
        "watch": True,
        "action": bool(confirmed),
    }
