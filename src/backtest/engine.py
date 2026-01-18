from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

EPS = 1e-9

@dataclass
class ExecModel:
    entry: str  # "next_open" or "close"
    exit: str   # "next_open" or "close"

@dataclass
class ExitRules:
    min_hold_days: int
    max_hold_days: int
    no_progress_days: int | None
    no_progress_min_r: float | None

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    stop_px: float
    size: float
    hold_days: int = 0

def simulate(
    prices: pd.DataFrame,
    signals: dict[pd.Timestamp, list[dict]],
    exec_model: ExecModel,
    exit_rules: ExitRules,
    max_new_per_day: int,
    max_positions: int,
    initial_equity: float = 100000.0,
    risk_per_trade: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = prices.sort_values(["date", "ticker"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df_idx = df.set_index(["date", "ticker"]).sort_index()
    calendar = sorted(df["date"].unique())

    equity_cash = initial_equity
    positions: dict[str, Position] = {}
    trades = []
    equity_rows = []

    def get_row(d: pd.Timestamp, t: str):
        try:
            return df_idx.loc[(d, t)]
        except KeyError:
            return None

    def next_trading_day(d: pd.Timestamp):
        i = calendar.index(d)
        if i + 1 >= len(calendar):
            return None
        return calendar[i + 1]

    for d in calendar:
        # mark-to-market at close
        mtm = equity_cash
        for pos in positions.values():
            row = get_row(d, pos.ticker)
            if row is None:
                continue
            mtm += pos.size * (float(row["close"]) - pos.entry_px)
        equity_rows.append({"date": d, "equity": mtm})

        # exits
        exit_list = []
        for t, pos in list(positions.items()):
            row = get_row(d, t)
            if row is None:
                continue

            low = float(row["low"])
            open_px = float(row["open"])
            close_px = float(row["close"])

            # stop logic (conservative)
            if low <= pos.stop_px:
                fill = open_px if open_px < pos.stop_px else pos.stop_px
                exit_list.append((t, fill, "stop"))
                continue

            pos.hold_days += 1

            # time stop
            if pos.hold_days >= exit_rules.max_hold_days:
                fill = close_px if exec_model.exit == "close" else open_px
                exit_list.append((t, fill, "time"))
                continue

            # no-progress exit (optional)
            if exit_rules.no_progress_days and exit_rules.no_progress_min_r is not None:
                if pos.hold_days >= exit_rules.no_progress_days:
                    risk = max(pos.entry_px - pos.stop_px, EPS)
                    r_mult = (close_px - pos.entry_px) / risk
                    if r_mult < exit_rules.no_progress_min_r and pos.hold_days >= exit_rules.min_hold_days:
                        fill = close_px if exec_model.exit == "close" else open_px
                        exit_list.append((t, fill, "no_progress"))
                        continue

        for t, fill, reason in exit_list:
            pos = positions.pop(t, None)
            if not pos:
                continue
            pnl = pos.size * (fill - pos.entry_px)
            equity_cash += pnl
            trades.append({
                "ticker": pos.ticker,
                "entry_date": pos.entry_date,
                "entry_px": pos.entry_px,
                "exit_date": d,
                "exit_px": fill,
                "size": pos.size,
                "pnl": pnl,
                "reason": reason,
                "hold_days": pos.hold_days,
                "stop_px": pos.stop_px,
            })

        # entries
        todays = signals.get(pd.Timestamp(d), [])
        if not todays:
            continue

        # rank by score
        cand = sorted(todays, key=lambda x: x.get("score", 0.0), reverse=True)
        slots = max_positions - len(positions)
        take = min(max_new_per_day, slots, len(cand))
        for s in cand[:take]:
            t = s["ticker"]
            if t in positions:
                continue

            if exec_model.entry == "close":
                row = get_row(d, t)
                if row is None:
                    continue
                entry_px = float(row["close"])
                entry_date = d
            else:
                d2 = next_trading_day(d)
                if d2 is None:
                    continue
                row2 = get_row(d2, t)
                if row2 is None:
                    continue
                entry_px = float(row2["open"])
                entry_date = d2

            stop_px = float(s["stop"])
            risk = max(entry_px - stop_px, EPS)
            dollars_risk = equity_cash * risk_per_trade
            size = dollars_risk / risk

            positions[t] = Position(
                ticker=t,
                entry_date=pd.Timestamp(entry_date),
                entry_px=entry_px,
                stop_px=stop_px,
                size=size,
                hold_days=0,
            )

    eq = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trades)
    return eq, trades_df
