from __future__ import annotations

from datetime import date, datetime, timedelta
import os
from zoneinfo import ZoneInfo

try:
  from alpaca.trading.client import TradingClient
  from alpaca.trading.requests import GetCalendarRequest
except Exception:
  TradingClient = None
  GetCalendarRequest = None

_ET = ZoneInfo("America/New_York")



def today_et() -> str:
  tz = ZoneInfo(os.environ.get("TZ", "America/New_York"))
  return datetime.now(tz).strftime("%Y-%m-%d")


def _alpaca_trading_days(end_date: date, lookback_days: int = 30) -> list[date] | None:
  api_key = os.environ.get("ALPACA_API_KEY")
  secret_key = os.environ.get("ALPACA_SECRET_KEY")
  if not api_key or not secret_key or TradingClient is None or GetCalendarRequest is None:
    return None

  start_date = end_date - timedelta(days=lookback_days)
  client = TradingClient(api_key, secret_key, paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true")
  try:
    calendars = client.get_calendar(
      GetCalendarRequest(start=start_date.isoformat(), end=end_date.isoformat())
    )
  except Exception:
    return None

  return [c.date for c in calendars]


def last_completed_trading_day_et(cutoff_hour: int = 18) -> str:
  """
  Returns YYYY-MM-DD of the last completed US trading day in ET.
  - Uses Alpaca market calendar when available (holiday-aware).
  - Falls back to weekend-only logic if calendar is unavailable.
  - Weekday before cutoff_hour -> previous trading day
  cutoff_hour=18 gives buffers for EOD bar finalization.
  """
  now = datetime.now(_ET)
  target = now.date()

  # if before cutoff, today's session isn't considered "completed"
  if now.hour < cutoff_hour:
    target = target - timedelta(days=1)

  trading_days = _alpaca_trading_days(target)
  if trading_days:
    eligible = [d for d in trading_days if d <= target]
    if eligible:
      return max(eligible).isoformat()

  # fallback: roll back weekends only
  d = target
  while d.weekday() >= 5:  # 5=Sat, 6=Sun
    d = d - timedelta(days=1)

  return d.isoformat()
