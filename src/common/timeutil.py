from datetime import datetime
from zoneinfo import ZoneInfo
import os
from datetime import datetime, timedelta

_ET = ZoneInfo("America/New_York")



def today_et() -> str:
  tz = ZoneInfo(os.environ.get("TZ", "America/New_York"))
  return datetime.now(tz).strftime("%Y-%m-%d")


def last_completed_trading_day_et(cutoff_hour: int = 18) -> str:
  """
  Returns YYYY-MM-DD of the last completed US trading day in ET.
  - Weekend -> Friday
  - Weekday before cutoff_hour -> previous trading day
  cutoff_hour=18 gives buffers for EOD bar finalization.
  """
  now = datetime.now(_ET)
  d = now.date()

  # if before cutoff, today's session isn't considered "completed"
  if now.hour < cutoff_hour:
    d = d - timedelta(days=1)

  # roll back weekends
  while d.weekday() >= 5:  # 5=Sat, 6=Sun
    d = d - timedelta(days=1)

  return d.isoformat()
