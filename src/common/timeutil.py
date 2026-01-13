from datetime import datetime
from zoneinfo import ZoneInfo
import os


def today_et() -> str:
  tz = ZoneInfo(os.environ.get("TZ", "America/New_York"))
  return datetime.now(tz).strftime("%Y-%m-%d")
