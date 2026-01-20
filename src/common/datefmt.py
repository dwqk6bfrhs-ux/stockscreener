from __future__ import annotations

from datetime import datetime

def normalize_date_str(s: str) -> str:
  """
  Accepts:
    - YYYY-MM-DD (ISO) -> returns same
    - MM-DD-YYYY -> returns YYYY-MM-DD
  Raises ValueError if unparseable.
  """
  s = (s or "").strip()
  if not s:
    raise ValueError("empty date")

  # ISO
  try:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")
  except Exception:
    pass

  # US style
  dt = datetime.strptime(s, "%m-%d-%Y")
  return dt.strftime("%Y-%m-%d")
