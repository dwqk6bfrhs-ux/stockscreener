import logging
import os
from datetime import datetime


def setup_logger(name: str) -> logging.Logger:
  log_dir = os.environ.get("LOG_DIR", "/app/logs")
  os.makedirs(log_dir, exist_ok=True)

  logger = logging.getLogger(name)
  if logger.handlers:
    return logger

  logger.setLevel(logging.INFO)
  fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

  sh = logging.StreamHandler()
  sh.setFormatter(fmt)
  logger.addHandler(sh)

  fname = datetime.now().strftime(f"{name}_%Y-%m-%d.log")
  fh = logging.FileHandler(os.path.join(log_dir, fname))
  fh.setFormatter(fmt)
  logger.addHandler(fh)

  return logger
