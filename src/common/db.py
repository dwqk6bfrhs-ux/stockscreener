import os
import sqlite3
from contextlib import contextmanager

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices_daily (
  ticker TEXT NOT NULL,
  date   TEXT NOT NULL,
  open   REAL,
  high   REAL,
  low    REAL,
  close  REAL,
  volume REAL,
  PRIMARY KEY (ticker, date)
);
"""


def get_db_path() -> str:
  return os.environ.get("DB_PATH", "/app/data/app.db")


@contextmanager
def connect():
  conn = sqlite3.connect(get_db_path())
  try:
    yield conn
  finally:
    conn.close()


def init_db():
  with connect() as conn:
    conn.executescript(DB_SCHEMA)
    conn.commit()
