import os
import sqlite3
from contextlib import contextmanager

RAW_SCHEMA = """
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

HOURLY_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices_hourly (
  ticker   TEXT NOT NULL,
  ts       TEXT NOT NULL,   -- ISO timestamp in UTC from Alpaca (e.g. 2026-01-16T15:00:00Z)
  date_et  TEXT NOT NULL,   -- derived ET trading date YYYY-MM-DD
  open     REAL,
  high     REAL,
  low      REAL,
  close    REAL,
  volume   REAL,
  PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_prices_hourly_date_ticker
  ON prices_hourly(date_et, ticker);

CREATE INDEX IF NOT EXISTS idx_prices_hourly_ticker_ts
  ON prices_hourly(ticker, ts);
"""


DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals_daily (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  strategy TEXT NOT NULL,
  state TEXT NOT NULL,
  score REAL,
  stop REAL,
  meta_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (date, ticker, strategy)
);

CREATE INDEX IF NOT EXISTS idx_signals_date_strategy
  ON signals_daily(date, strategy);

CREATE TABLE IF NOT EXISTS universe_daily (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  source TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (date, ticker, source)
);

CREATE INDEX IF NOT EXISTS idx_universe_date_source
  ON universe_daily(date, source);

CREATE INDEX IF NOT EXISTS idx_universe_date
  ON universe_daily(date);
"""

DB_SCHEMA = RAW_SCHEMA + "\n" + HOURLY_SCHEMA + "\n" + DERIVED_SCHEMA

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
