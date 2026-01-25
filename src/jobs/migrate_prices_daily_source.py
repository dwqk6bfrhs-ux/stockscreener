from __future__ import annotations

import argparse
import sqlite3

from src.common.db import connect, init_db
from src.common.logging import setup_logger

log = setup_logger("migrate_prices_daily_source")


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
  rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
  return any(r[1] == column for r in rows)


def migrate(default_source: str) -> None:
  with connect() as conn:
    if not _column_exists(conn, "prices_daily", "source"):
      log.info("Migrating prices_daily to include source column.")
      conn.executescript("""
      CREATE TABLE IF NOT EXISTS prices_daily_new (
        ticker TEXT NOT NULL,
        date   TEXT NOT NULL,
        source TEXT NOT NULL,
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL,
        volume REAL,
        PRIMARY KEY (ticker, date, source)
      );
      """)
      conn.execute(
        """
        INSERT INTO prices_daily_new (ticker, date, source, open, high, low, close, volume)
        SELECT ticker, date, ?, open, high, low, close, volume
        FROM prices_daily
        """,
        (default_source,),
      )
      conn.executescript("""
      DROP TABLE prices_daily;
      ALTER TABLE prices_daily_new RENAME TO prices_daily;
      CREATE INDEX IF NOT EXISTS idx_prices_daily_date_source
        ON prices_daily(date, source);
      CREATE INDEX IF NOT EXISTS idx_prices_daily_ticker_date_source
        ON prices_daily(ticker, date, source);
      """)
      conn.commit()
      log.info("Migration complete.")
    else:
      log.info("prices_daily already has source column; no migration needed.")

  init_db()


def main() -> None:
  ap = argparse.ArgumentParser(description="Migrate prices_daily to include source column.")
  ap.add_argument("--default-source", default="alpaca", help="Source label for existing rows")
  args = ap.parse_args()

  migrate(args.default_source)


if __name__ == "__main__":
  main()
