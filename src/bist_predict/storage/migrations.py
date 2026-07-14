"""Schema versioning and migrations.

Migrations are applied in order. Each migration is a SQL string that transforms
the schema from version N to N+1. The current schema version is stored in the
schema_version table.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: dict[int, str] = {
    2: """
    ALTER TABLE raw_prices ADD COLUMN open_quality TEXT NOT NULL DEFAULT 'observed';
    ALTER TABLE raw_prices ADD COLUMN volume_quality TEXT NOT NULL DEFAULT 'observed';
    ALTER TABLE raw_prices ADD COLUMN provider_symbol TEXT;
    ALTER TABLE raw_prices ADD COLUMN provider_record_id TEXT;
    ALTER TABLE raw_prices ADD COLUMN source_retrieved_at TEXT;
    ALTER TABLE raw_prices ADD COLUMN split_adj_open REAL;
    ALTER TABLE raw_prices ADD COLUMN split_adj_high REAL;
    ALTER TABLE raw_prices ADD COLUMN split_adj_low REAL;
    ALTER TABLE raw_prices ADD COLUMN split_adj_close REAL;
    ALTER TABLE raw_prices ADD COLUMN split_adj_volume INTEGER;
    ALTER TABLE raw_prices ADD COLUMN total_return_open REAL;
    ALTER TABLE raw_prices ADD COLUMN total_return_high REAL;
    ALTER TABLE raw_prices ADD COLUMN total_return_low REAL;
    ALTER TABLE raw_prices ADD COLUMN total_return_close REAL;
    ALTER TABLE raw_prices ADD COLUMN total_return_volume INTEGER;

    CREATE TABLE IF NOT EXISTS corporate_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        effective_date TEXT NOT NULL,
        action_type TEXT NOT NULL,
        source TEXT NOT NULL,
        ratio REAL,
        cash_amount REAL,
        currency TEXT,
        subscription_price REAL,
        new_ticker TEXT,
        delisting_price REAL,
        source_retrieved_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(ticker, effective_date, action_type, source)
    );

    CREATE INDEX IF NOT EXISTS idx_corporate_actions_ticker_date
    ON corporate_actions(ticker, effective_date);
    """,
}


def get_current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version."""
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] if row and row[0] else 0


def apply_pending_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the final schema version."""
    current = get_current_version(conn)
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            conn.executescript(MIGRATIONS[version])
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,),
            )
            conn.commit()
            current = version
    return current
