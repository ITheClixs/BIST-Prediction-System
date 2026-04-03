"""SQLite database connection and schema management."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_date ON raw_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker ON raw_prices(ticker);

CREATE TABLE IF NOT EXISTS macro_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(indicator, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_data_indicator_date ON macro_data(indicator, date);

CREATE TABLE IF NOT EXISTS sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    headline TEXT,
    sentiment_score REAL,
    raw_text TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_data(ticker, date);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    predicted_pct_move REAL NOT NULL,
    actual_pct_move REAL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, target_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_target ON predictions(ticker, target_date);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(ticker, date, feature_name, version)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON features(ticker, date);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_path TEXT NOT NULL,
    metrics_json TEXT,
    trained_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 0,
    UNIQUE(model_name, version)
);
"""


class Database:
    """SQLite database for BIST predictor data."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create database file, parent directories, and schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            existing = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            conn.commit()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def get_latest_date(self, ticker: str) -> str | None:
        """Return the most recent date for a ticker in raw_prices, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM raw_prices WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            return row[0] if row and row[0] else None
