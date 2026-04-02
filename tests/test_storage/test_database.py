"""Tests for SQLite database layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bist_predict.storage.database import Database


class TestDatabaseInit:
    def test_creates_db_file(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        assert tmp_db_path.exists()

    def test_creates_data_directory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "test.db"
        db = Database(db_path)
        db.initialize()
        assert db_path.exists()

    def test_creates_raw_prices_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_prices'"
            )
            assert cursor.fetchone() is not None

    def test_creates_macro_data_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='macro_data'"
            )
            assert cursor.fetchone() is not None

    def test_creates_sentiment_data_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'"
            )
            assert cursor.fetchone() is not None

    def test_creates_predictions_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            assert cursor.fetchone() is not None

    def test_creates_schema_version_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent_initialize(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        db.initialize()  # Should not raise
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            )
            count = cursor.fetchone()[0]
            assert count >= 5


class TestDatabaseOperations:
    def test_insert_and_query_raw_prices(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", "2026-04-01", 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM raw_prices WHERE ticker = ? AND date = ?",
                ("THYAO", "2026-04-01"),
            ).fetchone()
            assert row is not None
            assert row[1] == "THYAO"  # ticker
            assert row[6] == 312.5    # close

    def test_unique_constraint_ticker_date(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", "2026-04-01", 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO raw_prices
                       (ticker, date, open, high, low, close, adj_close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("THYAO", "2026-04-01", 311.0, 316.0, 309.0, 313.0, 313.0, 1100000, "yahoo"),
                )

    def test_get_latest_date_for_ticker(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            for date in ["2026-03-28", "2026-03-31", "2026-04-01"]:
                conn.execute(
                    """INSERT INTO raw_prices
                       (ticker, date, open, high, low, close, adj_close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("THYAO", date, 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
                )
            conn.commit()
        latest = db.get_latest_date("THYAO")
        assert latest == "2026-04-01"

    def test_get_latest_date_no_data(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        latest = db.get_latest_date("THYAO")
        assert latest is None
