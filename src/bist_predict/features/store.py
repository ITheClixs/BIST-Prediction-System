"""Feature store — persist computed features in SQLite."""

from __future__ import annotations

from bist_predict.storage.database import Database


class FeatureStore:
    """Read/write features keyed by (ticker, date, feature_name)."""

    def __init__(self, db: Database, version: int = 1) -> None:
        self._db = db
        self._version = version

    def save(self, ticker: str, date: str, features: dict[str, float]) -> None:
        """Save features for a ticker on a date. Overwrites existing values."""
        with self._db.connect() as conn:
            for name, value in features.items():
                conn.execute(
                    """INSERT INTO features (ticker, date, feature_name, value, version)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ticker, date, feature_name, version)
                       DO UPDATE SET value = excluded.value""",
                    (ticker, date, name, value, self._version),
                )
            conn.commit()

    def load(self, ticker: str, date: str) -> dict[str, float]:
        """Load all features for a ticker on a date."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT feature_name, value FROM features
                   WHERE ticker = ? AND date = ? AND version = ?""",
                (ticker, date, self._version),
            ).fetchall()
        return {name: value for name, value in rows}

    def load_range(
        self, ticker: str, start_date: str, end_date: str
    ) -> dict[str, dict[str, float]]:
        """Load features for a ticker across a date range. Returns {date: {name: value}}."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT date, feature_name, value FROM features
                   WHERE ticker = ? AND date >= ? AND date <= ? AND version = ?
                   ORDER BY date""",
                (ticker, start_date, end_date, self._version),
            ).fetchall()

        result: dict[str, dict[str, float]] = {}
        for date, name, value in rows:
            if date not in result:
                result[date] = {}
            result[date][name] = value
        return result

    def get_latest_date(self, ticker: str) -> str | None:
        """Return the most recent date with features for a ticker."""
        with self._db.connect() as conn:
            row = conn.execute(
                """SELECT MAX(date) FROM features
                   WHERE ticker = ? AND version = ?""",
                (ticker, self._version),
            ).fetchone()
        return row[0] if row and row[0] else None
