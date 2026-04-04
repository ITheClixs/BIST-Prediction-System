"""Tests for macro feature computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.macro_features import compute_macro_features
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    # Seed macro data
    with db.connect() as conn:
        macro_data = [
            ("USD_TRY", "2026-03-28", 37.80, "tcmb"),
            ("USD_TRY", "2026-03-31", 38.10, "tcmb"),
            ("USD_TRY", "2026-04-01", 38.45, "tcmb"),
            ("EUR_TRY", "2026-03-28", 40.50, "tcmb"),
            ("EUR_TRY", "2026-03-31", 40.80, "tcmb"),
            ("EUR_TRY", "2026-04-01", 41.20, "tcmb"),
            ("GOLD_TRY", "2026-03-28", 2800.0, "tcmb"),
            ("GOLD_TRY", "2026-03-31", 2850.0, "tcmb"),
            ("GOLD_TRY", "2026-04-01", 2830.0, "tcmb"),
        ]
        for indicator, date, value, source in macro_data:
            conn.execute(
                "INSERT INTO macro_data (indicator, date, value, source) VALUES (?, ?, ?, ?)",
                (indicator, date, value, source),
            )
        conn.commit()
    return db


class TestMacroFeatures:
    def test_computes_daily_deltas(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "usd_try_delta" in features
        assert abs(features["usd_try_delta"] - 0.35) < 0.01  # 38.45 - 38.10

    def test_computes_percentage_change(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "usd_try_pct" in features
        expected_pct = (38.45 - 38.10) / 38.10
        assert abs(features["usd_try_pct"] - expected_pct) < 0.001

    def test_includes_multiple_indicators(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "eur_try_delta" in features
        assert "gold_try_delta" in features

    def test_missing_data_returns_nan(self, db: Database) -> None:
        import math
        features = compute_macro_features(db, "2026-01-01")  # No data
        assert math.isnan(features.get("usd_try_delta", float("nan")))
