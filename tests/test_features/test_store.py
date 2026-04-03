"""Tests for the feature store."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


class TestFeatureStore:
    def test_save_and_load_features(self, db: Database) -> None:
        store = FeatureStore(db)
        features = {"rsi_14": 65.3, "sma_20": 312.5, "macd": 1.25}
        store.save("THYAO", "2026-04-01", features)

        loaded = store.load("THYAO", "2026-04-01")
        assert loaded["rsi_14"] == 65.3
        assert loaded["sma_20"] == 312.5
        assert loaded["macd"] == 1.25

    def test_load_nonexistent_returns_empty(self, db: Database) -> None:
        store = FeatureStore(db)
        loaded = store.load("THYAO", "2026-04-01")
        assert loaded == {}

    def test_save_overwrites_existing(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("THYAO", "2026-04-01", {"rsi_14": 70.0})

        loaded = store.load("THYAO", "2026-04-01")
        assert loaded["rsi_14"] == 70.0

    def test_load_multiple_tickers(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("GARAN", "2026-04-01", {"rsi_14": 55.0})

        thyao = store.load("THYAO", "2026-04-01")
        garan = store.load("GARAN", "2026-04-01")
        assert thyao["rsi_14"] == 65.3
        assert garan["rsi_14"] == 55.0

    def test_load_date_range(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-03-31", {"rsi_14": 60.0})
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("THYAO", "2026-04-02", {"rsi_14": 70.0})

        features = store.load_range("THYAO", "2026-03-31", "2026-04-02")
        assert len(features) == 3
        assert features["2026-03-31"]["rsi_14"] == 60.0
        assert features["2026-04-02"]["rsi_14"] == 70.0

    def test_get_latest_feature_date(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-03-31", {"rsi_14": 60.0})
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})

        latest = store.get_latest_date("THYAO")
        assert latest == "2026-04-01"

    def test_get_latest_date_no_data(self, db: Database) -> None:
        store = FeatureStore(db)
        assert store.get_latest_date("THYAO") is None
