"""Tests for the feature computation engine."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from bist_predict.features.engine import FeatureEngine
from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    # Seed with 30 days of price data for THYAO
    start = date(2026, 2, 1)
    with db.connect() as conn:
        for i in range(30):
            d = start + timedelta(days=i)
            price = 300.0 + i * 0.5 + (i % 3 - 1) * 2  # Slightly uptrending with noise
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "THYAO", d.isoformat(),
                    price - 1.0, price + 2.0, price - 2.0, price, price,
                    1000000 + i * 10000, "isyatirim",
                ),
            )
        conn.commit()
    return db


class TestFeatureEngine:
    def test_compute_features_for_ticker(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_includes_technical_indicators(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert "rsi_14" in features
        assert "sma_20" in features
        assert "ema_10" in features

    def test_includes_temporal_features(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert "day_of_week" in features
        assert "month" in features

    def test_compute_and_store(self, db: Database) -> None:
        engine = FeatureEngine(db)
        store = FeatureStore(db)

        engine.compute_and_store("THYAO", "2026-04-01")

        loaded = store.load("THYAO", "2026-04-01")
        assert len(loaded) > 0
        assert "rsi_14" in loaded

    def test_no_price_data_returns_empty(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("NONEXISTENT", "2026-04-01")
        assert features == {}

    def test_includes_quant_features(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        # With 30 data points, Kalman and O-U features should be present
        assert "kalman_trend" in features
        assert "ou_theta" in features
