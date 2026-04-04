"""Tests for model types and dataset helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bist_predict.models.types import (
    Prediction,
    TrainDataset,
    build_tabular_dataset,
    build_sequence_dataset,
)
from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


@pytest.fixture
def seeded_db(db: Database) -> Database:
    """DB with feature data and price data for dataset building."""
    store = FeatureStore(db)
    with db.connect() as conn:
        for i in range(60):
            date_str = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
            price = 100.0 + i * 0.5
            conn.execute(
                """INSERT OR IGNORE INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", date_str, price, price + 1, price - 1, price, price, 1000000, "test"),
            )
        conn.commit()

    for i in range(60):
        date_str = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        features = {
            "rsi_14": 50.0 + i * 0.3,
            "sma_20": 100.0 + i * 0.5,
            "macd": 0.1 * i,
            "volume_ratio": 1.0 + i * 0.01,
        }
        store.save("THYAO", date_str, features)

    return db


class TestPrediction:
    def test_create_prediction(self) -> None:
        pred = Prediction(
            ticker="THYAO",
            direction="UP",
            confidence=0.78,
            predicted_pct_move=1.5,
            model_name="xgboost",
        )
        assert pred.ticker == "THYAO"
        assert pred.direction == "UP"
        assert pred.confidence == 0.78
        assert pred.predicted_pct_move == 1.5

    def test_prediction_is_buy(self) -> None:
        pred = Prediction(
            ticker="THYAO", direction="UP", confidence=0.78,
            predicted_pct_move=1.5, model_name="xgboost",
        )
        assert pred.is_buy
        assert not pred.is_sell

    def test_prediction_signal_tier(self) -> None:
        strong_buy = Prediction(
            ticker="THYAO", direction="UP", confidence=0.85,
            predicted_pct_move=2.0, model_name="xgboost",
        )
        assert strong_buy.signal_tier == "STRONG BUY"

        buy = Prediction(
            ticker="THYAO", direction="UP", confidence=0.75,
            predicted_pct_move=1.0, model_name="xgboost",
        )
        assert buy.signal_tier == "BUY"


class TestBuildTabularDataset:
    def test_builds_feature_matrix(self, seeded_db: Database) -> None:
        X, y_dir, y_pct, dates = build_tabular_dataset(
            seeded_db, "THYAO", min_features=3,
        )
        assert X.shape[0] > 0
        assert X.shape[1] >= 3
        assert len(y_dir) == X.shape[0]
        assert len(y_pct) == X.shape[0]

    def test_labels_are_binary_direction(self, seeded_db: Database) -> None:
        X, y_dir, y_pct, dates = build_tabular_dataset(
            seeded_db, "THYAO", min_features=3,
        )
        assert set(np.unique(y_dir)).issubset({0, 1})


class TestBuildSequenceDataset:
    def test_builds_sequences(self, seeded_db: Database) -> None:
        X_seq, y_dir, y_pct, dates = build_sequence_dataset(
            seeded_db, "THYAO", seq_len=10, min_features=3,
        )
        assert X_seq.shape[0] > 0
        assert X_seq.shape[1] == 10
        assert X_seq.shape[2] >= 3
