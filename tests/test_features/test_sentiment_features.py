"""Tests for sentiment feature aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.sentiment_features import compute_sentiment_features
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    with db.connect() as conn:
        sentiments = [
            ("THYAO", "2026-04-01", "google_news", "THY yükseldi", 0.8, "text1"),
            ("THYAO", "2026-04-01", "google_news", "THY güçlendi", 0.6, "text2"),
            ("THYAO", "2026-04-01", "bloomberght", "THY bilançosu", -0.2, "text3"),
            ("THYAO", "2026-03-31", "google_news", "THY haberleri", 0.5, "text4"),
        ]
        for ticker, date, source, headline, score, raw in sentiments:
            conn.execute(
                """INSERT INTO sentiment_data
                   (ticker, date, source, headline, sentiment_score, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, date, source, headline, score, raw),
            )
        conn.commit()
    return db


class TestSentimentFeatures:
    def test_computes_average_sentiment(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert "sentiment_mean" in features
        # Mean of 0.8, 0.6, -0.2 = 0.4
        assert abs(features["sentiment_mean"] - 0.4) < 0.01

    def test_computes_sentiment_count(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert features["sentiment_count"] == 3

    def test_computes_positive_ratio(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert "sentiment_positive_ratio" in features
        # 2 out of 3 are positive
        assert abs(features["sentiment_positive_ratio"] - 2 / 3) < 0.01

    def test_no_data_returns_defaults(self, db: Database) -> None:
        import math
        features = compute_sentiment_features(db, "GARAN", "2026-04-01")
        assert features["sentiment_count"] == 0
        assert math.isnan(features["sentiment_mean"])
