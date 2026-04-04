"""Tests for live prediction accuracy tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.evaluation.tracker import AccuracyTracker
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


class TestAccuracyTracker:
    def test_log_prediction(self, db: Database) -> None:
        tracker = AccuracyTracker(db)
        tracker.log_prediction(
            ticker="THYAO", prediction_date="2026-04-01", target_date="2026-04-02",
            direction="UP", confidence=0.78, predicted_pct_move=1.5, model_version="v1",
        )
        preds = tracker.get_predictions("THYAO")
        assert len(preds) == 1
        assert preds[0]["direction"] == "UP"

    def test_record_actual(self, db: Database) -> None:
        tracker = AccuracyTracker(db)
        tracker.log_prediction(
            ticker="THYAO", prediction_date="2026-04-01", target_date="2026-04-02",
            direction="UP", confidence=0.78, predicted_pct_move=1.5, model_version="v1",
        )
        tracker.record_actual("THYAO", "2026-04-02", actual_pct_move=1.2, model_version="v1")

        preds = tracker.get_predictions("THYAO")
        assert preds[0]["actual_pct_move"] == 1.2

    def test_rolling_accuracy(self, db: Database) -> None:
        tracker = AccuracyTracker(db)
        for i in range(10):
            direction = "UP" if i % 2 == 0 else "DOWN"
            actual = 0.01 if i % 2 == 0 else -0.01  # All correct
            tracker.log_prediction(
                ticker="THYAO", prediction_date=f"2026-04-{i + 1:02d}",
                target_date=f"2026-04-{i + 2:02d}",
                direction=direction, confidence=0.75,
                predicted_pct_move=0.01 if direction == "UP" else -0.01,
                model_version="v1",
            )
            tracker.record_actual("THYAO", f"2026-04-{i + 2:02d}", actual, "v1")

        accuracy = tracker.rolling_accuracy("THYAO", window=10)
        assert accuracy == 1.0

    def test_rolling_accuracy_no_data(self, db: Database) -> None:
        tracker = AccuracyTracker(db)
        accuracy = tracker.rolling_accuracy("THYAO", window=30)
        assert accuracy == 0.0

    def test_confidence_bucket_analysis(self, db: Database) -> None:
        tracker = AccuracyTracker(db)
        for i in range(20):
            conf = 0.6 + i * 0.02
            direction = "UP"
            actual = 0.01 if i > 10 else -0.01
            tracker.log_prediction(
                ticker="THYAO", prediction_date=f"2026-04-{i + 1:02d}",
                target_date=f"2026-04-{i + 2:02d}",
                direction=direction, confidence=conf, predicted_pct_move=0.01,
                model_version="v1",
            )
            tracker.record_actual("THYAO", f"2026-04-{i + 2:02d}", actual, "v1")

        buckets = tracker.confidence_buckets("THYAO")
        assert isinstance(buckets, dict)
        assert len(buckets) > 0
