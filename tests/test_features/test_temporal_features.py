"""Tests for temporal/calendar feature computation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.features.temporal_features import compute_temporal_features


class TestTemporalFeatures:
    def test_day_of_week(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))  # Wednesday
        assert features["day_of_week"] == 2  # 0=Monday

    def test_month(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["month"] == 4

    def test_is_monday(self) -> None:
        features = compute_temporal_features(date(2026, 3, 30))  # Monday
        assert features["is_monday"] == 1

    def test_is_friday(self) -> None:
        features = compute_temporal_features(date(2026, 4, 3))  # Friday
        assert features["is_friday"] == 1

    def test_day_of_month(self) -> None:
        features = compute_temporal_features(date(2026, 4, 15))
        assert features["day_of_month"] == 15

    def test_is_month_start(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["is_month_start"] == 1
        features2 = compute_temporal_features(date(2026, 4, 15))
        assert features2["is_month_start"] == 0

    def test_is_month_end(self) -> None:
        features = compute_temporal_features(date(2026, 4, 30))
        assert features["is_month_end"] == 1

    def test_quarter(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["quarter"] == 2

    def test_week_of_year(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert "week_of_year" in features
