"""Tests for Platt scaling confidence calibration."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.models.calibration import PlattCalibrator


class TestPlattCalibrator:
    def test_calibrate_improves_probabilities(self) -> None:
        rng = np.random.default_rng(42)
        # Raw scores that need calibration
        raw_scores = rng.uniform(0.3, 0.7, 200)
        true_labels = (raw_scores > 0.5).astype(int)
        # Add noise
        flip_idx = rng.choice(200, 30, replace=False)
        true_labels[flip_idx] = 1 - true_labels[flip_idx]

        cal = PlattCalibrator()
        cal.fit(raw_scores, true_labels)
        calibrated = cal.transform(raw_scores)

        assert calibrated.shape == (200,)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_extreme_scores_stay_extreme(self) -> None:
        rng = np.random.default_rng(42)
        raw_scores = np.concatenate([
            rng.uniform(0.0, 0.1, 100),  # Clearly negative
            rng.uniform(0.9, 1.0, 100),  # Clearly positive
        ])
        true_labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

        cal = PlattCalibrator()
        cal.fit(raw_scores, true_labels)
        calibrated = cal.transform(raw_scores)

        # Low raw -> low calibrated, high raw -> high calibrated
        assert np.mean(calibrated[:100]) < 0.3
        assert np.mean(calibrated[100:]) > 0.7

    def test_not_fitted_raises(self) -> None:
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))

    def test_minimum_confidence_filter(self) -> None:
        cal = PlattCalibrator(min_confidence=0.70)
        rng = np.random.default_rng(42)
        raw_scores = rng.uniform(0, 1, 200)
        true_labels = (raw_scores > 0.5).astype(int)
        cal.fit(raw_scores, true_labels)

        calibrated = cal.transform(np.array([0.5]))
        assert len(calibrated) == 1
        assert cal.min_confidence == 0.70
