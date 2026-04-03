"""Tests for statistical methods — Kalman, GARCH, HMM, cointegration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.statistical import (
    compute_garch_volatility,
    compute_kalman_trend,
)


class TestKalmanTrend:
    def test_filters_noisy_uptrend(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        trend = np.linspace(100, 120, n)
        noisy = trend + rng.normal(0, 2, n)

        result = compute_kalman_trend(noisy)
        assert "kalman_trend" in result
        assert "kalman_variance" in result
        # Filtered trend at end should be close to true trend
        assert abs(result["kalman_trend"] - 120.0) < 5.0
        # Variance should be positive
        assert result["kalman_variance"] > 0

    def test_adapts_to_level_shift(self) -> None:
        rng = np.random.default_rng(42)
        # Flat at 100, then jumps to 120
        prices = np.concatenate([
            100 + rng.normal(0, 1, 100),
            120 + rng.normal(0, 1, 100),
        ])
        result = compute_kalman_trend(prices)
        # After shift, trend should be near 120
        assert result["kalman_trend"] > 115.0

    def test_short_series(self) -> None:
        prices = np.array([100.0, 101.0])
        result = compute_kalman_trend(prices)
        assert "kalman_trend" in result


class TestGARCH:
    def test_computes_volatility_forecast(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        # Add volatility cluster
        returns[200:250] *= 3.0

        result = compute_garch_volatility(returns)
        assert "garch_vol_forecast" in result
        assert "garch_vol_surprise" in result
        assert result["garch_vol_forecast"] > 0

    def test_insufficient_data_returns_nan(self) -> None:
        returns = np.array([0.01, -0.01, 0.02])
        result = compute_garch_volatility(returns)
        assert math.isnan(result["garch_vol_forecast"])

    def test_constant_returns(self) -> None:
        returns = np.zeros(200)
        result = compute_garch_volatility(returns)
        # Should not crash on zero-variance input
        assert "garch_vol_forecast" in result
