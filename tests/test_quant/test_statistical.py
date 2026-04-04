"""Tests for statistical methods — Kalman, GARCH, HMM, cointegration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.statistical import (
    compute_cointegration,
    compute_garch_volatility,
    compute_hmm_regime,
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


class TestHMMRegime:
    def test_detects_regimes(self) -> None:
        rng = np.random.default_rng(42)
        # Bull regime (positive returns, low vol) then bear (negative, high vol)
        bull = rng.normal(0.001, 0.01, 200)
        bear = rng.normal(-0.002, 0.03, 100)
        sideways = rng.normal(0.0, 0.005, 100)
        returns = np.concatenate([bull, bear, sideways])

        result = compute_hmm_regime(returns, n_states=3)
        assert "regime_current" in result
        assert "regime_bull_prob" in result
        assert "regime_bear_prob" in result
        assert "regime_sideways_prob" in result
        # Current regime should be one of 0, 1, 2
        assert result["regime_current"] in (0, 1, 2)
        # Probabilities should sum to ~1
        prob_sum = result["regime_bull_prob"] + result["regime_bear_prob"] + result["regime_sideways_prob"]
        assert abs(prob_sum - 1.0) < 0.01

    def test_insufficient_data(self) -> None:
        returns = np.array([0.01, -0.01])
        result = compute_hmm_regime(returns, n_states=3)
        assert math.isnan(result["regime_bull_prob"])


class TestCointegration:
    def test_cointegrated_pair(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        # Random walk
        x = np.cumsum(rng.normal(0, 1, n)) + 100.0
        # Cointegrated partner: y = 2*x + mean-reverting noise
        noise = np.zeros(n)
        noise[0] = rng.normal()
        for i in range(1, n):
            noise[i] = 0.8 * noise[i - 1] + rng.normal()
        y = 2 * x + noise + 50.0

        result = compute_cointegration(x, y)
        assert "coint_pvalue" in result
        assert "spread_zscore" in result
        assert "spread_halflife" in result
        # Should detect cointegration (p < 0.05)
        assert result["coint_pvalue"] < 0.05

    def test_non_cointegrated_pair(self) -> None:
        rng = np.random.default_rng(42)
        # Two independent random walks
        x = np.cumsum(rng.normal(0, 1, 500)) + 100.0
        y = np.cumsum(rng.normal(0, 1, 500)) + 100.0

        result = compute_cointegration(x, y)
        # Should NOT detect cointegration (p > 0.05)
        assert result["coint_pvalue"] > 0.05

    def test_short_series(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        result = compute_cointegration(x, y)
        assert "coint_pvalue" in result
