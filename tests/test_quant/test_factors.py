"""Tests for factor models — momentum rankings and mean reversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.factors import (
    compute_cross_sectional_momentum,
    compute_fama_french_factors,
    compute_mean_reversion_ou,
    compute_time_series_momentum,
)


class TestCrossSectionalMomentum:
    def test_ranks_by_trailing_returns(self) -> None:
        # 5 stocks, 252 days of returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (252, 5))
        # Make stock 0 a clear winner (high returns), stock 4 a clear loser
        returns[:, 0] += 0.005
        returns[:, 4] -= 0.005

        result = compute_cross_sectional_momentum(returns, periods=[63, 126, 252])
        assert "momentum_rank_63" in result
        assert "momentum_rank_126" in result
        assert "momentum_rank_252" in result
        # Each rank array should have 5 values (one per stock)
        assert len(result["momentum_rank_63"]) == 5

    def test_ranks_are_percentiles(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 5))
        result = compute_cross_sectional_momentum(returns, periods=[252])
        ranks = result["momentum_rank_252"]
        # Percentile ranks should be between 0 and 1
        assert all(0.0 <= r <= 1.0 for r in ranks)

    def test_insufficient_data_returns_nan(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.02, (10, 3))
        result = compute_cross_sectional_momentum(returns, periods=[252])
        ranks = result["momentum_rank_252"]
        assert all(math.isnan(r) for r in ranks)


class TestTimeSeriesMomentum:
    def test_positive_excess_returns_long_signal(self) -> None:
        # 253 prices (need period+1 for trailing return calculation)
        prices = np.cumsum(np.ones(253) * 0.5) + 100.0
        result = compute_time_series_momentum(prices, period=252)
        assert result["tsmom_signal"] == 1.0
        assert result["tsmom_magnitude"] > 0

    def test_negative_excess_returns_short_signal(self) -> None:
        # 253 prices (need period+1 for trailing return calculation)
        prices = 100.0 - np.cumsum(np.ones(253) * 0.5)
        result = compute_time_series_momentum(prices, period=252)
        assert result["tsmom_signal"] == -1.0
        assert result["tsmom_magnitude"] < 0

    def test_insufficient_data(self) -> None:
        prices = np.array([100.0, 101.0, 102.0])
        result = compute_time_series_momentum(prices, period=252)
        assert math.isnan(result["tsmom_signal"])


class TestMeanReversionOU:
    def test_estimates_ou_parameters(self) -> None:
        # Generate mean-reverting series: X(t+1) = X(t) + theta*(mu - X(t)) + noise
        rng = np.random.default_rng(42)
        n = 500
        theta_true = 0.1
        mu_true = 100.0
        sigma_true = 1.0
        x = np.zeros(n)
        x[0] = 95.0
        for i in range(1, n):
            x[i] = x[i - 1] + theta_true * (mu_true - x[i - 1]) + sigma_true * rng.normal()

        result = compute_mean_reversion_ou(x)
        assert "ou_theta" in result
        assert "ou_mu" in result
        assert "ou_deviation" in result
        assert "ou_signal" in result
        # theta should be positive (mean-reverting)
        assert result["ou_theta"] > 0
        # mu should be close to 100
        assert abs(result["ou_mu"] - mu_true) < 10.0

    def test_constant_series(self) -> None:
        x = np.ones(100) * 50.0
        result = compute_mean_reversion_ou(x)
        assert abs(result["ou_deviation"]) < 0.01


class TestFamaFrench:
    def test_computes_smb_hml(self) -> None:
        rng = np.random.default_rng(42)
        n_stocks = 20
        n_days = 252
        returns = rng.normal(0.001, 0.02, (n_days, n_stocks))
        market_caps = rng.uniform(1e9, 50e9, n_stocks)
        book_to_market = rng.uniform(0.3, 2.0, n_stocks)

        result = compute_fama_french_factors(returns, market_caps, book_to_market)
        assert "smb" in result
        assert "hml" in result
        assert "market_premium" in result
        # SMB and HML should be arrays of daily factor returns
        assert len(result["smb"]) == n_days
        assert len(result["hml"]) == n_days

    def test_factor_exposures(self) -> None:
        rng = np.random.default_rng(42)
        n_stocks = 20
        n_days = 252
        returns = rng.normal(0.001, 0.02, (n_days, n_stocks))
        market_caps = rng.uniform(1e9, 50e9, n_stocks)
        book_to_market = rng.uniform(0.3, 2.0, n_stocks)

        result = compute_fama_french_factors(returns, market_caps, book_to_market)
        assert "factor_exposures" in result
        # One exposure per stock
        assert result["factor_exposures"].shape == (n_stocks, 3)
