"""Tests for Rust cross-stock correlation functions."""

from __future__ import annotations

import numpy as np
import pytest

import bist_features


class TestCorrelationMatrix:
    def test_diagonal_is_one(self) -> None:
        returns = np.random.randn(5, 100).astype(np.float64)
        corr = bist_features.compute_correlation_matrix(returns)
        for i in range(5):
            assert abs(corr[i, i] - 1.0) < 0.001

    def test_symmetric(self) -> None:
        returns = np.random.randn(4, 50).astype(np.float64)
        corr = bist_features.compute_correlation_matrix(returns)
        for i in range(4):
            for j in range(4):
                assert abs(corr[i, j] - corr[j, i]) < 0.001

    def test_perfect_correlation(self) -> None:
        base = np.random.randn(1, 100).astype(np.float64)
        returns = np.vstack([base, base])
        corr = bist_features.compute_correlation_matrix(returns)
        assert abs(corr[0, 1] - 1.0) < 0.001

    def test_values_between_neg1_and_1(self) -> None:
        returns = np.random.randn(5, 100).astype(np.float64)
        corr = bist_features.compute_correlation_matrix(returns)
        assert np.all(corr >= -1.001)
        assert np.all(corr <= 1.001)


class TestBeta:
    def test_beta_of_market_is_one(self) -> None:
        market = np.random.randn(100).astype(np.float64)
        beta = bist_features.compute_beta(market, market)
        assert abs(beta - 1.0) < 0.001

    def test_beta_positive_for_correlated(self) -> None:
        market = np.random.randn(100).astype(np.float64)
        stock = (market * 1.5 + np.random.randn(100) * 0.1).astype(np.float64)
        beta = bist_features.compute_beta(stock, market)
        assert beta > 1.0

    def test_empty_returns_nan(self) -> None:
        beta = bist_features.compute_beta(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        assert np.isnan(beta)
