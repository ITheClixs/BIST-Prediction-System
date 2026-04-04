"""Tests for risk and position sizing — Kelly, Ledoit-Wolf covariance, PCA."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.risk import (
    compute_kelly_fraction,
    compute_ledoit_wolf_covariance,
    compute_pca_factors,
)


class TestKelly:
    def test_positive_edge(self) -> None:
        # 60% win rate, 1:1 win/loss ratio → f* = (0.6*1 - 0.4)/1 = 0.2
        result = compute_kelly_fraction(win_prob=0.6, win_loss_ratio=1.0, fraction=1.0)
        assert abs(result["kelly_full"] - 0.2) < 0.001
        assert abs(result["kelly_fraction"] - 0.2) < 0.001

    def test_fractional_kelly(self) -> None:
        result = compute_kelly_fraction(win_prob=0.6, win_loss_ratio=1.0, fraction=0.25)
        assert abs(result["kelly_fraction"] - 0.05) < 0.001

    def test_no_edge(self) -> None:
        # 50/50 with 1:1 → f* = 0
        result = compute_kelly_fraction(win_prob=0.5, win_loss_ratio=1.0, fraction=0.25)
        assert result["kelly_full"] == 0.0

    def test_negative_edge_clamps_to_zero(self) -> None:
        result = compute_kelly_fraction(win_prob=0.3, win_loss_ratio=1.0, fraction=0.25)
        assert result["kelly_fraction"] == 0.0

    def test_high_win_loss_ratio(self) -> None:
        # 50% win rate but 3:1 payout → f* = (0.5*3 - 0.5)/3 = 0.333
        result = compute_kelly_fraction(win_prob=0.5, win_loss_ratio=3.0, fraction=1.0)
        assert abs(result["kelly_full"] - 1.0 / 3) < 0.01


class TestLedoitWolf:
    def test_shrinks_covariance(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 10))

        result = compute_ledoit_wolf_covariance(returns)
        assert "covariance" in result
        assert "shrinkage_coefficient" in result
        assert result["covariance"].shape == (10, 10)
        # Shrinkage coefficient between 0 and 1
        assert 0.0 <= result["shrinkage_coefficient"] <= 1.0
        # Should be symmetric
        assert np.allclose(result["covariance"], result["covariance"].T)

    def test_positive_definite(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 10))
        result = compute_ledoit_wolf_covariance(returns)
        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(result["covariance"])
        assert all(ev > 0 for ev in eigenvalues)


class TestPCAFactors:
    def test_extracts_common_factor(self) -> None:
        rng = np.random.default_rng(42)
        # 20 stocks, 252 days, with a strong common factor
        common = rng.normal(0, 0.02, (252, 1))
        idiosyncratic = rng.normal(0, 0.01, (252, 20))
        returns = common + idiosyncratic

        result = compute_pca_factors(returns, n_components=5)
        assert "pca_components" in result
        assert "explained_variance_ratio" in result
        assert result["pca_components"].shape == (252, 5)
        assert len(result["explained_variance_ratio"]) == 5
        # First component should explain most variance (strong common factor)
        assert result["explained_variance_ratio"][0] > 0.2

    def test_loadings_shape(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 20))
        result = compute_pca_factors(returns, n_components=5)
        assert "loadings" in result
        assert result["loadings"].shape == (5, 20)

    def test_insufficient_data(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.02, (3, 20))
        result = compute_pca_factors(returns, n_components=5)
        assert result["pca_components"].shape[1] <= 3
