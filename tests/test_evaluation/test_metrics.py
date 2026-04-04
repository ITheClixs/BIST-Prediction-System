"""Tests for prediction and trading quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.evaluation.metrics import (
    compute_prediction_metrics,
    compute_trading_metrics,
)


class TestPredictionMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
        y_pct_true = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        y_pct_pred = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

        metrics = compute_prediction_metrics(y_true, y_prob, y_pct_true, y_pct_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["mae"] == 0.0
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "brier_score" in metrics

    def test_random_predictions(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        y_pct_true = rng.normal(0, 0.02, 1000)
        y_pct_pred = rng.normal(0, 0.02, 1000)

        metrics = compute_prediction_metrics(y_true, y_prob, y_pct_true, y_pct_pred)
        # Random predictions -> ~50% accuracy
        assert 0.4 < metrics["accuracy"] < 0.6


class TestTradingMetrics:
    def test_profitable_strategy(self) -> None:
        # Consistent small positive returns
        daily_returns = np.array([0.01, 0.005, 0.008, -0.002, 0.012, 0.003, -0.001, 0.009, 0.004, 0.006])
        metrics = compute_trading_metrics(daily_returns)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert metrics["win_rate"] > 0.5
        assert metrics["sharpe_ratio"] > 0

    def test_losing_strategy(self) -> None:
        daily_returns = np.array([-0.01, -0.005, -0.008, 0.002, -0.012])
        metrics = compute_trading_metrics(daily_returns)
        assert metrics["sharpe_ratio"] < 0
        assert metrics["win_rate"] < 0.5

    def test_empty_returns(self) -> None:
        metrics = compute_trading_metrics(np.array([]))
        assert metrics["sharpe_ratio"] == 0.0
