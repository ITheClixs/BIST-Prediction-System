"""Tests for walk-forward backtesting engine."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.evaluation.backtest import WalkForwardBacktest


class TestWalkForwardBacktest:
    def test_generates_folds(self) -> None:
        n_dates = 500
        bt = WalkForwardBacktest(
            train_window=252, val_window=63, step_size=21,
            commission=0.001, slippage=0.0005,
        )
        folds = bt.generate_folds(n_dates)
        assert len(folds) > 0
        for fold in folds:
            assert fold[0] < fold[1] <= fold[2] < fold[3]
            assert fold[1] - fold[0] == 252  # train window
            assert fold[3] - fold[2] == 63   # val window

    def test_no_future_leakage(self) -> None:
        bt = WalkForwardBacktest(train_window=252, val_window=63, step_size=21)
        folds = bt.generate_folds(500)
        for fold in folds:
            assert fold[1] <= fold[2]

    def test_apply_costs(self) -> None:
        bt = WalkForwardBacktest(commission=0.001, slippage=0.0005)
        gross_return = 0.05
        net_return = bt.apply_costs(gross_return)
        # Commission on entry + exit: 2 * 0.001 = 0.002
        # Slippage on entry + exit: 2 * 0.0005 = 0.001
        assert abs(net_return - (0.05 - 0.003)) < 0.0001

    def test_insufficient_data(self) -> None:
        bt = WalkForwardBacktest(train_window=252, val_window=63)
        folds = bt.generate_folds(100)  # Too short
        assert len(folds) == 0
