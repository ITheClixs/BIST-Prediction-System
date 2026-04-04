"""Tests for regime-aware routing — adjusts ensemble weights based on HMM regime."""

from __future__ import annotations

import pytest

from bist_predict.quant.regime import RegimeRouter


class TestRegimeRouter:
    def test_bull_regime_favors_momentum(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.9,
            regime_bear_prob=0.05,
            regime_sideways_prob=0.05,
        )
        assert weights["momentum_weight"] > weights["mean_reversion_weight"]
        # Kelly is blended: 0.9*0.5 + 0.05*0.25 + 0.05*0.25 = 0.475
        assert weights["kelly_fraction"] > 0.4

    def test_bear_regime_favors_mean_reversion(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.05,
            regime_bear_prob=0.9,
            regime_sideways_prob=0.05,
        )
        assert weights["mean_reversion_weight"] > weights["momentum_weight"]
        # Kelly is blended: mostly bear → close to 0.25
        assert weights["kelly_fraction"] < 0.3

    def test_sideways_regime_favors_pairs(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.05,
            regime_bear_prob=0.05,
            regime_sideways_prob=0.9,
        )
        assert weights["pairs_weight"] > weights["momentum_weight"]
        # Kelly is blended: mostly sideways → close to 0.25
        assert weights["kelly_fraction"] < 0.3

    def test_uncertain_regime_blends(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.33,
            regime_bear_prob=0.33,
            regime_sideways_prob=0.34,
        )
        # No extreme weighting when uncertain
        assert abs(weights["momentum_weight"] - weights["mean_reversion_weight"]) < 0.2

    def test_weights_sum_to_one(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.6,
            regime_bear_prob=0.3,
            regime_sideways_prob=0.1,
        )
        total = weights["momentum_weight"] + weights["mean_reversion_weight"] + weights["pairs_weight"]
        assert abs(total - 1.0) < 0.01

    def test_nan_regime_returns_defaults(self) -> None:
        import math
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=math.nan,
            regime_bear_prob=math.nan,
            regime_sideways_prob=math.nan,
        )
        assert abs(weights["momentum_weight"] - 1 / 3) < 0.01
        assert weights["kelly_fraction"] == 0.25
