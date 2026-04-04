"""Regime-aware routing — adjusts model ensemble weights based on HMM regime detection."""

from __future__ import annotations

import math


class RegimeRouter:
    """Map HMM regime probabilities to ensemble weight adjustments.

    Bull regime → increase momentum weight, Kelly at 0.5×
    Bear regime → increase mean-reversion weight, Kelly at 0.25×
    Sideways regime → increase pairs trading weight, Kelly at 0.25×
    """

    # Weight profiles per regime (momentum, mean_reversion, pairs)
    BULL_WEIGHTS = (0.50, 0.20, 0.30)
    BEAR_WEIGHTS = (0.20, 0.50, 0.30)
    SIDEWAYS_WEIGHTS = (0.15, 0.25, 0.60)
    DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

    KELLY_BULL = 0.5
    KELLY_BEAR = 0.25
    KELLY_SIDEWAYS = 0.25

    def get_weights(
        self,
        regime_bull_prob: float,
        regime_bear_prob: float,
        regime_sideways_prob: float,
    ) -> dict[str, float]:
        """Compute blended strategy weights from regime probabilities.

        Args:
            regime_bull_prob: probability of bull regime (0-1).
            regime_bear_prob: probability of bear regime (0-1).
            regime_sideways_prob: probability of sideways regime (0-1).

        Returns:
            momentum_weight: weight for momentum-based signals.
            mean_reversion_weight: weight for mean-reversion signals.
            pairs_weight: weight for pairs/cointegration signals.
            kelly_fraction: recommended fractional Kelly multiplier.
        """
        # Handle NaN inputs → return equal weights
        if (
            math.isnan(regime_bull_prob)
            or math.isnan(regime_bear_prob)
            or math.isnan(regime_sideways_prob)
        ):
            return {
                "momentum_weight": self.DEFAULT_WEIGHTS[0],
                "mean_reversion_weight": self.DEFAULT_WEIGHTS[1],
                "pairs_weight": self.DEFAULT_WEIGHTS[2],
                "kelly_fraction": 0.25,
            }

        # Blend weight profiles by regime probability
        momentum = (
            regime_bull_prob * self.BULL_WEIGHTS[0]
            + regime_bear_prob * self.BEAR_WEIGHTS[0]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[0]
        )
        mean_rev = (
            regime_bull_prob * self.BULL_WEIGHTS[1]
            + regime_bear_prob * self.BEAR_WEIGHTS[1]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[1]
        )
        pairs = (
            regime_bull_prob * self.BULL_WEIGHTS[2]
            + regime_bear_prob * self.BEAR_WEIGHTS[2]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[2]
        )

        # Normalize to sum to 1
        total = momentum + mean_rev + pairs
        if total > 0:
            momentum /= total
            mean_rev /= total
            pairs /= total

        # Blend Kelly fraction by regime
        kelly = (
            regime_bull_prob * self.KELLY_BULL
            + regime_bear_prob * self.KELLY_BEAR
            + regime_sideways_prob * self.KELLY_SIDEWAYS
        )

        return {
            "momentum_weight": momentum,
            "mean_reversion_weight": mean_rev,
            "pairs_weight": pairs,
            "kelly_fraction": kelly,
        }
