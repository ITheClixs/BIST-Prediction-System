"""Family-wise error control across a family of model comparisons.

Six fitted models are tested against the same benchmark on the same data.  The
probability that at least one of six independent tests clears ``p < 0.05`` by
chance alone is ``1 - 0.95**6 = 26.5%``, so an uncorrected table of six
p-values is not evidence about any single model.  Holm (1979) controls the
family-wise error rate without assuming independence between the tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["HolmCorrection", "holm_step_down"]


@dataclass(frozen=True)
class HolmCorrection:
    """Holm step-down adjusted p-values for one family of hypotheses."""

    alpha: float
    family_size: int
    raw_p_values: tuple[tuple[str, float], ...]
    adjusted_p_values: tuple[tuple[str, float], ...]
    rejected: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the correction."""
        return {
            "method": "holm_step_down",
            "alpha": self.alpha,
            "family_size": self.family_size,
            "raw_p_values": {name: value for name, value in self.raw_p_values},
            "adjusted_p_values": {name: value for name, value in self.adjusted_p_values},
            "rejected": list(self.rejected),
        }


def holm_step_down(p_values: Mapping[str, float], *, alpha: float = 0.05) -> HolmCorrection:
    r"""Return Holm (1979) step-down adjusted p-values.

    With ordered raw p-values :math:`p_{(1)} \le \dots \le p_{(m)}`, the
    adjusted values enforce monotonicity across the ordered family:

    .. math::
        \tilde p_{(i)} = \max_{j \le i} \min\left[(m - j + 1) p_{(j)},\; 1\right].

    A hypothesis is rejected when its adjusted p-value does not exceed
    ``alpha``.  Ties are broken by hypothesis name so the result is
    deterministic.
    """
    if not p_values:
        raise ValueError("holm_step_down requires at least one hypothesis")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    for name, value in p_values.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value must lie in [0, 1]: {name}={value}")

    family_size = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: list[tuple[str, float]] = []
    running_maximum = 0.0
    for position, (name, raw) in enumerate(ordered):
        scaled = min((family_size - position) * raw, 1.0)
        running_maximum = max(running_maximum, scaled)
        adjusted.append((name, running_maximum))
    rejected = tuple(name for name, value in adjusted if value <= alpha)
    return HolmCorrection(
        alpha=float(alpha),
        family_size=family_size,
        raw_p_values=tuple(ordered),
        adjusted_p_values=tuple(adjusted),
        rejected=rejected,
    )
