"""Cross-sectional dependence diagnostics for pooled panel evaluation.

A pooled panel of ``k`` tickers observed on ``T`` sessions has ``kT`` rows but
far fewer independent observations, because same-session returns share market
factors.  Treating the rows as independent shrinks every standard error by
roughly ``sqrt(VIF)`` and manufactures significance.  These helpers quantify the
inflation so the evaluation can report it instead of ignoring it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CrossSectionalDependence",
    "cross_sectional_dependence",
    "effective_sample_size",
    "variance_inflation_factor",
]


@dataclass(frozen=True)
class CrossSectionalDependence:
    """Average within-session correlation and the sample size it implies."""

    unit_count: int
    session_count: int
    row_count: int
    mean_pairwise_correlation: float
    variance_inflation_factor: float
    effective_row_count: float
    pair_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe record of the diagnostic."""
        return {
            "unit_count": self.unit_count,
            "session_count": self.session_count,
            "row_count": self.row_count,
            "mean_pairwise_correlation": self.mean_pairwise_correlation,
            "variance_inflation_factor": self.variance_inflation_factor,
            "effective_row_count": self.effective_row_count,
            "pair_count": self.pair_count,
        }


def variance_inflation_factor(unit_count: int, mean_pairwise_correlation: float) -> float:
    r"""Return the equicorrelated variance inflation factor.

    For ``k`` units with average pairwise correlation :math:`\bar\rho`, the
    variance of the cross-sectional mean is inflated relative to independence by

    .. math:: \mathrm{VIF} = 1 + (k - 1)\bar\rho.

    Values below one are clipped to one: a negative average correlation would
    imply the pooled sample carries *more* information than independent rows,
    which is not a claim this evaluation is willing to make.
    """
    if unit_count < 1:
        raise ValueError("unit_count must be positive")
    if not np.isfinite(mean_pairwise_correlation):
        raise ValueError("mean_pairwise_correlation must be finite")
    return float(max(1.0, 1.0 + (unit_count - 1) * mean_pairwise_correlation))


def effective_sample_size(row_count: int, inflation_factor: float) -> float:
    """Return ``row_count`` deflated by an equicorrelated inflation factor."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if inflation_factor <= 0.0 or not np.isfinite(inflation_factor):
        raise ValueError("inflation_factor must be positive and finite")
    return float(row_count / inflation_factor)


def cross_sectional_dependence(
    frame: pd.DataFrame,
    *,
    value_column: str,
    session_column: str = "date",
    unit_column: str = "ticker",
) -> CrossSectionalDependence:
    """Measure average within-session correlation across panel units.

    Correlations are computed pairwise on the sessions both units observe, then
    averaged with equal weight over unit pairs.  Pairs with fewer than three
    shared sessions, or with a constant series on either side, are skipped; a
    correlation is undefined there rather than zero.
    """
    for column in (value_column, session_column, unit_column):
        if column not in frame.columns:
            raise ValueError(f"frame is missing required column: {column}")
    if frame.duplicated([session_column, unit_column]).any():
        raise ValueError("frame must hold one row per session and unit")
    wide = frame.pivot(index=session_column, columns=unit_column, values=value_column)
    units = list(wide.columns)
    if len(units) < 2:
        raise ValueError("cross-sectional dependence requires at least two units")

    correlations: list[float] = []
    for first in range(len(units)):
        for second in range(first + 1, len(units)):
            pair = wide.iloc[:, [first, second]].dropna()
            if len(pair) < 3:
                continue
            left = pair.iloc[:, 0].to_numpy(dtype=np.float64)
            right = pair.iloc[:, 1].to_numpy(dtype=np.float64)
            if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
                continue
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    if not correlations:
        raise ValueError("no unit pair had enough shared sessions to estimate correlation")

    mean_correlation = float(np.mean(correlations))
    inflation = variance_inflation_factor(len(units), mean_correlation)
    row_count = int(len(frame))
    return CrossSectionalDependence(
        unit_count=len(units),
        session_count=int(wide.shape[0]),
        row_count=row_count,
        mean_pairwise_correlation=mean_correlation,
        variance_inflation_factor=inflation,
        effective_row_count=effective_sample_size(row_count, inflation),
        pair_count=len(correlations),
    )
