r"""A joint test for the best of a correlated configuration grid.

The deflated Sharpe ratio corrects the best of :math:`N` trials by an
*assumption* about how the trials disperse.  When the trials are 72 re-runs of
one experiment on overlapping sessions, that assumption is the weakest link:
neighbouring configurations share most of their data and their Sharpe ratios
move almost in lockstep, so the effective number of independent searches is far
below 72 and nobody knows by how much without measuring it.

This module measures it.  Every configuration's session-by-session net returns
are recentred to the joint null of zero expected return and resampled with a
*single* stationary-bootstrap index draw applied to all of them at once, which
preserves both the serial dependence within a configuration and the
cross-sectional dependence between configurations.  The maximum Sharpe ratio
across the grid is recorded on every draw.  The resulting distribution is the
object the deflated Sharpe ratio approximates, and it supplies

* an exact bootstrap p-value for the best configuration,
* a critical value at any level, rather than an expectation, and
* the number of genuinely independent trials the grid behaved like, obtained by
  inverting the False Strategy Theorem against the simulated expected maximum.

Politis and Romano (1994) is the resampling scheme; the joint-draw construction
is the multivariate form White (2000) uses for the Reality Check, applied here
to a Sharpe ratio rather than to a loss differential.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bist_predict.research.inference.detectability import false_strategy_quantile
from bist_predict.research.inference.snooping import (
    stationary_block_length,
    stationary_bootstrap_indices,
)

__all__ = ["JointSearchResult", "joint_search_test"]


@dataclass(frozen=True)
class JointSearchResult:
    """The null distribution of the best Sharpe ratio in a correlated grid."""

    trial_count: int
    session_count: int
    replications: int
    block_length: float
    seed: int
    best_trial: str
    best_per_period_sharpe: float
    null_expected_maximum: float
    null_quantile_95: float
    p_value: float
    mean_pairwise_correlation: float
    independent_equivalent_trials: float
    null_maximum_dispersion: float

    @property
    def verdict(self) -> str:
        """Return whether the best configuration survives the joint null at 5%."""
        return "best_trial_is_within_search_noise" if self.p_value > 0.05 else "best_trial_survives"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the joint test."""
        return {
            "trial_count": self.trial_count,
            "session_count": self.session_count,
            "replications": self.replications,
            "block_length": self.block_length,
            "seed": self.seed,
            "best_trial": self.best_trial,
            "best_per_period_sharpe": self.best_per_period_sharpe,
            "null_expected_maximum": self.null_expected_maximum,
            "null_quantile_95": self.null_quantile_95,
            "p_value": self.p_value,
            "mean_pairwise_correlation": self.mean_pairwise_correlation,
            "independent_equivalent_trials": self.independent_equivalent_trials,
            "null_maximum_dispersion": self.null_maximum_dispersion,
            "null": "every configuration has zero expected net return",
            "resampling": "stationary_bootstrap_synchronised_across_trials",
            "verdict": self.verdict,
        }


def _sharpe_along_axis(values: np.ndarray) -> np.ndarray:
    """Return per-period Sharpe ratios along the session axis of a 3-D block."""
    mean = values.mean(axis=1)
    deviation = values - mean[:, None, :]
    variance = np.square(deviation).sum(axis=1) / (values.shape[1] - 1)
    scale = np.sqrt(variance)
    return np.where(scale > 0.0, mean / np.where(scale > 0.0, scale, 1.0), 0.0)


def _independent_equivalent_trials(
    expected_maximum: float, dispersion: float, *, trial_count: int
) -> float:
    r"""Return the independent trial count matching a simulated expected maximum.

    The False Strategy Theorem states that the best of :math:`N` independent
    skill-free trials with Sharpe dispersion :math:`\sqrt V` is expected to show
    :math:`\sqrt V\,q(N)`.  Reading that identity backwards with the *simulated*
    expected maximum on the left returns the :math:`N` the correlated grid
    behaved like.  Because :math:`q` increases strictly, bisection is exact to
    tolerance.

    The dispersion must be the sampling standard deviation of a *single* trial's
    statistic across resamples, not the spread across trials within one
    resample.  The two coincide only when the trials are independent: as they
    become perfectly correlated, every trial moves together, the within-resample
    spread collapses towards zero and a ratio built on it would report the grid
    as *more* independent the more dependent it actually is.
    """
    if dispersion <= 0.0:
        return float("nan")
    target = expected_maximum / dispersion
    if target <= false_strategy_quantile(1.0 + 1e-6):
        return 1.0
    low, high = 1.0 + 1e-6, float(max(trial_count, 2))
    while false_strategy_quantile(high) < target and high < 1e6:
        high *= 2.0
    if false_strategy_quantile(high) < target:
        return float(high)
    for _ in range(200):
        middle = math.sqrt(low * high)
        if false_strategy_quantile(middle) < target:
            low = middle
        else:
            high = middle
    return float(math.sqrt(low * high))


def joint_search_test(
    returns: pd.DataFrame,
    *,
    replications: int = 10_000,
    seed: int = 42,
    block_length: float | None = None,
) -> JointSearchResult:
    """Bootstrap the distribution of the best Sharpe ratio across a grid.

    ``returns`` holds one column per configuration and one row per session, on
    the sessions every configuration evaluated.  Each column is recentred to
    mean zero, which imposes the joint null that no configuration earns a
    positive expected return, and every replication applies the *same* index
    draw to every column so the grid's co-movement survives the resample.
    """
    if returns.shape[1] < 2:
        raise ValueError("a joint search test requires at least two configurations")
    values = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("the trial return matrix must be finite")
    count = values.shape[0]
    if count < 8:
        raise ValueError("a joint search test requires at least eight sessions")
    if replications < 100:
        raise ValueError("a joint search test requires at least one hundred replications")

    observed_mean = values.mean(axis=0)
    observed_scale = values.std(axis=0, ddof=1)
    observed_sharpe = np.where(observed_scale > 0.0, observed_mean / observed_scale, 0.0)
    centred = values - observed_mean

    chosen_block = stationary_block_length(centred) if block_length is None else float(block_length)
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(
        count, block_length=chosen_block, replications=replications, rng=rng
    )
    draws = _sharpe_along_axis(centred[indices])
    maxima = draws.max(axis=1)

    best_position = int(np.argmax(observed_sharpe))
    best_sharpe = float(observed_sharpe[best_position])
    correlation = np.corrcoef(values, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    dispersion = float(np.mean(draws.std(axis=0, ddof=1)))
    return JointSearchResult(
        trial_count=int(values.shape[1]),
        session_count=count,
        replications=replications,
        block_length=chosen_block,
        seed=seed,
        best_trial=str(returns.columns[best_position]),
        best_per_period_sharpe=best_sharpe,
        null_expected_maximum=float(maxima.mean()),
        null_quantile_95=float(np.quantile(maxima, 0.95)),
        p_value=float(np.mean(maxima >= best_sharpe)),
        mean_pairwise_correlation=float(np.mean(upper)) if upper.size else float("nan"),
        independent_equivalent_trials=_independent_equivalent_trials(
            float(maxima.mean()), dispersion, trial_count=int(values.shape[1])
        ),
        null_maximum_dispersion=float(maxima.std(ddof=1)),
    )
