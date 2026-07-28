"""Sharpe-ratio inference that accounts for non-normality, memory and search.

A Sharpe ratio quoted without an interval is a point estimate of a random
variable whose sampling distribution depends on the higher moments of the
return series, on its autocorrelation, and on how many strategy configurations
were examined before this one was reported.  The estimators here follow Lo
(2002) for the first two effects and Bailey and Lopez de Prado (2012, 2014) for
the third.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from bist_predict.research.inference.hac import automatic_bartlett_bandwidth

__all__ = [
    "EULER_MASCHERONI",
    "SharpeInference",
    "annualisation_factor",
    "deflated_sharpe_ratio",
    "per_period_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_inference",
]

EULER_MASCHERONI = 0.5772156649015329
_RELATIVE_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class SharpeInference:
    """Every Sharpe quantity the report is allowed to quote, in one record."""

    observation_count: int
    periods_per_year: int
    per_period_sharpe: float
    annualised_sharpe: float
    autocorrelation_adjusted_annualised_sharpe: float
    autocorrelation_lags: int
    standard_error: float
    skewness: float
    kurtosis: float
    probabilistic_sharpe_ratio: float
    trial_count: int
    trial_sharpe_variance: float
    deflated_sharpe_threshold: float
    deflated_sharpe_ratio: float

    @property
    def verdict(self) -> str:
        """Return whether the deflated Sharpe clears the conventional 95% bar."""
        return "skill_not_established" if self.deflated_sharpe_ratio < 0.95 else "skill_established"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe record of the inference."""
        return {
            "observation_count": self.observation_count,
            "periods_per_year": self.periods_per_year,
            "per_period_sharpe": self.per_period_sharpe,
            "annualised_sharpe": self.annualised_sharpe,
            "autocorrelation_adjusted_annualised_sharpe": (
                self.autocorrelation_adjusted_annualised_sharpe
            ),
            "autocorrelation_lags": self.autocorrelation_lags,
            "standard_error": self.standard_error,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "probabilistic_sharpe_ratio": self.probabilistic_sharpe_ratio,
            "trial_count": self.trial_count,
            "trial_sharpe_variance": self.trial_sharpe_variance,
            "deflated_sharpe_threshold": self.deflated_sharpe_threshold,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "verdict": self.verdict,
        }


def _returns(values: Sequence[float] | np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size < 3:
        raise ValueError("Sharpe inference requires at least three return observations")
    if not np.isfinite(series).all():
        raise ValueError("returns must be finite")
    return series


def per_period_sharpe_ratio(returns: Sequence[float] | np.ndarray) -> float:
    """Return the per-period Sharpe ratio with a scale-relative variance guard.

    A constant series assembled from floating-point price ratios has a sample
    standard deviation near ``1e-16`` rather than exactly zero.  Guarding with
    ``std > 0`` therefore admits a division that produces a Sharpe ratio of
    order ``1e14``.  The guard compares the deviation against the scale of the
    returns themselves.
    """
    series = _returns(returns)
    deviation = float(np.std(series, ddof=1))
    scale = float(np.max(np.abs(series)))
    if deviation <= _RELATIVE_VARIANCE_FLOOR * max(scale, _RELATIVE_VARIANCE_FLOOR):
        return 0.0
    return float(np.mean(series) / deviation)


def annualisation_factor(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int,
    max_lags: int | None = None,
) -> float:
    r"""Return Lo's (2002) autocorrelation-aware annualisation factor.

    .. math::
        \hat\eta(q) = \frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\hat\rho_k}}

    Independent returns give :math:`\hat\eta(q) = \sqrt q`, recovering the
    conventional square-root rule.  Positive autocorrelation makes the honest
    factor smaller than :math:`\sqrt q`, so the conventional rule overstates the
    annualised Sharpe ratio.

    The sum runs to :math:`q - 1` in the population, but a sample of ``n``
    observations cannot estimate ``q - 1`` autocorrelations: at ``q = 252`` and
    ``n = 120`` the high lags are noise multiplied by a weight of order ``q``,
    and the factor becomes arbitrary.  Lags are therefore truncated at
    ``max_lags``, defaulting to the Newey and West (1994) bandwidth, with higher
    lags treated as zero.  The radicand is floored because a finite sample can
    still drive it non-positive.
    """
    series = _returns(returns)
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    horizon = int(periods_per_year)
    bandwidth = automatic_bartlett_bandwidth(series.size) if max_lags is None else int(max_lags)
    if bandwidth < 0:
        raise ValueError("max_lags must be non-negative")
    usable_lags = min(horizon - 1, series.size - 1, bandwidth)
    centred = series - series.mean()
    variance = float(np.dot(centred, centred))
    if variance <= 0.0:
        return float(math.sqrt(horizon))
    total = float(horizon)
    for lag in range(1, usable_lags + 1):
        autocorrelation = float(np.dot(centred[lag:], centred[:-lag]) / variance)
        total += 2.0 * (horizon - lag) * autocorrelation
    if total <= 0.0:
        return float(math.sqrt(horizon))
    return float(horizon / math.sqrt(total))


def sharpe_standard_error(per_period_sharpe: float, observation_count: int) -> float:
    r"""Return Lo's (2002) large-sample standard error for iid normal returns.

    .. math:: \mathrm{SE}(\hat{SR}) = \sqrt{(1 + \hat{SR}^2 / 2) / n}
    """
    if observation_count < 2:
        raise ValueError("standard error requires at least two observations")
    return float(math.sqrt((1.0 + 0.5 * per_period_sharpe**2) / observation_count))


def probabilistic_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.0,
) -> float:
    r"""Return the probability that the true per-period Sharpe exceeds ``threshold``.

    .. math::
        \widehat{PSR}(SR^{*}) = \Phi\!\left[
            \frac{(\hat{SR} - SR^{*})\sqrt{n-1}}
                 {\sqrt{1 - \hat\gamma_3 \hat{SR} + \frac{\hat\gamma_4 - 1}{4}\hat{SR}^2}}
        \right]

    ``threshold`` and the estimate are both **per period**.  Passing an
    annualised Sharpe ratio here silently inflates the result, so the caller is
    responsible for supplying a per-period benchmark.
    """
    series = _returns(returns)
    estimate = per_period_sharpe_ratio(series)
    skewness = float(stats.skew(series, bias=True))
    kurtosis = float(stats.kurtosis(series, fisher=False, bias=True))
    variance = 1.0 - skewness * estimate + 0.25 * (kurtosis - 1.0) * estimate**2
    if variance <= 0.0:
        raise ValueError("estimated Sharpe variance is non-positive; the sample is degenerate")
    statistic = (estimate - threshold) * math.sqrt(series.size - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(statistic))


def deflated_sharpe_threshold(trial_count: int, trial_sharpe_variance: float) -> float:
    r"""Return the expected maximum Sharpe ratio under the null of no skill.

    .. math::
        SR^{*}_0 = \sqrt{V[\hat{SR}]}\left[
            (1 - \gamma)\Phi^{-1}\!\left(1 - \frac{1}{N}\right)
            + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{N e}\right)\right]

    with :math:`\gamma` the Euler-Mascheroni constant.  ``N`` counts the
    strategy configurations actually examined and ``V[\hat{SR}]`` is the
    variance of the per-period Sharpe estimates across them.  A single trial
    gives a threshold of zero, which is the undeflated case.
    """
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    if trial_sharpe_variance < 0.0 or not np.isfinite(trial_sharpe_variance):
        raise ValueError("trial_sharpe_variance must be non-negative and finite")
    if trial_count == 1 or trial_sharpe_variance == 0.0:
        return 0.0
    first = float(stats.norm.ppf(1.0 - 1.0 / trial_count))
    second = float(stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e)))
    return float(
        math.sqrt(trial_sharpe_variance)
        * ((1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)
    )


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    trial_count: int,
    trial_sharpe_variance: float,
) -> float:
    """Return the probabilistic Sharpe ratio measured against the search threshold."""
    threshold = deflated_sharpe_threshold(trial_count, trial_sharpe_variance)
    return probabilistic_sharpe_ratio(returns, threshold=threshold)


def sharpe_inference(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int,
    trial_count: int,
    trial_sharpe_variance: float,
) -> SharpeInference:
    """Assemble every Sharpe quantity the report quotes from one return series."""
    series = _returns(returns)
    estimate = per_period_sharpe_ratio(series)
    factor = annualisation_factor(series, periods_per_year=periods_per_year)
    lag_bandwidth = automatic_bartlett_bandwidth(series.size)
    threshold = deflated_sharpe_threshold(trial_count, trial_sharpe_variance)
    return SharpeInference(
        observation_count=int(series.size),
        periods_per_year=int(periods_per_year),
        per_period_sharpe=estimate,
        annualised_sharpe=float(estimate * math.sqrt(periods_per_year)),
        autocorrelation_adjusted_annualised_sharpe=float(estimate * factor),
        autocorrelation_lags=lag_bandwidth,
        standard_error=sharpe_standard_error(estimate, series.size),
        skewness=float(stats.skew(series, bias=True)),
        kurtosis=float(stats.kurtosis(series, fisher=False, bias=True)),
        probabilistic_sharpe_ratio=probabilistic_sharpe_ratio(series, threshold=0.0),
        trial_count=int(trial_count),
        trial_sharpe_variance=float(trial_sharpe_variance),
        deflated_sharpe_threshold=threshold,
        deflated_sharpe_ratio=probabilistic_sharpe_ratio(series, threshold=threshold),
    )
