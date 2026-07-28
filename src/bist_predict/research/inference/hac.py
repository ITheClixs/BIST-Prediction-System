"""Autocorrelation-consistent variance estimation for dependent research series.

Every estimator here is written for a mean statistic computed on a single time
series of per-session observations.  The formulas follow Newey and West (1987)
for the Bartlett-kernel long-run variance and Newey and West (1994) for the
automatic bandwidth rule.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "automatic_bartlett_bandwidth",
    "bartlett_long_run_variance",
    "mean_standard_error",
    "sample_autocovariance",
]


def _as_series(values: object, *, name: str) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if not np.isfinite(series).all():
        raise ValueError(f"{name} must be finite")
    return series


def sample_autocovariance(values: object, lag: int) -> float:
    r"""Return the biased sample autocovariance at ``lag``.

    Uses the ``1/n`` normalisation shared by Newey and West (1987) and by
    Diebold and Mariano (1995):

    .. math:: \gamma_k = n^{-1} \sum_{t=k+1}^{n} (d_t - \bar d)(d_{t-k} - \bar d)
    """
    series = _as_series(values, name="values")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    count = series.size
    if lag >= count:
        raise ValueError("lag must be smaller than the observation count")
    centred = series - series.mean()
    if lag == 0:
        return float(np.dot(centred, centred) / count)
    return float(np.dot(centred[lag:], centred[:-lag]) / count)


def automatic_bartlett_bandwidth(count: int) -> int:
    r"""Return the Newey and West (1994) rule-of-thumb Bartlett bandwidth.

    .. math:: m = \lfloor 4 (n/100)^{2/9} \rfloor

    The result is clipped to ``[0, n - 1]`` so it stays a usable lag count.
    """
    if count < 1:
        raise ValueError("count must be positive")
    bandwidth = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    return int(min(max(bandwidth, 0), count - 1))


def bartlett_long_run_variance(values: object, *, lags: int | None = None) -> float:
    r"""Return the Bartlett-kernel long-run variance of a dependent series.

    .. math::
        \hat\Omega = \gamma_0 + 2 \sum_{k=1}^{m} \left(1 - \frac{k}{m+1}\right) \gamma_k

    ``lags=None`` selects :func:`automatic_bartlett_bandwidth`.  ``lags=0``
    reduces to the plain sample variance, which is the Diebold and Mariano
    (1995) choice for a one-session forecast horizon.

    The Bartlett weights guarantee a non-negative estimate in population, but a
    finite sample can still return a small negative value; it is clipped to zero
    because a variance estimate must not be negative.
    """
    series = _as_series(values, name="values")
    count = series.size
    bandwidth = automatic_bartlett_bandwidth(count) if lags is None else int(lags)
    if bandwidth < 0:
        raise ValueError("lags must be non-negative")
    if bandwidth >= count:
        raise ValueError("lags must be smaller than the observation count")
    total = sample_autocovariance(series, 0)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        total += 2.0 * weight * sample_autocovariance(series, lag)
    return float(max(total, 0.0))


def mean_standard_error(values: object, *, lags: int | None = None) -> float:
    """Return the autocorrelation-consistent standard error of the sample mean."""
    series = _as_series(values, name="values")
    long_run_variance = bartlett_long_run_variance(series, lags=lags)
    return float(np.sqrt(long_run_variance / series.size))
