r"""Encompassing test for a fitted forecast nested in a zero forecast.

Comparing a fitted model against a zero forecast under squared-error loss asks
a question the model cannot win fairly.  Write :math:`d_{t} = \hat y_t^2 - 2
r_t \hat y_t` for the differential against a zero benchmark.  Its expectation is

.. math:: \mathbb{E}[d] = \mathbb{E}[\hat y^2] - 2\,\mathrm{cov}(r, \hat y),

so a forecast carrying genuine but small covariance with the target still loses
whenever its estimation variance exceeds twice that covariance.  Under the null
that the population forecast *is* zero, the fitted forecast is not zero---it is
noise---and the Diebold--Mariano test therefore rejects towards the benchmark
by construction.  Clark and West (2007) remove exactly that term:

.. math:: f_t = d^{\text{MSE}}_t + (\hat y_t - 0)^2 \;\Longrightarrow\;
    f_t = 2 r_t \hat y_t,

whose expectation is zero under the null and positive whenever the forecast
covaries with the target, however weakly.  The two tests answer different
questions and this study reports both: Diebold--Mariano asks whether the
forecast would have been worth using, Clark and West asks whether it contains
any predictive information at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from bist_predict.research.inference.hac import bartlett_long_run_variance

__all__ = [
    "ClarkWestResult",
    "clark_west",
    "encompassing_adjustment",
]


@dataclass(frozen=True)
class ClarkWestResult:
    """A one-sided test that a forecast carries predictive information."""

    candidate: str
    benchmark: str
    observation_count: int
    mean_adjusted_differential: float
    standard_error: float
    statistic: float
    p_value: float
    lags: int
    aggregation: str
    convention: str = "f = 2 * target * forecast; a positive mean favours the candidate"

    @property
    def verdict(self) -> str:
        """Return whether the forecast shows predictive content at 5%."""
        return "predictive_content" if self.p_value <= 0.05 else "no_predictive_content"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe record of the test."""
        return {
            "candidate": self.candidate,
            "benchmark": self.benchmark,
            "observation_count": self.observation_count,
            "mean_adjusted_differential": self.mean_adjusted_differential,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "lags": self.lags,
            "aggregation": self.aggregation,
            "convention": self.convention,
            "verdict": self.verdict,
        }


def encompassing_adjustment(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    benchmark: str,
    aggregation: str = "session",
) -> pd.Series:
    r"""Return the Clark--West adjusted differential for a nested comparison.

    The benchmark must be the zero forecast: the adjustment
    :math:`f_t = 2 r_t \hat y_t` is derived under that restriction, and applying
    it to a non-nested pair would silently test something else.  This function
    verifies the restriction rather than documenting it.
    """
    required = {"date", "ticker", "model_name", "target", "predicted_return"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {', '.join(missing)}")
    if candidate == benchmark:
        raise ValueError("candidate and benchmark must be different models")

    benchmark_rows = predictions.loc[predictions["model_name"] == benchmark]
    if benchmark_rows.empty:
        raise ValueError(f"predictions contain no rows for model: {benchmark}")
    if not np.allclose(
        benchmark_rows["predicted_return"].to_numpy(dtype=np.float64), 0.0, atol=1e-12
    ):
        raise ValueError(
            "the Clark-West adjustment requires a zero-forecast benchmark; "
            f"model {benchmark} forecasts non-zero values"
        )

    rows = predictions.loc[predictions["model_name"] == candidate]
    if rows.empty:
        raise ValueError(f"predictions contain no rows for model: {candidate}")
    if rows.duplicated(["date", "ticker"]).any():
        raise ValueError(f"model has duplicate date-ticker rows: {candidate}")

    adjusted = (
        2.0
        * rows["target"].to_numpy(dtype=np.float64)
        * rows["predicted_return"].to_numpy(dtype=np.float64)
    )
    working = pd.DataFrame(
        {
            "date": rows["date"].to_numpy(),
            "ticker": rows["ticker"].to_numpy(),
            "adjusted": adjusted,
        }
    )
    if aggregation == "row":
        ordered = working.sort_values(["date", "ticker"], kind="stable")
        index = pd.Index(
            ordered["date"].astype(str) + "|" + ordered["ticker"].astype(str),
            name="sample_id",
        )
        return pd.Series(
            ordered["adjusted"].to_numpy(dtype=np.float64), index=index, name="adjusted"
        )
    if aggregation != "session":
        raise ValueError("aggregation must be 'session' or 'row'")
    by_session = working.groupby("date", sort=True)["adjusted"].mean()
    by_session.name = "adjusted"
    return by_session


def clark_west(
    adjusted: pd.Series | np.ndarray | list[float],
    *,
    candidate: str,
    benchmark: str,
    lags: int = 0,
    aggregation: str = "session",
) -> ClarkWestResult:
    r"""Test one-sided that an adjusted differential has a positive mean.

    Clark and West (2007) show the statistic is approximately standard normal
    under the null even though the models are nested, which is precisely the
    case where the Diebold--Mariano reference distribution fails.  The test is
    one-sided by construction: negative covariance with the target is evidence
    of no predictive content, not evidence against the alternative.
    """
    series = np.asarray(adjusted, dtype=np.float64).ravel()
    count = series.size
    if count < 2:
        raise ValueError("Clark-West requires at least two observations")
    if not np.isfinite(series).all():
        raise ValueError("adjusted differentials must be finite")
    if lags < 0:
        raise ValueError("lags must be non-negative")
    if lags >= count:
        raise ValueError("lags must be smaller than the observation count")

    mean = float(series.mean())
    long_run_variance = bartlett_long_run_variance(series, lags=lags)
    scale = float(np.sqrt(long_run_variance / count))
    reference = float(np.mean(np.abs(series)))
    if scale <= 1e-12 * max(reference, 1e-12):
        raise ValueError("adjusted differential has no usable variation")
    statistic = mean / scale
    return ClarkWestResult(
        candidate=candidate,
        benchmark=benchmark,
        observation_count=count,
        mean_adjusted_differential=mean,
        standard_error=scale,
        statistic=statistic,
        p_value=float(stats.norm.sf(statistic)),
        lags=lags,
        aggregation=aggregation,
    )
