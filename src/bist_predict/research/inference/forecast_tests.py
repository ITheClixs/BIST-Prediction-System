"""Equal-predictive-accuracy testing for saved out-of-sample predictions.

The test implemented here is Diebold and Mariano (1995) with the small-sample
modification of Harvey, Leybourne and Newbold (1997).  Loss differentials are
aggregated to one value per trading session before testing, because the panel's
same-session rows are strongly cross-correlated and are not independent draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from bist_predict.research.inference.hac import bartlett_long_run_variance

__all__ = [
    "DieboldMarianoResult",
    "LOSS_DIFFERENTIAL_CONVENTION",
    "diebold_mariano",
    "squared_error_differential",
]

LOSS_DIFFERENTIAL_CONVENTION = (
    "d = loss(candidate) - loss(benchmark); a negative mean favours the candidate"
)

Aggregation = Literal["session", "row"]


@dataclass(frozen=True)
class DieboldMarianoResult:
    """One equal-predictive-accuracy test rendered with its sign convention."""

    candidate: str
    benchmark: str
    observation_count: int
    mean_differential: float
    standard_error: float
    statistic: float
    p_value: float
    horizon: int
    lags: int
    aggregation: str
    convention: str = LOSS_DIFFERENTIAL_CONVENTION

    @property
    def verdict(self) -> str:
        """Return the direction of the comparison, not the sign of a number.

        Two tests in one table can carry opposite sign conventions.  Comparing
        verdicts rather than raw statistics removes that failure mode.
        """
        if self.p_value > 0.05:
            return "indistinguishable"
        return "candidate_better" if self.mean_differential < 0.0 else "benchmark_better"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe record of the test."""
        return {
            "candidate": self.candidate,
            "benchmark": self.benchmark,
            "observation_count": self.observation_count,
            "mean_differential": self.mean_differential,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "horizon": self.horizon,
            "lags": self.lags,
            "aggregation": self.aggregation,
            "convention": self.convention,
            "verdict": self.verdict,
        }


def squared_error_differential(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    benchmark: str,
    aggregation: Aggregation = "session",
) -> pd.Series:
    """Return squared-error loss differentials for two models on shared rows.

    ``aggregation="session"`` averages the differential across the tickers
    observed on each date and returns one value per date.  ``aggregation="row"``
    keeps every panel row, which overstates the independent observation count
    and exists only so the inflation can be measured and reported.
    """
    required = {"date", "ticker", "model_name", "target", "predicted_return"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {', '.join(missing)}")
    if candidate == benchmark:
        raise ValueError("candidate and benchmark must be different models")

    frames: dict[str, pd.DataFrame] = {}
    for name in (candidate, benchmark):
        rows = predictions.loc[predictions["model_name"] == name]
        if rows.empty:
            raise ValueError(f"predictions contain no rows for model: {name}")
        if rows.duplicated(["date", "ticker"]).any():
            raise ValueError(f"model has duplicate date-ticker rows: {name}")
        frames[name] = rows

    merged = frames[candidate].merge(
        frames[benchmark][["date", "ticker", "target", "predicted_return"]],
        on=["date", "ticker"],
        how="inner",
        suffixes=("_candidate", "_benchmark"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("candidate and benchmark share no evaluated rows")
    if not np.allclose(
        merged["target_candidate"].to_numpy(dtype=np.float64),
        merged["target_benchmark"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("candidate and benchmark disagree on the realised target")

    candidate_error = merged["target_candidate"] - merged["predicted_return_candidate"]
    benchmark_error = merged["target_benchmark"] - merged["predicted_return_benchmark"]
    merged["differential"] = np.square(candidate_error) - np.square(benchmark_error)

    if aggregation == "row":
        ordered = merged.sort_values(["date", "ticker"], kind="stable")
        index = pd.Index(
            ordered["date"].astype(str) + "|" + ordered["ticker"].astype(str),
            name="sample_id",
        )
        return pd.Series(
            ordered["differential"].to_numpy(dtype=np.float64), index=index, name="differential"
        )
    if aggregation != "session":
        raise ValueError("aggregation must be 'session' or 'row'")
    by_session = merged.groupby("date", sort=True)["differential"].mean()
    by_session.name = "differential"
    return by_session


def diebold_mariano(
    differential: pd.Series | np.ndarray | list[float],
    *,
    candidate: str,
    benchmark: str,
    horizon: int = 1,
    lags: int | None = None,
    aggregation: str = "session",
) -> DieboldMarianoResult:
    r"""Test equal predictive accuracy on a series of loss differentials.

    The statistic is

    .. math:: DM = \bar d \big/ \sqrt{\hat\Omega / n},

    with :math:`\hat\Omega` the Bartlett long-run variance.  Harvey, Leybourne
    and Newbold (1997) rescale it for finite samples,

    .. math::
        DM^{*} = DM \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}},

    and refer the result to a Student ``t`` distribution with ``n - 1`` degrees
    of freedom.  ``lags=None`` uses ``horizon - 1`` truncation lags, which is
    the original Diebold and Mariano choice; pass an explicit lag count or
    ``"automatic"`` bandwidth from :mod:`hac` for a serially robust variant.
    """
    series = np.asarray(differential, dtype=np.float64).ravel()
    count = series.size
    if count < 2:
        raise ValueError("Diebold-Mariano requires at least two observations")
    if not np.isfinite(series).all():
        raise ValueError("loss differentials must be finite")
    if horizon < 1:
        raise ValueError("horizon must be a positive number of sessions")
    truncation = horizon - 1 if lags is None else int(lags)
    if truncation < 0:
        raise ValueError("lags must be non-negative")
    if truncation >= count:
        raise ValueError("lags must be smaller than the observation count")

    mean_differential = float(series.mean())
    long_run_variance = bartlett_long_run_variance(series, lags=truncation)
    scale = float(np.sqrt(long_run_variance / count))
    reference = float(np.mean(np.abs(series)))
    if scale <= 1e-12 * max(reference, 1e-12):
        raise ValueError(
            "loss differential has no usable variation; the two models are numerically identical"
        )

    statistic = mean_differential / scale
    correction = np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count)
    if not np.isfinite(correction) or correction <= 0.0:
        raise ValueError("horizon is too long for the available observation count")
    corrected = float(statistic * correction)
    p_value = float(2.0 * stats.t.sf(abs(corrected), df=count - 1))
    return DieboldMarianoResult(
        candidate=candidate,
        benchmark=benchmark,
        observation_count=count,
        mean_differential=mean_differential,
        standard_error=scale,
        statistic=corrected,
        p_value=p_value,
        horizon=horizon,
        lags=truncation,
        aggregation=aggregation,
    )
