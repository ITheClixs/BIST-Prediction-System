"""Monte Carlo size and power for the equal-predictive-accuracy test.

Everything in this module answers one question: when the evaluation stack says
``p = 0.03``, how often is that a false alarm?  The answer is not a property of
the Diebold--Mariano statistic in the abstract.  It is a property of the
statistic applied to *these* data, whose rows are cross-sectionally correlated,
and it can only be obtained by running the estimator that the study actually
uses on panels whose truth is known.

The loss differentials are formed here in closed form rather than through the
pandas path used on real predictions.  ``test_matches_the_production_estimator``
asserts the two agree to floating-point tolerance, so the speed is bought
without letting the calibration drift away from the code it is calibrating.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from bist_predict.research.inference.forecast_tests import diebold_mariano
from bist_predict.research.inference.nested import clark_west
from bist_predict.research.simulation.panels import (
    PanelDesign,
    forecast_family,
    noise_forecast_family,
    simulate_panel,
)

__all__ = [
    "NestedNullCell",
    "RejectionRate",
    "SizePowerCell",
    "asymptotic_pooled_size",
    "loss_differentials",
    "minimum_detectable_effect",
    "nested_null_cell",
    "rejection_rate",
    "size_power_cell",
]


@dataclass(frozen=True)
class RejectionRate:
    """A Monte Carlo rejection frequency with an exact binomial interval.

    Reporting a simulated rate without an interval invites the reader to treat
    ``0.061`` and ``0.050`` as different when a thousand replications cannot
    separate them.  The interval is Clopper--Pearson, which is conservative and
    stays valid at the boundaries where a normal approximation fails.
    """

    trials: int
    rejections: int
    rate: float
    lower: float
    upper: float
    confidence_level: float

    def covers(self, value: float) -> bool:
        """Return whether the interval contains a stated rate."""
        return bool(self.lower <= value <= self.upper)

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe record of the rate."""
        return {
            "trials": self.trials,
            "rejections": self.rejections,
            "rate": self.rate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
        }


def rejection_rate(
    rejections: int, trials: int, *, confidence_level: float = 0.95
) -> RejectionRate:
    """Return a Clopper--Pearson interval for a simulated rejection frequency."""
    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= rejections <= trials:
        raise ValueError("rejections must lie between zero and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    tail = (1.0 - confidence_level) / 2.0
    lower = (
        0.0 if rejections == 0 else float(stats.beta.ppf(tail, rejections, trials - rejections + 1))
    )
    upper = (
        1.0
        if rejections == trials
        else float(stats.beta.ppf(1.0 - tail, rejections + 1, trials - rejections))
    )
    return RejectionRate(
        trials=trials,
        rejections=rejections,
        rate=rejections / trials,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def loss_differentials(returns: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    r"""Return the squared-error differential of a forecast against zero.

    With a benchmark that predicts zero the differential collapses to

    .. math:: d_{i,t} = (r_{i,t} - \hat y_{i,t})^2 - r_{i,t}^2
        = \hat y_{i,t}^2 - 2 r_{i,t}\hat y_{i,t},

    which is the same quantity ``squared_error_differential`` computes from a
    prediction frame, in the same sign convention: positive favours the
    benchmark.
    """
    if returns.shape != forecast.shape:
        raise ValueError("returns and forecast must share a shape")
    return np.square(forecast) - 2.0 * returns * forecast


def asymptotic_pooled_size(unit_count: int, correlation: float, *, alpha: float = 0.05) -> float:
    r"""Return the limiting size of a pooled test at nominal level ``alpha``.

    A test that divides by :math:`\sqrt{\hat\sigma^2 / (kT)}` when the true
    variance of the mean is :math:`\sigma^2(1 + (k-1)\bar\rho)/(kT)` produces a
    statistic that is :math:`\sqrt{1 + (k-1)\bar\rho}` times too large.  Its
    two-sided rejection probability under the null therefore converges to

    .. math::
        \alpha^{\ast}(k, \bar\rho)
        = 2\Phi\!\left(-\frac{z_{1-\alpha/2}}{\sqrt{1 + (k-1)\bar\rho}}\right),

    which is Proposition~1 of the accompanying manuscript.  It is stated here
    rather than in the manuscript alone so the simulation can be checked against
    it directly.
    """
    if unit_count < 1:
        raise ValueError("unit_count must be positive")
    if not -1.0 < correlation <= 1.0:
        raise ValueError("correlation must lie in (-1, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    # A negative average correlation is admissible and predicts an *under*sized
    # pooled test rather than an oversized one. It is not clipped away here,
    # because a prediction that only ever errs in one direction cannot be
    # falsified by the simulation it is supposed to be checked against.
    inflation = 1.0 + (unit_count - 1) * correlation
    if inflation <= 0.0:
        raise ValueError("the implied variance inflation factor is not positive")
    quantile = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(2.0 * stats.norm.cdf(-quantile / np.sqrt(inflation)))


@dataclass(frozen=True)
class SizePowerCell:
    """One design point: how often each test fires at a known effect size."""

    design: PanelDesign
    population_r_squared: float
    population_covariance_ratio: float
    replications: int
    alpha: float
    session_rejection: RejectionRate
    row_rejection: RejectionRate
    session_superiority: RejectionRate
    row_superiority: RejectionRate
    clark_west_session: RejectionRate
    mean_loss_differential_correlation: float
    predicted_row_size: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the cell."""
        return {
            "design": self.design.to_dict(),
            "population_r_squared": self.population_r_squared,
            "population_covariance_ratio": self.population_covariance_ratio,
            "replications": self.replications,
            "alpha": self.alpha,
            "session_rejection": self.session_rejection.to_dict(),
            "row_rejection": self.row_rejection.to_dict(),
            "session_superiority": self.session_superiority.to_dict(),
            "row_superiority": self.row_superiority.to_dict(),
            "clark_west_session": self.clark_west_session.to_dict(),
            "mean_loss_differential_correlation": self.mean_loss_differential_correlation,
            "predicted_row_size": self.predicted_row_size,
        }


def _mean_pairwise_correlation(values: np.ndarray) -> float:
    """Return the mean off-diagonal correlation of a sessions-by-units panel."""
    if values.shape[1] < 2:
        return float("nan")
    if np.any(np.ptp(values, axis=0) == 0.0):
        return float("nan")
    matrix = np.corrcoef(values, rowvar=False)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.mean(upper))


def size_power_cell(
    design: PanelDesign,
    *,
    population_r_squared: float,
    replications: int,
    seed: int,
    alpha: float = 0.05,
) -> SizePowerCell:
    """Run one design point of the size-and-power experiment.

    Each replication draws a fresh panel, builds a single forecast of the
    requested population accuracy, and applies the Diebold--Mariano test twice:
    once to session-aggregated differentials, and once to the pooled rows.  The
    second is not a straw man.  It is what a study reports when it treats a
    panel of ``k T`` rows as ``k T`` observations, which is the default in this
    literature.
    """
    if replications < 2:
        raise ValueError("replications must be at least two")
    generator = np.random.default_rng(seed)
    session_rejections = 0
    row_rejections = 0
    session_superior = 0
    row_superior = 0
    clark_west_rejections = 0
    correlations: list[float] = []
    for _ in range(replications):
        panel = simulate_panel(design, generator)
        forecast = forecast_family(
            panel,
            population_r_squared=population_r_squared,
            family_size=1,
            rng=generator,
        )[0]
        differential = loss_differentials(panel.returns, forecast)
        session = diebold_mariano(
            differential.mean(axis=1), candidate="simulated", benchmark="zero_return"
        )
        row = diebold_mariano(
            differential.reshape(-1),
            candidate="simulated",
            benchmark="zero_return",
            aggregation="row",
        )
        nested = clark_west(
            (2.0 * panel.returns * forecast).mean(axis=1),
            candidate="simulated",
            benchmark="zero_return",
        )
        session_rejections += int(session.p_value <= alpha)
        row_rejections += int(row.p_value <= alpha)
        session_superior += int(session.verdict == "candidate_better")
        row_superior += int(row.verdict == "candidate_better")
        clark_west_rejections += int(nested.p_value <= alpha)
        measured = _mean_pairwise_correlation(differential)
        if np.isfinite(measured):
            correlations.append(measured)
    return SizePowerCell(
        design=design,
        population_r_squared=population_r_squared,
        # The two tests answer to different effect sizes on the same forecast.
        # Diebold-Mariano fires on the population R-squared, which nets the
        # forecast's own variance off its covariance with the target; Clark-West
        # fires on the covariance alone. Recording both makes the two power
        # curves comparable on one axis.
        population_covariance_ratio=population_r_squared + design.forecast_variance_ratio,
        replications=replications,
        alpha=alpha,
        session_rejection=rejection_rate(session_rejections, replications),
        row_rejection=rejection_rate(row_rejections, replications),
        session_superiority=rejection_rate(session_superior, replications),
        row_superiority=rejection_rate(row_superior, replications),
        clark_west_session=rejection_rate(clark_west_rejections, replications),
        mean_loss_differential_correlation=(
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        predicted_row_size=asymptotic_pooled_size(
            design.unit_count,
            float(np.mean(correlations)) if correlations else design.target_correlation,
            alpha=alpha,
        ),
    )


@dataclass(frozen=True)
class NestedNullCell:
    """What each test does when the restricted model is the true one.

    ``diebold_mariano_against_candidate`` is the quantity of interest.  It is
    the rate at which a squared-error comparison declares a *correctly
    specified* zero forecast significantly better than a fitted model whose
    only defect is that it was estimated.  A reader who interprets such a
    rejection as evidence about the market has misread the test.
    """

    design: PanelDesign
    variance_ratio: float
    replications: int
    alpha: float
    clark_west_session: RejectionRate
    clark_west_row: RejectionRate
    diebold_mariano_against_candidate: RejectionRate
    diebold_mariano_for_candidate: RejectionRate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the cell."""
        return {
            "design": self.design.to_dict(),
            "variance_ratio": self.variance_ratio,
            "replications": self.replications,
            "alpha": self.alpha,
            "clark_west_session": self.clark_west_session.to_dict(),
            "clark_west_row": self.clark_west_row.to_dict(),
            "diebold_mariano_against_candidate": (self.diebold_mariano_against_candidate.to_dict()),
            "diebold_mariano_for_candidate": self.diebold_mariano_for_candidate.to_dict(),
        }


def nested_null_cell(
    design: PanelDesign,
    *,
    variance_ratio: float,
    replications: int,
    seed: int,
    alpha: float = 0.05,
) -> NestedNullCell:
    """Measure both tests when the fitted forecast is pure estimation noise.

    The Clark--West null holds exactly here and its rejection rate is a size.
    The Diebold--Mariano null does not hold, and its rejection rate towards the
    benchmark is not a size but a measurement of the penalty a squared-error
    comparison charges for having estimated anything at all.
    """
    if replications < 2:
        raise ValueError("replications must be at least two")
    generator = np.random.default_rng(seed)
    session_hits = 0
    row_hits = 0
    against = 0
    favour = 0
    for _ in range(replications):
        panel = simulate_panel(design, generator)
        forecast = noise_forecast_family(
            panel, variance_ratio=variance_ratio, family_size=1, rng=generator
        )[0]
        adjusted = 2.0 * panel.returns * forecast
        session_hits += int(
            clark_west(
                adjusted.mean(axis=1), candidate="simulated", benchmark="zero_return"
            ).p_value
            <= alpha
        )
        row_hits += int(
            clark_west(
                adjusted.reshape(-1),
                candidate="simulated",
                benchmark="zero_return",
                aggregation="row",
            ).p_value
            <= alpha
        )
        differential = loss_differentials(panel.returns, forecast)
        result = diebold_mariano(
            differential.mean(axis=1), candidate="simulated", benchmark="zero_return"
        )
        against += int(result.verdict == "benchmark_better")
        favour += int(result.verdict == "candidate_better")
    return NestedNullCell(
        design=design,
        variance_ratio=variance_ratio,
        replications=replications,
        alpha=alpha,
        clark_west_session=rejection_rate(session_hits, replications),
        clark_west_row=rejection_rate(row_hits, replications),
        diebold_mariano_against_candidate=rejection_rate(against, replications),
        diebold_mariano_for_candidate=rejection_rate(favour, replications),
    )


def minimum_detectable_effect(
    cells: Sequence[SizePowerCell],
    *,
    power: float = 0.80,
    test: str = "diebold_mariano",
    scale: str = "r_squared",
) -> float | None:
    """Return the smallest effect the design detects at a stated power.

    The crossing is located by linear interpolation between the two adjacent
    grid points that bracket it, on the session-aggregated one-sided rate.  A
    ``None`` return means the grid never reached the requested power, which is
    reported as such rather than extrapolated: an effect size read off beyond
    the simulated range would be a guess wearing a decimal point.

    ``scale`` selects the axis the answer is expressed on.  A squared-error
    comparison is powered against the population :math:`R^2`; an encompassing
    test is powered against the covariance ratio, and quoting the second on the
    first's axis would credit it with detecting effects that are negative there.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie in (0, 1)")
    rates = {
        "diebold_mariano": lambda cell: cell.session_superiority.rate,
        "clark_west": lambda cell: cell.clark_west_session.rate,
    }
    axes = {
        "r_squared": lambda cell: cell.population_r_squared,
        "covariance_ratio": lambda cell: cell.population_covariance_ratio,
    }
    if test not in rates:
        raise ValueError("test must be 'diebold_mariano' or 'clark_west'")
    if scale not in axes:
        raise ValueError("scale must be 'r_squared' or 'covariance_ratio'")
    rate_of, effect_of = rates[test], axes[scale]
    ordered = sorted(cells, key=effect_of)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        low_rate, high_rate = rate_of(lower), rate_of(upper)
        if low_rate < power <= high_rate:
            if high_rate == low_rate:
                return float(effect_of(upper))
            weight = (power - low_rate) / (high_rate - low_rate)
            return float(effect_of(lower) + weight * (effect_of(upper) - effect_of(lower)))
    return None
