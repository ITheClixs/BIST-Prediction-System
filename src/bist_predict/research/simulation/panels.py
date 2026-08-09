r"""Synthetic return panels with a known predictable component.

The evaluation stack this package calibrates is applied to panels of equity
returns that are cross-sectionally correlated, serially heteroskedastic and
heavy-tailed.  None of those features is optional decoration: each one changes
what a nominal five percent test actually does.  The generators here reproduce
them with parameters that can be matched to a measured panel, and---the reason
they exist---with a predictable component whose share of return variance is set
by construction rather than estimated afterwards.

Returns are assembled from four orthogonal unit-variance channels: a common
predictable factor :math:`z^{p}`, a common unpredictable factor :math:`z^{u}`,
and their idiosyncratic counterparts :math:`e^{p}_i` and :math:`e^{u}_i`.  With
:math:`\bar\rho` the cross-sectional correlation and :math:`\theta` the
predictable share,

.. math::
    r_{i,t} = \sigma\,\nu_t\Big[
    \sqrt{\bar\rho\theta}\,z^{p}_t + \sqrt{(1-\bar\rho)\theta}\,e^{p}_{i,t}
    + \sqrt{\bar\rho(1-\theta)}\,z^{u}_t
    + \sqrt{(1-\bar\rho)(1-\theta)}\,e^{u}_{i,t}\Big],

where :math:`\nu_t` is a unit-mean-square volatility path shared by every name.
Pairwise correlation is then exactly :math:`\bar\rho` whatever that path does,
and the predictable part carries exactly a share :math:`\theta` of the variance.

Building the panel from named channels rather than from a single draw is what
lets a forecast be specified by its *population* moments.  A forecast is a
linear combination of the two predictable channels and two fresh noise
channels, and its variance, its cross-sectional correlation and its
out-of-sample :math:`R^2` against a zero benchmark are then all available in
closed form and can be set to measured values simultaneously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

__all__ = [
    "ForecastMoments",
    "InnovationLaw",
    "PanelDesign",
    "SimulatedPanel",
    "VolatilityLaw",
    "attainable_r_squared",
    "forecast_family",
    "forecast_moments",
    "noise_forecast_family",
    "population_loss_differential",
    "simulate_panel",
    "standardised_innovations",
]

InnovationLaw = Literal["gaussian", "student_t"]
VolatilityLaw = Literal["constant", "garch", "regime"]


@dataclass(frozen=True)
class PanelDesign:
    """Every quantity that fixes the law of a simulated panel and its forecasts.

    ``predictable_share`` is the population out-of-sample :math:`R^2` available
    to a forecaster who knows the predictable component exactly.  It bounds the
    effect sizes an experiment on this panel can inject, so it is set above any
    of them rather than at the value under test.  ``forecast_variance_ratio``
    and ``forecast_correlation`` describe the *fitted* forecast rather than the
    ideal one, and exist so that a simulated family can be matched to a measured
    one instead of assumed well behaved.
    """

    unit_count: int = 4
    session_count: int = 120
    target_correlation: float = 0.57
    target_volatility: float = 0.019
    predictable_share: float = 0.136
    forecast_variance_ratio: float = 0.272
    forecast_correlation: float = 0.947
    innovation: InnovationLaw = "gaussian"
    degrees_of_freedom: float = 5.0
    volatility: VolatilityLaw = "constant"
    garch_persistence: float = 0.90
    garch_shock: float = 0.08
    regime_probability: float = 0.02
    regime_volatility_ratio: float = 2.5

    def __post_init__(self) -> None:
        if self.unit_count < 1:
            raise ValueError("unit_count must be positive")
        if self.session_count < 8:
            raise ValueError("session_count must be at least eight")
        if not 0.0 <= self.target_correlation < 1.0:
            raise ValueError("target_correlation must lie in [0, 1)")
        if not 0.0 <= self.forecast_correlation <= 1.0:
            raise ValueError("forecast_correlation must lie in [0, 1]")
        if self.target_volatility <= 0.0:
            raise ValueError("target_volatility must be positive")
        if not 0.0 < self.predictable_share <= 1.0:
            raise ValueError("predictable_share must lie in (0, 1]")
        if self.forecast_variance_ratio <= 0.0:
            raise ValueError("forecast_variance_ratio must be positive")
        if self.innovation == "student_t" and self.degrees_of_freedom <= 4.0:
            raise ValueError("degrees_of_freedom must exceed four for a finite kurtosis")
        if not 0.0 < self.garch_persistence + self.garch_shock < 1.0:
            raise ValueError("the GARCH recursion must be stationary")
        if not 0.0 < self.regime_probability < 1.0:
            raise ValueError("regime_probability must lie in (0, 1)")
        if self.regime_volatility_ratio <= 1.0:
            raise ValueError("regime_volatility_ratio must exceed one")

    @property
    def row_count(self) -> int:
        """Return the number of panel rows a pooled analysis would see."""
        return self.unit_count * self.session_count

    def variance_inflation(self) -> float:
        """Return the equicorrelated inflation factor of the session mean."""
        return 1.0 + (self.unit_count - 1) * self.target_correlation

    def with_(self, **changes: object) -> PanelDesign:
        """Return a copy of the design with the named fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe record of the design."""
        return {
            "unit_count": self.unit_count,
            "session_count": self.session_count,
            "target_correlation": self.target_correlation,
            "target_volatility": self.target_volatility,
            "predictable_share": self.predictable_share,
            "forecast_variance_ratio": self.forecast_variance_ratio,
            "forecast_correlation": self.forecast_correlation,
            "innovation": self.innovation,
            "degrees_of_freedom": self.degrees_of_freedom,
            "volatility": self.volatility,
        }


@dataclass(frozen=True)
class SimulatedPanel:
    """One realised panel, with the channels a forecast is built from."""

    design: PanelDesign
    returns: np.ndarray
    common_signal: np.ndarray
    idiosyncratic_signal: np.ndarray
    volatility: np.ndarray

    @property
    def session_count(self) -> int:
        """Return the number of simulated sessions."""
        return int(self.returns.shape[0])

    @property
    def unit_count(self) -> int:
        """Return the number of simulated panel units."""
        return int(self.returns.shape[1])

    @property
    def predictable(self) -> np.ndarray:
        """Return the conditional mean a perfect forecaster would report."""
        design = self.design
        scale = design.target_volatility * self.volatility
        return scale * (
            math.sqrt(design.target_correlation * design.predictable_share) * self.common_signal
            + math.sqrt((1.0 - design.target_correlation) * design.predictable_share)
            * self.idiosyncratic_signal
        )


def standardised_innovations(
    shape: tuple[int, ...],
    *,
    law: InnovationLaw,
    degrees_of_freedom: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return unit-variance innovations from the requested law.

    A Student ``t`` with :math:`\nu` degrees of freedom has variance
    :math:`\nu/(\nu-2)`, so it is rescaled before use.  Without that the tail
    parameter would silently change the panel's volatility as well as its
    kurtosis, and any size distortion attributed to the tails would be partly a
    scale effect.
    """
    if law == "gaussian":
        return rng.standard_normal(shape)
    if law == "student_t":
        if degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed two for a finite variance")
        draws = rng.standard_t(degrees_of_freedom, size=shape)
        return draws / math.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
    raise ValueError(f"unknown innovation law: {law}")


def _volatility_path(design: PanelDesign, rng: np.random.Generator) -> np.ndarray:
    """Return a unit-mean-square volatility path of the requested law.

    Each path is rescaled to mean square one so that the panel's unconditional
    variance stays at ``target_volatility`` squared regardless of which law is
    in force.  Comparing a heteroskedastic cell against a constant one then
    isolates the clustering rather than confounding it with a scale change.
    """
    length = design.session_count
    if design.volatility == "constant":
        return np.ones(length)
    if design.volatility == "garch":
        shock = design.garch_shock
        persistence = design.garch_persistence
        intercept = 1.0 - shock - persistence
        variance = np.empty(length)
        state = 1.0
        innovations = rng.standard_normal(length)
        for step in range(length):
            variance[step] = state
            state = intercept + shock * state * innovations[step] ** 2 + persistence * state
        path = np.sqrt(variance)
    elif design.volatility == "regime":
        switches = rng.random(length) < design.regime_probability
        elevated = np.zeros(length, dtype=bool)
        current = False
        for step in range(length):
            if switches[step]:
                current = not current
            elevated[step] = current
        path = np.where(elevated, design.regime_volatility_ratio, 1.0)
    else:
        raise ValueError(f"unknown volatility law: {design.volatility}")
    return path / math.sqrt(float(np.mean(np.square(path))))


def _channels(design: PanelDesign, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return one common and one idiosyncratic unit-variance channel."""
    common = standardised_innovations(
        (design.session_count, 1),
        law=design.innovation,
        degrees_of_freedom=design.degrees_of_freedom,
        rng=rng,
    )
    idiosyncratic = standardised_innovations(
        (design.session_count, design.unit_count),
        law=design.innovation,
        degrees_of_freedom=design.degrees_of_freedom,
        rng=rng,
    )
    return np.broadcast_to(common, (design.session_count, design.unit_count)).copy(), idiosyncratic


def simulate_panel(design: PanelDesign, rng: np.random.Generator) -> SimulatedPanel:
    """Draw one panel of returns together with the channels behind it."""
    common_signal, idiosyncratic_signal = _channels(design, rng)
    common_noise, idiosyncratic_noise = _channels(design, rng)
    volatility = _volatility_path(design, rng)[:, None]
    correlation = design.target_correlation
    share = design.predictable_share
    standardised = (
        math.sqrt(correlation * share) * common_signal
        + math.sqrt((1.0 - correlation) * share) * idiosyncratic_signal
        + math.sqrt(correlation * (1.0 - share)) * common_noise
        + math.sqrt((1.0 - correlation) * (1.0 - share)) * idiosyncratic_noise
    )
    return SimulatedPanel(
        design=design,
        returns=design.target_volatility * volatility * standardised,
        common_signal=common_signal,
        idiosyncratic_signal=idiosyncratic_signal,
        volatility=volatility,
    )


@dataclass(frozen=True)
class ForecastMoments:
    """The loadings that deliver a requested population accuracy."""

    signal_share: float
    common_signal_loading: float
    idiosyncratic_signal_loading: float
    common_noise_loading: float
    idiosyncratic_noise_loading: float
    population_r_squared: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe record of the loadings."""
        return {
            "signal_share": self.signal_share,
            "common_signal_loading": self.common_signal_loading,
            "idiosyncratic_signal_loading": self.idiosyncratic_signal_loading,
            "common_noise_loading": self.common_noise_loading,
            "idiosyncratic_noise_loading": self.idiosyncratic_noise_loading,
            "population_r_squared": self.population_r_squared,
        }


def _covariance_coefficient(design: PanelDesign) -> float:
    r"""Return the covariance a fully invested unit-variance forecast attains.

    Writing a forecast's signal loadings as :math:`\sqrt{s\varphi v}` on the
    common channel and :math:`\sqrt{s(1-\varphi)v}` on the idiosyncratic one,
    its covariance with the return is :math:`\sigma^2\sqrt{s v\theta}\,\kappa`
    with

    .. math:: \kappa = \sqrt{\bar\rho\varphi} + \sqrt{(1-\bar\rho)(1-\varphi)}.

    Everything the accuracy of a forecast depends on enters through
    :math:`\kappa`, so it is computed once.
    """
    return math.sqrt(design.target_correlation * design.forecast_correlation) + math.sqrt(
        (1.0 - design.target_correlation) * (1.0 - design.forecast_correlation)
    )


def attainable_r_squared(design: PanelDesign) -> float:
    r"""Return the largest population :math:`R^2` this forecast shape can reach.

    A forecast constrained to a variance ratio :math:`v` and a cross-sectional
    correlation :math:`\varphi` cannot be the conditional mean unless those two
    happen to match it.  Its accuracy is maximised at a full signal share,

    .. math:: R^2_{\max} = 2\sqrt{v\theta}\,\kappa - v,

    and requesting more than that is a specification error rather than a hard
    problem, so it raises rather than silently rescaling.
    """
    ratio = design.forecast_variance_ratio
    return (
        2.0 * math.sqrt(ratio * design.predictable_share) * _covariance_coefficient(design) - ratio
    )


def forecast_moments(design: PanelDesign, *, population_r_squared: float) -> ForecastMoments:
    r"""Return the loadings of a forecast with a stated population accuracy.

    The variance ratio and cross-sectional correlation are held at the design's
    values and the signal share :math:`s` is solved for:

    .. math:: R^2 = 2\sqrt{s v \theta}\,\kappa - v
        \;\Longrightarrow\;
        s = \left(\frac{R^2 + v}{2\kappa\sqrt{v\theta}}\right)^{2}.

    Setting ``population_r_squared=0`` therefore does *not* produce a degenerate
    forecast.  It produces one whose genuine signal is exactly cancelled by its
    estimation noise, which is the null a fitted model is actually tested
    against and a far harder case for a test than a constant.
    """
    ratio = design.forecast_variance_ratio
    kappa = _covariance_coefficient(design)
    if kappa <= 0.0:
        raise ValueError("this design admits no forecast correlated with the target")
    root = (population_r_squared + ratio) / (
        2.0 * kappa * math.sqrt(ratio * design.predictable_share)
    )
    signal_share = root**2
    if signal_share > 1.0 + 1e-9:
        raise ValueError(
            "population_r_squared exceeds what this forecast shape can deliver: "
            f"requested {population_r_squared}, attainable {attainable_r_squared(design)}"
        )
    signal_share = min(signal_share, 1.0)
    correlation = design.forecast_correlation
    return ForecastMoments(
        signal_share=signal_share,
        common_signal_loading=math.sqrt(signal_share * correlation * ratio),
        idiosyncratic_signal_loading=math.sqrt(signal_share * (1.0 - correlation) * ratio),
        common_noise_loading=math.sqrt((1.0 - signal_share) * correlation * ratio),
        idiosyncratic_noise_loading=math.sqrt((1.0 - signal_share) * (1.0 - correlation) * ratio),
        population_r_squared=population_r_squared,
    )


def _assemble(
    panel: SimulatedPanel, moments: ForecastMoments, *, family_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a family sharing one signal component and differing in noise."""
    design = panel.design
    scale = design.target_volatility * panel.volatility
    signal = (
        moments.common_signal_loading * panel.common_signal
        + moments.idiosyncratic_signal_loading * panel.idiosyncratic_signal
    )
    forecasts = np.empty((family_size, panel.session_count, panel.unit_count))
    for member in range(family_size):
        common_noise, idiosyncratic_noise = _channels(design, rng)
        forecasts[member] = scale * (
            signal
            + moments.common_noise_loading * common_noise
            + moments.idiosyncratic_noise_loading * idiosyncratic_noise
        )
    return forecasts


def forecast_family(
    panel: SimulatedPanel,
    *,
    population_r_squared: float,
    family_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``family_size`` forecasts of a stated population accuracy.

    The members share their signal component and differ only in their noise
    draw, which is how a real family of models behaves: several specifications
    fitted to one dataset chase the same predictable variation and disagree
    mainly through estimation error.  A family of independent forecasts would
    make every multiplicity correction look better than it is.
    """
    if family_size < 1:
        raise ValueError("family_size must be positive")
    moments = forecast_moments(panel.design, population_r_squared=population_r_squared)
    return _assemble(panel, moments, family_size=family_size, rng=rng)


def noise_forecast_family(
    panel: SimulatedPanel,
    *,
    family_size: int,
    rng: np.random.Generator,
    variance_ratio: float | None = None,
) -> np.ndarray:
    r"""Return forecasts with no predictive content and a stated variance.

    This is the situation a nested comparison is really about.  The restricted
    model is correct---the population forecast is zero---but the researcher fits
    an unrestricted model anyway, and what comes out is estimation noise with
    variance :math:`v\sigma^2`.  Then

    .. math:: \mathbb{E}[d] = v\sigma^2 > 0, \qquad \mathbb{E}[2r\hat y] = 0,

    so the null of the Diebold--Mariano test is *false* while the null of the
    Clark--West test is *true*.  A test that rejects here in the direction of
    the benchmark is not detecting the absence of skill; it is detecting the
    fact that the model was estimated.
    """
    if family_size < 1:
        raise ValueError("family_size must be positive")
    ratio = panel.design.forecast_variance_ratio if variance_ratio is None else variance_ratio
    if ratio <= 0.0:
        raise ValueError("variance_ratio must be positive")
    moments = ForecastMoments(
        signal_share=0.0,
        common_signal_loading=0.0,
        idiosyncratic_signal_loading=0.0,
        common_noise_loading=math.sqrt(panel.design.forecast_correlation * ratio),
        idiosyncratic_noise_loading=math.sqrt((1.0 - panel.design.forecast_correlation) * ratio),
        population_r_squared=-ratio,
    )
    return _assemble(panel, moments, family_size=family_size, rng=rng)


def population_loss_differential(design: PanelDesign, *, population_r_squared: float) -> float:
    r"""Return the population mean loss differential against a zero forecast.

    Positive values favour the zero benchmark.  The identity
    :math:`\mathbb{E}[d] = -R^2\sigma^2` is what makes the null of the simulated
    experiments exact rather than approximate, and it is asserted directly by
    the tests.
    """
    return -population_r_squared * design.target_volatility**2
