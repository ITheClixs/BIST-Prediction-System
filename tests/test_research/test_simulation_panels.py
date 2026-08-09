"""The simulated panel must hit its stated moments, or nothing built on it holds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.research.simulation.panels import (
    PanelDesign,
    attainable_r_squared,
    forecast_family,
    forecast_moments,
    noise_forecast_family,
    population_loss_differential,
    simulate_panel,
    standardised_innovations,
)


def _mean_pairwise_correlation(values: np.ndarray) -> float:
    matrix = np.corrcoef(values, rowvar=False)
    return float(np.mean(matrix[np.triu_indices_from(matrix, k=1)]))


@pytest.fixture
def long_panel() -> tuple[PanelDesign, np.ndarray, np.ndarray]:
    """Return one very long realisation so sample moments approximate population ones."""
    design = PanelDesign(unit_count=6, session_count=120_000)
    rng = np.random.default_rng(20260806)
    panel = simulate_panel(design, rng)
    forecast = forecast_family(panel, population_r_squared=0.0, family_size=1, rng=rng)[0]
    return design, panel.returns, forecast


def test_returns_carry_the_requested_correlation(
    long_panel: tuple[PanelDesign, np.ndarray, np.ndarray],
) -> None:
    design, returns, _ = long_panel
    assert _mean_pairwise_correlation(returns) == pytest.approx(design.target_correlation, abs=0.01)


def test_returns_carry_the_requested_volatility(
    long_panel: tuple[PanelDesign, np.ndarray, np.ndarray],
) -> None:
    design, returns, _ = long_panel
    assert float(returns.std(ddof=1)) == pytest.approx(design.target_volatility, rel=0.01)


def test_the_forecast_hits_its_variance_and_correlation(
    long_panel: tuple[PanelDesign, np.ndarray, np.ndarray],
) -> None:
    """These two are measured from a real run, so a mismatch invalidates the anchor."""
    design, returns, forecast = long_panel
    ratio = float(np.mean(np.square(forecast))) / design.target_volatility**2
    assert ratio == pytest.approx(design.forecast_variance_ratio, rel=0.02)
    assert _mean_pairwise_correlation(forecast) == pytest.approx(
        design.forecast_correlation, abs=0.01
    )


def test_the_population_loss_differential_is_exactly_zero_at_zero_accuracy(
    long_panel: tuple[PanelDesign, np.ndarray, np.ndarray],
) -> None:
    """The null of every size experiment is this identity and nothing else."""
    design, returns, forecast = long_panel
    differential = np.square(forecast) - 2.0 * returns * forecast
    assert float(differential.mean()) == pytest.approx(
        population_loss_differential(design, population_r_squared=0.0),
        abs=0.02 * design.target_volatility**2,
    )


@pytest.mark.parametrize("requested", [0.0, 0.01, 0.05])
def test_a_requested_accuracy_is_delivered(requested: float) -> None:
    design = PanelDesign(unit_count=4, session_count=200_000, predictable_share=0.30)
    rng = np.random.default_rng(11)
    panel = simulate_panel(design, rng)
    forecast = forecast_family(panel, population_r_squared=requested, family_size=1, rng=rng)[0]
    differential = np.square(forecast) - 2.0 * panel.returns * forecast
    realised = -float(differential.mean()) / design.target_volatility**2
    assert realised == pytest.approx(requested, abs=0.01)


def test_a_zero_accuracy_forecast_is_not_a_constant() -> None:
    """Otherwise the null would be trivially easy and the measured size meaningless."""
    design = PanelDesign(session_count=500)
    rng = np.random.default_rng(3)
    panel = simulate_panel(design, rng)
    forecast = forecast_family(panel, population_r_squared=0.0, family_size=1, rng=rng)[0]
    assert float(forecast.std()) > 0.3 * design.target_volatility
    moments = forecast_moments(design, population_r_squared=0.0)
    assert 0.0 < moments.signal_share < 1.0


def test_accuracy_beyond_the_attainable_bound_is_refused() -> None:
    design = PanelDesign()
    rng = np.random.default_rng(4)
    panel = simulate_panel(design, rng)
    limit = attainable_r_squared(design)
    with pytest.raises(ValueError, match="exceeds what this forecast shape can deliver"):
        forecast_family(panel, population_r_squared=limit + 0.05, family_size=1, rng=rng)


def test_a_noise_forecast_has_no_covariance_with_the_target() -> None:
    """The Clark-West null is exactly this and it must hold by construction."""
    design = PanelDesign(session_count=200_000)
    rng = np.random.default_rng(19)
    panel = simulate_panel(design, rng)
    forecast = noise_forecast_family(panel, family_size=1, rng=rng)[0]
    scale = design.target_volatility**2
    assert float(np.mean(2.0 * panel.returns * forecast)) == pytest.approx(0.0, abs=0.01 * scale)
    assert float(np.mean(np.square(forecast))) / scale == pytest.approx(
        design.forecast_variance_ratio, rel=0.02
    )


def test_family_members_share_their_signal_and_differ_in_noise() -> None:
    """An independent family would flatter every multiplicity correction."""
    design = PanelDesign(session_count=400)
    rng = np.random.default_rng(23)
    panel = simulate_panel(design, rng)
    family = forecast_family(panel, population_r_squared=0.0, family_size=4, rng=rng)
    pairwise = [
        float(np.corrcoef(family[first].ravel(), family[second].ravel())[0, 1])
        for first in range(4)
        for second in range(first + 1, 4)
    ]
    assert min(pairwise) > 0.3
    assert max(pairwise) < 0.999


@pytest.mark.parametrize("volatility", ["constant", "garch", "regime"])
def test_every_volatility_law_preserves_the_unconditional_variance(volatility: str) -> None:
    design = PanelDesign(session_count=60_000, volatility=volatility)  # type: ignore[arg-type]
    returns = simulate_panel(design, np.random.default_rng(29)).returns
    assert float(returns.std(ddof=1)) == pytest.approx(design.target_volatility, rel=0.05)


def test_heavy_tails_raise_kurtosis_without_moving_the_variance() -> None:
    gaussian = standardised_innovations(
        (400_000,), law="gaussian", degrees_of_freedom=5.0, rng=np.random.default_rng(31)
    )
    student = standardised_innovations(
        (400_000,), law="student_t", degrees_of_freedom=5.0, rng=np.random.default_rng(31)
    )
    assert float(gaussian.std()) == pytest.approx(1.0, abs=0.01)
    assert float(student.std()) == pytest.approx(1.0, abs=0.05)
    excess = float(np.mean(student**4)) - 3.0
    assert excess > 2.0


def test_the_predictable_component_carries_its_stated_share() -> None:
    design = PanelDesign(unit_count=4, session_count=200_000, predictable_share=0.25)
    panel = simulate_panel(design, np.random.default_rng(37))
    share = float(np.var(panel.predictable)) / float(np.var(panel.returns))
    assert share == pytest.approx(design.predictable_share, rel=0.05)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unit_count", 0, "unit_count must be positive"),
        ("session_count", 4, "session_count must be at least eight"),
        ("target_correlation", 1.0, "target_correlation must lie"),
        ("target_volatility", 0.0, "target_volatility must be positive"),
        ("predictable_share", 0.0, "predictable_share must lie"),
        ("forecast_variance_ratio", 0.0, "forecast_variance_ratio must be positive"),
        ("forecast_correlation", 1.5, "forecast_correlation must lie"),
        ("regime_volatility_ratio", 1.0, "regime_volatility_ratio must exceed one"),
    ],
)
def test_invalid_designs_are_refused(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PanelDesign(**{field: value})  # type: ignore[arg-type]


def test_a_student_t_with_four_degrees_of_freedom_is_refused() -> None:
    """Its kurtosis is infinite, so any simulated size would be a sample artifact."""
    with pytest.raises(ValueError, match="degrees_of_freedom must exceed four"):
        PanelDesign(innovation="student_t", degrees_of_freedom=4.0)


def test_the_attainable_bound_matches_its_closed_form() -> None:
    design = PanelDesign()
    kappa = math.sqrt(design.target_correlation * design.forecast_correlation) + math.sqrt(
        (1.0 - design.target_correlation) * (1.0 - design.forecast_correlation)
    )
    ratio = design.forecast_variance_ratio
    expected = 2.0 * math.sqrt(ratio * design.predictable_share) * kappa - ratio
    assert attainable_r_squared(design) == pytest.approx(expected)
