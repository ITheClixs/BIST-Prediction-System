"""Whether the calibration measures what it claims, and on the production code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bist_predict.research.inference.forecast_tests import squared_error_differential
from bist_predict.research.simulation.calibration import (
    asymptotic_pooled_size,
    loss_differentials,
    minimum_detectable_effect,
    nested_null_cell,
    rejection_rate,
    size_power_cell,
)
from bist_predict.research.simulation.panels import PanelDesign


def test_matches_the_production_estimator() -> None:
    """The fast path and the pandas path must compute the same differential.

    Without this the study would calibrate a private reimplementation and report
    the answer as though it applied to the estimator the manuscript uses.
    """
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.02, size=(9, 3))
    forecast = rng.normal(0.0, 0.01, size=(9, 3))
    dates = [f"2026-01-{day:02d}" for day in range(1, 10)]
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    for name, values in (("fitted", forecast), ("zero_return", np.zeros_like(forecast))):
        for session, date in enumerate(dates):
            for unit, ticker in enumerate(tickers):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model_name": name,
                        "target": returns[session, unit],
                        "predicted_return": values[session, unit],
                    }
                )
    frame = pd.DataFrame.from_records(rows)

    expected_session = squared_error_differential(
        frame, candidate="fitted", benchmark="zero_return"
    ).to_numpy()
    expected_row = squared_error_differential(
        frame, candidate="fitted", benchmark="zero_return", aggregation="row"
    ).to_numpy()
    fast = loss_differentials(returns, forecast)

    assert fast.mean(axis=1) == pytest.approx(expected_session, abs=1e-15)
    assert np.sort(fast.reshape(-1)) == pytest.approx(np.sort(expected_row), abs=1e-15)


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="must share a shape"):
        loss_differentials(np.zeros((3, 2)), np.zeros((3, 3)))


def test_the_closed_form_size_reduces_to_the_nominal_level_without_dependence() -> None:
    assert asymptotic_pooled_size(1, 0.9, alpha=0.05) == pytest.approx(0.05)
    assert asymptotic_pooled_size(10, 0.0, alpha=0.05) == pytest.approx(0.05)


def test_the_closed_form_size_matches_a_direct_computation() -> None:
    inflation = 1.0 + 3 * 0.5697
    expected = 2.0 * stats.norm.cdf(-stats.norm.ppf(0.975) / np.sqrt(inflation))
    assert asymptotic_pooled_size(4, 0.5697) == pytest.approx(expected)


def test_the_closed_form_size_grows_with_both_arguments() -> None:
    sizes = [asymptotic_pooled_size(units, 0.5) for units in (2, 4, 10, 30, 100)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 0.5
    by_correlation = [asymptotic_pooled_size(30, rho) for rho in (0.0, 0.2, 0.5, 0.9)]
    assert by_correlation == sorted(by_correlation)


def test_negative_dependence_predicts_an_undersized_test() -> None:
    """A prediction that can only err upwards cannot be falsified by simulation."""
    assert asymptotic_pooled_size(4, -0.2) < 0.05


def test_an_impossible_inflation_is_refused() -> None:
    with pytest.raises(ValueError, match="variance inflation factor is not positive"):
        asymptotic_pooled_size(10, -0.5)


def test_the_binomial_interval_brackets_the_point_estimate() -> None:
    rate = rejection_rate(50, 1000)
    assert rate.lower < rate.rate < rate.upper
    assert rate.covers(0.05)
    assert not rate.covers(0.2)


def test_the_binomial_interval_stays_valid_at_the_boundaries() -> None:
    assert rejection_rate(0, 200).lower == 0.0
    assert rejection_rate(200, 200).upper == 1.0
    assert rejection_rate(0, 200).upper > 0.0


@pytest.mark.parametrize(
    ("rejections", "trials", "message"),
    [(1, 0, "trials must be positive"), (5, 4, "rejections must lie")],
)
def test_impossible_rejection_counts_are_refused(
    rejections: int, trials: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rejection_rate(rejections, trials)


def test_the_session_test_holds_its_nominal_size_and_the_pooled_one_does_not() -> None:
    """This is the manuscript's headline result reproduced at test scale."""
    design = PanelDesign(unit_count=4, session_count=120)
    cell = size_power_cell(design, population_r_squared=0.0, replications=600, seed=101)
    assert cell.session_rejection.covers(0.05)
    assert cell.row_rejection.rate > 0.10
    assert not cell.row_rejection.covers(0.05)


def test_the_pooled_size_lands_where_the_proposition_says_it_will() -> None:
    design = PanelDesign(unit_count=10, session_count=120)
    cell = size_power_cell(design, population_r_squared=0.0, replications=800, seed=102)
    assert abs(cell.row_rejection.rate - cell.predicted_row_size) < 0.08


def test_independent_rows_leave_the_pooled_test_calibrated() -> None:
    """The distortion must come from the dependence and from nothing else."""
    design = PanelDesign(unit_count=8, target_correlation=0.0, forecast_correlation=0.0)
    cell = size_power_cell(design, population_r_squared=0.0, replications=600, seed=103)
    assert cell.row_rejection.covers(0.05)
    assert cell.session_rejection.covers(0.05)


def test_power_rises_with_the_effect() -> None:
    design = PanelDesign(session_count=250, predictable_share=0.35)
    rates = [
        size_power_cell(
            design, population_r_squared=effect, replications=300, seed=200 + index
        ).session_superiority.rate
        for index, effect in enumerate((0.0, 0.05, 0.15))
    ]
    assert rates[0] < rates[1] < rates[2]


def test_the_clark_west_null_holds_when_the_forecast_is_pure_noise() -> None:
    design = PanelDesign(unit_count=4, session_count=120)
    cell = nested_null_cell(design, variance_ratio=0.272, replications=800, seed=301)
    assert cell.clark_west_session.covers(0.05)


def test_a_squared_error_comparison_condemns_a_correctly_specified_model() -> None:
    """The zero benchmark is the true model here, so this rate is pure artifact."""
    design = PanelDesign(unit_count=4, session_count=120)
    cell = nested_null_cell(design, variance_ratio=0.272, replications=600, seed=302)
    assert cell.diebold_mariano_against_candidate.rate > 0.5
    assert cell.diebold_mariano_for_candidate.rate < 0.05


def test_pooling_rows_breaks_the_clark_west_test_too() -> None:
    design = PanelDesign(unit_count=10, session_count=120)
    cell = nested_null_cell(design, variance_ratio=0.272, replications=600, seed=303)
    assert cell.clark_west_row.rate > cell.clark_west_session.rate
    assert not cell.clark_west_row.covers(0.05)


def _cell(effect: float, rate: float) -> object:
    class _Rate:
        def __init__(self, value: float) -> None:
            self.rate = value

    class _Cell:
        def __init__(self) -> None:
            self.population_r_squared = effect
            self.population_covariance_ratio = effect + 0.272
            self.session_superiority = _Rate(rate)
            self.clark_west_session = _Rate(rate)

    return _Cell()


def test_the_detectable_effect_interpolates_between_bracketing_points() -> None:
    cells = [_cell(0.0, 0.2), _cell(0.10, 0.6), _cell(0.20, 1.0)]
    assert minimum_detectable_effect(cells) == pytest.approx(0.15)  # type: ignore[arg-type]


def test_an_unreached_power_returns_nothing_rather_than_an_extrapolation() -> None:
    cells = [_cell(0.0, 0.05), _cell(0.10, 0.3)]
    assert minimum_detectable_effect(cells) is None  # type: ignore[arg-type]


def test_the_two_tests_are_reported_on_their_own_effect_axes() -> None:
    cells = [_cell(0.0, 0.2), _cell(0.10, 0.9)]
    on_r_squared = minimum_detectable_effect(cells)  # type: ignore[arg-type]
    on_covariance = minimum_detectable_effect(
        cells,  # type: ignore[arg-type]
        test="clark_west",
        scale="covariance_ratio",
    )
    assert on_covariance == pytest.approx(on_r_squared + 0.272)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"power": 1.0}, "power must lie"),
        ({"test": "wald"}, "test must be"),
        ({"scale": "sharpe"}, "scale must be"),
    ],
)
def test_invalid_detectability_requests_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        minimum_detectable_effect([_cell(0.0, 0.1)], **kwargs)  # type: ignore[arg-type]
