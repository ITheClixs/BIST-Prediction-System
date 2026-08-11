"""Whether the multiplicity layer's measured error rates mean what they say."""

from __future__ import annotations

import pytest

from bist_predict.research.simulation.panels import PanelDesign
from bist_predict.research.simulation.search_calibration import (
    family_wise_cell,
    search_threshold_cell,
)

DESIGN = PanelDesign(
    unit_count=4,
    session_count=40,
    target_correlation=0.57,
    target_volatility=0.019,
    predictable_share=0.136,
    forecast_variance_ratio=0.272,
    forecast_correlation=0.947,
)


def test_a_family_needs_at_least_two_members() -> None:
    with pytest.raises(ValueError, match="at least two members"):
        family_wise_cell(DESIGN, family_size=1, replications=5, seed=1)


def test_a_family_cell_needs_repeated_draws() -> None:
    with pytest.raises(ValueError, match="replications must be at least two"):
        family_wise_cell(DESIGN, family_size=3, replications=1, seed=1)


def test_the_family_cell_is_reproducible_from_its_seed() -> None:
    """A study that cannot be replayed cell by cell cannot be audited."""
    first = family_wise_cell(
        DESIGN, family_size=3, replications=12, seed=7, bootstrap_replications=199
    )
    second = family_wise_cell(
        DESIGN, family_size=3, replications=12, seed=7, bootstrap_replications=199
    )
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_moves_the_family_cell() -> None:
    first = family_wise_cell(
        DESIGN, family_size=3, replications=12, seed=7, bootstrap_replications=199
    )
    second = family_wise_cell(
        DESIGN, family_size=3, replications=12, seed=8, bootstrap_replications=199
    )
    assert first.to_dict() != second.to_dict()


def test_every_family_rate_is_a_proportion_with_a_bracketing_interval() -> None:
    cell = family_wise_cell(
        DESIGN, family_size=3, replications=20, seed=11, bootstrap_replications=199
    )
    rates = (
        cell.uncorrected_any,
        cell.holm_session,
        cell.holm_row,
        cell.reality_check,
        cell.spa_untruncated,
        cell.spa_hansen,
    )
    for rate in rates:
        assert 0.0 <= rate.rate <= 1.0
        assert rate.lower <= rate.rate <= rate.upper
        assert rate.trials == 20


def test_correcting_never_rejects_more_often_than_not_correcting() -> None:
    """Holm is a step-down on the same p-values, so it cannot reject more.

    This is the one relation between the measured rates that holds replication
    by replication rather than only on average, which makes it a real guard
    rather than a restatement of the expected ordering.
    """
    cell = family_wise_cell(
        DESIGN, family_size=4, replications=25, seed=3, bootstrap_replications=199
    )
    assert cell.holm_session.rate <= cell.uncorrected_any.rate


def test_the_search_cell_recovers_the_nominal_count_when_trials_are_independent() -> None:
    """With uncorrelated trials the grid should look like the grid it is."""
    cell = search_threshold_cell(
        trial_count=16,
        session_count=60,
        trial_correlation=0.0,
        replications=60,
        seed=5,
        bootstrap_replications=199,
    )
    assert cell.mean_independent_equivalent_trials == pytest.approx(16.0, rel=0.45)


def test_dependence_between_trials_lowers_the_equivalent_count() -> None:
    """The diagnostic is only useful if it moves in the right direction."""
    independent = search_threshold_cell(
        trial_count=16,
        session_count=60,
        trial_correlation=0.0,
        replications=60,
        seed=5,
        bootstrap_replications=199,
    )
    dependent = search_threshold_cell(
        trial_count=16,
        session_count=60,
        trial_correlation=0.95,
        replications=60,
        seed=5,
        bootstrap_replications=199,
    )
    assert (
        dependent.mean_independent_equivalent_trials
        < independent.mean_independent_equivalent_trials
    )


def test_the_joint_bootstrap_quantile_is_closer_to_nominal_than_the_expectation() -> None:
    """The False Strategy threshold is an expectation, not a critical value.

    A skill-free grid clears an expected maximum about half the time by
    definition. The joint bootstrap quantile is a quantile and should sit near
    the nominal level instead. Reporting the first as though it were the second
    is the error this cell exists to measure.
    """
    cell = search_threshold_cell(
        trial_count=16,
        session_count=60,
        trial_correlation=0.5,
        replications=80,
        seed=9,
        bootstrap_replications=199,
    )
    assert cell.false_strategy_expectation.rate > cell.joint_bootstrap_quantile.rate
    assert cell.joint_bootstrap_quantile.rate < 0.25


def test_the_search_cell_is_reproducible_from_its_seed() -> None:
    first = search_threshold_cell(
        trial_count=8,
        session_count=40,
        trial_correlation=0.5,
        replications=30,
        seed=13,
        bootstrap_replications=199,
    )
    second = search_threshold_cell(
        trial_count=8,
        session_count=40,
        trial_correlation=0.5,
        replications=30,
        seed=13,
        bootstrap_replications=199,
    )
    assert first.to_dict() == second.to_dict()
