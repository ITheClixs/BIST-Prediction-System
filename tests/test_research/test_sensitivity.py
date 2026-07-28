"""Configuration-grid construction and the search-threshold summary."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from bist_predict.research.inference.sharpe import EULER_MASCHERONI
from bist_predict.research.sensitivity import (
    SensitivityTrial,
    configuration_grid,
    summarise_sensitivity,
)


def _trial(sharpe: float, net_return: float, *, top_k: int = 3) -> SensitivityTrial:
    return SensitivityTrial(
        min_train_dates=24,
        validation_dates=10,
        step_dates=10,
        embargo_dates=1,
        top_k=top_k,
        fold_count=12,
        evaluated_sample_count=480,
        session_count=120,
        gross_return=net_return + 0.1,
        net_return=net_return,
        annualised_return=net_return * 2.0,
        per_period_sharpe=sharpe,
        annualised_sharpe=sharpe * math.sqrt(252.0),
        maximum_drawdown=-0.1,
        turnover=100.0,
        trade_count=140,
        portfolio_model_zero_mean_r_squared=-0.2,
        best_model="zero_return",
        best_zero_mean_r_squared=0.0,
    )


def test_grid_is_the_full_cross_product() -> None:
    grid = configuration_grid(
        min_train_dates=[24, 36], validation_dates=[5, 10], embargo_dates=[1], top_k=[1, 2, 3]
    )
    assert len(grid) == 2 * 2 * 1 * 3


def test_step_is_tied_to_the_validation_width() -> None:
    """Otherwise the grid would silently vary how often each session is reused."""
    grid = configuration_grid(
        min_train_dates=[24], validation_dates=[5, 20], embargo_dates=[1], top_k=[2]
    )
    assert {point["step_dates"] for point in grid} == {5, 20}
    assert all(point["step_dates"] == point["validation_dates"] for point in grid)


def test_grid_is_deterministic_and_deduplicated() -> None:
    first = configuration_grid(
        min_train_dates=[36, 24, 24], validation_dates=[10], embargo_dates=[1], top_k=[2]
    )
    second = configuration_grid(
        min_train_dates=[24, 36], validation_dates=[10], embargo_dates=[1], top_k=[2]
    )
    assert first == second
    assert len(first) == 2


def test_an_empty_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="must contain at least one value"):
        configuration_grid(min_train_dates=[], validation_dates=[10], embargo_dates=[1], top_k=[2])


def test_search_threshold_matches_the_false_strategy_theorem() -> None:
    trials = [
        _trial(sharpe, 0.01 * index) for index, sharpe in enumerate((0.01, 0.05, -0.02, 0.03))
    ]
    summary = summarise_sensitivity(trials)
    variance = float(np.var([trial.per_period_sharpe for trial in trials], ddof=1))
    count = len(trials)
    expected = math.sqrt(variance) * (
        (1 - EULER_MASCHERONI) * float(stats.norm.ppf(1 - 1 / count))
        + EULER_MASCHERONI * float(stats.norm.ppf(1 - 1 / (count * math.e)))
    )
    assert summary["trial_sharpe_variance"] == pytest.approx(variance)
    assert summary["expected_maximum_sharpe_under_no_skill"] == pytest.approx(expected)


def test_the_threshold_is_an_expectation_and_not_a_critical_value() -> None:
    """Verified by simulation: a skill-free grid exceeds it about half the time.

    The False Strategy Theorem returns the mean of the maximum, so treating it
    as a 95% bar would reject roughly half of all genuinely skill-free searches.
    The deflated Sharpe ratio, not this flag, is the test.
    """
    exceedances = 0
    replications = 200
    for replication in range(replications):
        rng = np.random.default_rng(1000 + replication)
        draws = rng.normal(0.0, 0.05, size=60)
        summary = summarise_sensitivity([_trial(float(value), float(value)) for value in draws])
        exceedances += int(bool(summary["best_trial_exceeds_expected_maximum"]))
    assert 0.3 <= exceedances / replications <= 0.7


def test_a_genuine_outlier_still_exceeds_the_expected_maximum() -> None:
    trials = [_trial(0.005 * index, 0.001 * index) for index in range(20)]
    trials.append(_trial(3.0, 1.0))
    summary = summarise_sensitivity(trials)
    assert summary["best_trial_exceeds_expected_maximum"] is True


def test_reported_rank_counts_strictly_better_trials() -> None:
    trials = [_trial(0.05, 0.1), _trial(0.02, 0.0), _trial(-0.01, -0.1)]
    summary = summarise_sensitivity(trials, reported=trials[1])
    assert summary["reported_rank_by_sharpe"] == 2
    assert summary["trial_count"] == 3


def test_share_positive_counts_profitable_configurations() -> None:
    trials = [_trial(0.01, 0.02), _trial(-0.01, -0.02), _trial(-0.02, -0.03), _trial(-0.03, -0.04)]
    summary = summarise_sensitivity(trials)
    assert summary["net_return"]["share_positive"] == pytest.approx(0.25)


def test_a_single_trial_has_no_dispersion_and_no_threshold() -> None:
    summary = summarise_sensitivity([_trial(0.4, 0.2)])
    assert summary["trial_sharpe_variance"] == 0.0
    assert summary["expected_maximum_sharpe_under_no_skill"] == 0.0


def test_an_empty_grid_summary_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one trial"):
        summarise_sensitivity([])


def test_trial_identity_encodes_every_varied_axis() -> None:
    """Two grid points must never collide on the identity used to find the reported one."""
    identities = {
        SensitivityTrial(
            min_train_dates=train,
            validation_dates=validation,
            step_dates=validation,
            embargo_dates=embargo,
            top_k=breadth,
            fold_count=1,
            evaluated_sample_count=1,
            session_count=1,
            gross_return=0.0,
            net_return=0.0,
            annualised_return=0.0,
            per_period_sharpe=0.0,
            annualised_sharpe=0.0,
            maximum_drawdown=0.0,
            turnover=0.0,
            trade_count=0,
            portfolio_model_zero_mean_r_squared=0.0,
            best_model="zero_return",
            best_zero_mean_r_squared=0.0,
        ).trial_id
        for train in (24, 36)
        for validation in (5, 10)
        for embargo in (1, 2)
        for breadth in (1, 2)
    }
    assert len(identities) == 16
