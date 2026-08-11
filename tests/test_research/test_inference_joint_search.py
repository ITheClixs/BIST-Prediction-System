"""The joint search test must be calibrated and must feel the grid's dependence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_predict.research.inference.joint_search import joint_search_test


def _grid(
    *, trials: int, sessions: int, correlation: float, seed: int, drift: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((sessions, 1))
    idiosyncratic = rng.standard_normal((sessions, trials))
    values = np.sqrt(correlation) * common + np.sqrt(1.0 - correlation) * idiosyncratic
    values[:, 0] += drift
    return pd.DataFrame(
        values * 0.01,
        columns=[f"trial_{index}" for index in range(trials)],
        index=[f"2026-03-{index:03d}" for index in range(sessions)],
    )


def test_a_skill_free_grid_is_not_rejected() -> None:
    result = joint_search_test(
        _grid(trials=20, sessions=120, correlation=0.9, seed=7), replications=2000, seed=1
    )
    assert result.p_value > 0.05
    assert result.verdict == "best_trial_is_within_search_noise"


def test_a_genuinely_strong_configuration_is_rejected() -> None:
    result = joint_search_test(
        _grid(trials=20, sessions=120, correlation=0.9, seed=7, drift=0.6),
        replications=2000,
        seed=1,
    )
    assert result.p_value < 0.05
    assert result.best_trial == "trial_0"
    assert result.verdict == "best_trial_survives"


def test_the_effective_trial_count_falls_as_the_grid_becomes_redundant() -> None:
    """A grid whose members move together has searched less than its size suggests."""
    counts = [
        joint_search_test(
            _grid(trials=32, sessions=150, correlation=correlation, seed=11),
            replications=1500,
            seed=2,
        ).independent_equivalent_trials
        for correlation in (0.0, 0.5, 0.9, 0.99)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 10.0
    assert counts[-1] < 5.0


def test_an_independent_grid_recovers_roughly_its_own_size() -> None:
    result = joint_search_test(
        _grid(trials=32, sessions=400, correlation=0.0, seed=13), replications=2000, seed=3
    )
    assert 16.0 < result.independent_equivalent_trials < 64.0


def test_the_null_quantile_exceeds_the_null_expectation() -> None:
    result = joint_search_test(
        _grid(trials=20, sessions=120, correlation=0.5, seed=17), replications=2000, seed=4
    )
    assert result.null_quantile_95 > result.null_expected_maximum
    assert result.null_maximum_dispersion > 0.0


def test_the_measured_grid_correlation_is_reported() -> None:
    result = joint_search_test(
        _grid(trials=12, sessions=200, correlation=0.8, seed=19), replications=1000, seed=5
    )
    assert result.mean_pairwise_correlation == pytest.approx(0.8, abs=0.06)


def test_the_result_is_reproducible_from_its_seed() -> None:
    grid = _grid(trials=12, sessions=120, correlation=0.7, seed=23)
    first = joint_search_test(grid, replications=800, seed=99)
    second = joint_search_test(grid, replications=800, seed=99)
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize(
    ("frame", "replications", "message"),
    [
        (_grid(trials=1, sessions=40, correlation=0.0, seed=1), 500, "at least two configurations"),
        (
            _grid(trials=4, sessions=8, correlation=0.0, seed=1).head(4),
            500,
            "at least eight sessions",
        ),
        (_grid(trials=4, sessions=40, correlation=0.0, seed=1), 10, "one hundred replications"),
    ],
)
def test_degenerate_requests_are_refused(
    frame: pd.DataFrame, replications: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        joint_search_test(frame, replications=replications)


def test_a_non_finite_grid_is_refused() -> None:
    frame = _grid(trials=4, sessions=40, correlation=0.0, seed=1)
    frame.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        joint_search_test(frame, replications=500)
