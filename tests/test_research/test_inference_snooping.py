"""Reality Check and SPA: bootstrap mechanics, calibration, and an external pin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_predict.research.inference.snooping import (
    reality_check_and_spa,
    stationary_block_length,
    stationary_bootstrap_indices,
)


def _loss_panel(
    rng: np.random.Generator, *, sessions: int, candidates: int, edge: float
) -> pd.DataFrame:
    """Return a loss panel where every candidate beats the benchmark by ``edge``."""
    benchmark = rng.normal(1.0, 0.3, size=sessions)
    data = {"benchmark": benchmark}
    for index in range(candidates):
        data[f"candidate_{index}"] = benchmark - edge + rng.normal(0.0, 0.05, size=sessions)
    return pd.DataFrame(data)


def test_bootstrap_indices_stay_inside_the_sample() -> None:
    rng = np.random.default_rng(0)
    indices = stationary_bootstrap_indices(50, block_length=5.0, replications=200, rng=rng)
    assert indices.shape == (200, 50)
    assert indices.min() >= 0
    assert indices.max() < 50


def test_bootstrap_block_lengths_are_geometric_with_the_requested_mean() -> None:
    """Verified by simulation rather than asserted in a comment."""
    rng = np.random.default_rng(1)
    block_length = 8.0
    indices = stationary_bootstrap_indices(
        400, block_length=block_length, replications=400, rng=rng
    )
    continuations = (indices[:, 1:] - indices[:, :-1]) % 400 == 1
    observed_restart_rate = 1.0 - continuations.mean()
    assert observed_restart_rate == pytest.approx(1.0 / block_length, rel=0.08)


def test_a_block_length_of_one_reduces_to_an_iid_bootstrap() -> None:
    rng = np.random.default_rng(2)
    indices = stationary_bootstrap_indices(200, block_length=1.0, replications=300, rng=rng)
    continuations = (indices[:, 1:] - indices[:, :-1]) % 200 == 1
    assert continuations.mean() == pytest.approx(1.0 / 200.0, abs=0.01)


def test_persistent_series_get_a_longer_selected_block() -> None:
    rng = np.random.default_rng(3)
    innovations = rng.normal(size=(1500, 1))
    persistent = np.zeros((1500, 1))
    for index in range(1, 1500):
        persistent[index] = 0.8 * persistent[index - 1] + innovations[index]
    assert stationary_block_length(persistent) > stationary_block_length(innovations)


def test_a_clearly_superior_candidate_is_detected() -> None:
    rng = np.random.default_rng(4)
    panel = _loss_panel(rng, sessions=250, candidates=5, edge=0.15)
    result = reality_check_and_spa(panel, benchmark="benchmark", replications=2000, seed=7)
    assert result.best_mean_outperformance > 0.0
    assert result.reality_check_p_value < 0.05
    assert result.spa_p_value_consistent < 0.05
    assert result.verdict == "at_least_one_candidate_beats_benchmark"


def test_pure_noise_candidates_are_not_declared_superior() -> None:
    rng = np.random.default_rng(5)
    panel = _loss_panel(rng, sessions=250, candidates=20, edge=0.0)
    result = reality_check_and_spa(panel, benchmark="benchmark", replications=2000, seed=7)
    assert result.spa_p_value_consistent > 0.10
    assert result.verdict == "no_candidate_beats_benchmark"


def test_the_hansen_recentring_bounds_are_ordered() -> None:
    rng = np.random.default_rng(6)
    panel = _loss_panel(rng, sessions=200, candidates=8, edge=0.02)
    result = reality_check_and_spa(panel, benchmark="benchmark", replications=2000, seed=11)
    assert result.spa_p_value_lower <= result.spa_p_value_consistent + 1e-12
    assert result.spa_p_value_consistent <= result.spa_p_value_upper + 1e-12


def test_adding_useless_candidates_weakens_the_evidence_for_the_good_one() -> None:
    """The whole point of the Reality Check: a wider search costs significance."""
    rng = np.random.default_rng(8)
    benchmark = rng.normal(1.0, 0.3, size=220)
    winner = benchmark - 0.05 + rng.normal(0.0, 0.28, size=220)
    small = pd.DataFrame({"benchmark": benchmark, "winner": winner})
    wide = small.copy()
    for index in range(60):
        wide[f"noise_{index}"] = benchmark + rng.normal(0.0, 0.28, size=220)
    narrow_p = reality_check_and_spa(
        small, benchmark="benchmark", replications=3000, seed=3
    ).reality_check_p_value
    wide_p = reality_check_and_spa(
        wide, benchmark="benchmark", replications=3000, seed=3
    ).reality_check_p_value
    assert wide_p > narrow_p


def test_the_null_distribution_is_not_degenerate_under_no_skill() -> None:
    """Calibration by simulation: rejection rate at 5% must stay near nominal.

    Omitting the bootstrap recentring leaves the draws centred on the observed
    means instead of on the null, which destroys the calibration this checks.
    """
    rejections = 0
    trials = 40
    for trial in range(trials):
        rng = np.random.default_rng(100 + trial)
        panel = _loss_panel(rng, sessions=150, candidates=6, edge=0.0)
        result = reality_check_and_spa(
            panel, benchmark="benchmark", replications=400, seed=trial, block_length=4.0
        )
        rejections += int(result.spa_p_value_consistent <= 0.05)
    assert rejections / trials <= 0.15


def test_p_values_agree_with_the_arch_reference_implementation() -> None:
    """Independent recomputation against ``arch.bootstrap.SPA``.

    ``arch`` uses the loss-differential variance only to decide which columns
    are relevant for the consistent recentring, and then compares unstudentized
    means. ``studentize=False`` reproduces that comparison exactly, so the two
    implementations should differ by Monte Carlo error alone. With 5000
    replications the standard error of a p-value near one half is about 0.007.
    """
    from arch.bootstrap import SPA

    rng = np.random.default_rng(12)
    panel = _loss_panel(rng, sessions=300, candidates=6, edge=0.01)
    block_length = stationary_block_length(
        panel[["benchmark"]].to_numpy() - panel.drop(columns="benchmark").to_numpy()
    )
    ours = reality_check_and_spa(
        panel,
        benchmark="benchmark",
        replications=5000,
        seed=5,
        block_length=block_length,
        studentize=False,
    )
    reference = SPA(
        panel["benchmark"].to_numpy(),
        panel.drop(columns="benchmark").to_numpy(),
        block_size=int(round(block_length)),
        reps=5000,
        seed=5,
    )
    reference.compute()
    for name, value in (
        ("lower", ours.spa_p_value_lower),
        ("consistent", ours.spa_p_value_consistent),
        ("upper", ours.spa_p_value_upper),
    ):
        assert value == pytest.approx(float(reference.pvalues[name]), abs=0.05)


def test_missing_benchmark_column_is_rejected() -> None:
    rng = np.random.default_rng(13)
    panel = _loss_panel(rng, sessions=60, candidates=2, edge=0.0)
    with pytest.raises(ValueError, match="benchmark column is missing"):
        reality_check_and_spa(panel, benchmark="absent")


def test_a_benchmark_without_candidates_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        reality_check_and_spa(pd.DataFrame({"benchmark": np.arange(30.0)}), benchmark="benchmark")


def test_too_few_sessions_are_rejected() -> None:
    rng = np.random.default_rng(14)
    panel = _loss_panel(rng, sessions=5, candidates=2, edge=0.0)
    with pytest.raises(ValueError, match="at least eight evaluation sessions"):
        reality_check_and_spa(panel, benchmark="benchmark")


def test_too_few_replications_are_rejected() -> None:
    rng = np.random.default_rng(15)
    panel = _loss_panel(rng, sessions=60, candidates=2, edge=0.0)
    with pytest.raises(ValueError, match="at least one hundred replications"):
        reality_check_and_spa(panel, benchmark="benchmark", replications=10)


def test_candidates_identical_to_the_benchmark_are_rejected() -> None:
    rng = np.random.default_rng(16)
    losses = rng.normal(size=80)
    panel = pd.DataFrame({"benchmark": losses, "clone": losses})
    with pytest.raises(ValueError, match="numerically identical to the benchmark"):
        reality_check_and_spa(panel, benchmark="benchmark", replications=200)


def _inferior_panel(rng: np.random.Generator, *, sessions: int, candidates: int) -> pd.DataFrame:
    """Return a panel where every candidate is strictly worse than the benchmark."""
    benchmark = np.abs(rng.normal(1.0, 0.2, size=sessions))
    data = {"benchmark": benchmark}
    for index in range(candidates):
        data[f"candidate_{index}"] = benchmark + 0.2 + np.abs(rng.normal(0.0, 0.1, size=sessions))
    return pd.DataFrame(data)


def test_a_family_of_inferior_candidates_reports_no_evidence() -> None:
    """Boundary case: the observed maximum is negative and the statistic floors at zero.

    Comparing floored draws against a floored statistic collapses both sides
    onto the atom at zero and returns whatever share of draws happened to be
    strictly positive. The comparison must use the untruncated maxima.
    """
    rng = np.random.default_rng(41)
    panel = _inferior_panel(rng, sessions=200, candidates=6)
    result = reality_check_and_spa(panel, benchmark="benchmark", replications=4000, seed=9)
    assert result.best_mean_outperformance < 0.0
    assert result.spa_statistic == 0.0
    assert result.spa_p_value_consistent > 0.5
    assert result.reality_check_p_value > 0.5
    assert result.verdict == "no_candidate_beats_benchmark"
    # The upper recentring shifts every draw to a zero mean, so a negative
    # observed maximum must sit deep in the left tail. Skipping the recentring
    # leaves the draws centred on the observed means and collapses this to 0.5.
    assert result.spa_p_value_upper > 0.9


def test_inferior_family_p_values_also_agree_with_arch() -> None:
    """The external pin has to hold in the regime the accepted run actually lands in."""
    from arch.bootstrap import SPA

    rng = np.random.default_rng(43)
    panel = _inferior_panel(rng, sessions=200, candidates=6)
    block_length = 4.0
    ours = reality_check_and_spa(
        panel,
        benchmark="benchmark",
        replications=5000,
        seed=6,
        block_length=block_length,
        studentize=False,
    )
    reference = SPA(
        panel["benchmark"].to_numpy(),
        panel.drop(columns="benchmark").to_numpy(),
        block_size=int(block_length),
        reps=5000,
        seed=6,
    )
    reference.compute()
    for name, value in (
        ("lower", ours.spa_p_value_lower),
        ("consistent", ours.spa_p_value_consistent),
        ("upper", ours.spa_p_value_upper),
    ):
        assert value == pytest.approx(float(reference.pvalues[name]), abs=0.05)


def test_studentizing_changes_the_answer_when_variances_differ() -> None:
    """Hansen's studentization is not cosmetic when the candidates differ in scale."""
    rng = np.random.default_rng(47)
    benchmark = rng.normal(1.0, 0.3, size=250)
    panel = pd.DataFrame(
        {
            "benchmark": benchmark,
            "steady": benchmark - 0.02 + rng.normal(0.0, 0.05, size=250),
            "wild": benchmark - 0.02 + rng.normal(0.0, 1.20, size=250),
        }
    )
    studentized = reality_check_and_spa(
        panel, benchmark="benchmark", replications=4000, seed=2, block_length=4.0
    )
    raw = reality_check_and_spa(
        panel,
        benchmark="benchmark",
        replications=4000,
        seed=2,
        block_length=4.0,
        studentize=False,
    )
    assert studentized.spa_p_value_consistent != raw.spa_p_value_consistent
    assert studentized.spa_p_value_consistent < raw.spa_p_value_consistent
