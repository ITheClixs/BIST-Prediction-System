"""Sharpe-ratio inference: variance guard, annualisation, PSR and DSR."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from bist_predict.research.inference.hac import automatic_bartlett_bandwidth
from bist_predict.research.inference.sharpe import (
    EULER_MASCHERONI,
    annualisation_factor,
    deflated_sharpe_ratio,
    deflated_sharpe_threshold,
    per_period_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_inference,
    sharpe_standard_error,
)


def _constant_returns_from_price_ratios(rate: float, count: int) -> np.ndarray:
    """Build a constant return series the way a backtest ledger builds one."""
    prices = np.array([100.0 * (1.0 + rate) ** step for step in range(count + 1)])
    return prices[1:] / prices[:-1] - 1.0


def test_constant_return_series_has_a_nonzero_floating_point_deviation() -> None:
    """The premise of the variance-guard test: ``std > 0`` really does hold here."""
    returns = _constant_returns_from_price_ratios(0.0007, 60)
    deviation = float(np.std(returns, ddof=1))
    assert deviation > 0.0
    assert deviation < 1e-15


def test_scale_relative_guard_returns_zero_for_a_constant_series() -> None:
    """A ``std > 0`` guard would divide by 1.2e-16 and report Sharpe of order 1e14."""
    returns = _constant_returns_from_price_ratios(0.0007, 60)
    naive = float(np.mean(returns) / np.std(returns, ddof=1))
    assert abs(naive) > 1e10
    assert per_period_sharpe_ratio(returns) == 0.0


def test_exactly_zero_returns_give_a_zero_sharpe() -> None:
    assert per_period_sharpe_ratio(np.zeros(40)) == 0.0


def test_the_guard_does_not_suppress_a_genuinely_small_but_real_sharpe() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0.00001, 0.01, size=2000)
    assert per_period_sharpe_ratio(returns) != 0.0


def test_per_period_sharpe_matches_the_direct_ratio() -> None:
    rng = np.random.default_rng(8)
    returns = rng.normal(0.001, 0.01, size=300)
    expected = float(np.mean(returns) / np.std(returns, ddof=1))
    assert per_period_sharpe_ratio(returns) == pytest.approx(expected, rel=1e-12)


def test_independent_returns_recover_the_square_root_rule() -> None:
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0004, 0.01, size=20_000)
    assert annualisation_factor(returns, periods_per_year=252) == pytest.approx(
        math.sqrt(252.0), rel=0.06
    )


def test_positive_autocorrelation_lowers_the_annualisation_factor() -> None:
    """Lo (2002): the square-root rule overstates the Sharpe of a persistent series."""
    rng = np.random.default_rng(23)
    innovations = rng.normal(0.0, 0.01, size=20_000)
    returns = np.zeros(20_000)
    for index in range(1, returns.size):
        returns[index] = 0.4 * returns[index - 1] + innovations[index]
    assert annualisation_factor(returns, periods_per_year=252) < math.sqrt(252.0)


def test_negative_autocorrelation_raises_the_annualisation_factor() -> None:
    rng = np.random.default_rng(29)
    innovations = rng.normal(0.0, 0.01, size=20_000)
    returns = np.zeros(20_000)
    for index in range(1, returns.size):
        returns[index] = -0.4 * returns[index - 1] + innovations[index]
    assert annualisation_factor(returns, periods_per_year=252) > math.sqrt(252.0)


def test_annualisation_factor_for_one_period_is_one() -> None:
    rng = np.random.default_rng(31)
    assert annualisation_factor(rng.normal(size=200), periods_per_year=1) == pytest.approx(1.0)


def test_probabilistic_sharpe_matches_the_gaussian_closed_form() -> None:
    """For symmetric mesokurtic input, PSR reduces to a Lo (2002) z-score."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0006, 0.01, size=5000)
    estimate = per_period_sharpe_ratio(returns)
    skewness = float(stats.skew(returns, bias=True))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=True))
    variance = 1.0 - skewness * estimate + 0.25 * (kurtosis - 1.0) * estimate**2
    expected = float(stats.norm.cdf(estimate * math.sqrt(returns.size - 1) / math.sqrt(variance)))
    assert probabilistic_sharpe_ratio(returns) == pytest.approx(expected, rel=1e-12)


def test_negative_skew_lowers_the_probabilistic_sharpe() -> None:
    """Two series with the same Sharpe ratio are not equally trustworthy."""
    rng = np.random.default_rng(13)
    symmetric = rng.normal(0.0, 1.0, size=4000)
    skewed = -(np.abs(rng.normal(0.0, 1.0, size=4000)) ** 1.7)
    symmetric = (symmetric - symmetric.mean()) / symmetric.std(ddof=1)
    skewed = (skewed - skewed.mean()) / skewed.std(ddof=1)
    target = 0.05
    symmetric = symmetric * 0.01 + target * 0.01
    skewed = skewed * 0.01 + target * 0.01
    assert per_period_sharpe_ratio(symmetric) == pytest.approx(
        per_period_sharpe_ratio(skewed), abs=1e-9
    )
    assert stats.skew(skewed, bias=True) < -0.5
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)


def test_standard_error_matches_the_lo_formula() -> None:
    assert sharpe_standard_error(0.2, 100) == pytest.approx(math.sqrt((1 + 0.02) / 100), rel=1e-15)


def test_deflated_threshold_matches_the_hand_computation() -> None:
    trials, variance = 36, 0.04
    first = float(stats.norm.ppf(1.0 - 1.0 / trials))
    second = float(stats.norm.ppf(1.0 - 1.0 / (trials * math.e)))
    expected = math.sqrt(variance) * ((1 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)
    assert deflated_sharpe_threshold(trials, variance) == pytest.approx(expected, rel=1e-12)


def test_deflated_threshold_grows_with_the_number_of_trials() -> None:
    """More configurations searched means a higher bar for the winner."""
    thresholds = [deflated_sharpe_threshold(count, 0.04) for count in (2, 10, 100, 1000)]
    assert all(later > earlier for earlier, later in zip(thresholds, thresholds[1:], strict=False))
    assert thresholds[0] > 0.0


def test_a_single_trial_is_not_deflated() -> None:
    assert deflated_sharpe_threshold(1, 0.04) == 0.0


def test_zero_dispersion_across_trials_is_not_deflated() -> None:
    assert deflated_sharpe_threshold(50, 0.0) == 0.0


def test_deflated_sharpe_never_exceeds_the_undeflated_one() -> None:
    rng = np.random.default_rng(19)
    returns = rng.normal(0.0008, 0.01, size=250)
    undeflated = probabilistic_sharpe_ratio(returns)
    deflated = deflated_sharpe_ratio(returns, trial_count=64, trial_sharpe_variance=0.02)
    assert deflated < undeflated


def test_inference_record_is_internally_consistent() -> None:
    rng = np.random.default_rng(37)
    returns = rng.normal(0.0005, 0.012, size=200)
    record = sharpe_inference(
        returns, periods_per_year=252, trial_count=36, trial_sharpe_variance=0.03
    )
    assert record.observation_count == 200
    assert record.annualised_sharpe == pytest.approx(record.per_period_sharpe * math.sqrt(252))
    assert record.deflated_sharpe_ratio <= record.probabilistic_sharpe_ratio
    assert record.verdict == "skill_not_established"
    assert set(record.to_dict()) >= {"deflated_sharpe_ratio", "verdict", "kurtosis"}


def test_a_short_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least three return observations"):
        per_period_sharpe_ratio([0.01, 0.02])


def test_non_finite_returns_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        per_period_sharpe_ratio([0.01, float("inf"), 0.02])


def test_non_positive_trial_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="trial_count must be positive"):
        deflated_sharpe_threshold(0, 0.04)


def test_annualisation_lags_are_truncated_to_an_estimable_bandwidth() -> None:
    """A 120-session sample cannot estimate 251 autocorrelations.

    Summing to ``q - 1`` weights each noisy high-lag estimate by ``q - k``,
    which is of order 250 here, so the untruncated factor is dominated by
    sampling noise and can land multiple times away from the square-root rule.
    """
    rng = np.random.default_rng(101)
    returns = rng.normal(0.0004, 0.012, size=120)
    truncated = annualisation_factor(returns, periods_per_year=252)
    untruncated = annualisation_factor(returns, periods_per_year=252, max_lags=119)
    assert truncated == pytest.approx(math.sqrt(252.0), rel=0.35)
    assert abs(untruncated - math.sqrt(252.0)) > abs(truncated - math.sqrt(252.0))


def test_the_default_lag_count_is_the_newey_west_bandwidth() -> None:
    rng = np.random.default_rng(103)
    returns = rng.normal(0.0004, 0.012, size=120)
    record = sharpe_inference(
        returns, periods_per_year=252, trial_count=8, trial_sharpe_variance=0.01
    )
    assert record.autocorrelation_lags == automatic_bartlett_bandwidth(120)
    assert record.autocorrelation_adjusted_annualised_sharpe == pytest.approx(
        record.per_period_sharpe * annualisation_factor(returns, periods_per_year=252)
    )


def test_negative_max_lags_is_rejected() -> None:
    rng = np.random.default_rng(105)
    with pytest.raises(ValueError, match="max_lags must be non-negative"):
        annualisation_factor(rng.normal(size=50), periods_per_year=252, max_lags=-1)
