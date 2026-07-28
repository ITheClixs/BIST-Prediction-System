"""Autocovariance and long-run variance pinned to hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.research.inference.hac import (
    automatic_bartlett_bandwidth,
    bartlett_long_run_variance,
    mean_standard_error,
    sample_autocovariance,
)

# mean 2.5, centred deviations (-1.5, -0.5, 0.5, 1.5)
SERIES = (1.0, 2.0, 3.0, 4.0)
# gamma_0 = (2.25 + 0.25 + 0.25 + 2.25) / 4
HAND_GAMMA_0 = 1.25
# gamma_1 = ((-0.5)(-1.5) + (0.5)(-0.5) + (1.5)(0.5)) / 4
HAND_GAMMA_1 = 0.3125


def test_lag_zero_autocovariance_matches_hand_computation() -> None:
    assert sample_autocovariance(SERIES, 0) == pytest.approx(HAND_GAMMA_0, abs=1e-15)


def test_lag_one_autocovariance_matches_hand_computation() -> None:
    assert sample_autocovariance(SERIES, 1) == pytest.approx(HAND_GAMMA_1, abs=1e-15)


def test_autocovariance_uses_the_full_sample_normalisation() -> None:
    """Divide by ``n``, not by ``n - k``.

    Diebold and Mariano (1995) and Newey and West (1987) share the ``1/n``
    convention. The ``1/(n-k)`` alternative would return 0.4166... at lag one.
    """
    assert sample_autocovariance(SERIES, 1) != pytest.approx(HAND_GAMMA_1 * 4.0 / 3.0)


def test_bartlett_variance_with_zero_lags_is_the_sample_variance() -> None:
    assert bartlett_long_run_variance(SERIES, lags=0) == pytest.approx(HAND_GAMMA_0, abs=1e-15)


def test_bartlett_variance_with_one_lag_matches_hand_computation() -> None:
    # Omega = gamma_0 + 2 * (1 - 1/2) * gamma_1
    expected = HAND_GAMMA_0 + HAND_GAMMA_1
    assert bartlett_long_run_variance(SERIES, lags=1) == pytest.approx(expected, abs=1e-15)


def test_bartlett_weights_decay_linearly_in_the_lag() -> None:
    """Two lags weight gamma_1 by 2/3 and gamma_2 by 1/3."""
    gamma_2 = sample_autocovariance(SERIES, 2)
    expected = HAND_GAMMA_0 + 2.0 * (2.0 / 3.0) * HAND_GAMMA_1 + 2.0 * (1.0 / 3.0) * gamma_2
    assert bartlett_long_run_variance(SERIES, lags=2) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (100, 4),  # 4 * 1.0000 ** (2/9) = 4.000
        (120, 4),  # 4 * 1.2000 ** (2/9) = 4.165
        (25, 2),  # 4 * 0.2500 ** (2/9) = 2.939
        (1000, 6),  # 4 * 10.000 ** (2/9) = 6.672
    ],
)
def test_automatic_bandwidth_matches_the_newey_west_rule(count: int, expected: int) -> None:
    assert automatic_bartlett_bandwidth(count) == expected
    assert automatic_bartlett_bandwidth(count) == int(np.floor(4.0 * (count / 100.0) ** (2 / 9)))


def test_positive_autocorrelation_inflates_the_long_run_variance() -> None:
    """An AR(1) series has a long-run variance above its sample variance."""
    rng = np.random.default_rng(11)
    innovations = rng.normal(size=4000)
    series = np.zeros(4000)
    for index in range(1, 4000):
        series[index] = 0.6 * series[index - 1] + innovations[index]
    short_run = bartlett_long_run_variance(series, lags=0)
    long_run = bartlett_long_run_variance(series, lags=None)
    assert long_run > 1.5 * short_run


def test_mean_standard_error_shrinks_with_the_square_root_of_the_count() -> None:
    rng = np.random.default_rng(3)
    small = rng.normal(size=400)
    large = np.concatenate([small] * 4)
    ratio = mean_standard_error(small, lags=0) / mean_standard_error(large, lags=0)
    assert ratio == pytest.approx(2.0, rel=1e-9)


def test_negative_lag_is_rejected() -> None:
    with pytest.raises(ValueError, match="lag must be non-negative"):
        sample_autocovariance(SERIES, -1)


def test_lag_at_or_beyond_the_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the observation count"):
        sample_autocovariance(SERIES, 4)


def test_non_finite_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        bartlett_long_run_variance((1.0, float("nan"), 3.0))


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        sample_autocovariance((), 0)
