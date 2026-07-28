"""Holm step-down correction pinned to a hand-worked family."""

from __future__ import annotations

import pytest

from bist_predict.research.inference.multiplicity import holm_step_down

# m = 4. Ordered raw values 0.01, 0.02, 0.03, 0.04 scale to
# 4*0.01 = 0.04, 3*0.02 = 0.06, 2*0.03 = 0.06, 1*0.04 = 0.04,
# and the running maximum enforces 0.04, 0.06, 0.06, 0.06.
HAND_FAMILY = {"alpha": 0.01, "beta": 0.02, "gamma": 0.03, "delta": 0.04}
HAND_ADJUSTED = {"alpha": 0.04, "beta": 0.06, "gamma": 0.06, "delta": 0.06}


def test_adjusted_values_match_the_hand_computation() -> None:
    correction = holm_step_down(HAND_FAMILY)
    assert dict(correction.adjusted_p_values) == pytest.approx(HAND_ADJUSTED, abs=1e-15)


def test_monotonicity_is_enforced_across_the_ordered_family() -> None:
    """Without the running maximum, ``delta`` would adjust to 0.04 and be rejected.

    That is the classic Holm implementation error: the raw step-down scale
    factor is not monotone in the ordered index, so a large raw p-value can end
    up with a smaller adjusted value than a smaller one.
    """
    correction = holm_step_down(HAND_FAMILY)
    adjusted = dict(correction.adjusted_p_values)
    assert adjusted["delta"] == pytest.approx(0.06)
    assert correction.rejected == ("alpha",)


def test_holm_is_uniformly_less_conservative_than_bonferroni() -> None:
    correction = holm_step_down(HAND_FAMILY)
    for name, adjusted in correction.adjusted_p_values:
        assert adjusted <= min(len(HAND_FAMILY) * HAND_FAMILY[name], 1.0) + 1e-15


def test_adjusted_values_are_non_decreasing_in_the_raw_order() -> None:
    correction = holm_step_down({"a": 0.001, "b": 0.2, "c": 0.02, "d": 0.9, "e": 0.05})
    values = [value for _, value in correction.adjusted_p_values]
    assert values == sorted(values)


def test_a_single_hypothesis_is_left_unadjusted() -> None:
    correction = holm_step_down({"only": 0.03})
    assert dict(correction.adjusted_p_values)["only"] == pytest.approx(0.03)
    assert correction.rejected == ("only",)


def test_six_borderline_p_values_all_survive_correction() -> None:
    """Six models each at p = 0.04 give a 26.5% chance of one false positive."""
    family = {f"model_{index}": 0.04 for index in range(6)}
    correction = holm_step_down(family)
    assert correction.rejected == ()
    assert all(value == pytest.approx(0.24) for _, value in correction.adjusted_p_values)


def test_adjusted_values_are_capped_at_one() -> None:
    correction = holm_step_down({"a": 0.6, "b": 0.7, "c": 0.8})
    assert all(value <= 1.0 for _, value in correction.adjusted_p_values)


def test_out_of_range_p_value_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"p-value must lie in \[0, 1\]"):
        holm_step_down({"a": 1.5})


def test_empty_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one hypothesis"):
        holm_step_down({})


def test_out_of_range_alpha_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must lie strictly"):
        holm_step_down(HAND_FAMILY, alpha=1.0)
