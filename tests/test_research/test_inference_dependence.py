"""Cross-sectional dependence diagnostics on constructed panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_predict.research.inference.dependence import (
    cross_sectional_dependence,
    effective_sample_size,
    variance_inflation_factor,
)


def _panel(values: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    sessions, units = values.shape
    assert units == len(tickers)
    rows = [
        {
            "date": f"2025-01-{session + 1:03d}",
            "ticker": tickers[unit],
            "value": values[session, unit],
        }
        for session in range(sessions)
        for unit in range(units)
    ]
    return pd.DataFrame.from_records(rows)


def test_variance_inflation_matches_the_equicorrelation_formula() -> None:
    assert variance_inflation_factor(4, 0.5) == pytest.approx(1.0 + 3 * 0.5)


def test_negative_average_correlation_is_clipped_to_one() -> None:
    """The evaluation never claims pooling buys more information than independence."""
    assert variance_inflation_factor(4, -0.3) == 1.0


def test_effective_sample_size_divides_by_the_inflation_factor() -> None:
    assert effective_sample_size(480, 2.5) == pytest.approx(192.0)


def test_perfectly_correlated_units_collapse_to_one_effective_unit() -> None:
    rng = np.random.default_rng(1)
    common = rng.normal(size=(200, 1))
    values = np.hstack([common, common, common, common])
    diagnostic = cross_sectional_dependence(
        _panel(values, ["a", "b", "c", "d"]), value_column="value"
    )
    assert diagnostic.mean_pairwise_correlation == pytest.approx(1.0, abs=1e-12)
    assert diagnostic.variance_inflation_factor == pytest.approx(4.0, abs=1e-12)
    assert diagnostic.effective_row_count == pytest.approx(diagnostic.row_count / 4.0)


def test_independent_units_leave_the_sample_size_almost_intact() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4000, 4))
    diagnostic = cross_sectional_dependence(
        _panel(values, ["a", "b", "c", "d"]), value_column="value"
    )
    assert abs(diagnostic.mean_pairwise_correlation) < 0.05
    assert diagnostic.effective_row_count == pytest.approx(diagnostic.row_count, rel=0.2)


def test_a_common_factor_is_detected_at_roughly_its_true_loading() -> None:
    """Equicorrelated construction: rho = w^2 / (w^2 + 1) for unit-variance parts."""
    rng = np.random.default_rng(3)
    weight = 1.0
    factor = rng.normal(size=(6000, 1))
    idiosyncratic = rng.normal(size=(6000, 4))
    values = weight * factor + idiosyncratic
    diagnostic = cross_sectional_dependence(
        _panel(values, ["a", "b", "c", "d"]), value_column="value"
    )
    expected = weight**2 / (weight**2 + 1.0)
    assert diagnostic.mean_pairwise_correlation == pytest.approx(expected, abs=0.03)


def test_pair_count_and_shape_are_reported() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(50, 4))
    diagnostic = cross_sectional_dependence(
        _panel(values, ["a", "b", "c", "d"]), value_column="value"
    )
    assert diagnostic.unit_count == 4
    assert diagnostic.session_count == 50
    assert diagnostic.row_count == 200
    assert diagnostic.pair_count == 6
    assert set(diagnostic.to_dict()) == {
        "unit_count",
        "session_count",
        "row_count",
        "mean_pairwise_correlation",
        "variance_inflation_factor",
        "effective_row_count",
        "pair_count",
    }


def test_duplicate_session_unit_rows_are_rejected() -> None:
    rng = np.random.default_rng(5)
    frame = _panel(rng.normal(size=(10, 2)), ["a", "b"])
    duplicated = pd.concat([frame, frame.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="one row per session and unit"):
        cross_sectional_dependence(duplicated, value_column="value")


def test_a_single_unit_is_rejected() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="at least two units"):
        cross_sectional_dependence(_panel(rng.normal(size=(10, 1)), ["a"]), value_column="value")


def test_missing_value_column_is_rejected() -> None:
    rng = np.random.default_rng(7)
    frame = _panel(rng.normal(size=(10, 2)), ["a", "b"])
    with pytest.raises(ValueError, match="missing required column"):
        cross_sectional_dependence(frame, value_column="absent")
