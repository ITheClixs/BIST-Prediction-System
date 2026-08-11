"""The encompassing adjustment, and the restriction it is only valid under."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bist_predict.research.inference.nested import (
    clark_west,
    encompassing_adjustment,
)


def _frame(
    targets: np.ndarray, forecasts: dict[str, np.ndarray], tickers: tuple[str, ...] = ("AAA", "BBB")
) -> pd.DataFrame:
    dates = [f"2026-02-{day:02d}" for day in range(1, targets.shape[0] + 1)]
    rows = []
    for name, values in forecasts.items():
        for session, date in enumerate(dates):
            for unit, ticker in enumerate(tickers):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model_name": name,
                        "target": float(targets[session, unit]),
                        "predicted_return": float(values[session, unit]),
                    }
                )
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def nested_frame() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    targets = rng.normal(0.0, 0.02, size=(40, 2))
    fitted = 0.3 * targets + rng.normal(0.0, 0.01, size=(40, 2))
    return _frame(targets, {"fitted": fitted, "zero_return": np.zeros_like(targets)})


def test_the_adjustment_is_twice_the_product_of_target_and_forecast(
    nested_frame: pd.DataFrame,
) -> None:
    adjusted = encompassing_adjustment(
        nested_frame, candidate="fitted", benchmark="zero_return", aggregation="row"
    )
    rows = nested_frame.loc[nested_frame["model_name"] == "fitted"].sort_values(
        ["date", "ticker"], kind="stable"
    )
    expected = 2.0 * rows["target"].to_numpy() * rows["predicted_return"].to_numpy()
    assert adjusted.to_numpy() == pytest.approx(expected)


def test_session_aggregation_averages_over_the_cross_section(
    nested_frame: pd.DataFrame,
) -> None:
    row_level = encompassing_adjustment(
        nested_frame, candidate="fitted", benchmark="zero_return", aggregation="row"
    )
    session = encompassing_adjustment(nested_frame, candidate="fitted", benchmark="zero_return")
    assert len(session) == len(row_level) // 2
    assert float(session.mean()) == pytest.approx(float(row_level.mean()))


def test_a_non_zero_benchmark_is_refused() -> None:
    """The adjustment is derived under nesting; applying it elsewhere tests nothing."""
    rng = np.random.default_rng(17)
    targets = rng.normal(0.0, 0.02, size=(20, 2))
    frame = _frame(
        targets,
        {"fitted": 0.2 * targets, "rolling_mean": np.full_like(targets, 0.001)},
    )
    with pytest.raises(ValueError, match="requires a zero-forecast benchmark"):
        encompassing_adjustment(frame, candidate="fitted", benchmark="rolling_mean")


def test_a_forecast_with_information_is_detected(nested_frame: pd.DataFrame) -> None:
    result = clark_west(
        encompassing_adjustment(nested_frame, candidate="fitted", benchmark="zero_return"),
        candidate="fitted",
        benchmark="zero_return",
    )
    assert result.verdict == "predictive_content"
    assert result.p_value < 0.05


def test_the_test_is_one_sided(nested_frame: pd.DataFrame) -> None:
    """Negative covariance is absence of content, not evidence for the alternative."""
    adjusted = encompassing_adjustment(nested_frame, candidate="fitted", benchmark="zero_return")
    flipped = clark_west(-adjusted, candidate="fitted", benchmark="zero_return")
    assert flipped.p_value > 0.95
    assert flipped.verdict == "no_predictive_content"


def test_the_statistic_matches_an_independent_recomputation(nested_frame: pd.DataFrame) -> None:
    adjusted = encompassing_adjustment(nested_frame, candidate="fitted", benchmark="zero_return")
    result = clark_west(adjusted, candidate="fitted", benchmark="zero_return")
    values = adjusted.to_numpy(dtype=np.float64)
    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / values.size)
    expected = float(values.mean() / np.sqrt(variance / values.size))
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(float(stats.norm.sf(expected)))


def test_a_pure_noise_forecast_is_not_flagged() -> None:
    rng = np.random.default_rng(23)
    targets = rng.normal(0.0, 0.02, size=(200, 2))
    noise = rng.normal(0.0, 0.01, size=(200, 2))
    frame = _frame(targets, {"fitted": noise, "zero_return": np.zeros_like(targets)})
    result = clark_west(
        encompassing_adjustment(frame, candidate="fitted", benchmark="zero_return"),
        candidate="fitted",
        benchmark="zero_return",
    )
    assert result.verdict == "no_predictive_content"


@pytest.mark.parametrize(
    ("series", "message"),
    [
        ([1.0], "at least two observations"),
        ([1.0, float("nan"), 2.0], "must be finite"),
        ([0.0, 0.0, 0.0], "no usable variation"),
    ],
)
def test_degenerate_inputs_are_refused(series: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        clark_west(series, candidate="fitted", benchmark="zero_return")


def test_a_lag_count_beyond_the_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="smaller than the observation count"):
        clark_west([0.1, 0.2, 0.3], candidate="fitted", benchmark="zero_return", lags=3)


def test_the_record_carries_its_sign_convention(nested_frame: pd.DataFrame) -> None:
    result = clark_west(
        encompassing_adjustment(nested_frame, candidate="fitted", benchmark="zero_return"),
        candidate="fitted",
        benchmark="zero_return",
    )
    record = result.to_dict()
    assert "positive mean favours the candidate" in str(record["convention"])
    assert record["verdict"] == result.verdict
