"""Diebold-Mariano behaviour, sign conventions, and session aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bist_predict.research.inference.forecast_tests import (
    diebold_mariano,
    squared_error_differential,
)


def _prediction_frame(
    dates: list[str],
    tickers: list[str],
    targets: dict[tuple[str, str], float],
    predictions: dict[str, dict[tuple[str, str], float]],
) -> pd.DataFrame:
    rows = []
    for model_name, values in predictions.items():
        for date in dates:
            for ticker in tickers:
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model_name": model_name,
                        "target": targets[(date, ticker)],
                        "predicted_return": values[(date, ticker)],
                    }
                )
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def two_model_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = [f"2025-01-{day:02d}" for day in range(1, 41)]
    tickers = ["AAA", "BBB"]
    targets = {(date, ticker): float(rng.normal(0.0, 0.02)) for date in dates for ticker in tickers}
    sharp = {key: value * 0.7 for key, value in targets.items()}
    flat = {key: 0.0 for key in targets}
    return _prediction_frame(dates, tickers, targets, {"sharp": sharp, "flat": flat})


def test_statistic_matches_an_independent_recomputation(two_model_frame: pd.DataFrame) -> None:
    """Recompute the Harvey-Leybourne-Newbold statistic from first principles."""
    differential = squared_error_differential(
        two_model_frame, candidate="sharp", benchmark="flat", aggregation="session"
    )
    result = diebold_mariano(differential, candidate="sharp", benchmark="flat")

    values = differential.to_numpy(dtype=np.float64)
    count = values.size
    centred = values - values.mean()
    gamma_zero = float(np.dot(centred, centred) / count)
    raw = float(values.mean() / np.sqrt(gamma_zero / count))
    corrected = raw * np.sqrt((count + 1 - 2 + 0) / count)
    expected_p = 2.0 * stats.t.sf(abs(corrected), df=count - 1)

    assert result.observation_count == count
    assert result.statistic == pytest.approx(corrected, rel=1e-12)
    assert result.p_value == pytest.approx(expected_p, rel=1e-12)


def test_small_sample_correction_is_actually_applied(two_model_frame: pd.DataFrame) -> None:
    """Dropping the HLN factor would leave the statistic larger in magnitude."""
    differential = squared_error_differential(
        two_model_frame, candidate="sharp", benchmark="flat", aggregation="session"
    )
    result = diebold_mariano(differential, candidate="sharp", benchmark="flat")
    values = differential.to_numpy(dtype=np.float64)
    count = values.size
    centred = values - values.mean()
    uncorrected = float(values.mean() / np.sqrt(float(np.dot(centred, centred) / count) / count))
    assert abs(result.statistic) < abs(uncorrected)
    assert abs(result.statistic) == pytest.approx(
        abs(uncorrected) * np.sqrt((count - 1) / count), rel=1e-12
    )


def test_session_aggregation_collapses_same_date_rows(two_model_frame: pd.DataFrame) -> None:
    by_session = squared_error_differential(
        two_model_frame, candidate="sharp", benchmark="flat", aggregation="session"
    )
    by_row = squared_error_differential(
        two_model_frame, candidate="sharp", benchmark="flat", aggregation="row"
    )
    assert len(by_session) == two_model_frame["date"].nunique()
    assert len(by_row) == len(two_model_frame) // 2
    assert by_session.mean() == pytest.approx(by_row.mean(), rel=1e-12)


def test_row_aggregation_reports_a_smaller_p_value_than_session_aggregation(
    two_model_frame: pd.DataFrame,
) -> None:
    """Treating correlated same-date rows as independent manufactures precision."""
    session = diebold_mariano(
        squared_error_differential(two_model_frame, candidate="sharp", benchmark="flat"),
        candidate="sharp",
        benchmark="flat",
    )
    row = diebold_mariano(
        squared_error_differential(
            two_model_frame, candidate="sharp", benchmark="flat", aggregation="row"
        ),
        candidate="sharp",
        benchmark="flat",
        aggregation="row",
    )
    assert row.p_value < session.p_value


def test_a_strictly_better_candidate_gets_a_negative_mean_and_the_right_verdict(
    two_model_frame: pd.DataFrame,
) -> None:
    result = diebold_mariano(
        squared_error_differential(two_model_frame, candidate="sharp", benchmark="flat"),
        candidate="sharp",
        benchmark="flat",
    )
    assert result.mean_differential < 0.0
    assert result.p_value < 0.05
    assert result.verdict == "candidate_better"


def test_swapping_the_roles_flips_the_verdict_but_not_the_p_value(
    two_model_frame: pd.DataFrame,
) -> None:
    """Guards against reading a signed statistic without its convention."""
    forward = diebold_mariano(
        squared_error_differential(two_model_frame, candidate="sharp", benchmark="flat"),
        candidate="sharp",
        benchmark="flat",
    )
    reverse = diebold_mariano(
        squared_error_differential(two_model_frame, candidate="flat", benchmark="sharp"),
        candidate="flat",
        benchmark="sharp",
    )
    assert reverse.statistic == pytest.approx(-forward.statistic, rel=1e-12)
    assert reverse.p_value == pytest.approx(forward.p_value, rel=1e-12)
    assert forward.verdict == "candidate_better"
    assert reverse.verdict == "benchmark_better"


def test_indistinguishable_models_report_no_winner() -> None:
    rng = np.random.default_rng(21)
    differential = rng.normal(0.0, 1e-5, size=90)
    result = diebold_mariano(differential, candidate="a", benchmark="b")
    assert result.p_value > 0.05
    assert result.verdict == "indistinguishable"


def test_numerically_identical_models_are_rejected_rather_than_divided_by_zero() -> None:
    """A zero-variance differential must not produce an astronomical statistic."""
    with pytest.raises(ValueError, match="no usable variation"):
        diebold_mariano(np.zeros(50), candidate="a", benchmark="b")


def test_an_economically_constant_differential_is_rejected() -> None:
    """Scale-relative guard, on a series whose variance is nonzero but negligible.

    ``np.zeros(50)`` and ``np.full(50, x)`` both have a sample variance of
    exactly zero, so a guard written against zero catches them too and proves
    nothing. Two models that agree to fourteen digits give a differential of
    order one whose standard error is of order 1e-14: strictly positive, and
    meaningless. Only a guard measured against the scale of the series rejects
    it, and without one the statistic is of order 1e13.
    """
    differential = 1.0 + 1e-14 * np.arange(50, dtype=np.float64)
    scale = float(np.std(differential, ddof=1)) / np.sqrt(differential.size)
    assert 0.0 < scale < 1e-12 * float(np.mean(np.abs(differential)))
    assert abs(differential.mean() / scale) > 1e13
    with pytest.raises(ValueError, match="no usable variation"):
        diebold_mariano(differential, candidate="a", benchmark="b")


def test_mismatched_targets_are_rejected(two_model_frame: pd.DataFrame) -> None:
    corrupted = two_model_frame.copy()
    mask = corrupted["model_name"] == "flat"
    corrupted.loc[mask, "target"] = corrupted.loc[mask, "target"] + 0.01
    with pytest.raises(ValueError, match="disagree on the realised target"):
        squared_error_differential(corrupted, candidate="sharp", benchmark="flat")


def test_duplicate_date_ticker_rows_are_rejected(two_model_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([two_model_frame, two_model_frame.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate date-ticker rows"):
        squared_error_differential(duplicated, candidate="sharp", benchmark="flat")


def test_unknown_model_is_rejected(two_model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows for model"):
        squared_error_differential(two_model_frame, candidate="missing", benchmark="flat")


def test_identical_candidate_and_benchmark_is_rejected(two_model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be different models"):
        squared_error_differential(two_model_frame, candidate="flat", benchmark="flat")
