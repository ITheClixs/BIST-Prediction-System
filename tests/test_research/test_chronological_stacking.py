"""Strict OOF stacking and chronological calibration tests."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pandas.testing as pdt
import pytest

from bist_predict.research.chronological_stacking import (
    ChronologicalStackingPipeline,
    StackingLeakageError,
    StackingPeriods,
)


def _periods() -> StackingPeriods:
    return StackingPeriods(
        base_training_end="2023-12-29",
        stacking_start="2024-01-02",
        stacking_end="2024-01-31",
        calibration_start="2024-02-01",
        calibration_end="2024-02-29",
        test_start="2024-03-01",
        test_end="2024-03-29",
    )


def _long_predictions(start: str, periods: int, label: str) -> pd.DataFrame:
    sessions = pd.bdate_range(start, periods=periods, tz="UTC")
    records: list[dict[str, object]] = []
    for index, session in enumerate(sessions):
        row_id = f"{label}-{index:03d}"
        target = 0.01 if index % 2 == 0 else -0.008
        for model_index, model in enumerate(("ridge", "xgboost")):
            probability = 0.68 if target > 0.0 else 0.32
            probability += (model_index - 0.5) * 0.04
            predicted_return = target * (0.7 + model_index * 0.1)
            records.append(
                {
                    "row_id": row_id,
                    "date": session.date().isoformat(),
                    "ticker": ("GARAN", "THYAO")[index % 2],
                    "base_model": model,
                    "base_model_training_end": "2023-12-29",
                    "oof_fold_id": f"{label}-fold-{index // 4:02d}",
                    "prediction_timestamp": session + pd.Timedelta(hours=18, minutes=10),
                    "target": target,
                    "target_direction": int(target > 0.0),
                    "predicted_probability": probability,
                    "predicted_return": predicted_return,
                    "base_training_row_ids": ("base-001", "base-002"),
                }
            )
    return pd.DataFrame.from_records(records)


def test_stacker_rejects_base_prediction_for_its_own_training_row() -> None:
    stacking = _long_predictions("2024-01-02", 20, "stack")
    stacking.at[0, "base_training_row_ids"] = ("stack-000", "base-002")

    with pytest.raises(StackingLeakageError, match="stack-000.*ridge"):
        ChronologicalStackingPipeline(_periods()).fit_predict(
            stacking,
            _long_predictions("2024-02-01", 20, "calibration"),
            _long_predictions("2024-03-01", 20, "test"),
        )


def test_periods_must_be_strictly_chronological_and_non_overlapping() -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        replace(_periods(), calibration_end="2024-03-05", test_start="2024-03-01")


def test_pipeline_persists_oof_lineage_and_calibrates_before_test() -> None:
    stacking = _long_predictions("2024-01-02", 20, "stack")
    calibration = _long_predictions("2024-02-01", 20, "calibration")
    test = _long_predictions("2024-03-01", 20, "test")
    pipeline = ChronologicalStackingPipeline(_periods())

    result = pipeline.fit_predict(stacking, calibration, test)

    assert list(result.stacking_lineage.columns) == [
        "row_id",
        "base_model",
        "base_model_training_end",
        "oof_fold_id",
        "prediction_timestamp",
    ]
    assert len(result.stacking_lineage) == len(stacking)
    assert result.test_predictions["predicted_probability"].between(0.0, 1.0).all()
    assert set(result.calibration_metrics) == {
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "calibration_slope",
        "calibration_intercept",
        "reliability_buckets",
    }


def test_test_labels_cannot_change_stacker_or_calibrator_outputs() -> None:
    stacking = _long_predictions("2024-01-02", 20, "stack")
    calibration = _long_predictions("2024-02-01", 20, "calibration")
    test = _long_predictions("2024-03-01", 20, "test")
    expected = ChronologicalStackingPipeline(_periods()).fit_predict(stacking, calibration, test)
    changed_test = test.copy()
    changed_test["target"] *= -100.0
    changed_test["target_direction"] = 1 - changed_test["target_direction"]

    actual = ChronologicalStackingPipeline(_periods()).fit_predict(
        stacking, calibration, changed_test
    )

    pdt.assert_series_equal(
        actual.test_predictions["predicted_probability"],
        expected.test_predictions["predicted_probability"],
    )
    pdt.assert_series_equal(
        actual.test_predictions["predicted_return"],
        expected.test_predictions["predicted_return"],
    )
    assert actual.calibration_metrics == expected.calibration_metrics
