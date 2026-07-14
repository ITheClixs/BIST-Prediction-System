"""Strict out-of-fold stacking with a separate chronological calibrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from bist_predict.models.calibration import PlattCalibrator
from bist_predict.models.ensemble import EnsembleCombiner

LINEAGE_COLUMNS = (
    "row_id",
    "base_model",
    "base_model_training_end",
    "oof_fold_id",
    "prediction_timestamp",
)


class StackingLeakageError(ValueError):
    """Raised when a meta-feature was generated in-sample or out of sequence."""


@dataclass(frozen=True)
class StackingPeriods:
    """Non-overlapping base, stacking, calibration, and final-test periods."""

    base_training_end: str
    stacking_start: str
    stacking_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        boundaries = [
            pd.Timestamp(self.base_training_end),
            pd.Timestamp(self.stacking_start),
            pd.Timestamp(self.stacking_end),
            pd.Timestamp(self.calibration_start),
            pd.Timestamp(self.calibration_end),
            pd.Timestamp(self.test_start),
            pd.Timestamp(self.test_end),
        ]
        if not (
            boundaries[0]
            < boundaries[1]
            <= boundaries[2]
            < boundaries[3]
            <= boundaries[4]
            < boundaries[5]
            <= boundaries[6]
        ):
            raise ValueError("stacking periods must be strictly ordered and non-overlapping")


@dataclass(frozen=True)
class ChronologicalStackingResult:
    """Final predictions plus auditable stacking and calibration evidence."""

    test_predictions: pd.DataFrame
    stacking_lineage: pd.DataFrame
    calibration_fit_metrics: dict[str, object]
    final_test_calibration_metrics: dict[str, object]


def _validated_long(frame: pd.DataFrame, periods: StackingPeriods) -> pd.DataFrame:
    required = {
        *LINEAGE_COLUMNS,
        "date",
        "ticker",
        "target",
        "target_direction",
        "predicted_probability",
        "predicted_return",
        "base_training_row_ids",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"stacking input missing columns: {', '.join(missing)}")
    if frame.duplicated(["row_id", "base_model"]).any():
        raise ValueError("duplicate base prediction for stacking row")

    working = frame.copy()
    working["_date"] = pd.to_datetime(working["date"], errors="coerce")
    working["_training_end"] = pd.to_datetime(working["base_model_training_end"], errors="coerce")
    working["_prediction_time"] = pd.to_datetime(
        working["prediction_timestamp"], utc=True, errors="coerce"
    )
    if working[["_date", "_training_end", "_prediction_time"]].isna().any().any():
        raise ValueError("stacking timestamps must be valid")
    if (working["_training_end"] > pd.Timestamp(periods.base_training_end)).any():
        raise StackingLeakageError("base model training extends beyond base-training period")
    prediction_dates = working["_prediction_time"].dt.tz_convert(None).dt.normalize()
    working["_prediction_date"] = prediction_dates
    if (working["_training_end"] >= prediction_dates).any():
        raise StackingLeakageError("base model training must precede its prediction")
    if (prediction_dates > working["_date"].dt.normalize()).any():
        raise StackingLeakageError("prediction timestamp cannot postdate its row session")

    for row in working.itertuples(index=False):
        training_ids = row.base_training_row_ids
        if not isinstance(training_ids, (tuple, list, set, frozenset)):
            raise ValueError("base_training_row_ids must be an explicit collection")
        if row.row_id in training_ids:
            raise StackingLeakageError(
                f"row {row.row_id} was included in {row.base_model} base-model training"
            )

    grouped = working.groupby("row_id", sort=False)
    if grouped["target"].nunique().gt(1).any():
        raise ValueError("base models disagree on stacking target")
    if grouped["target_direction"].nunique().gt(1).any():
        raise ValueError("base models disagree on stacking direction")
    if grouped["oof_fold_id"].nunique().gt(1).any():
        raise ValueError("base models disagree on OOF fold identity")
    model_counts = grouped["base_model"].nunique()
    if model_counts.nunique() != 1:
        raise ValueError("every stacking row must contain the same base models")
    if not working["predicted_probability"].between(0.0, 1.0).all():
        raise ValueError("base probabilities must lie in [0, 1]")
    return working


def _validate_period(
    frame: pd.DataFrame,
    start: str,
    end: str,
    label: str,
) -> None:
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    if not frame["_date"].between(lower, upper, inclusive="both").all():
        raise StackingLeakageError(f"{label} rows fall outside their declared period")
    if not frame["_prediction_date"].between(lower, upper, inclusive="both").all():
        raise StackingLeakageError(
            f"{label} prediction timestamps fall outside their declared period"
        )


def _model_matrices(
    frame: pd.DataFrame,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    pd.DataFrame,
]:
    metadata = (
        frame.sort_values(["_date", "ticker", "row_id", "base_model"], kind="stable")
        .drop_duplicates("row_id")[["row_id", "date", "ticker", "target", "target_direction"]]
        .reset_index(drop=True)
    )
    row_ids = list(metadata["row_id"])
    model_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, model_rows in frame.groupby("base_model", sort=True):
        indexed = model_rows.set_index("row_id").loc[row_ids]
        model_predictions[str(model_name)] = (
            indexed["predicted_probability"].to_numpy(dtype=np.float64),
            indexed["predicted_return"].to_numpy(dtype=np.float64),
        )
    return model_predictions, metadata


def _reliability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    buckets: list[dict[str, float | int]] = []
    expected_error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        include_upper = index == len(edges) - 2
        mask = (clipped >= lower) & ((clipped <= upper) if include_upper else (clipped < upper))
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_probability = float(np.mean(clipped[mask]))
        observed_rate = float(np.mean(labels[mask]))
        expected_error += count / len(labels) * abs(mean_probability - observed_rate)
        buckets.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )

    slope: float | None = None
    intercept: float | None = None
    if len(np.unique(labels)) == 2:
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        diagnostic = LogisticRegression(C=1e6, max_iter=1_000, random_state=42)
        diagnostic.fit(logits, labels)
        slope = float(diagnostic.coef_[0, 0])
        intercept = float(diagnostic.intercept_[0])
    return {
        "brier_score": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "expected_calibration_error": float(expected_error),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_buckets": buckets,
    }


class ChronologicalStackingPipeline:
    """Fit a stacker, then calibrate it, then touch final-test features once."""

    def __init__(self, periods: StackingPeriods) -> None:
        self._periods = periods

    def fit_predict(
        self,
        stacking_rows: pd.DataFrame,
        calibration_rows: pd.DataFrame,
        test_rows: pd.DataFrame,
    ) -> ChronologicalStackingResult:
        stacking = _validated_long(stacking_rows, self._periods)
        calibration = _validated_long(calibration_rows, self._periods)
        test = _validated_long(test_rows, self._periods)
        _validate_period(
            stacking,
            self._periods.stacking_start,
            self._periods.stacking_end,
            "stacking",
        )
        _validate_period(
            calibration,
            self._periods.calibration_start,
            self._periods.calibration_end,
            "calibration",
        )
        _validate_period(test, self._periods.test_start, self._periods.test_end, "test")

        stacking_predictions, stacking_metadata = _model_matrices(stacking)
        calibration_predictions, calibration_metadata = _model_matrices(calibration)
        test_predictions, test_metadata = _model_matrices(test)
        if not (set(stacking_predictions) == set(calibration_predictions) == set(test_predictions)):
            raise ValueError("base model identities must remain constant across periods")

        combiner = EnsembleCombiner()
        combiner.train(
            stacking_predictions,
            stacking_metadata["target_direction"].to_numpy(dtype=np.int64),
            stacking_metadata["target"].to_numpy(dtype=np.float64),
        )
        raw_calibration_probability, _ = combiner.predict(calibration_predictions)
        calibrator = PlattCalibrator()
        calibration_labels = calibration_metadata["target_direction"].to_numpy(dtype=np.int64)
        calibrator.fit(raw_calibration_probability, calibration_labels)
        calibrated_probability = (
            calibrator.transform(raw_calibration_probability)
            if calibrator.is_fitted
            else raw_calibration_probability
        )
        calibration_fit_metrics = _reliability_metrics(calibration_labels, calibrated_probability)

        raw_test_probability, test_return = combiner.predict(test_predictions)
        test_probability = (
            calibrator.transform(raw_test_probability)
            if calibrator.is_fitted
            else raw_test_probability
        )
        final_predictions = test_metadata.copy()
        final_predictions["predicted_probability"] = test_probability
        final_predictions["predicted_return"] = test_return
        final_test_calibration_metrics = _reliability_metrics(
            test_metadata["target_direction"].to_numpy(dtype=np.int64),
            final_predictions["predicted_probability"].to_numpy(dtype=np.float64),
        )

        lineage = stacking[list(LINEAGE_COLUMNS)].copy().reset_index(drop=True)
        return ChronologicalStackingResult(
            test_predictions=final_predictions,
            stacking_lineage=lineage,
            calibration_fit_metrics=calibration_fit_metrics,
            final_test_calibration_metrics=final_test_calibration_metrics,
        )
