"""Metrics recomputed exclusively from saved out-of-sample prediction rows."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
)

from bist_predict.research.predictions import validate_predictions


def _correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    rank: bool,
) -> float | None:
    if len(actual) < 2 or np.ptp(actual) == 0.0 or np.ptp(predicted) == 0.0:
        return None
    value = spearmanr(actual, predicted).statistic if rank else pearsonr(actual, predicted).statistic
    return float(value) if math.isfinite(float(value)) else None


def _model_metrics(rows: pd.DataFrame) -> dict[str, float | int | None]:
    target = rows["target"].to_numpy(dtype=np.float64)
    predicted_return = rows["predicted_return"].to_numpy(dtype=np.float64)
    actual_direction = (target > 0.0).astype(np.int64)
    predicted_direction = rows["prediction"].to_numpy(dtype=np.int64)
    probability = np.clip(
        rows["predicted_probability"].to_numpy(dtype=np.float64), 1e-12, 1.0 - 1e-12
    )
    errors = target - predicted_return
    denominator = float(np.sum(np.square(target)))
    return {
        "sample_count": int(len(rows)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "zero_mean_r_squared": (
            1.0 - float(np.sum(np.square(errors))) / denominator
            if denominator > 0.0
            else None
        ),
        "pearson_ic": _correlation(target, predicted_return, rank=False),
        "spearman_ic": _correlation(target, predicted_return, rank=True),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(actual_direction, predicted_direction))
            if len(np.unique(actual_direction)) == 2
            else float(np.mean(actual_direction == predicted_direction))
        ),
        "log_loss": float(log_loss(actual_direction, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual_direction, probability)),
        "pr_auc": (
            float(average_precision_score(actual_direction, probability))
            if np.any(actual_direction == 1)
            else None
        ),
        "mcc": (
            float(matthews_corrcoef(actual_direction, predicted_direction))
            if len(np.unique(actual_direction)) == 2
            else 0.0
        ),
    }


def recompute_prediction_metrics(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, float | int | None]]:
    """Return per-model metrics using no state outside the prediction rows."""
    validate_predictions(predictions)
    return {
        str(model_name): _model_metrics(rows)
        for model_name, rows in predictions.groupby("model_name", sort=True)
    }
