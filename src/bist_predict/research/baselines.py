"""Date-grouped accepted baselines under one immutable panel contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from bist_predict.features.manifest import FeatureManifest
from bist_predict.features.preprocessing import TrainOnlyPreprocessor
from bist_predict.research.predictions import PREDICTION_COLUMNS, validate_predictions
from bist_predict.research.splits import ExpandingWindowSplitter, WalkForwardFold

ACCEPTED_BASELINES = (
    "zero_return",
    "majority_direction",
    "previous_return",
    "market_direction",
    "rolling_mean",
    "logistic",
    "ridge",
)

MODEL_VERSIONS = {
    "zero_return": "zero-return-v1",
    "majority_direction": "train-prevalence-v1",
    "previous_return": "last-matured-ticker-return-v1",
    "market_direction": "last-matured-market-return-v1",
    "rolling_mean": "last-20-matured-ticker-returns-v1",
    "logistic": "sklearn-logistic-c1-seed42-v1",
    "ridge": "sklearn-ridge-alpha1-v1",
}


@dataclass(frozen=True)
class BaselineBenchmarkResult:
    """Out-of-sample rows and exact folds used to generate them."""

    predictions: pd.DataFrame
    folds: tuple[WalkForwardFold, ...]


def _normal_probability(values: np.ndarray, scale: float) -> np.ndarray:
    safe_scale = scale if scale > 1e-12 else 1e-12
    return np.asarray(
        [0.5 * (1.0 + math.erf(value / (safe_scale * math.sqrt(2.0)))) for value in values],
        dtype=np.float64,
    )


def _point_in_time_baselines(
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    fallback_return: float,
) -> dict[str, np.ndarray]:
    predictions = {name: [] for name in ("previous_return", "market_direction", "rolling_mean")}
    for _, row in validation.iterrows():
        matured = panel.loc[panel["_target_time"] < row["_feature_time"]]
        ticker_history = matured.loc[matured["ticker"] == row["ticker"]].sort_values(
            ["_target_time", "date", "ticker"], kind="stable"
        )
        previous = (
            float(ticker_history.iloc[-1]["target_return"])
            if not ticker_history.empty
            else fallback_return
        )
        rolling = (
            float(ticker_history["target_return"].tail(20).mean())
            if not ticker_history.empty
            else fallback_return
        )
        if matured.empty:
            market = fallback_return
        else:
            latest_feature_date = matured["date"].max()
            market = float(
                matured.loc[matured["date"] == latest_feature_date, "target_return"].mean()
            )
        predictions["previous_return"].append(previous)
        predictions["market_direction"].append(market)
        predictions["rolling_mean"].append(rolling)
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in predictions.items()
    }


def _validated_panel(panel: pd.DataFrame, manifest: FeatureManifest) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "feature_available_at",
        "target_end",
        "target_return",
        "target_direction",
        "feature_manifest_hash",
        *manifest.ordered_feature_names,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(missing)}")
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("panel must have one row per date and ticker")
    if not panel["feature_manifest_hash"].eq(manifest.manifest_hash).all():
        raise ValueError("panel feature manifest hash mismatch")

    working = panel.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.date.astype(str)
    working["_feature_time"] = pd.to_datetime(
        working["feature_available_at"], utc=True, errors="coerce"
    )
    working["_target_time"] = pd.to_datetime(
        working["target_end"], utc=True, errors="coerce"
    )
    if working[["_feature_time", "_target_time"]].isna().any().any():
        raise ValueError("panel timestamps must be valid")
    working = working.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    working.index = pd.Index(
        working["date"].astype(str) + "|" + working["ticker"].astype(str),
        name="sample_id",
    )
    return working


def _prediction_rows(
    validation: pd.DataFrame,
    fold: WalkForwardFold,
    manifest: FeatureManifest,
    model_name: str,
    predicted_return: np.ndarray,
    probability: np.ndarray,
    direction: np.ndarray | None = None,
) -> list[dict[str, object]]:
    predicted_direction = (
        np.asarray(direction, dtype=np.int64)
        if direction is not None
        else (predicted_return > 0.0).astype(np.int64)
    )
    return [
        {
            "date": str(row["date"]),
            "ticker": str(row["ticker"]),
            "fold_id": fold.fold_id,
            "model_name": model_name,
            "model_version": MODEL_VERSIONS[model_name],
            "training_end": fold.train_window.date_end,
            "feature_manifest_hash": manifest.manifest_hash,
            "target": float(row["target_return"]),
            "prediction": int(predicted_direction[index]),
            "predicted_probability": float(probability[index]),
            "predicted_return": float(predicted_return[index]),
        }
        for index, (_, row) in enumerate(validation.iterrows())
    ]


def run_baseline_benchmark(
    panel: pd.DataFrame,
    manifest: FeatureManifest,
    splitter: ExpandingWindowSplitter,
) -> BaselineBenchmarkResult:
    """Fit the seven accepted baselines on identical point-in-time folds."""
    working = _validated_panel(panel, manifest)
    folds = tuple(splitter.split(working))
    if not folds:
        raise ValueError("split configuration produced no validation folds")
    records: list[dict[str, object]] = []
    feature_names = manifest.ordered_feature_names

    for fold in folds:
        train = working.loc[list(fold.train_indices)]
        validation = working.loc[list(fold.validation_indices)]
        y_train_return = train["target_return"].to_numpy(dtype=np.float64)
        y_train_direction = train["target_direction"].to_numpy(dtype=np.int64)
        mean_return = float(np.mean(y_train_return))
        return_scale = float(np.std(y_train_return))
        positive_prevalence = float(np.mean(y_train_direction))
        positive_returns = y_train_return[y_train_direction == 1]
        negative_returns = y_train_return[y_train_direction == 0]
        positive_mean = float(np.mean(positive_returns)) if len(positive_returns) else mean_return
        negative_mean = float(np.mean(negative_returns)) if len(negative_returns) else mean_return
        row_count = len(validation)

        zero = np.zeros(row_count, dtype=np.float64)
        records.extend(
            _prediction_rows(
                validation,
                fold,
                manifest,
                "zero_return",
                zero,
                np.full(row_count, 0.5, dtype=np.float64),
            )
        )

        majority_direction = int(positive_prevalence >= 0.5)
        majority_return = positive_mean if majority_direction else negative_mean
        records.extend(
            _prediction_rows(
                validation,
                fold,
                manifest,
                "majority_direction",
                np.full(row_count, majority_return, dtype=np.float64),
                np.full(row_count, positive_prevalence, dtype=np.float64),
                np.full(row_count, majority_direction, dtype=np.int64),
            )
        )

        for model_name, predicted_return in _point_in_time_baselines(
            working, validation, mean_return
        ).items():
            records.extend(
                _prediction_rows(
                    validation,
                    fold,
                    manifest,
                    model_name,
                    predicted_return,
                    _normal_probability(predicted_return, return_scale),
                )
            )

        train_matrix = train[list(feature_names)].to_numpy(dtype=np.float64)
        validation_matrix = validation[list(feature_names)].to_numpy(dtype=np.float64)
        preprocessor = TrainOnlyPreprocessor(manifest, model_family="linear").fit(
            train_matrix,
            feature_names,
            manifest_hash=manifest.manifest_hash,
        )
        prepared_train = preprocessor.transform(
            train_matrix, feature_names, manifest_hash=manifest.manifest_hash
        )
        prepared_validation = preprocessor.transform(
            validation_matrix, feature_names, manifest_hash=manifest.manifest_hash
        )

        if len(np.unique(y_train_direction)) == 2:
            classifier = LogisticRegression(C=1.0, max_iter=1_000, random_state=42)
            classifier.fit(prepared_train, y_train_direction)
            logistic_probability = classifier.predict_proba(prepared_validation)[:, 1]
        else:
            logistic_probability = np.full(row_count, positive_prevalence)
        logistic_return = (
            logistic_probability * positive_mean
            + (1.0 - logistic_probability) * negative_mean
        )
        records.extend(
            _prediction_rows(
                validation,
                fold,
                manifest,
                "logistic",
                logistic_return,
                logistic_probability,
                (logistic_probability >= 0.5).astype(np.int64),
            )
        )

        regressor = Ridge(alpha=1.0)
        regressor.fit(prepared_train, y_train_return)
        ridge_return = regressor.predict(prepared_validation).astype(np.float64)
        residual_scale = float(
            np.std(y_train_return - regressor.predict(prepared_train))
        )
        records.extend(
            _prediction_rows(
                validation,
                fold,
                manifest,
                "ridge",
                ridge_return,
                _normal_probability(
                    ridge_return, residual_scale if residual_scale > 0.0 else return_scale
                ),
            )
        )

    predictions = pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)
    predictions = predictions.sort_values(
        ["fold_id", "date", "ticker", "model_name"], kind="stable"
    ).reset_index(drop=True)
    validate_predictions(predictions)
    return BaselineBenchmarkResult(predictions=predictions, folds=folds)

