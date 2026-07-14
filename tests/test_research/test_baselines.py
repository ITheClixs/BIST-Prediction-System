"""Accepted baseline benchmark and prediction artifact invariants."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.research.baselines import ACCEPTED_BASELINES, run_baseline_benchmark
from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.predictions import (
    PREDICTION_COLUMNS,
    read_prediction_artifact,
    write_prediction_artifact,
)
from bist_predict.research.splits import ExpandingWindowSplitter


def _manifest() -> FeatureManifest:
    return FeatureManifest(
        schema_version="1",
        features=(
            FeatureSpec("momentum", "synthetic", "1", 2, "after close", "preserve", "train_zscore"),
            FeatureSpec(
                "volatility", "synthetic", "1", 3, "after close", "preserve", "train_zscore"
            ),
        ),
    )


def _panel() -> pd.DataFrame:
    sessions = pd.bdate_range("2024-01-02", periods=42, tz="Europe/Istanbul")
    records: list[dict[str, object]] = []
    for date_index, session in enumerate(sessions):
        for ticker_index, ticker in enumerate(("GARAN", "THYAO", "TUPRS")):
            momentum = np.sin(date_index / 3.0) + ticker_index * 0.1
            volatility = 0.1 + (date_index % 5) * 0.02 + ticker_index * 0.01
            target_return = 0.012 * momentum - 0.004 * volatility
            feature_time = session + pd.Timedelta(hours=18, minutes=10)
            target_end = sessions[min(date_index + 1, len(sessions) - 1)] + pd.Timedelta(hours=18)
            records.append(
                {
                    "date": session.date().isoformat(),
                    "ticker": ticker,
                    "feature_available_at": feature_time,
                    "target_end": target_end,
                    "target_return": target_return,
                    "target_direction": int(target_return > 0.0),
                    "feature_manifest_hash": _manifest().manifest_hash,
                    "momentum": momentum,
                    "volatility": volatility,
                }
            )
    return pd.DataFrame.from_records(records)


def _splitter() -> ExpandingWindowSplitter:
    return ExpandingWindowSplitter(
        min_train_dates=20,
        validation_dates=7,
        step_dates=7,
        embargo_dates=1,
    )


def _sorted_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.sort_values(
        ["fold_id", "date", "ticker", "model_name"], kind="stable"
    ).reset_index(drop=True)


def test_all_baselines_share_folds_features_targets_and_oos_rows() -> None:
    manifest = _manifest()
    result = run_baseline_benchmark(_panel(), manifest, _splitter())

    assert tuple(result.predictions.columns) == PREDICTION_COLUMNS
    assert set(result.predictions["model_name"]) == set(ACCEPTED_BASELINES)
    assert result.predictions["feature_manifest_hash"].eq(manifest.manifest_hash).all()
    assert result.predictions["predicted_probability"].between(0.0, 1.0).all()
    counts = result.predictions.groupby(["fold_id", "model_name"]).size().unstack()
    assert counts.nunique(axis=1).eq(1).all()
    assert all(
        fold.train_window.date_end == training_end
        for fold, training_end in zip(
            result.folds,
            result.predictions.groupby("fold_id", sort=True)["training_end"].first(),
            strict=True,
        )
    )


def test_fitted_model_states_are_json_safe_and_reconstruct_linear_predictions() -> None:
    manifest = _manifest()
    panel = _panel()
    result = run_baseline_benchmark(panel, manifest, _splitter())

    encoded = json.dumps(result.fitted_model_states, sort_keys=True, allow_nan=False)
    states = json.loads(encoded)
    assert len(states) == len(result.folds) * len(ACCEPTED_BASELINES)
    assert {(state["fold_id"], state["model_name"]) for state in states} == {
        (fold.fold_id, model_name) for fold in result.folds for model_name in ACCEPTED_BASELINES
    }

    fold = result.folds[0]
    validation = panel.copy()
    validation["date"] = pd.to_datetime(validation["date"]).dt.date.astype(str)
    validation.index = pd.Index(
        validation["date"].astype(str) + "|" + validation["ticker"].astype(str),
        name="sample_id",
    )
    validation = validation.loc[list(fold.validation_indices)]
    raw_matrix = validation[list(manifest.ordered_feature_names)].to_numpy(dtype=np.float64)

    for model_name in ("logistic", "ridge"):
        state = next(
            item
            for item in states
            if item["fold_id"] == fold.fold_id and item["model_name"] == model_name
        )
        assert state["schema_version"] == "1"
        assert state["model_version"]
        assert state["training_end"] == fold.train_window.date_end
        assert state["feature_manifest_hash"] == manifest.manifest_hash
        assert state["ordered_feature_names"] == list(manifest.ordered_feature_names)
        assert state["training_row_count"] == len(fold.train_indices)

        preprocessing = state["preprocessing"]
        imputation = np.asarray(preprocessing["imputation_values"], dtype=np.float64)
        means = np.asarray(preprocessing["means"], dtype=np.float64)
        scales = np.asarray(preprocessing["scales"], dtype=np.float64)
        missing = np.isnan(raw_matrix)
        normalized = (np.where(missing, imputation, raw_matrix) - means) / scales
        prepared = np.concatenate([normalized, missing.astype(np.float64)], axis=1)
        assert preprocessing["transformed_feature_names"] == [
            *manifest.ordered_feature_names,
            *(f"{name}__missing" for name in manifest.ordered_feature_names),
        ]

        estimator = state["estimator"]
        expected = result.predictions.loc[
            (result.predictions["fold_id"] == fold.fold_id)
            & (result.predictions["model_name"] == model_name)
        ].sort_values(["date", "ticker"], kind="stable")
        if model_name == "logistic":
            coefficients = np.asarray(estimator["coefficients"], dtype=np.float64)[0]
            intercept = float(estimator["intercept"][0])
            probability = 1.0 / (1.0 + np.exp(-(prepared @ coefficients + intercept)))
            mapping = state["prediction_mapping"]
            predicted_return = probability * float(mapping["positive_mean_return"]) + (
                1.0 - probability
            ) * float(mapping["negative_mean_return"])
            np.testing.assert_allclose(
                probability,
                expected["predicted_probability"].to_numpy(dtype=np.float64),
            )
        else:
            coefficients = np.asarray(estimator["coefficients"], dtype=np.float64)
            intercept = float(estimator["intercept"])
            predicted_return = prepared @ coefficients + intercept
            probability_scale = float(state["prediction_mapping"]["probability_scale"])
            probability = np.asarray(
                [
                    0.5 * (1.0 + math.erf(value / (max(probability_scale, 1e-12) * math.sqrt(2.0))))
                    for value in predicted_return
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                probability,
                expected["predicted_probability"].to_numpy(dtype=np.float64),
            )
        np.testing.assert_allclose(
            predicted_return,
            expected["predicted_return"].to_numpy(dtype=np.float64),
        )


def test_ticker_and_row_order_cannot_change_baseline_predictions() -> None:
    panel = _panel()
    manifest = _manifest()
    expected = run_baseline_benchmark(panel, manifest, _splitter())
    shuffled = panel.sample(frac=1.0, random_state=991).reset_index(drop=True)

    actual = run_baseline_benchmark(shuffled, manifest, _splitter())

    pdt.assert_frame_equal(
        _sorted_predictions(actual.predictions),
        _sorted_predictions(expected.predictions),
    )
    assert actual.fitted_model_states == expected.fitted_model_states


def test_future_target_changes_cannot_rewrite_prior_predictions() -> None:
    panel = _panel()
    manifest = _manifest()
    cutoff = "2024-02-12"
    expected = run_baseline_benchmark(panel, manifest, _splitter()).predictions
    perturbed = panel.copy()
    future = perturbed["date"] > cutoff
    perturbed.loc[future, "target_return"] *= -100.0
    perturbed.loc[future, "target_direction"] = (
        perturbed.loc[future, "target_return"] > 0.0
    ).astype(int)

    actual = run_baseline_benchmark(perturbed, manifest, _splitter()).predictions

    prior_expected = _sorted_predictions(expected.loc[expected["date"] <= cutoff])
    prior_actual = _sorted_predictions(actual.loc[actual["date"] <= cutoff])
    pdt.assert_frame_equal(prior_actual, prior_expected)


def test_parquet_predictions_round_trip_and_recompute_metrics(tmp_path) -> None:
    result = run_baseline_benchmark(_panel(), _manifest(), _splitter())
    path = tmp_path / "predictions.parquet"

    artifact = write_prediction_artifact(result.predictions, path)
    loaded = read_prediction_artifact(path)

    assert artifact.row_count == len(result.predictions)
    assert len(artifact.sha256) == 64
    pdt.assert_frame_equal(loaded, result.predictions)
    assert recompute_prediction_metrics(loaded) == recompute_prediction_metrics(result.predictions)
    with pytest.raises(FileExistsError):
        write_prediction_artifact(result.predictions, path)
