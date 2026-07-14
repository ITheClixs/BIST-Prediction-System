"""Train-only preprocessing and missing-value behavior."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.features.preprocessing import PreprocessingError, TrainOnlyPreprocessor


@pytest.fixture
def manifest() -> FeatureManifest:
    return FeatureManifest(
        schema_version="1.0.0",
        features=(
            FeatureSpec(
                name="return_1d",
                formula="close / lag(close) - 1",
                formula_version="1",
                lookback=2,
                availability_rule="after_session_close",
                missing_value_policy="train_median_with_indicator",
                normalization_policy="train_zscore",
            ),
            FeatureSpec(
                name="atr_over_close",
                formula="atr_14 / close",
                formula_version="1",
                lookback=15,
                availability_rule="after_session_close",
                missing_value_policy="train_median_with_indicator",
                normalization_policy="train_zscore",
            ),
        ),
    )


def test_linear_preprocessor_fits_imputation_and_scaling_on_train_only(
    manifest: FeatureManifest,
) -> None:
    train = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    validation = np.array([[1_000_000.0, np.nan], [np.nan, -1_000_000.0]])
    preprocessor = TrainOnlyPreprocessor(manifest, model_family="linear")
    preprocessor.fit(
        train,
        manifest.ordered_feature_names,
        manifest_hash=manifest.manifest_hash,
    )
    state_before = copy.deepcopy(preprocessor.state)

    transformed = preprocessor.transform(
        validation,
        manifest.ordered_feature_names,
        manifest_hash=manifest.manifest_hash,
    )

    assert preprocessor.state == state_before
    assert preprocessor.state.imputation_values == pytest.approx((2.0, 20.0))
    assert preprocessor.state.means == pytest.approx((2.0, 20.0))
    assert preprocessor.state.scales == pytest.approx(
        (np.std([1.0, 2.0, 3.0]), np.std([10.0, 20.0, 30.0]))
    )
    assert transformed.shape == (2, 4)
    assert transformed[0, 3] == 1.0
    assert transformed[1, 2] == 1.0
    assert np.isfinite(transformed).all()


def test_tree_preprocessor_preserves_native_missing_values(
    manifest: FeatureManifest,
) -> None:
    train = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    preprocessor = TrainOnlyPreprocessor(manifest, model_family="tree")
    preprocessor.fit(
        train,
        manifest.ordered_feature_names,
        manifest_hash=manifest.manifest_hash,
    )

    transformed = preprocessor.transform(
        np.array([[np.nan, 12.0]]),
        manifest.ordered_feature_names,
        manifest_hash=manifest.manifest_hash,
    )

    assert np.isnan(transformed[0, 0])
    assert transformed[0, 1] == 12.0
    assert transformed.shape == (1, 2)


def test_preprocessor_fails_when_a_feature_disappears_from_training(
    manifest: FeatureManifest,
) -> None:
    train = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    preprocessor = TrainOnlyPreprocessor(manifest, model_family="linear")

    with pytest.raises(PreprocessingError, match="entirely missing"):
        preprocessor.fit(
            train,
            manifest.ordered_feature_names,
            manifest_hash=manifest.manifest_hash,
        )


def test_preprocessor_rejects_infinite_values_instead_of_silently_zeroing(
    manifest: FeatureManifest,
) -> None:
    preprocessor = TrainOnlyPreprocessor(manifest, model_family="linear")

    with pytest.raises(PreprocessingError, match="infinite"):
        preprocessor.fit(
            np.array([[1.0, 2.0], [np.inf, 3.0]]),
            manifest.ordered_feature_names,
            manifest_hash=manifest.manifest_hash,
        )
