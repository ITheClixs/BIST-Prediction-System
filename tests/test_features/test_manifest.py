"""Tests for immutable feature schema manifests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bist_predict.features.manifest import (
    FeatureManifest,
    FeatureSchemaError,
    FeatureSpec,
)


@pytest.fixture
def manifest() -> FeatureManifest:
    return FeatureManifest(
        schema_version="1.0.0",
        features=(
            FeatureSpec(
                name="log_return_1d",
                formula="log(adjusted_close_t / adjusted_close_t_minus_1)",
                formula_version="1",
                lookback=1,
                availability_rule="available after session close",
                missing_value_policy="preserve",
                normalization_policy="none",
            ),
            FeatureSpec(
                name="atr_over_close_14d",
                formula="atr_14 / adjusted_close",
                formula_version="1",
                lookback=14,
                availability_rule="available after session close",
                missing_value_policy="train-only median with indicator",
                normalization_policy="none",
            ),
        ),
    )


def test_manifest_is_immutable_and_preserves_feature_order(
    manifest: FeatureManifest,
) -> None:
    assert manifest.ordered_feature_names == (
        "log_return_1d",
        "atr_over_close_14d",
    )

    with pytest.raises(FrozenInstanceError):
        manifest.schema_version = "2.0.0"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        manifest.features[0].lookback = 2  # type: ignore[misc]


def test_manifest_hash_is_deterministic_and_covers_feature_contract(
    manifest: FeatureManifest,
) -> None:
    identical = FeatureManifest(
        schema_version=manifest.schema_version,
        features=manifest.features,
    )
    changed_policy = FeatureManifest(
        schema_version=manifest.schema_version,
        features=(
            manifest.features[0],
            FeatureSpec(
                name="atr_over_close_14d",
                formula="atr_14 / adjusted_close",
                formula_version="1",
                lookback=14,
                availability_rule="available after session close",
                missing_value_policy="preserve",
                normalization_policy="none",
            ),
        ),
    )

    assert identical.manifest_hash == manifest.manifest_hash
    assert len(manifest.manifest_hash) == 64
    assert changed_policy.manifest_hash != manifest.manifest_hash


def test_manifest_json_round_trip_preserves_hash(manifest: FeatureManifest) -> None:
    restored = FeatureManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.manifest_hash == manifest.manifest_hash


def test_json_round_trip_rejects_tampered_contract(manifest: FeatureManifest) -> None:
    tampered = manifest.to_json().replace('"lookback":1', '"lookback":2')

    with pytest.raises(FeatureSchemaError, match="manifest hash mismatch"):
        FeatureManifest.from_json(tampered)


def test_validate_matrix_schema_accepts_exact_identity_and_hash(
    manifest: FeatureManifest,
) -> None:
    manifest.validate_matrix_schema(
        ["log_return_1d", "atr_over_close_14d"],
        manifest_hash=manifest.manifest_hash,
    )


def test_validate_matrix_schema_rejects_missing_feature(
    manifest: FeatureManifest,
) -> None:
    with pytest.raises(FeatureSchemaError, match="missing required features"):
        manifest.validate_matrix_schema(
            ["log_return_1d"],
            manifest_hash=manifest.manifest_hash,
        )


def test_validate_matrix_schema_rejects_unknown_feature_at_equal_width(
    manifest: FeatureManifest,
) -> None:
    with pytest.raises(FeatureSchemaError, match="unknown features"):
        manifest.validate_matrix_schema(
            ["log_return_1d", "rsi_14"],
            manifest_hash=manifest.manifest_hash,
        )


def test_validate_matrix_schema_rejects_reordered_features(
    manifest: FeatureManifest,
) -> None:
    with pytest.raises(FeatureSchemaError, match="feature order differs"):
        manifest.validate_matrix_schema(
            ["atr_over_close_14d", "log_return_1d"],
            manifest_hash=manifest.manifest_hash,
        )


def test_validate_matrix_schema_rejects_manifest_hash_mismatch(
    manifest: FeatureManifest,
) -> None:
    with pytest.raises(FeatureSchemaError, match="manifest hash mismatch"):
        manifest.validate_matrix_schema(
            list(manifest.ordered_feature_names),
            manifest_hash="0" * 64,
        )


def test_manifest_rejects_duplicate_feature_names(manifest: FeatureManifest) -> None:
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeatureManifest(
            schema_version="1.0.0",
            features=(manifest.features[0], manifest.features[0]),
        )
