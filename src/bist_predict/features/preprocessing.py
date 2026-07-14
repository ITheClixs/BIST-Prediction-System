"""Fold-local preprocessing for model families that cannot consume missing values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bist_predict.features.manifest import FeatureManifest


class PreprocessingError(ValueError):
    """Raised when a matrix cannot be prepared without masking data defects."""


@dataclass(frozen=True)
class PreprocessorState:
    """Immutable statistics learned exclusively from a training partition."""

    manifest_hash: str
    model_family: str
    imputation_values: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]


class TrainOnlyPreprocessor:
    """Fit fold-local missing-value and normalization statistics."""

    def __init__(self, manifest: FeatureManifest, *, model_family: str) -> None:
        if model_family not in {"linear", "neural", "tree"}:
            raise ValueError(f"unsupported model family: {model_family}")
        self._manifest = manifest
        self._model_family = model_family
        self._state: PreprocessorState | None = None

    @property
    def state(self) -> PreprocessorState:
        if self._state is None:
            raise RuntimeError("preprocessor is not fitted")
        return self._state

    def fit(
        self,
        matrix: NDArray[np.float64],
        feature_names: Sequence[str],
        *,
        manifest_hash: str,
    ) -> TrainOnlyPreprocessor:
        values = self._validated_matrix(matrix, feature_names, manifest_hash)
        entirely_missing = np.isnan(values).all(axis=0)
        if entirely_missing.any():
            names = [
                self._manifest.ordered_feature_names[index]
                for index in np.flatnonzero(entirely_missing)
            ]
            raise PreprocessingError(
                f"required features entirely missing in training: {', '.join(names)}"
            )

        if self._model_family == "tree":
            self._state = PreprocessorState(
                manifest_hash=self._manifest.manifest_hash,
                model_family=self._model_family,
                imputation_values=(),
                means=(),
                scales=(),
            )
            return self

        imputation = np.nanmedian(values, axis=0)
        imputed = np.where(np.isnan(values), imputation, values)
        means = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for index, spec in enumerate(self._manifest.features):
            if spec.normalization_policy == "train_zscore":
                means[index] = float(np.mean(imputed[:, index]))
                scale = float(np.std(imputed[:, index]))
                scales[index] = scale if scale > 0.0 else 1.0

        self._state = PreprocessorState(
            manifest_hash=self._manifest.manifest_hash,
            model_family=self._model_family,
            imputation_values=tuple(float(value) for value in imputation),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
        )
        return self

    def transform(
        self,
        matrix: NDArray[np.float64],
        feature_names: Sequence[str],
        *,
        manifest_hash: str,
    ) -> NDArray[np.float64]:
        state = self.state
        values = self._validated_matrix(matrix, feature_names, manifest_hash)
        if state.model_family == "tree":
            return values.copy()

        missing = np.isnan(values)
        imputation = np.asarray(state.imputation_values, dtype=np.float64)
        means = np.asarray(state.means, dtype=np.float64)
        scales = np.asarray(state.scales, dtype=np.float64)
        imputed = np.where(missing, imputation, values)
        normalized = (imputed - means) / scales
        return np.concatenate([normalized, missing.astype(np.float64)], axis=1)

    def _validated_matrix(
        self,
        matrix: NDArray[np.float64],
        feature_names: Sequence[str],
        manifest_hash: str,
    ) -> NDArray[np.float64]:
        self._manifest.validate_matrix_schema(
            feature_names,
            manifest_hash=manifest_hash,
        )
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise PreprocessingError("feature matrix must be two-dimensional")
        if values.shape[1] != len(self._manifest.features):
            raise PreprocessingError("feature matrix width differs from manifest")
        if np.isinf(values).any():
            raise PreprocessingError("feature matrix contains infinite values")
        return values
