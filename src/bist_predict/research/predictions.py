"""Immutable out-of-sample prediction artifact contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PREDICTION_COLUMNS = (
    "date",
    "ticker",
    "fold_id",
    "model_name",
    "model_version",
    "training_end",
    "feature_manifest_hash",
    "target",
    "prediction",
    "predicted_probability",
    "predicted_return",
)


@dataclass(frozen=True)
class PredictionArtifact:
    """Content identity for one immutable prediction file."""

    path: str
    sha256: str
    row_count: int


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Reject incomplete, ambiguous, or non-finite prediction rows."""
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("prediction columns differ from the immutable schema")
    required_text = PREDICTION_COLUMNS[:7]
    if predictions[list(required_text)].isna().any().any():
        raise ValueError("prediction identity columns must not be missing")
    if predictions.duplicated(["date", "ticker", "fold_id", "model_name"]).any():
        raise ValueError("duplicate out-of-sample prediction identity")
    numeric = predictions[
        ["target", "prediction", "predicted_probability", "predicted_return"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction values must be finite")
    if not predictions["predicted_probability"].between(0.0, 1.0).all():
        raise ValueError("predicted probabilities must lie in [0, 1]")
    if not predictions["prediction"].isin([0, 1]).all():
        raise ValueError("direction predictions must be binary")
    if not predictions["feature_manifest_hash"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("feature manifest hashes must be SHA-256 values")
    if pd.to_datetime(predictions["date"], errors="coerce").isna().any():
        raise ValueError("prediction dates must be parseable")
    if pd.to_datetime(predictions["training_end"], errors="coerce").isna().any():
        raise ValueError("training_end dates must be parseable")


def write_prediction_artifact(
    predictions: pd.DataFrame,
    path: Path,
) -> PredictionArtifact:
    """Write a new Parquet artifact without overwriting prior evidence."""
    validate_predictions(predictions)
    if path.exists():
        raise FileExistsError(f"prediction artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"bist_prediction_schema_version"] = b"1"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PredictionArtifact(str(path), digest, len(predictions))


def read_prediction_artifact(path: Path) -> pd.DataFrame:
    """Read and revalidate an immutable Parquet prediction artifact."""
    table = pq.read_table(path)
    version = (table.schema.metadata or {}).get(b"bist_prediction_schema_version")
    if version != b"1":
        raise ValueError("unknown prediction artifact schema version")
    predictions = table.to_pandas()
    validate_predictions(predictions)
    return predictions
