"""Content-addressed feature artifact lineage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from bist_predict.features.lineage import FeatureArtifactLineage, write_feature_artifact


def _lineage(*, code_version: str = "1") -> FeatureArtifactLineage:
    return FeatureArtifactLineage(
        feature_manifest_hash="feature-hash",
        git_commit="49eca4b",
        input_data_manifest_hash="data-hash",
        calculation_timestamp=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        code_version=code_version,
        ticker="THYAO",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 6, 30),
    )


def test_lineage_hash_is_deterministic_and_sensitive_to_contract() -> None:
    original = _lineage()
    same = _lineage()
    revised = _lineage(code_version="2")

    assert original.artifact_id == same.artifact_id
    assert original.artifact_id != revised.artifact_id


def test_feature_artifacts_are_content_addressed_and_never_overwritten(tmp_path) -> None:
    rows = [{"date": "2026-01-05", "log_return_1d": 0.01}]
    first_path = write_feature_artifact(tmp_path, _lineage(), rows)
    first_bytes = first_path.read_bytes()
    second_path = write_feature_artifact(tmp_path, _lineage(code_version="2"), rows)

    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert first_path.exists() and second_path.exists()

    payload = json.loads(first_path.read_text())
    assert payload["lineage"] == {
        "calculation_timestamp": "2026-07-14T12:00:00+00:00",
        "code_version": "1",
        "end_date": "2026-06-30",
        "feature_manifest_hash": "feature-hash",
        "git_commit": "49eca4b",
        "input_data_manifest_hash": "data-hash",
        "start_date": "2026-01-01",
        "ticker": "THYAO",
    }
    assert payload["rows"] == rows
