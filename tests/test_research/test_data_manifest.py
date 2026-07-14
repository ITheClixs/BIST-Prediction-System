"""Immutable dataset manifest governance."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from bist_predict.research.data_manifest import DataManifest, write_data_manifest


def _manifest(*, dataset_hash: str = "a" * 64) -> DataManifest:
    return DataManifest(
        dataset_id="fixed-bist-smoke",
        sources=("yahoo", "borsa_istanbul_calendar"),
        universe_version="fixed_bist_large_cap_prototype:v1",
        start=date(2026, 1, 1),
        end=date(2026, 3, 31),
        row_count=120,
        sha256=dataset_hash,
        created_at=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        missing_sessions=("2026-02-17",),
        quality_summary={"proxy_open_rows": 0, "provider_overlap_rows": 80},
    )


def test_data_manifest_hash_covers_dataset_and_quality_contract() -> None:
    original = _manifest()
    same = _manifest()
    changed = _manifest(dataset_hash="b" * 64)

    assert original.manifest_hash == same.manifest_hash
    assert original.manifest_hash != changed.manifest_hash


def test_data_manifest_is_written_to_a_content_addressed_path(tmp_path) -> None:
    manifest = _manifest()
    path = write_data_manifest(tmp_path, manifest)
    original = path.read_bytes()

    assert path.name == f"{manifest.dataset_id}-{manifest.manifest_hash}.json"
    assert write_data_manifest(tmp_path, manifest) == path
    assert path.read_bytes() == original

    payload = json.loads(path.read_text())
    assert payload["dataset_id"] == "fixed-bist-smoke"
    assert payload["row_count"] == 120
    assert payload["sha256"] == "a" * 64
    assert payload["missing_sessions"] == ["2026-02-17"]
    assert payload["quality_summary"]["proxy_open_rows"] == 0
