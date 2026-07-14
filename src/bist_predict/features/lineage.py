"""Immutable lineage records for generated feature artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class FeatureArtifactLineage:
    """Provenance required to reproduce one feature artifact."""

    feature_manifest_hash: str
    git_commit: str
    input_data_manifest_hash: str
    calculation_timestamp: datetime
    code_version: str
    ticker: str
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.calculation_timestamp.tzinfo is None:
            raise ValueError("calculation_timestamp must be timezone-aware")
        if self.start_date > self.end_date:
            raise ValueError("feature artifact start_date must not exceed end_date")

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible lineage payload."""
        return {
            "feature_manifest_hash": self.feature_manifest_hash,
            "git_commit": self.git_commit,
            "input_data_manifest_hash": self.input_data_manifest_hash,
            "calculation_timestamp": self.calculation_timestamp.isoformat(),
            "code_version": self.code_version,
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    @property
    def artifact_id(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def write_feature_artifact(
    root: Path,
    lineage: FeatureArtifactLineage,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Persist a content-addressed feature artifact without overwriting versions."""
    target = (
        root
        / lineage.ticker
        / f"{lineage.start_date.isoformat()}_{lineage.end_date.isoformat()}"
        / f"{lineage.artifact_id}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lineage": lineage.to_dict(), "rows": list(rows)}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with target.open("x", encoding="utf-8") as artifact:
            artifact.write(encoded)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(
                f"feature artifact collision without overwrite: {target}"
            ) from None
    return target
