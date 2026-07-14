"""Immutable dataset manifests for reproducible research inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class DataManifest:
    dataset_id: str
    sources: tuple[str, ...]
    universe_version: str
    start: date
    end: date
    row_count: int
    sha256: str
    created_at: datetime
    missing_sessions: tuple[str, ...]
    quality_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("data manifest start must not exceed end")
        if self.row_count < 0:
            raise ValueError("data manifest row_count cannot be negative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("data manifest sha256 must be a 64-character hex digest")
        if self.created_at.tzinfo is None:
            raise ValueError("data manifest created_at must be timezone-aware")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "missing_sessions", tuple(self.missing_sessions))
        object.__setattr__(
            self,
            "quality_summary",
            MappingProxyType(dict(self.quality_summary)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "sources": list(self.sources),
            "universe_version": self.universe_version,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "row_count": self.row_count,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat(),
            "missing_sessions": list(self.missing_sessions),
            "quality_summary": dict(self.quality_summary),
        }

    @property
    def manifest_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def write_data_manifest(root: Path, manifest: DataManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{manifest.dataset_id}-{manifest.manifest_hash}.json"
    encoded = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        with target.open("x", encoding="utf-8") as output:
            output.write(encoded)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"data manifest collision without overwrite: {target}")
    return target
