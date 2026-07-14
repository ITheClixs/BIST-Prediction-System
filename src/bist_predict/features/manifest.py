"""Immutable contracts for feature identities and matrix schemas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence


class FeatureSchemaError(ValueError):
    """Raised when feature metadata does not match a manifest."""


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one ordered model feature."""

    name: str
    formula: str
    formula_version: str
    lookback: int
    availability_rule: str
    missing_value_policy: str
    normalization_policy: str


@dataclass(frozen=True)
class FeatureManifest:
    """Versioned collection of ordered feature definitions."""

    schema_version: str
    features: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        features = tuple(self.features)
        object.__setattr__(self, "features", features)

        names = self.ordered_feature_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {', '.join(duplicates)}")

    @property
    def ordered_feature_names(self) -> tuple[str, ...]:
        """Return feature identities in the only accepted matrix order."""
        return tuple(feature.name for feature in self.features)

    def _contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "features": [asdict(feature) for feature in self.features],
        }

    @staticmethod
    def _canonical_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def manifest_hash(self) -> str:
        """SHA-256 identity of the complete, ordered feature contract."""
        encoded = self._canonical_json(self._contract_payload()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> str:
        """Serialize the manifest with its verifiable contract hash."""
        payload = self._contract_payload()
        payload["manifest_hash"] = self.manifest_hash
        return self._canonical_json(payload)

    @classmethod
    def from_json(cls, value: str) -> FeatureManifest:
        """Deserialize a manifest and reject a stale or tampered hash."""
        payload = json.loads(value)
        features = tuple(FeatureSpec(**feature) for feature in payload["features"])
        manifest = cls(schema_version=payload["schema_version"], features=features)
        if payload.get("manifest_hash") != manifest.manifest_hash:
            raise FeatureSchemaError("manifest hash mismatch")
        return manifest

    def validate_matrix_schema(
        self,
        feature_names: Sequence[str],
        *,
        manifest_hash: str,
    ) -> None:
        """Require exact feature identities, order, and manifest provenance."""
        actual = tuple(feature_names)
        expected = self.ordered_feature_names

        unknown = tuple(name for name in actual if name not in expected)
        if unknown:
            raise FeatureSchemaError(f"unknown features: {', '.join(unknown)}")

        missing = tuple(name for name in expected if name not in actual)
        if missing:
            raise FeatureSchemaError(f"missing required features: {', '.join(missing)}")

        if actual != expected:
            raise FeatureSchemaError("feature order differs from manifest")

        if manifest_hash != self.manifest_hash:
            raise FeatureSchemaError("manifest hash mismatch")
