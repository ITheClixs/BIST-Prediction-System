"""Read-only access to one immutable run bundle for the figure builders.

Every figure is drawn from this object, so no figure can drift from the table
it illustrates: both read the same artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from bist_predict.research.run_artifacts import verify_artifact_hashes

__all__ = ["RunArtifacts"]


@dataclass(frozen=True)
class RunArtifacts:
    """One accepted run, hash-verified on load."""

    path: Path

    @classmethod
    def load(cls, path: Path | str) -> RunArtifacts:
        """Load a run after confirming every recorded artifact hash still holds."""
        directory = Path(path)
        if not (directory / "artifact_hashes.json").is_file():
            raise FileNotFoundError(f"not an accepted run directory: {directory}")
        failures = verify_artifact_hashes(directory)
        if failures:
            detail = ", ".join(f"{name}={reason}" for name, reason in sorted(failures.items()))
            raise ValueError(f"run artifact integrity check failed: {detail}")
        return cls(directory)

    def _json(self, name: str) -> Any:
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def _parquet(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path / f"{name}.parquet")

    @cached_property
    def metrics(self) -> dict[str, Any]:
        """Return ``metrics.json``."""
        return dict(self._json("metrics.json"))

    @cached_property
    def config(self) -> dict[str, Any]:
        """Return the exact configuration the run was created with."""
        return dict(self._json("config.yaml"))

    @cached_property
    def run_manifest(self) -> dict[str, Any]:
        """Return the run identity and provenance record."""
        return dict(self._json("run_manifest.json"))

    @cached_property
    def data_manifest(self) -> dict[str, Any]:
        """Return the input dataset identity."""
        return dict(self._json("data_manifest.json"))

    @cached_property
    def folds(self) -> list[dict[str, Any]]:
        """Return the executed walk-forward folds."""
        return list(self._json("folds.json"))

    @cached_property
    def predictions(self) -> pd.DataFrame:
        """Return every saved out-of-sample prediction row."""
        return self._parquet("predictions")

    @cached_property
    def daily_equity(self) -> pd.DataFrame:
        """Return the portfolio's per-session ledger summary."""
        return self._parquet("daily_equity")

    @cached_property
    def sensitivity(self) -> pd.DataFrame:
        """Return one row per configuration-grid trial."""
        return self._parquet("configuration_sensitivity")

    @cached_property
    def panel(self) -> pd.DataFrame:
        """Return the canonical date-ticker research panel."""
        return self._parquet("panel")

    @cached_property
    def trading_dates(self) -> list[str]:
        """Return every session the panel covers, in order."""
        return sorted(str(value) for value in self.panel["date"].unique())

    def target_panel(self) -> pd.DataFrame:
        """Return the executable target as a session-by-ticker matrix."""
        rows = self.predictions.loc[self.predictions["model_name"] == "zero_return"]
        return rows.pivot(index="date", columns="ticker", values="target")

    def squared_error_panel(self) -> pd.DataFrame:
        """Return per-session mean squared error with one column per model."""
        working = self.predictions.copy()
        working["squared_error"] = (working["target"] - working["predicted_return"]) ** 2
        panel = working.pivot_table(
            index="date", columns="model_name", values="squared_error", aggfunc="mean"
        )
        panel.columns = [str(name) for name in panel.columns]
        return panel.sort_index()
