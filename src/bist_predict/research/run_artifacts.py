"""Create-only research run bundles with complete artifact hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from bist_predict.features.manifest import FeatureManifest
from bist_predict.features.lineage import FeatureArtifactLineage, write_feature_artifact
from bist_predict.research.portfolio_backtest import PortfolioBacktestResult
from bist_predict.research.inference.snooping import stationary_block_length
from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.predictions import write_prediction_artifact
from bist_predict.research.reporting import (
    block_bootstrap_intervals,
    block_size_sensitivity,
    compute_portfolio_metrics,
    grouped_prediction_metrics,
)

_SAFE_ARTIFACT_STEM = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_ARTIFACT_STEMS = {
    "artifact_hashes",
    "config",
    "corporate_action_ledger",
    "costs",
    "cash_ledger",
    "daily_equity",
    "data_manifest",
    "environment",
    "feature_manifest",
    "feature_artifacts",
    "fills",
    "folds",
    "model_artifact",
    "orders",
    "positions",
    "predictions",
    "run_manifest",
    "signals",
    "trials",
    "universe_manifest",
}


@dataclass(frozen=True)
class RunBundle:
    """Identity and location of one immutable research run."""

    run_id: str
    path: Path


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _current_git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _environment() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in (
        "bist-predict",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "xgboost",
        "lightgbm",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": os.name,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


class RunBundleWriter:
    """Materialize every accepted result under a content-described run ID."""

    def __init__(
        self,
        runs_root: Path,
        *,
        git_sha: str | None = None,
        dirty_working_tree: bool | None = None,
        now: datetime | None = None,
    ) -> None:
        detected_sha, detected_dirty = _current_git_state()
        self._runs_root = runs_root
        self._git_sha = git_sha or detected_sha
        self._dirty = detected_dirty if dirty_working_tree is None else dirty_working_tree
        self._now = now or datetime.now(UTC)
        if self._now.tzinfo is None:
            raise ValueError("run timestamp must be timezone-aware")

    def _run_id(self, config: Mapping[str, object]) -> str:
        config_hash = hashlib.sha256(_canonical_json(config).encode()).hexdigest()[:6]
        timestamp = self._now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{self._git_sha[:7]}-{config_hash}"

    @property
    def git_sha(self) -> str:
        """Return the exact Git identity that will be persisted for this run."""
        return self._git_sha

    def write(
        self,
        *,
        config: Mapping[str, object],
        data_manifest: Mapping[str, object],
        universe_manifest: Mapping[str, object],
        feature_manifest: FeatureManifest,
        folds: Iterable[Mapping[str, object]],
        predictions: pd.DataFrame,
        portfolio: PortfolioBacktestResult,
        model_artifact: Mapping[str, object],
        trials: Iterable[Mapping[str, object]],
        seeds: Iterable[int],
        command: str,
        input_frames: Mapping[str, pd.DataFrame] | None = None,
        feature_artifacts: Iterable[tuple[FeatureArtifactLineage, Sequence[dict[str, Any]]]] = (),
        sample_metadata: pd.DataFrame | None = None,
        benchmark_returns: Sequence[float] | None = None,
        additional_metrics: Mapping[str, object] | None = None,
        bootstrap_iterations: int = 10_000,
        bootstrap_block_sizes: Sequence[int] = (1, 2, 3, 5, 8, 13, 21),
    ) -> RunBundle:
        """Write a complete run once; any existing run ID is immutable."""
        seed_values = tuple(seeds)
        frames = dict(input_frames or {})
        for name in frames:
            if not _SAFE_ARTIFACT_STEM.fullmatch(name):
                raise ValueError(f"input artifact name must be a safe stem: {name}")
            if name in _RESERVED_ARTIFACT_STEMS:
                raise ValueError(f"input artifact name is reserved: {name}")
        run_id = self._run_id(config)
        run_path = self._runs_root / run_id
        if run_path.exists():
            raise FileExistsError(f"research run already exists: {run_path}")
        run_path.mkdir(parents=True)

        _write_json(run_path / "config.yaml", dict(config))
        _write_json(run_path / "data_manifest.json", dict(data_manifest))
        _write_json(run_path / "universe_manifest.json", dict(universe_manifest))
        (run_path / "feature_manifest.json").write_text(
            json.dumps(json.loads(feature_manifest.to_json()), indent=2, sort_keys=True) + "\n"
        )
        fold_payload = list(folds)
        _write_json(run_path / "folds.json", fold_payload)
        trial_payload = list(trials)
        (run_path / "trials.jsonl").write_text(
            "".join(_canonical_json(trial) + "\n" for trial in trial_payload)
        )
        prediction_artifact = write_prediction_artifact(
            predictions, run_path / "predictions.parquet"
        )
        _write_json(run_path / "model_artifact.json", dict(model_artifact))

        for name, frame in sorted(frames.items()):
            frame.to_parquet(run_path / f"{name}.parquet", index=False, compression="zstd")

        for lineage, rows in feature_artifacts:
            write_feature_artifact(run_path / "feature_artifacts", lineage, rows)

        for name, frame in portfolio.artifact_frames().items():
            frame.to_parquet(run_path / f"{name}.parquet", index=False, compression="zstd")

        net_returns = [snapshot.net_return for snapshot in portfolio.daily_snapshots]
        seed = seed_values[0] if seed_values else 42
        usable_blocks = [size for size in bootstrap_block_sizes if 1 <= size <= len(net_returns)]
        primary_block = (
            stationary_block_length(np.asarray(net_returns, dtype=np.float64).reshape(-1, 1))
            if len(net_returns) >= 4
            else 1.0
        )
        metrics: dict[str, object] = {
            "prediction": recompute_prediction_metrics(predictions),
            "portfolio": compute_portfolio_metrics(portfolio, benchmark_returns=benchmark_returns),
            "bootstrap": (
                {
                    **block_bootstrap_intervals(
                        net_returns,
                        block_size=max(1, min(round(primary_block), len(net_returns))),
                        iterations=bootstrap_iterations,
                        seed=seed,
                    ),
                    "block_size_selection": "politis_white_2009",
                    "selected_block_length": float(primary_block),
                }
                if net_returns
                else {"status": "no_portfolio_sessions"}
            ),
            "bootstrap_block_sensitivity": (
                block_size_sensitivity(
                    net_returns,
                    block_sizes=usable_blocks,
                    iterations=max(2_000, bootstrap_iterations // 5),
                    seed=seed,
                )
                if net_returns and usable_blocks
                else {}
            ),
        }
        if sample_metadata is not None:
            metrics["grouped"] = grouped_prediction_metrics(predictions, sample_metadata)
        if additional_metrics is not None:
            overlap = set(metrics).intersection(additional_metrics)
            if overlap:
                raise ValueError(
                    f"additional metric names are reserved: {', '.join(sorted(overlap))}"
                )
            metrics.update(additional_metrics)
        _write_json(run_path / "metrics.json", metrics)
        environment = _environment()
        _write_json(run_path / "environment.json", environment)

        model_hash = _sha256(run_path / "model_artifact.json")
        run_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": self._now.astimezone(UTC).isoformat(),
            "git_sha": self._git_sha,
            "dirty_working_tree": self._dirty,
            "config_hash": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "data_manifest_sha256": _sha256(run_path / "data_manifest.json"),
            "universe_manifest_sha256": _sha256(run_path / "universe_manifest.json"),
            "feature_manifest_hash": feature_manifest.manifest_hash,
            "model_artifact_sha256": model_hash,
            "predictions_sha256": prediction_artifact.sha256,
            "random_seeds": list(seed_values),
            "training_command": command,
            "environment": {
                "python_version": environment["python_version"],
                "platform": environment["platform"],
                "machine": environment["machine"],
            },
        }
        _write_json(run_path / "run_manifest.json", run_manifest)

        hashes = {
            path.relative_to(run_path).as_posix(): _sha256(path)
            for path in sorted(run_path.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }
        _write_json(run_path / "artifact_hashes.json", hashes)
        return RunBundle(run_id=run_id, path=run_path)


def verify_artifact_hashes(run_path: Path) -> dict[str, str]:
    """Return missing, changed, or unexpected artifacts for an immutable run."""
    expected = json.loads((run_path / "artifact_hashes.json").read_text())
    failures: dict[str, str] = {}
    for name, digest in expected.items():
        path = run_path / name
        if not path.exists():
            failures[name] = "missing"
        elif _sha256(path) != digest:
            failures[name] = "sha256_mismatch"
    actual_names = {
        path.relative_to(run_path).as_posix()
        for path in run_path.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    for name in sorted(actual_names.difference(expected)):
        failures[name] = "unexpected"
    return failures
