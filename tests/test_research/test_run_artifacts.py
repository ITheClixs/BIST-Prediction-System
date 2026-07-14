"""Immutable run bundle and artifact hash tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.research.portfolio_backtest import (
    DailySnapshot,
    Portfolio,
    PortfolioBacktestResult,
)
from bist_predict.research.predictions import PREDICTION_COLUMNS
from bist_predict.research.run_artifacts import RunBundleWriter, verify_artifact_hashes


def _predictions() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "date": "2024-01-02",
                "ticker": "THYAO",
                "fold_id": "fold_0001",
                "model_name": "zero_return",
                "model_version": "zero-return-v1",
                "training_end": "2023-12-29",
                "feature_manifest_hash": "a" * 64,
                "target": 0.01,
                "prediction": 0,
                "predicted_probability": 0.5,
                "predicted_return": 0.0,
            }
        ],
        columns=PREDICTION_COLUMNS,
    )


def _manifest() -> FeatureManifest:
    return FeatureManifest(
        "1",
        (FeatureSpec("return_1d", "log return", "1", 2, "after close", "preserve", "none"),),
    )


def _portfolio() -> PortfolioBacktestResult:
    snapshot = DailySnapshot(
        "2024-01-03", 100_000.0, 100_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    return PortfolioBacktestResult(
        signals=(),
        orders=(),
        fills=(),
        positions=(),
        cash_ledger=(),
        costs=(),
        daily_snapshots=(snapshot,),
        portfolio=Portfolio(100_000.0, 100_000.0, 100_000.0, ()),
    )


def test_run_bundle_contains_required_recomputable_and_hashed_artifacts(tmp_path) -> None:
    writer = RunBundleWriter(
        tmp_path / "runs",
        git_sha="abcdef123456",
        dirty_working_tree=True,
        now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
    )
    result = writer.write(
        config={"scope": "fixed_bist_large_cap_prototype", "top_k": 10},
        data_manifest={"dataset_id": "synthetic-001", "sha256": "b" * 64},
        universe_manifest={"universe_version": "fixed-v1", "tickers": ["THYAO"]},
        feature_manifest=_manifest(),
        folds=[{"fold_id": "fold_0001", "train_dates": ["2023-12-29"]}],
        predictions=_predictions().assign(feature_manifest_hash=_manifest().manifest_hash),
        portfolio=_portfolio(),
        model_artifact={"accepted_models": ["zero_return"]},
        trials=(),
        seeds=(42,),
        command="make reproduce-smoke",
        input_frames={"input_prices": pd.DataFrame({"ticker": ["THYAO"]})},
    )

    assert result.run_id.startswith("20240405T120000Z-abcdef1-")
    required = {
        "config.yaml",
        "run_manifest.json",
        "data_manifest.json",
        "universe_manifest.json",
        "feature_manifest.json",
        "folds.json",
        "trials.jsonl",
        "predictions.parquet",
        "metrics.json",
        "model_artifact.json",
        "environment.json",
        "artifact_hashes.json",
        "signals.parquet",
        "orders.parquet",
        "fills.parquet",
        "positions.parquet",
        "daily_equity.parquet",
        "costs.parquet",
        "input_prices.parquet",
    }
    assert required.issubset({path.name for path in result.path.iterdir()})
    assert verify_artifact_hashes(result.path) == {}
    run_manifest = json.loads((result.path / "run_manifest.json").read_text())
    assert run_manifest["git_sha"] == "abcdef123456"
    assert run_manifest["dirty_working_tree"] is True
    assert run_manifest["training_command"] == "make reproduce-smoke"
    metrics = json.loads((result.path / "metrics.json").read_text())
    assert metrics["prediction"]["zero_return"]["sample_count"] == 1
    with pytest.raises(FileExistsError):
        writer.write(
            config={"scope": "fixed_bist_large_cap_prototype", "top_k": 10},
            data_manifest={"dataset_id": "synthetic-001", "sha256": "b" * 64},
            universe_manifest={"universe_version": "fixed-v1", "tickers": ["THYAO"]},
            feature_manifest=_manifest(),
            folds=[],
            predictions=_predictions().assign(feature_manifest_hash=_manifest().manifest_hash),
            portfolio=_portfolio(),
            model_artifact={"accepted_models": ["zero_return"]},
            trials=(),
            seeds=(42,),
            command="make reproduce-smoke",
            input_frames={"input_prices": pd.DataFrame({"ticker": ["THYAO"]})},
        )


def test_run_bundle_rejects_unsafe_or_reserved_input_artifact_names(tmp_path) -> None:
    writer = RunBundleWriter(
        tmp_path / "runs",
        git_sha="abcdef123456",
        dirty_working_tree=False,
        now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
    )
    common = {
        "config": {"scope": "synthetic_methodology_smoke"},
        "data_manifest": {"dataset_id": "synthetic-001", "sha256": "b" * 64},
        "universe_manifest": {"universe_version": "fixed-v1", "tickers": ["THYAO"]},
        "feature_manifest": _manifest(),
        "folds": (),
        "predictions": _predictions().assign(
            feature_manifest_hash=_manifest().manifest_hash
        ),
        "portfolio": _portfolio(),
        "model_artifact": {"accepted_models": ["zero_return"]},
        "trials": (),
        "seeds": (42,),
        "command": "make reproduce-smoke",
    }

    with pytest.raises(ValueError, match="safe stem"):
        writer.write(**common, input_frames={"../prices": pd.DataFrame()})
    with pytest.raises(ValueError, match="reserved"):
        writer.write(**common, input_frames={"predictions": pd.DataFrame()})
