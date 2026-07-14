"""End-to-end accepted benchmark and exact replay tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from bist_predict.research import run_artifacts
from bist_predict.research.accepted_benchmark import (
    AcceptedBenchmarkConfig,
    generate_synthetic_prices,
    reproduce_run,
    run_accepted_benchmark,
)
from bist_predict.research.prediction_tracking import (
    ImmutablePredictionStore,
    persist_run_signal_predictions,
)
from bist_predict.research.run_artifacts import verify_artifact_hashes


def test_synthetic_methodology_smoke_runs_and_replays_from_bundled_inputs(
    tmp_path,
) -> None:
    now = datetime(2024, 4, 5, 12, 0, tzinfo=UTC)
    config = AcceptedBenchmarkConfig.synthetic_smoke()
    bundle = run_accepted_benchmark(
        generate_synthetic_prices(),
        runs_root=tmp_path / "runs",
        config=config,
        now=now,
        git_sha="abcdef123456",
        dirty_working_tree=False,
        command="make reproduce-smoke",
    )

    assert verify_artifact_hashes(bundle.path) == {}
    metrics = json.loads((bundle.path / "metrics.json").read_text())
    assert set(metrics["prediction"]) == {
        "logistic",
        "majority_direction",
        "market_direction",
        "previous_return",
        "ridge",
        "rolling_mean",
        "zero_return",
    }
    assert set(metrics["grouped"]) == {
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    }
    assert set(metrics["portfolio_grouped"]["dimensions"]) == {
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    }
    assert all(item["passed"] for item in metrics["portfolio_grouped"]["reconciliation"].values())
    sensitivity = metrics["cost_sensitivity"]
    assert (
        sensitivity["0.0x"]["metrics"]["trade_count"]
        == sensitivity["1.0x"]["metrics"]["trade_count"]
        == sensitivity["2.0x"]["metrics"]["trade_count"]
    )
    assert (
        sensitivity["0.0x"]["metrics"]["net_return"]
        >= sensitivity["1.0x"]["metrics"]["net_return"]
        >= sensitivity["2.0x"]["metrics"]["net_return"]
    )
    assert (bundle.path / "input_prices.parquet").exists()
    assert (bundle.path / "corporate_action_coverage.parquet").exists()
    assert (bundle.path / "corporate_actions.parquet").exists()
    assert (bundle.path / "official_calendar.parquet").exists()
    assert (bundle.path / "feature_lineage.parquet").exists()
    assert (bundle.path / "missingness.parquet").exists()
    assert (bundle.path / "panel.parquet").exists()
    feature_artifacts = tuple((bundle.path / "feature_artifacts").rglob("*.json"))
    assert len(feature_artifacts) == 4
    missingness = pd.read_parquet(bundle.path / "missingness.parquet")
    lineage = pd.read_parquet(bundle.path / "feature_lineage.parquet")
    assert missingness["feature_manifest_hash"].nunique() == 1
    assert set(missingness["fold"]).issuperset({"pre_validation_history"})
    assert set(lineage["git_commit"]) == {"abcdef123456"}
    assert lineage["artifact_id"].is_unique
    data_manifest = json.loads((bundle.path / "data_manifest.json").read_text())
    model_artifact = json.loads((bundle.path / "model_artifact.json").read_text())
    folds = json.loads((bundle.path / "folds.json").read_text())
    assert len(model_artifact["fitted_model_states"]) == 7 * len(folds)
    assert {state["model_name"] for state in model_artifact["fitted_model_states"]} == set(
        metrics["prediction"]
    )
    assert data_manifest["quality_summary"]["calendar_validation"] == {
        "duplicate_sessions": [],
        "missing_expected_sessions": [],
        "unexpected_sessions": [],
        "unexpected_weekend_rows": [],
    }

    tracking_store = ImmutablePredictionStore(tmp_path / "tracking")
    tracked = persist_run_signal_predictions(bundle.path, tracking_store)

    assert tracked
    assert {record.model_run_id for record in tracking_store.records()} == {bundle.run_id}

    replay = reproduce_run(bundle.path, scratch_root=tmp_path / "replay")

    assert replay == {}


def test_replay_allows_provenance_environment_drift_but_not_scientific_drift(
    tmp_path,
    monkeypatch,
) -> None:
    original_environment = {
        "python_version": "3.13.0",
        "python_executable": "/original/.venv/bin/python",
        "platform": "macOS-original",
        "os": "posix",
        "machine": "arm64",
        "processor": "arm",
        "logical_cpu_count": 10,
        "packages": {"pandas": "2.3.0"},
    }
    replay_environment = {
        **original_environment,
        "python_version": "3.12.0",
        "python_executable": "/runner/.venv/bin/python",
        "platform": "Linux-runner",
        "machine": "x86_64",
        "processor": "x86_64",
        "logical_cpu_count": 4,
    }
    monkeypatch.setattr(run_artifacts, "_environment", lambda: original_environment)
    bundle = run_accepted_benchmark(
        generate_synthetic_prices(),
        runs_root=tmp_path / "runs",
        config=AcceptedBenchmarkConfig.synthetic_smoke(),
        now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
        git_sha="abcdef123456",
        dirty_working_tree=False,
    )

    monkeypatch.setattr(run_artifacts, "_environment", lambda: replay_environment)

    assert reproduce_run(bundle.path, scratch_root=tmp_path / "replay") == {}


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"provider_record_id": None}, "stable record ID"),
        ({"split_adjusted_prices": None}, "split-adjusted"),
        ({"total_return_prices": None}, "total-return"),
    ],
)
def test_accepted_inputs_require_provenance_and_explicit_price_representations(
    tmp_path,
    replacement,
    message,
) -> None:
    prices = list(generate_synthetic_prices())
    prices[0] = replace(prices[0], **replacement)

    with pytest.raises(ValueError, match=message):
        run_accepted_benchmark(
            prices,
            runs_root=tmp_path / "runs",
            config=AcceptedBenchmarkConfig.synthetic_smoke(),
            now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
            git_sha="abcdef123456",
            dirty_working_tree=False,
        )


def test_market_benchmark_requires_corporate_action_snapshot(tmp_path) -> None:
    config = replace(
        AcceptedBenchmarkConfig.synthetic_smoke(),
        experiment_scope="fixed_bist_large_cap_prototype",
    )

    with pytest.raises(ValueError, match="corporate-action snapshot"):
        run_accepted_benchmark(
            generate_synthetic_prices(),
            runs_root=tmp_path / "runs",
            config=config,
            now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
            git_sha="abcdef123456",
            dirty_working_tree=False,
        )

    with pytest.raises(ValueError, match="corporate-action coverage"):
        run_accepted_benchmark(
            generate_synthetic_prices(),
            corporate_actions=(),
            runs_root=tmp_path / "runs",
            config=config,
            now=datetime(2024, 4, 5, 12, 0, tzinfo=UTC),
            git_sha="abcdef123456",
            dirty_working_tree=False,
        )
