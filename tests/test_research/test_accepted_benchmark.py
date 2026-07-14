"""End-to-end accepted benchmark and exact replay tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from bist_predict.research.accepted_benchmark import (
    AcceptedBenchmarkConfig,
    generate_synthetic_prices,
    reproduce_run,
    run_accepted_benchmark,
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
    assert (bundle.path / "input_prices.parquet").exists()
    assert (bundle.path / "official_calendar.parquet").exists()
    assert (bundle.path / "panel.parquet").exists()
    data_manifest = json.loads((bundle.path / "data_manifest.json").read_text())
    assert data_manifest["quality_summary"]["calendar_validation"] == {
        "duplicate_sessions": [],
        "missing_expected_sessions": [],
        "unexpected_sessions": [],
        "unexpected_weekend_rows": [],
    }

    replay = reproduce_run(bundle.path, scratch_root=tmp_path / "replay")

    assert replay == {}
