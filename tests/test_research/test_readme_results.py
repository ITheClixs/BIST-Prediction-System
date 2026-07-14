"""README accepted-results generation tests."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.predictions import PREDICTION_COLUMNS
from bist_predict.research.readme_results import (
    ReadmeResultsError,
    main,
    render_accepted_results,
    update_readme_results,
)


START_MARKER = "<!-- ACCEPTED_RESULTS:START -->"
END_MARKER = "<!-- ACCEPTED_RESULTS:END -->"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_directory(tmp_path: Path) -> Path:
    run_path = tmp_path / "runs" / "20260714T120000Z-abc1234-deadbe"
    run_path.mkdir(parents=True)
    predictions = pd.DataFrame.from_records(
        [
            {
                "date": f"2025-01-{day:02d}",
                "ticker": "THYAO",
                "fold_id": "fold_0001",
                "model_name": model_name,
                "model_version": f"{model_name}-v1",
                "training_end": "2024-12-31",
                "feature_manifest_hash": "a" * 64,
                "target": target,
                "prediction": int(predicted_return > 0.0),
                "predicted_probability": probability,
                "predicted_return": predicted_return,
            }
            for model_name, predicted_return, probability in (
                ("ridge", 0.008, 0.6),
                ("zero_return", 0.0, 0.5),
            )
            for day, target in enumerate((0.01, -0.02, 0.015, -0.005), start=2)
        ],
        columns=PREDICTION_COLUMNS,
    )
    predictions.to_parquet(run_path / "predictions.parquet", index=False)
    _write_json(
        run_path / "metrics.json",
        {
            "prediction": recompute_prediction_metrics(predictions),
            "portfolio": {
                "gross_return": -0.01,
                "net_return": -0.06,
                "annualized_return": -0.12,
                "annualized_volatility": 0.16,
                "sharpe": -0.75,
                "maximum_drawdown": -0.08,
                "turnover": 15.5,
                "trade_count": 20,
                "benchmark_return": -0.04,
                "benchmark_relative_return": -0.02,
                "cost_decomposition": {"total": 5100.25},
            },
            "cost_sensitivity": {
                "2.0x": {
                    "cost_multiplier": 2.0,
                    "metrics": {
                        "gross_return": -0.011,
                        "net_return": -0.11,
                        "trade_count": 20,
                        "cost_decomposition": {"total": 10000.0},
                    },
                },
                "0.0x": {
                    "cost_multiplier": 0.0,
                    "metrics": {
                        "gross_return": -0.009,
                        "net_return": -0.009,
                        "trade_count": 20,
                        "cost_decomposition": {"total": 0.0},
                    },
                },
                "1.0x": {
                    "cost_multiplier": 1.0,
                    "metrics": {
                        "gross_return": -0.01,
                        "net_return": -0.06,
                        "trade_count": 20,
                        "cost_decomposition": {"total": 5100.25},
                    },
                },
            },
            "benchmarks": {
                "cash": {"total_return": 0.0},
                "equal_weight_eligible_universe": {"total_return": -0.04},
                "relevant_bist_index": {"status": "not_available_in_input_dataset"},
            },
            "bootstrap": {
                "annualized_return": {
                    "estimate": -0.12,
                    "lower": -0.30,
                    "upper": 0.10,
                    "confidence_level": 0.95,
                }
            },
        },
    )
    _write_json(
        run_path / "run_manifest.json",
        {
            "run_id": run_path.name,
            "git_sha": "abc1234567890",
            "dirty_working_tree": False,
        },
    )
    _write_json(
        run_path / "data_manifest.json",
        {
            "dataset_id": "dataset-123",
            "universe_version": "fixed_bist_large_cap_prototype",
            "start": "2025-01-02",
            "end": "2025-12-31",
            "row_count": 1000,
        },
    )
    _write_json(
        run_path / "universe_manifest.json",
        {
            "universe_version": "fixed_bist_large_cap_prototype",
            "membership_type": "fixed_prototype_not_historical_index_membership",
            "tickers": ["GARAN", "ISCTR", "KCHOL", "THYAO"],
        },
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_path.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(run_path / "artifact_hashes.json", hashes)
    return run_path


def test_render_accepted_results_is_deterministic_and_evidence_based(tmp_path: Path) -> None:
    run_path = _run_directory(tmp_path)

    first = render_accepted_results(run_path)
    second = render_accepted_results(run_path)

    assert first == second
    assert "`20260714T120000Z-abc1234-deadbe`" in first
    assert "fixed_bist_large_cap_prototype" in first
    assert "GARAN, ISCTR, KCHOL, THYAO" in first
    assert "| ridge | 4 |" in first
    assert "| zero_return | 4 |" in first
    assert first.index("| 0.0x |") < first.index("| 1.0x |") < first.index("| 2.0x |")
    assert "No evaluated model achieved positive zero-mean R-squared" in first
    assert "The 95% block-bootstrap interval for annualized return spans zero" in first
    assert "No relevant BIST index benchmark was available in the accepted input dataset" in first


def test_update_replaces_only_marker_content_and_is_idempotent(tmp_path: Path) -> None:
    run_path = _run_directory(tmp_path)
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        f"before\n{START_MARKER}\nstale generated text\n{END_MARKER}\nafter\n",
        encoding="utf-8",
    )

    update_readme_results(readme_path, run_path)
    first = readme_path.read_text(encoding="utf-8")
    update_readme_results(readme_path, run_path)

    assert readme_path.read_text(encoding="utf-8") == first
    assert first.startswith(f"before\n{START_MARKER}\n")
    assert first.endswith(f"\n{END_MARKER}\nafter\n")
    assert "stale generated text" not in first


@pytest.mark.parametrize(
    "content",
    [
        "no markers\n",
        f"{START_MARKER}\nmissing end\n",
        f"{START_MARKER}\none\n{START_MARKER}\ntwo\n{END_MARKER}\n",
        f"{START_MARKER}\none\n{END_MARKER}\n{END_MARKER}\n",
    ],
)
def test_update_rejects_missing_or_duplicate_markers(tmp_path: Path, content: str) -> None:
    run_path = _run_directory(tmp_path)
    readme_path = tmp_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")

    with pytest.raises(ReadmeResultsError, match="exactly one accepted-results marker pair"):
        update_readme_results(readme_path, run_path)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "artifact_hashes.json",
        "metrics.json",
        "run_manifest.json",
        "data_manifest.json",
        "universe_manifest.json",
        "predictions.parquet",
    ],
)
def test_render_rejects_missing_required_artifacts(tmp_path: Path, artifact_name: str) -> None:
    run_path = _run_directory(tmp_path)
    (run_path / artifact_name).unlink()

    with pytest.raises(ReadmeResultsError, match="missing|integrity"):
        render_accepted_results(run_path)


def test_render_rejects_metrics_that_do_not_recompute_from_predictions(tmp_path: Path) -> None:
    run_path = _run_directory(tmp_path)
    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["prediction"]["ridge"]["mae"] = 999.0
    _write_json(metrics_path, metrics)
    hashes_path = run_path / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes["metrics.json"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    _write_json(hashes_path, hashes)

    with pytest.raises(ReadmeResultsError, match="does not match recomputation"):
        render_accepted_results(run_path)


def test_cli_updates_the_requested_readme(tmp_path: Path) -> None:
    run_path = _run_directory(tmp_path)
    readme_path = tmp_path / "README.md"
    readme_path.write_text(f"{START_MARKER}\nold\n{END_MARKER}\n", encoding="utf-8")

    exit_code = main(["--readme", str(readme_path), "--run", str(run_path)])

    assert exit_code == 0
    assert "old" not in readme_path.read_text(encoding="utf-8")
    assert run_path.name in readme_path.read_text(encoding="utf-8")
