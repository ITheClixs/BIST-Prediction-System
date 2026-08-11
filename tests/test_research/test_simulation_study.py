"""Whether the study assembles, replays and hashes the way the paper needs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bist_predict.research.simulation.study import (
    SCHEMA_VERSION,
    StudyConfiguration,
    run_study,
    write_study,
)


@pytest.fixture(scope="module")
def study() -> dict[str, object]:
    """Run the reduced study once and share it across the assertions."""
    return run_study(StudyConfiguration().quick(factor=200))


def test_quick_shrinks_every_replication_count() -> None:
    full = StudyConfiguration()
    quick = full.quick(factor=10)
    assert quick.size_replications < full.size_replications
    assert quick.power_replications < full.power_replications
    assert quick.nested_replications < full.nested_replications
    assert quick.family_replications < full.family_replications
    assert quick.search_replications < full.search_replications


def test_quick_preserves_the_design_it_is_calibrating() -> None:
    """Shrinking the replication budget must not move the design point.

    A quick study that also moved the anchor would be exercising a different
    experiment from the one the manuscript reports, and the smoke check would
    stop guarding anything.
    """
    full = StudyConfiguration()
    assert full.quick(factor=10).anchor_design() == full.anchor_design()


def test_a_non_positive_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="factor must be positive"):
        StudyConfiguration().quick(factor=0)


def test_the_study_carries_every_experiment(study: dict[str, object]) -> None:
    experiments = study["experiments"]
    assert isinstance(experiments, dict)
    assert set(experiments) == {
        "dependence",
        "robustness",
        "power",
        "nested",
        "family",
        "search",
    }
    assert study["schema_version"] == SCHEMA_VERSION


def test_the_study_records_the_environment_that_produced_it(study: dict[str, object]) -> None:
    environment = study["environment"]
    assert isinstance(environment, dict)
    assert set(environment) == {"python", "numpy", "scipy"}


def test_the_anchor_size_is_reported_in_closed_form(study: dict[str, object]) -> None:
    """The prediction has to be stated to be falsifiable by the measurement."""
    predicted = study["closed_form_anchor_size"]
    assert isinstance(predicted, float)
    assert 0.05 < predicted < 1.0


def test_the_closed_form_tracks_the_measured_row_level_size(study: dict[str, object]) -> None:
    """The measured size must follow the prediction, cell by cell.

    This is the study's central empirical claim. At the reduced replication
    count the Monte Carlo error is large, so the tolerance is loose; the full
    study tightens it, but the direction and the rough magnitude have to hold
    here or the closed form is not describing this estimator at all.
    """
    experiments = study["experiments"]
    assert isinstance(experiments, dict)
    cells = experiments["dependence"]
    assert isinstance(cells, list)
    swept = [cell for cell in cells if cell["varied"] == "unit_count"]
    assert len(swept) >= 3
    for cell in swept:
        assert cell["row_rejection"]["rate"] == pytest.approx(cell["predicted_row_size"], abs=0.12)


def test_session_aggregation_is_the_correction_the_study_recommends(
    study: dict[str, object],
) -> None:
    """Whatever the cross-section does to the row-level test, sessions survive it."""
    experiments = study["experiments"]
    assert isinstance(experiments, dict)
    cells = experiments["dependence"]
    assert isinstance(cells, list)
    for cell in cells:
        if cell["varied"] != "unit_count":
            continue
        assert cell["session_rejection"]["rate"] < 0.20


def test_the_study_replays_from_its_seed() -> None:
    """Two runs of the same configuration must agree on every number."""
    configuration = StudyConfiguration().quick(factor=400)
    first = run_study(configuration)
    second = run_study(configuration)
    del first["elapsed_seconds"], second["elapsed_seconds"]
    assert first == second


def test_writing_records_a_hash_that_ignores_the_clock(tmp_path: Path) -> None:
    """The content hash identifies the numbers, not the machine's speed."""
    configuration = StudyConfiguration().quick(factor=400)
    first = run_study(configuration)
    second = run_study(configuration)
    second["elapsed_seconds"] = float(first["elapsed_seconds"]) + 123.0

    first_path = write_study(tmp_path / "a" / "study.json", first)
    second_path = write_study(tmp_path / "b" / "study.json", second)
    first_record = json.loads(first_path.read_text(encoding="utf-8"))
    second_record = json.loads(second_path.read_text(encoding="utf-8"))

    assert first_record["content_hash"] == second_record["content_hash"]
    assert first_record["elapsed_seconds"] != second_record["elapsed_seconds"]


def test_a_changed_number_changes_the_hash(tmp_path: Path) -> None:
    configuration = StudyConfiguration().quick(factor=400)
    study = run_study(configuration)
    original = json.loads(write_study(tmp_path / "a.json", study).read_text(encoding="utf-8"))
    study["closed_form_anchor_size"] = float(study["closed_form_anchor_size"]) + 0.01
    mutated = json.loads(write_study(tmp_path / "b.json", study).read_text(encoding="utf-8"))
    assert original["content_hash"] != mutated["content_hash"]


def test_writing_creates_the_destination_directory(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "deeper" / "study.json"
    written = write_study(destination, run_study(StudyConfiguration().quick(factor=400)))
    assert written == destination
    assert destination.exists()
