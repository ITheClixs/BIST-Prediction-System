"""Figures must build from a real run and report what they drew."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bist_predict.figures import FIGURE_BUILDERS, RunArtifacts, build_all_figures

RUN = Path(__file__).resolve().parents[2] / "runs" / "20260728T223101Z-8b27df3-2a71b8"

pytestmark = pytest.mark.skipif(not RUN.is_dir(), reason="committed accepted run is unavailable")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return build_all_figures(RUN, tmp_path_factory.mktemp("figures"))


def test_every_builder_produces_both_a_raster_and_a_vector_copy(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    directory = tmp_path_factory.mktemp("formats")
    manifest = build_all_figures(RUN, directory)
    assert manifest["figure_count"] == len(FIGURE_BUILDERS)
    for record in manifest["figures"]:
        assert (directory / str(record["png"])).stat().st_size > 10_000
        assert (directory / str(record["pdf"])).stat().st_size > 5_000


def test_figure_facts_are_written_beside_the_images(built: dict[str, object]) -> None:
    """A caption can then be checked against what the figure computed."""
    assert built["run_id"] == "20260728T223101Z-8b27df3-2a71b8"
    assert len(built["figures"]) == len(FIGURE_BUILDERS)


def test_no_fitted_model_reaches_the_null(built: dict[str, object]) -> None:
    record = next(item for item in built["figures"] if item["figure"].startswith("fig03"))
    assert record["models_above_null"] == 0


def test_every_holm_rejection_is_against_the_model(built: dict[str, object]) -> None:
    """A rejection with the opposite sign would mean a model beat the null."""
    record = next(item for item in built["figures"] if item["figure"].startswith("fig04"))
    assert record["all_rejections_adverse"] is True


def test_the_reality_check_figure_recomputes_the_stored_p_value(
    built: dict[str, object],
) -> None:
    """The histogram is regenerated, so it must land on the persisted number."""
    record = next(item for item in built["figures"] if item["figure"].startswith("fig05"))
    assert record["recomputed_p_value"] == pytest.approx(record["reported_p_value"], abs=1e-12)


def test_the_cost_curve_is_monotone(built: dict[str, object]) -> None:
    record = next(item for item in built["figures"] if item["figure"].startswith("fig07"))
    assert record["net_is_monotone"] is True
    assert 0.0 < float(record["breakeven_multiplier"]) < 1.0


def test_the_block_length_sweep_never_excludes_zero(built: dict[str, object]) -> None:
    record = next(item for item in built["figures"] if item["figure"].startswith("fig09"))
    assert record["intervals_spanning_zero"] == record["interval_count"]


def test_a_tampered_run_is_refused(tmp_path: Path) -> None:
    """Figures are only ever drawn from a bundle whose hashes still verify."""
    import shutil

    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    metrics = json.loads((copied / "metrics.json").read_text())
    metrics["portfolio"]["net_return"] = 0.99
    (copied / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="integrity check failed"):
        RunArtifacts.load(copied)
