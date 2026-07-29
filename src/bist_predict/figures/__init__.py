"""Report figures, every one generated from an immutable run bundle."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bist_predict.figures.accuracy import (
    plot_equal_accuracy_tests,
    plot_out_of_sample_r_squared,
)
from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.design import plot_effective_sample_size, plot_fold_geometry
from bist_predict.figures.detectability import (
    plot_breadth_cost_feasibility,
    plot_detectable_effect,
    plot_search_threshold,
)
from bist_predict.figures.portfolio import plot_cost_sensitivity, plot_equity_curve
from bist_predict.figures.search import (
    plot_block_length_sensitivity,
    plot_configuration_search,
)
from bist_predict.figures.snooping import plot_reality_check

__all__ = ["FIGURE_BUILDERS", "RunArtifacts", "build_all_figures"]

FIGURE_BUILDERS: tuple[Callable[[RunArtifacts, Path], dict[str, Any]], ...] = (
    plot_fold_geometry,
    plot_effective_sample_size,
    plot_out_of_sample_r_squared,
    plot_equal_accuracy_tests,
    plot_reality_check,
    plot_equity_curve,
    plot_cost_sensitivity,
    plot_configuration_search,
    plot_block_length_sensitivity,
    plot_detectable_effect,
    plot_breadth_cost_feasibility,
    plot_search_threshold,
)


def build_all_figures(run_path: Path | str, output_directory: Path | str) -> dict[str, Any]:
    """Build every figure and return the facts each one computed.

    The returned record is written beside the images so a caption can be
    checked against the numbers the figure actually drew, rather than against
    what the caption asserts.
    """
    artifacts = RunArtifacts.load(run_path)
    directory = Path(output_directory)
    facts = [builder(artifacts, directory) for builder in FIGURE_BUILDERS]
    manifest = {
        "run_id": artifacts.run_manifest["run_id"],
        "dataset_id": artifacts.data_manifest["dataset_id"],
        "figure_count": len(facts),
        "figures": facts,
    }
    (directory / "figure_facts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
