"""Figures comparing forecast accuracy against the zero-return null."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure

__all__ = ["plot_out_of_sample_r_squared", "plot_equal_accuracy_tests"]


def plot_out_of_sample_r_squared(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Rank the models by zero-mean out-of-sample R-squared.

    The null sits at exactly zero by construction, so the only question the
    chart has to answer is whether any bar reaches the right of it.
    """
    prediction = artifacts.metrics["prediction"]
    scores = {
        name: float(values["zero_mean_r_squared"])
        for name, values in prediction.items()
        if values.get("zero_mean_r_squared") is not None
    }
    ordered = sorted(scores, key=lambda name: scores[name])
    values = [scores[name] for name in ordered]
    colours = [COLOURS["null"] if name == "zero_return" else COLOURS["model"] for name in ordered]

    with figure(7.2, 3.6) as fig:
        axes = fig.add_subplot(111)
        axes.barh(ordered, values, height=0.62, color=colours, zorder=3)
        axes.axvline(0.0, color=COLOURS["ink"], linewidth=1.0, zorder=4)
        axes.set_xlabel("zero-mean out-of-sample $R^2$")
        axes.grid(axis="y", visible=False)
        axes.set_title(
            "No fitted model reaches the zero-return null", loc="left", color=COLOURS["ink"]
        )
        span = max(abs(min(values)), 0.05)
        axes.set_xlim(-span * 1.28, span * 0.30)
        for name, value in zip(ordered, values, strict=True):
            offset = -span * 0.02 if value < 0 else span * 0.02
            axes.text(
                value + offset,
                name,
                f"{value:.4f}",
                va="center",
                ha="right" if value < 0 else "left",
                fontsize=8,
                color=COLOURS["ink"],
            )
        above = sum(1 for value in values if value > 0.0)
        caption(
            fig,
            f"{above} of {len(values)} models sit to the right of the null. "
            f"The worst, `{ordered[0]}`, is {abs(values[0]):.3f} below it.",
        )
        png, pdf = save_figure(fig, directory, "fig03_out_of_sample_r_squared")
    return {
        "figure": "fig03_out_of_sample_r_squared",
        "png": png.name,
        "pdf": pdf.name,
        "models_above_null": above,
        "model_count": len(values),
        "worst_model": ordered[0],
        "worst_value": values[0],
    }


def plot_equal_accuracy_tests(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Contrast session-aggregated and row-level p-values for the same tests.

    Both panels test the same six hypotheses on the same data. The left panel
    respects the cross-sectional dependence; the right panel is what treating
    every panel row as an independent draw produces.
    """
    accuracy = artifacts.metrics["inference"]["equal_predictive_accuracy"]
    session = accuracy["session_aggregated"]
    row_level = accuracy["row_level_for_comparison"]
    adjusted = accuracy["family_wise_correction"]["adjusted_p_values"]
    models = sorted(session)

    with figure(7.2, 3.8) as fig:
        left, right = fig.subplots(1, 2, sharey=True)

        statistics = [float(session[name]["statistic"]) for name in models]
        colours = [
            COLOURS["adverse"] if float(adjusted[name]) <= 0.05 else COLOURS["neutral"]
            for name in models
        ]
        indices_left = np.arange(len(models), dtype=float)
        left.barh(indices_left, statistics, height=0.6, color=colours, zorder=3)
        left.set_yticks(indices_left, models)
        left.set_ylim(-1.1, len(models) - 0.4)
        left.axvline(0.0, color=COLOURS["ink"], linewidth=1.0, zorder=4)
        left.set_xlabel("Diebold-Mariano statistic")
        left.grid(axis="y", visible=False)
        left.set_title("Session-aggregated", loc="left", color=COLOURS["ink"])
        for position, value in zip(indices_left, statistics, strict=True):
            left.text(
                value + 0.12,
                position,
                f"{value:+.2f}",
                va="center",
                fontsize=8,
                color=COLOURS["ink"],
            )
        left.set_xlim(min(0.0, min(statistics)) - 0.5, max(statistics) * 1.28)

        session_p = np.array([float(session[name]["p_value"]) for name in models])
        row_p = np.array([float(row_level[name]["p_value"]) for name in models])
        floor = 1e-7
        indices = np.arange(len(models), dtype=float)
        drawn_session = np.maximum(session_p, floor)
        drawn_row = np.maximum(row_p, floor)
        right.hlines(
            indices,
            drawn_row,
            drawn_session,
            color=COLOURS["neutral"],
            linewidth=1.4,
            zorder=3,
        )
        right.scatter(
            drawn_session,
            indices,
            s=46,
            color=COLOURS["portfolio"],
            zorder=4,
            label="session-aggregated (120 observations)",
        )
        right.scatter(
            drawn_row,
            indices,
            s=46,
            color=COLOURS["reference"],
            zorder=4,
            label="row-level (480 observations)",
        )
        right.set_xscale("log")
        right.set_xlim(floor * 0.5, 1.0)
        right.axvline(0.05, color=COLOURS["ink"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        right.text(0.055, -0.75, "p = 0.05", fontsize=7.5, color=COLOURS["muted"])
        right.set_xlabel("p-value (log scale)")
        right.grid(axis="y", visible=False)
        right.set_yticks(indices, models)
        right.set_ylim(-1.1, len(models) - 0.4)
        right.set_title("Same tests, two sample-size assumptions", loc="left", color=COLOURS["ink"])
        right.legend(loc="upper left", bbox_to_anchor=(0.0, 0.16))

        shrunk = int(np.sum(row_p < session_p))
        ratio = float(np.median(session_p / np.maximum(row_p, floor)))
        caption(
            fig,
            "Left: bars in red are rejected by Holm at the 5% family-wise level. A positive "
            "statistic means the model loses to the null, and every rejection here is of that "
            f"sign. Right: treating panel rows as independent shrinks {shrunk} of {len(models)} "
            f"p-values, by a median factor of {ratio:.0f}.",
        )
        png, pdf = save_figure(fig, directory, "fig04_equal_accuracy_tests")
    return {
        "figure": "fig04_equal_accuracy_tests",
        "png": png.name,
        "pdf": pdf.name,
        "rejected_count": sum(1 for name in models if float(adjusted[name]) <= 0.05),
        "shrunk_count": shrunk,
        "median_shrink_factor": ratio,
        "all_rejections_adverse": all(
            float(session[name]["statistic"]) > 0.0
            for name in models
            if float(adjusted[name]) <= 0.05
        ),
    }
