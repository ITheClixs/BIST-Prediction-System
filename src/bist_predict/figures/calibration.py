"""Figures for what the evaluation stack does when the answer is known.

Every other figure in this report draws a measurement of the market. These draw
measurements of the instrument: each one places a rejection rate the estimator
produced against the rate it was supposed to produce, on data whose truth was
fixed before the estimator saw it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bist_predict.figures.style import COLOURS, caption, figure, save_figure

__all__ = [
    "CALIBRATION_FIGURE_BUILDERS",
    "build_all_calibration_figures",
    "plot_family_wise_error",
    "plot_nested_null",
    "plot_search_thresholds",
    "plot_size_by_cross_section",
]

_NOMINAL = 0.05


def _nominal_line(axis: Any, label: str = "nominal 5%") -> None:
    """Draw the level every rate on the panel is claiming to have."""
    axis.axhline(_NOMINAL, color=COLOURS["null"], linewidth=1.4, linestyle="--", zorder=2)
    axis.text(
        0.995,
        _NOMINAL,
        label,
        transform=axis.get_yaxis_transform(),
        fontsize=7.0,
        color=COLOURS["null"],
        ha="right",
        va="bottom",
    )


def _interval(record: Mapping[str, Any]) -> tuple[float, float, float]:
    rate = float(record["rate"])
    return rate, rate - float(record["lower"]), float(record["upper"]) - rate


def plot_size_by_cross_section(study: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Show the row-level test failing predictably, and the fix holding.

    The left panel is the study's central claim: the size of a pooled test is
    not an unknown finite-sample quirk but a closed-form function of the
    cross-section's width and correlation, and the simulation lands on that
    function. The right panel shows the session-aggregated test surviving every
    stress the generator can apply.
    """
    cells = [cell for cell in study["experiments"]["dependence"] if cell["varied"] == "unit_count"]
    units = np.array([int(cell["design"]["unit_count"]) for cell in cells])
    predicted = np.array([float(cell["predicted_row_size"]) for cell in cells])
    measured = np.array([_interval(cell["row_rejection"]) for cell in cells])
    session = np.array([_interval(cell["session_rejection"]) for cell in cells])

    robustness = study["experiments"]["robustness"]
    labels = {
        "gaussian_constant": "Gaussian",
        "student_t5": "$t_5$",
        "garch": "GARCH",
        "regime_switching": "regime",
        "heavy_tailed_garch": "$t_5$+GARCH",
        "sixty_sessions": "$n{=}60$",
        "five_hundred_sessions": "$n{=}500$",
        "idiosyncratic_forecast": "idio.\\ fc.",
    }
    names = [labels.get(str(cell["variant"]), str(cell["variant"])) for cell in robustness]
    stressed = np.array([_interval(cell["session_rejection"]) for cell in robustness])
    stressed_row = np.array([_interval(cell["row_rejection"]) for cell in robustness])

    with figure(7.2, 3.6) as fig:
        left, right = fig.subplots(1, 2)

        grid = np.geomspace(2, 120, 200)
        correlation = float(study["anchor_design"]["target_correlation"])
        from scipy import stats as _stats

        quantile = float(_stats.norm.ppf(1.0 - _NOMINAL / 2.0))
        curve = 2.0 * _stats.norm.cdf(-quantile / np.sqrt(1.0 + (grid - 1.0) * correlation))
        left.plot(
            grid,
            curve,
            color=COLOURS["neutral"],
            linewidth=1.6,
            zorder=2,
            label="closed form",
        )
        left.errorbar(
            units,
            measured[:, 0],
            yerr=measured[:, 1:].T,
            fmt="o",
            markersize=6,
            color=COLOURS["adverse"],
            ecolor=COLOURS["adverse"],
            capsize=3,
            zorder=4,
            label="measured, row level",
        )
        left.errorbar(
            units,
            session[:, 0],
            yerr=session[:, 1:].T,
            fmt="s",
            markersize=5,
            color=COLOURS["model"],
            ecolor=COLOURS["model"],
            capsize=3,
            zorder=4,
            label="measured, session level",
        )
        _nominal_line(left)
        left.set_xscale("log")
        left.set_xlabel("names in the cross-section")
        left.set_ylabel("rejection rate under the null")
        left.set_ylim(0.0, 0.9)
        left.set_title("Size against the closed form")
        left.legend(loc="upper left")

        positions = np.arange(len(names))
        right.errorbar(
            positions,
            stressed_row[:, 0],
            yerr=stressed_row[:, 1:].T,
            fmt="o",
            markersize=5,
            color=COLOURS["adverse"],
            ecolor=COLOURS["adverse"],
            capsize=3,
            zorder=4,
            label="row level",
        )
        right.errorbar(
            positions,
            stressed[:, 0],
            yerr=stressed[:, 1:].T,
            fmt="s",
            markersize=5,
            color=COLOURS["model"],
            ecolor=COLOURS["model"],
            capsize=3,
            zorder=4,
            label="session level",
        )
        _nominal_line(right)
        right.set_xticks(positions)
        right.set_xticklabels(names, rotation=35, ha="right", fontsize=7.0)
        right.set_ylabel("rejection rate under the null")
        right.set_ylim(0.0, 0.32)
        right.set_title("Session level under stress")
        right.legend(loc="upper right")

        caption(
            fig,
            "Left: the size of the equal-accuracy test applied to panel rows is a closed-form "
            "function of the cross-section's width and correlation, and the simulation lands on "
            "it at every width. Right: the session-aggregated test holds its nominal level under "
            "heavy tails, clustered volatility, a switching level, and record lengths from 60 to "
            "500 sessions.",
        )
        png, pdf = save_figure(fig, directory, "fig13_calibration_size")

    return {
        "figure": "fig13_calibration_size",
        "png": png.name,
        "pdf": pdf.name,
        "maximum_row_size": float(measured[:, 0].max()),
        "maximum_session_size": float(max(session[:, 0].max(), stressed[:, 0].max())),
        "largest_closed_form_error": float(np.abs(measured[:, 0] - predicted).max()),
    }


def plot_nested_null(study: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Show the squared-error comparison convicting an innocent forecast.

    Under the nested null the benchmark is right and the fitted model is noise.
    A correct test rejects five percent of the time in each direction. The
    squared-error comparison instead convicts the fitted model almost always,
    at a rate governed by the forecast's own variance rather than by any
    property of the data.
    """
    cells = study["experiments"]["nested"]
    labels = {
        "anchor": "anchor",
        "quiet_forecast": "quiet\nforecast",
        "loud_forecast": "loud\nforecast",
        "independent_rows": "uncorrelated\nnames",
        "thirty_names": "30 names",
        "five_hundred_sessions": "500\nsessions",
    }
    names = [labels.get(str(cell["variant"]), str(cell["variant"])) for cell in cells]
    against = np.array([float(cell["diebold_mariano_against_candidate"]["rate"]) for cell in cells])
    clark_west = np.array([_interval(cell["clark_west_session"]) for cell in cells])
    positions = np.arange(len(names))
    width = 0.38

    with figure(7.2, 3.4) as fig:
        axis = fig.subplots(1, 1)
        axis.bar(
            positions - width / 2,
            against,
            width,
            color=COLOURS["adverse"],
            zorder=3,
            label="Diebold--Mariano rejects towards the null",
        )
        axis.bar(
            positions + width / 2,
            clark_west[:, 0],
            width,
            yerr=clark_west[:, 1:].T,
            color=COLOURS["model"],
            ecolor=COLOURS["ink"],
            capsize=3,
            zorder=3,
            label="Clark--West rejects",
        )
        _nominal_line(axis)
        for position, value in zip(positions, against, strict=True):
            axis.text(
                position - width / 2,
                value + 0.015,
                f"{value:.3f}",
                fontsize=7.0,
                color=COLOURS["adverse"],
                ha="center",
                va="bottom",
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(names, fontsize=7.5)
        axis.set_ylabel("rejection rate under the nested null")
        # The value labels sit just above bars that reach 1.0, so the legend has
        # to clear them rather than share the headroom.
        axis.set_ylim(0.0, 1.30)
        axis.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
        caption(
            fig,
            "The zero benchmark is the correct population forecast and the fitted model is pure "
            "estimation noise, so neither test should fire more than five percent of the time. "
            "The squared-error comparison declares the fitted model significantly worse in up to "
            "all of the replications: the finding that a fitted model 'loses to the null' is "
            "produced by the act of fitting, not by the data. Clark--West holds its level.",
        )
        png, pdf = save_figure(fig, directory, "fig14_calibration_nested")

    return {
        "figure": "fig14_calibration_nested",
        "png": png.name,
        "pdf": pdf.name,
        "maximum_diebold_mariano_rate": float(against.max()),
        "minimum_diebold_mariano_rate": float(against.min()),
        "maximum_clark_west_rate": float(clark_west[:, 0].max()),
    }


def plot_family_wise_error(study: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Compare every multiplicity correction on a family with no skill in it."""
    cells = study["experiments"]["family"]
    labels = {
        "anchor": "4 names\ncorrelated",
        "independent_rows": "4 names\nindependent",
        "thirty_names": "30 names\ncorrelated",
        "thirty_names_independent": "30 names\nindependent",
    }
    corrections = (
        ("uncorrected_any", "uncorrected", COLOURS["neutral"]),
        ("holm_row", "Holm (row)", COLOURS["adverse"]),
        ("holm_session", "Holm (session)", COLOURS["model"]),
        ("reality_check", "Reality Check", COLOURS["reference"]),
        ("spa_hansen", "SPA", COLOURS["portfolio"]),
    )
    names = [labels.get(str(cell["variant"]), str(cell["variant"])) for cell in cells]
    positions = np.arange(len(names))
    width = 0.16

    with figure(7.2, 3.4) as fig:
        axis = fig.subplots(1, 1)
        for index, (key, label, colour) in enumerate(corrections):
            values = np.array([_interval(cell[key]) for cell in cells])
            offset = (index - (len(corrections) - 1) / 2) * width
            axis.bar(
                positions + offset,
                values[:, 0],
                width,
                yerr=values[:, 1:].T,
                color=colour,
                ecolor=COLOURS["ink"],
                capsize=2,
                zorder=3,
                label=label,
            )
        _nominal_line(axis)
        axis.set_xticks(positions)
        axis.set_xticklabels(names, fontsize=7.5)
        axis.set_ylabel("family-wise error rate")
        axis.legend(loc="upper left", ncol=3)
        caption(
            fig,
            "Every member of the family is skill-free by construction, so any rejection is a "
            "false one. Holm controls multiplicity but not dependence: on row-level p-values "
            "from a correlated panel its family-wise error reaches well over half, while the "
            "identical procedure applied to session-aggregated p-values is correct. The "
            "bootstrap corrections are mildly liberal but never fail.",
        )
        png, pdf = save_figure(fig, directory, "fig15_calibration_family")

    worst = max(float(cell["holm_row"]["rate"]) for cell in cells)
    return {
        "figure": "fig15_calibration_family",
        "png": png.name,
        "pdf": pdf.name,
        "worst_holm_row_rate": worst,
        "worst_holm_session_rate": max(float(cell["holm_session"]["rate"]) for cell in cells),
    }


def plot_search_thresholds(study: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Show the False Strategy expectation failing as a test, and the fix."""
    cells = study["experiments"]["search"]
    correlations = np.array([float(cell["trial_correlation"]) for cell in cells])
    expectation = np.array([_interval(cell["false_strategy_expectation"]) for cell in cells])
    bootstrap = np.array([_interval(cell["joint_bootstrap_quantile"]) for cell in cells])
    effective = np.array([float(cell["mean_independent_equivalent_trials"]) for cell in cells])
    nominal_trials = int(study["configuration"]["grid_trial_count"])

    with figure(7.2, 3.4) as fig:
        left, right = fig.subplots(1, 2)

        left.errorbar(
            correlations,
            expectation[:, 0],
            yerr=expectation[:, 1:].T,
            fmt="o-",
            markersize=6,
            color=COLOURS["adverse"],
            ecolor=COLOURS["adverse"],
            capsize=3,
            zorder=4,
            label=r"clears $\mathbb{E}[\max]$",
        )
        left.errorbar(
            correlations,
            bootstrap[:, 0],
            yerr=bootstrap[:, 1:].T,
            fmt="s-",
            markersize=5,
            color=COLOURS["model"],
            ecolor=COLOURS["model"],
            capsize=3,
            zorder=4,
            label=r"clears bootstrap $q_{0.95}$",
        )
        _nominal_line(left)
        left.set_xlabel("correlation between trials")
        left.set_ylabel("false-positive rate")
        left.set_ylim(0.0, 0.62)
        left.set_title("An expectation is not a critical value")
        left.legend(loc="center left")

        right.plot(
            correlations,
            effective,
            "o-",
            markersize=6,
            color=COLOURS["portfolio"],
            zorder=4,
        )
        right.axhline(
            nominal_trials,
            color=COLOURS["neutral"],
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )
        right.text(
            0.995,
            nominal_trials,
            f"nominal {nominal_trials}",
            transform=right.get_yaxis_transform(),
            fontsize=7.0,
            color=COLOURS["muted"],
            ha="right",
            va="bottom",
        )
        for x, y in zip(correlations, effective, strict=True):
            right.annotate(
                f"{y:.1f}",
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=7.0,
                color=COLOURS["portfolio"],
                ha="center",
            )
        right.set_yscale("log")
        right.set_xlabel("correlation between trials")
        right.set_ylabel("independent-equivalent trials")
        right.set_title("What the grid really searched")

        caption(
            fig,
            "Left: a skill-free grid clears the False Strategy Theorem's expected maximum about "
            "half the time at every level of dependence, because that threshold is an "
            "expectation; the synchronised bootstrap quantile of the same grid holds the nominal "
            f"level. Right: the {nominal_trials} configurations behave like the independent-trial "
            "count shown, which collapses towards one as the trials come to share their data.",
        )
        png, pdf = save_figure(fig, directory, "fig16_calibration_search")

    return {
        "figure": "fig16_calibration_search",
        "png": png.name,
        "pdf": pdf.name,
        "worst_false_strategy_rate": float(expectation[:, 0].max()),
        "worst_bootstrap_rate": float(bootstrap[:, 0].max()),
        "effective_trials_at_highest_correlation": float(effective[-1]),
    }


CALIBRATION_FIGURE_BUILDERS: tuple[Callable[[Mapping[str, Any], Path], dict[str, Any]], ...] = (
    plot_size_by_cross_section,
    plot_nested_null,
    plot_family_wise_error,
    plot_search_thresholds,
)


def build_all_calibration_figures(
    study_path: Path | str, output_directory: Path | str
) -> dict[str, Any]:
    """Build every calibration figure and return what each one computed."""
    study = json.loads(Path(study_path).read_text(encoding="utf-8"))
    directory = Path(output_directory)
    facts = [builder(study, directory) for builder in CALIBRATION_FIGURE_BUILDERS]
    manifest = {
        "content_hash": study["content_hash"],
        "schema_version": study["schema_version"],
        "figure_count": len(facts),
        "figures": facts,
    }
    (directory / "calibration_figure_facts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
