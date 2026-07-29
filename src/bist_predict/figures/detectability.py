"""Figures for what the design could have found, rather than what it found."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure
from bist_predict.research.inference.detectability import (
    expected_top_selection_score,
    false_strategy_quantile,
    minimum_detectable_mean,
    panel_information_ceiling,
)

__all__ = [
    "plot_detectable_effect",
    "plot_breadth_cost_feasibility",
    "plot_search_threshold",
]

# The band of out-of-sample R-squared that the daily equity forecasting
# literature reports for horizons of one session. Anything a study claims above
# this range is more likely a leak than a discovery.
_PLAUSIBLE_R_SQUARED = (0.001, 0.01)


def plot_detectable_effect(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Place the smallest detectable effect beside the effects worth detecting.

    The left panel inverts the test that was actually run: at each sample size
    it shows the out-of-sample R-squared a two-sided test at 5% would reject the
    null for four times in five. The right panel asks whether a wider panel
    could close the gap, and answers that correlated names run into a ceiling.
    """
    report = artifacts.metrics["inference"]["detectability"]
    sessions = int(report["session_count"])
    standard_error = float(report["reference_standard_error"])
    benchmark_mse = float(report["benchmark_mean_squared_error"])
    achieved = float(report["minimum_detectable_r_squared"])
    required = int(report["sessions_required_for_reference_r_squared"])
    reference = float(report["reference_r_squared"])
    panel = report["panel"]
    correlation = float(panel["mean_pairwise_correlation"])

    grid = np.unique(np.round(np.geomspace(20, 400_000, 220)).astype(int))
    detectable = np.array(
        [
            minimum_detectable_mean(
                standard_error * math.sqrt(sessions / count), observations=count
            )
            / benchmark_mse
            for count in grid
        ]
    )

    with figure(7.2, 3.5) as fig:
        left, right = fig.subplots(1, 2)

        left.axhspan(
            _PLAUSIBLE_R_SQUARED[0],
            _PLAUSIBLE_R_SQUARED[1],
            color=COLOURS["band"],
            alpha=0.75,
            zorder=1,
        )
        left.text(
            grid.max() * 0.7,
            _PLAUSIBLE_R_SQUARED[0] * 1.15,
            "effects a daily forecast plausibly has",
            fontsize=7.0,
            color=COLOURS["muted"],
            ha="right",
            va="bottom",
        )
        left.plot(grid, detectable, color=COLOURS["model"], linewidth=1.8, zorder=3)
        left.scatter([sessions], [achieved], s=60, color=COLOURS["adverse"], zorder=5)
        left.annotate(
            f"this study\n{sessions} sessions, {achieved:.3f}",
            xy=(sessions, achieved),
            xytext=(sessions * 2.2, achieved * 2.4),
            fontsize=7.5,
            color=COLOURS["adverse"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["adverse"], "linewidth": 0.8},
        )
        left.scatter([required], [reference], s=60, color=COLOURS["reference"], zorder=5)
        left.annotate(
            f"{required:,} sessions\nwould reach {reference:.2f}",
            xy=(required, reference),
            xytext=(required / 26.0, reference / 9.0),
            fontsize=7.5,
            color=COLOURS["reference"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["reference"], "linewidth": 0.8},
        )
        left.set_xscale("log")
        left.set_yscale("log")
        left.set_xlabel("evaluation sessions")
        left.set_ylabel("smallest detectable out-of-sample $R^2$")
        left.set_title("Power, not evidence", loc="left", color=COLOURS["ink"])

        sizes = np.arange(2, 201)
        rows = np.array(
            [
                panel_information_ceiling(correlation, int(size))["independent_rows_per_session"]
                for size in sizes
            ]
        )
        ceiling = float(panel["independent_rows_per_session_ceiling"])
        right.plot(sizes, rows, color=COLOURS["portfolio"], linewidth=1.8, zorder=3)
        right.axhline(
            ceiling,
            color=COLOURS["adverse"],
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        right.text(
            200,
            ceiling,
            f"ceiling $1/\\bar\\rho$ = {ceiling:.2f}",
            fontsize=7.5,
            color=COLOURS["adverse"],
            ha="right",
            va="bottom",
        )
        right.scatter(
            [int(panel["unit_count"])],
            [float(panel["independent_rows_per_session"])],
            s=60,
            color=COLOURS["adverse"],
            zorder=5,
        )
        right.annotate(
            f"this universe\n{int(panel['unit_count'])} names, "
            f"{float(panel['independent_rows_per_session']):.2f}",
            xy=(int(panel["unit_count"]), float(panel["independent_rows_per_session"])),
            xytext=(14, float(panel["independent_rows_per_session"]) - 0.34),
            fontsize=7.5,
            color=COLOURS["adverse"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["adverse"], "linewidth": 0.8},
        )
        right.set_xscale("log")
        right.set_xlabel("names in the cross-section")
        right.set_ylabel("independent rows per session")
        right.set_ylim(0.0, ceiling * 1.22)
        right.set_title("Breadth cannot buy precision", loc="left", color=COLOURS["ink"])

        caption(
            fig,
            f"At {sessions} sessions the design separates an out-of-sample $R^2$ from zero only "
            f"once it exceeds {achieved:.3f}, roughly {achieved / reference:.0f} times the largest "
            f"effect the literature reports; {required:,} sessions would be needed for "
            f"{reference:.2f}. Widening the cross-section raises the independent rows a session "
            f"carries from {float(panel['independent_rows_per_session']):.2f} only as far as "
            f"{ceiling:.2f}, a standard-error gain of "
            f"{(float(panel['standard_error_headroom']) - 1.0) * 100:.0f}% for unlimited names.",
        )
        png, pdf = save_figure(fig, directory, "fig10_detectable_effect")
    return {
        "figure": "fig10_detectable_effect",
        "png": png.name,
        "pdf": pdf.name,
        "session_count": sessions,
        "minimum_detectable_r_squared": achieved,
        "reference_r_squared": reference,
        "sessions_required": required,
        "independent_rows_per_session": float(panel["independent_rows_per_session"]),
        "independent_rows_ceiling": ceiling,
        "standard_error_headroom": float(panel["standard_error_headroom"]),
    }


def plot_breadth_cost_feasibility(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Draw the information coefficient a top-k rule needs before it can pay costs.

    Proposition 1 makes the requirement a function of three things only: the
    round-trip cost, the volatility of the target, and how sharply the rule
    selects. None of them is a modelling choice, so the curve is a property of
    the experiment rather than of the forecaster placed inside it.
    """
    feasibility = artifacts.metrics["inference"]["detectability"]["feasibility"]
    realised = float(
        artifacts.metrics["inference"]["detectability"]["realised_information_coefficient"]
    )
    cost = float(feasibility["round_trip_cost_rate"])
    volatility = float(feasibility["target_volatility"])
    universe = int(feasibility["universe_size"])
    selected = int(feasibility["selected"])
    required = float(feasibility["required_information_coefficient"])

    fractions = (0.5, 0.25, 0.1, 0.02)
    largest = 500

    with figure(7.2, 4.0) as fig:
        axes = fig.add_subplot(111)
        shades = (COLOURS["null"], COLOURS["model"], COLOURS["portfolio"], COLOURS["reference"])
        for fraction, colour in zip(fractions, shades, strict=True):
            # Step in whole held names so that every plotted point is an exactly
            # achievable ratio; rounding a fixed grid of universe sizes instead
            # makes the curve saw-tooth on the rounding rather than on the
            # economics.
            widths, curve = [], []
            for hold in range(1, int(largest * fraction) + 1):
                size = int(round(hold / fraction))
                if size > largest or hold >= size:
                    continue
                score = expected_top_selection_score(size, hold)
                widths.append(size)
                curve.append(cost / (volatility * score))
            axes.plot(
                widths,
                curve,
                color=colour,
                linewidth=1.7,
                label=f"hold the top {fraction:.0%}",
                zorder=3,
            )
        axes.axhline(
            realised,
            color=COLOURS["ink"],
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        axes.text(
            4.2,
            realised * 0.94,
            f"information coefficient this study achieved = {realised:.3f}",
            fontsize=7.5,
            color=COLOURS["ink"],
            ha="left",
            va="top",
        )
        axes.scatter([universe], [required], s=90, color=COLOURS["adverse"], zorder=6)
        axes.annotate(
            f"this design: {universe} names, top {selected}\nneeds {required:.3f}",
            xy=(universe, required),
            xytext=(universe * 1.9, required * 1.06),
            fontsize=7.5,
            color=COLOURS["adverse"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["adverse"], "linewidth": 0.8},
        )
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_ylim(0.02, 0.6)
        axes.set_xlabel("names ranked")
        axes.set_ylabel("information coefficient required to break even")
        axes.set_title(
            "What the cost schedule demands of a forecast, before any model is chosen",
            loc="left",
            color=COLOURS["ink"],
        )
        axes.legend(loc="upper right")
        caption(
            fig,
            f"At a round-trip cost of {cost * 1e4:.1f} basis points against a target volatility of "
            f"{volatility * 100:.2f}%, holding the top {selected} of {universe} names requires a "
            f"cross-sectional information coefficient of {required:.3f}. The portfolio model "
            f"achieved {realised:.3f}, short by a factor of {required / realised:.1f}. Breadth "
            "relaxes the requirement sharply at first and then hardly at all: at a fixed holding "
            "fraction the selection score converges to the mean of that tail of the normal, so "
            "past a few dozen names only a more selective rule helps.",
        )
        png, pdf = save_figure(fig, directory, "fig11_breadth_cost_feasibility")
    return {
        "figure": "fig11_breadth_cost_feasibility",
        "png": png.name,
        "pdf": pdf.name,
        "round_trip_cost_rate": cost,
        "target_volatility": volatility,
        "required_information_coefficient": required,
        "realised_information_coefficient": realised,
        "shortfall_factor": float(required / realised),
    }


def plot_search_threshold(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Bracket the search correction between its two defensible readings.

    Treating 72 overlapping configurations as 72 independent searches sets one
    bar; using the dispersion those configurations actually showed sets a lower
    one. The grid maximum sits below both, so the conclusion does not rest on
    which reading is preferred.
    """
    detectability = artifacts.metrics["inference"]["detectability"]
    trials = int(detectability["trial_count"])
    realised_variance = float(detectability["realised_trial_variance"])
    independent_variance = float(detectability["independent_trial_variance"])
    effective = float(detectability["effective_trial_count"])
    realised_threshold = float(detectability["deflated_sharpe_threshold"])
    independent_threshold = float(detectability["independent_trial_threshold"])
    grid_maximum = float(detectability["grid_maximum_sharpe"])

    counts = np.unique(np.round(np.geomspace(2, 500, 160)).astype(int))
    realised_curve = np.array(
        [math.sqrt(realised_variance) * false_strategy_quantile(float(count)) for count in counts]
    )
    independent_curve = np.array(
        [
            math.sqrt(independent_variance) * false_strategy_quantile(float(count))
            for count in counts
        ]
    )

    with figure(7.2, 4.0) as fig:
        axes = fig.add_subplot(111)
        axes.fill_between(
            counts,
            realised_curve,
            independent_curve,
            color=COLOURS["band"],
            alpha=0.6,
            zorder=1,
            label="range spanned by the two readings",
        )
        axes.plot(
            counts,
            independent_curve,
            color=COLOURS["adverse"],
            linewidth=1.7,
            zorder=3,
            label="trials treated as independent",
        )
        axes.plot(
            counts,
            realised_curve,
            color=COLOURS["portfolio"],
            linewidth=1.7,
            zorder=3,
            label="trials at their realised dispersion",
        )
        axes.axhline(grid_maximum, color=COLOURS["ink"], linewidth=1.3, zorder=4)
        axes.text(
            2.2,
            grid_maximum,
            f"best configuration in the grid = {grid_maximum:.4f}",
            fontsize=7.5,
            color=COLOURS["ink"],
            va="bottom",
        )
        axes.scatter([trials], [realised_threshold], s=70, color=COLOURS["portfolio"], zorder=6)
        axes.scatter([trials], [independent_threshold], s=70, color=COLOURS["adverse"], zorder=6)
        axes.vlines(
            effective,
            0.0,
            realised_threshold,
            color=COLOURS["muted"],
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            zorder=5,
        )
        axes.annotate(
            f"the {trials} configurations disperse\nlike {effective:.1f} independent searches",
            xy=(effective, realised_threshold),
            xytext=(3.2, realised_threshold * 1.42),
            fontsize=7.5,
            color=COLOURS["muted"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["muted"], "linewidth": 0.8},
        )
        axes.set_xscale("log")
        axes.set_xlabel("configurations searched")
        axes.set_ylabel("per-session Sharpe ratio luck alone would reach")
        axes.set_ylim(0.0, max(independent_curve) * 1.12)
        axes.set_title(
            "Both readings of the search correction clear the grid maximum",
            loc="left",
            color=COLOURS["ink"],
        )
        axes.legend(loc="upper left")
        caption(
            fig,
            f"Under the False Strategy Theorem, {trials} independent trials would set a bar of "
            f"{independent_threshold:.4f}; the dispersion the grid actually showed sets "
            f"{realised_threshold:.4f}, the bar a search of {effective:.1f} independent trials "
            f"would set. The grid maximum of {grid_maximum:.4f} falls below both, so nothing in "
            "the search survives either accounting.",
        )
        png, pdf = save_figure(fig, directory, "fig12_search_threshold")
    return {
        "figure": "fig12_search_threshold",
        "png": png.name,
        "pdf": pdf.name,
        "trial_count": trials,
        "effective_trial_count": effective,
        "realised_threshold": realised_threshold,
        "independent_threshold": independent_threshold,
        "grid_maximum": grid_maximum,
        "grid_maximum_below_both": bool(
            grid_maximum < realised_threshold and grid_maximum < independent_threshold
        ),
    }
