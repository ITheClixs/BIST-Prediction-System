"""Figures for the configuration search and the arbitrary-choice sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure

__all__ = ["plot_configuration_search", "plot_block_length_sensitivity"]


def plot_configuration_search(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Put every configuration's Sharpe ratio next to the luck threshold.

    Each dot is a complete re-run of the evaluation under one fold geometry and
    one portfolio breadth. The vertical line is the Sharpe ratio the best of
    this many skill-free configurations would be expected to reach by chance
    alone, so a grid maximum to its left is what no skill looks like.
    """
    summary = artifacts.metrics["configuration_sensitivity"]
    trials = artifacts.sensitivity
    sharpe = trials["per_period_sharpe"].to_numpy(dtype=np.float64)
    threshold = float(summary["expected_maximum_sharpe_under_no_skill"])
    reported = float(summary["reported_trial"]["per_period_sharpe"])
    breadths = sorted(int(value) for value in trials["top_k"].unique())

    with figure(7.2, 4.0) as fig:
        axes = fig.add_subplot(111)
        rng = np.random.default_rng(0)
        for index, breadth in enumerate(breadths):
            mask = trials["top_k"].to_numpy() == breadth
            jitter = rng.uniform(-0.16, 0.16, size=int(mask.sum()))
            axes.scatter(
                sharpe[mask],
                index + jitter,
                s=34,
                color=COLOURS["model"],
                edgecolor=COLOURS["surface"],
                linewidth=0.7,
                zorder=3,
            )
        axes.axvline(0.0, color=COLOURS["ink"], linewidth=1.0, zorder=4)
        axes.axvline(
            threshold,
            color=COLOURS["adverse"],
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        axes.text(
            threshold - 0.006,
            len(breadths) - 1.15,
            f"expected maximum under\nno skill = {threshold:.3f}",
            fontsize=7.5,
            color=COLOURS["adverse"],
            ha="right",
            va="top",
        )
        axes.scatter(
            [reported],
            [breadths.index(int(summary["reported_trial"]["top_k"]))],
            s=150,
            facecolor="none",
            edgecolor=COLOURS["ink"],
            linewidth=1.6,
            zorder=5,
        )
        axes.annotate(
            "reported configuration",
            xy=(reported, breadths.index(int(summary["reported_trial"]["top_k"]))),
            xytext=(
                reported - 0.075,
                breadths.index(int(summary["reported_trial"]["top_k"])) - 0.55,
            ),
            fontsize=7.5,
            color=COLOURS["ink"],
            arrowprops={"arrowstyle": "-", "color": COLOURS["ink"], "linewidth": 0.8},
        )
        axes.set_yticks(range(len(breadths)), [f"top-{value}" for value in breadths])
        axes.set_xlabel("per-session Sharpe ratio")
        axes.set_xlim(min(sharpe.min(), 0.0) - 0.03, threshold + 0.035)
        axes.set_ylim(-0.75, len(breadths) - 0.35)
        axes.grid(axis="y", visible=False)
        axes.set_title(
            "Every defensible configuration, and the bar luck alone would clear",
            loc="left",
            color=COLOURS["ink"],
        )
        positive = int(np.sum(trials["net_return"].to_numpy(dtype=np.float64) > 0.0))
        above = int(np.sum(sharpe > threshold))
        caption(
            fig,
            f"{len(sharpe)} configurations. {positive} produced a positive net return and "
            f"{above} exceeded the no-skill threshold. The grid maximum is "
            f"{sharpe.max():.4f} against a threshold of {threshold:.4f}.",
        )
        png, pdf = save_figure(fig, directory, "fig08_configuration_search")
    return {
        "figure": "fig08_configuration_search",
        "png": png.name,
        "pdf": pdf.name,
        "trial_count": int(len(sharpe)),
        "positive_net_return_count": positive,
        "above_threshold_count": above,
        "grid_maximum": float(sharpe.max()),
        "threshold": threshold,
    }


def plot_block_length_sensitivity(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Repeat the confidence interval across every candidate block length.

    A block length is an arbitrary choice. If the interval moved with it, the
    interval would be a statement about the choice rather than about the data.
    """
    blocks = artifacts.metrics["bootstrap_block_sensitivity"]
    selected = float(artifacts.metrics["bootstrap"]["selected_block_length"])
    lengths = sorted(int(key) for key in blocks)
    lower = [float(blocks[str(key)]["annualized_return"]["lower"]) * 100 for key in lengths]
    upper = [float(blocks[str(key)]["annualized_return"]["upper"]) * 100 for key in lengths]
    estimate = float(blocks[str(lengths[0])]["annualized_return"]["estimate"]) * 100

    with figure(7.2, 3.6) as fig:
        axes = fig.add_subplot(111)
        positions = np.arange(len(lengths), dtype=float)
        axes.vlines(
            positions,
            lower,
            upper,
            color=COLOURS["portfolio"],
            linewidth=6.0,
            alpha=0.35,
            zorder=3,
        )
        axes.scatter(positions, lower, s=22, color=COLOURS["portfolio"], zorder=4)
        axes.scatter(positions, upper, s=22, color=COLOURS["portfolio"], zorder=4)
        axes.axhline(
            estimate,
            color=COLOURS["adverse"],
            linewidth=1.4,
            zorder=5,
            label=f"point estimate {estimate:+.2f}%",
        )
        axes.axhline(0.0, color=COLOURS["ink"], linewidth=1.0, zorder=5)
        marker = int(np.argmin([abs(value - selected) for value in lengths]))
        axes.scatter(
            [positions[marker]],
            [estimate],
            s=140,
            facecolor="none",
            edgecolor=COLOURS["ink"],
            linewidth=1.5,
            zorder=6,
        )
        axes.set_xticks(positions, [str(value) for value in lengths])
        axes.set_xlabel("mean bootstrap block length (sessions)")
        axes.set_ylabel("annualised return (%)")
        axes.set_title(
            "The interval spans zero at every block length", loc="left", color=COLOURS["ink"]
        )
        axes.legend(loc="upper right")
        spanning = sum(1 for low, high in zip(lower, upper, strict=True) if low <= 0.0 <= high)
        width = [high - low for low, high in zip(lower, upper, strict=True)]
        caption(
            fig,
            f"{spanning} of {len(lengths)} intervals contain zero. Interval width varies from "
            f"{min(width):.1f} to {max(width):.1f} percentage points, so the conclusion does not "
            f"depend on the block length; the circled marker is the Politis-White selection "
            f"({selected:.2f}).",
        )
        png, pdf = save_figure(fig, directory, "fig09_block_length_sensitivity")
    return {
        "figure": "fig09_block_length_sensitivity",
        "png": png.name,
        "pdf": pdf.name,
        "intervals_spanning_zero": spanning,
        "interval_count": len(lengths),
        "minimum_width": min(width),
        "maximum_width": max(width),
        "selected_block_length": selected,
    }
