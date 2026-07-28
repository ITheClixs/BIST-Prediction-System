"""Figures describing the evaluation design itself."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure

__all__ = ["plot_fold_geometry", "plot_effective_sample_size"]


def plot_fold_geometry(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Draw the actual walk-forward partition, read from ``folds.json``.

    Nothing here is illustrative. Each bar spans the dates the fold really
    used, so the purge, the embargo and the expanding training window are
    visible as executed rather than as described.
    """
    folds = artifacts.folds
    sessions = artifacts.trading_dates
    position = {value: index for index, value in enumerate(sessions)}

    with figure(7.2, 4.2) as fig:
        axes = fig.add_subplot(111)
        for row, fold in enumerate(folds):
            for label, key, colour in (
                ("train", "train_dates", COLOURS["model"]),
                ("embargo", "embargo_dates", COLOURS["adverse"]),
                ("validation", "validation_dates", COLOURS["null"]),
            ):
                dates = [value for value in fold[key] if value in position]
                if not dates:
                    continue
                start = position[min(dates)]
                width = position[max(dates)] - start + 1
                axes.barh(
                    row,
                    width,
                    left=start,
                    height=0.62,
                    color=colour,
                    edgecolor=COLOURS["surface"],
                    linewidth=1.0,
                    label=label if row == 0 else None,
                    zorder=3,
                )
        purged = [len(fold["train_dates"]) for fold in folds]
        axes.set_yticks(range(len(folds)))
        axes.set_yticklabels([f"fold {index + 1}" for index in range(len(folds))])
        axes.invert_yaxis()
        axes.set_xlabel("trading session index")
        axes.set_xlim(0, len(sessions))
        axes.grid(axis="y", visible=False)
        axes.set_title(
            "Date-grouped expanding-window partition, as executed",
            loc="left",
            color=COLOURS["ink"],
        )
        axes.legend(loc="upper right", ncol=1)
        gap = min(
            position[min(fold["validation_dates"])] - position[max(fold["train_dates"])] - 1
            for fold in folds
        )
        caption(
            fig,
            f"{len(folds)} folds over {len(sessions)} sessions. Training grows from "
            f"{min(purged)} to {max(purged)} dates; the smallest gap between the last "
            f"training date and the first validation date is {gap} session"
            f"{'s' if gap != 1 else ''}.",
        )
        png, pdf = save_figure(fig, directory, "fig01_fold_geometry")
    return {
        "figure": "fig01_fold_geometry",
        "png": png.name,
        "pdf": pdf.name,
        "fold_count": len(folds),
        "session_count": len(sessions),
        "minimum_gap_sessions": gap,
        "smallest_training_window": min(purged),
        "largest_training_window": max(purged),
    }


def plot_effective_sample_size(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Show why 480 rows are not 480 observations.

    The left panel is the realised correlation matrix of the executable target
    across tickers; the right panel converts the average off-diagonal entry
    into the sample size the panel actually carries.
    """
    dependence = artifacts.metrics["inference"]["cross_sectional_dependence"]
    wide = artifacts.target_panel()
    tickers = list(wide.columns)
    correlation = wide.corr().to_numpy()

    with figure(7.2, 3.3) as fig:
        left, right = fig.subplots(1, 2, width_ratios=(1.0, 1.15))

        image = left.imshow(correlation, cmap="Blues", vmin=0.0, vmax=1.0)
        left.set_xticks(range(len(tickers)), tickers, rotation=45, ha="right")
        left.set_yticks(range(len(tickers)), tickers)
        left.grid(visible=False)
        for row in range(len(tickers)):
            for column in range(len(tickers)):
                value = correlation[row, column]
                left.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=COLOURS["surface"] if value > 0.6 else COLOURS["ink"],
                )
        left.set_title("Target correlation across tickers", loc="left", color=COLOURS["ink"])
        fig.colorbar(image, ax=left, fraction=0.046, pad=0.04)

        rows = float(dependence["row_count"])
        effective = float(dependence["effective_row_count"])
        right.barh(
            [1, 0],
            [rows, effective],
            height=0.5,
            color=[COLOURS["neutral"], COLOURS["portfolio"]],
            zorder=3,
        )
        right.set_yticks([1, 0], ["panel rows", "effective\nobservations"])
        right.set_xlabel("count")
        right.grid(axis="y", visible=False)
        for position_index, value in ((1, rows), (0, effective)):
            right.text(
                value + rows * 0.02,
                position_index,
                f"{value:.0f}",
                va="center",
                fontsize=9,
                color=COLOURS["ink"],
            )
        right.set_xlim(0, rows * 1.18)
        right.set_title("Independent information in the panel", loc="left", color=COLOURS["ink"])
        caption(
            fig,
            f"Mean pairwise correlation {float(dependence['mean_pairwise_correlation']):.3f} "
            f"inflates the variance of any cross-sectional mean by "
            f"{float(dependence['variance_inflation_factor']):.2f}, so standard errors "
            f"computed on {rows:.0f} rows are too small by a factor of "
            f"{np.sqrt(rows / effective):.2f}.",
        )
        png, pdf = save_figure(fig, directory, "fig02_effective_sample_size")
    return {
        "figure": "fig02_effective_sample_size",
        "png": png.name,
        "pdf": pdf.name,
        "mean_pairwise_correlation": float(dependence["mean_pairwise_correlation"]),
        "row_count": int(rows),
        "effective_row_count": float(effective),
        "standard_error_factor": float(np.sqrt(rows / effective)),
        "minimum_pairwise_correlation": float(
            pd.DataFrame(correlation).where(~np.eye(len(tickers), dtype=bool)).min().min()
        ),
    }
