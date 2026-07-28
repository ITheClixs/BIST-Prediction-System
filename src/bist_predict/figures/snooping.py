"""Figures for the joint data-snooping tests across the model family."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure
from bist_predict.research.inference.snooping import stationary_bootstrap_indices

__all__ = ["plot_reality_check"]


def plot_reality_check(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Place the observed best model inside its own null distribution.

    The histogram is the recentred stationary-bootstrap distribution of the
    maximum outperformance across the whole family, regenerated here with the
    seed and block length the run recorded. The observed maximum is the single
    number a naive "our best model beat the benchmark" claim would quote.
    """
    snooping = artifacts.metrics["inference"]["data_snooping"]
    panel = artifacts.squared_error_panel()
    benchmark = str(snooping["benchmark"])
    candidates = [str(name) for name in snooping["candidates"]]
    relative = panel[[benchmark]].to_numpy(dtype=np.float64) - panel[candidates].to_numpy(
        dtype=np.float64
    )
    observed_means = relative.mean(axis=0)
    count = relative.shape[0]

    rng = np.random.default_rng(int(snooping["seed"]))
    indices = stationary_bootstrap_indices(
        count,
        block_length=float(snooping["block_length"]),
        replications=int(snooping["replications"]),
        rng=rng,
    )
    resampled = relative[indices].mean(axis=1)
    root = np.sqrt(count)
    draws = np.max(root * (resampled - observed_means), axis=1)
    observed = float(np.max(root * observed_means))
    reported = float(snooping["reality_check"]["p_value"])
    recomputed = float(np.mean(draws >= observed))

    with figure(7.2, 3.6) as fig:
        axes = fig.add_subplot(111)
        axes.hist(
            draws,
            bins=60,
            color=COLOURS["band"],
            edgecolor=COLOURS["surface"],
            linewidth=0.4,
            zorder=3,
        )
        axes.axvline(
            observed,
            color=COLOURS["adverse"],
            linewidth=2.0,
            zorder=4,
            label="observed best model",
        )
        critical = float(np.percentile(draws, 95.0))
        axes.axvline(
            critical,
            color=COLOURS["ink"],
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=4,
            label="95th percentile of the null",
        )
        axes.set_xlabel(r"$\sqrt{n}\,\times$ mean outperformance over the null")
        axes.set_ylabel("bootstrap replications")
        axes.set_title(
            "The best of six models, placed in its own null distribution",
            loc="left",
            color=COLOURS["ink"],
        )
        axes.legend(loc="upper left")
        share_right = float(np.mean(draws >= observed))
        caption(
            fig,
            f"{share_right * 100:.1f}% of {len(draws):,} recentred replications land at or above "
            f"the observed maximum, so the Reality Check p-value is {recomputed:.4f}. The null "
            f"distribution sits above zero because it is the distribution of a maximum over six "
            f"candidates: that offset is the data-snooping correction made visible. The observed "
            f"maximum is itself negative, so the best candidate loses to the null before any "
            f"correction is applied.",
        )
        png, pdf = save_figure(fig, directory, "fig05_reality_check")
    return {
        "figure": "fig05_reality_check",
        "png": png.name,
        "pdf": pdf.name,
        "observed_maximum": observed,
        "critical_value": critical,
        "reported_p_value": reported,
        "recomputed_p_value": recomputed,
        "observed_maximum_is_negative": observed < 0.0,
    }
