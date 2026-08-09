"""Monte Carlo family-wise error for the multiple-comparison layer.

A study that fits several models and searches several configurations makes two
multiplicity corrections: Holm across the models, and a Reality Check, an SPA
test or a deflated Sharpe ratio across the configurations.  Each is justified
by an asymptotic argument that assumes something about the dependence between
the things being compared.  On a correlated panel searched over overlapping
folds, none of those assumptions is obviously satisfied, and the failure mode
is silent---the correction still returns a number.

This module runs each correction on data where the null is true by
construction, so that the number it returns can be compared against the number
it promises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bist_predict.research.inference.detectability import false_strategy_quantile
from bist_predict.research.inference.forecast_tests import diebold_mariano
from bist_predict.research.inference.multiplicity import holm_step_down
from bist_predict.research.inference.snooping import (
    reality_check_and_spa,
    stationary_bootstrap_indices,
)
from bist_predict.research.simulation.calibration import (
    RejectionRate,
    loss_differentials,
    rejection_rate,
)
from bist_predict.research.simulation.panels import (
    PanelDesign,
    forecast_family,
    simulate_panel,
)

__all__ = [
    "FamilyWiseCell",
    "SearchThresholdCell",
    "family_wise_cell",
    "search_threshold_cell",
]


@dataclass(frozen=True)
class FamilyWiseCell:
    """False-positive rates of every multiplicity correction at one design."""

    design: PanelDesign
    family_size: int
    replications: int
    alpha: float
    uncorrected_any: RejectionRate
    holm_session: RejectionRate
    holm_row: RejectionRate
    reality_check: RejectionRate
    spa_untruncated: RejectionRate
    spa_hansen: RejectionRate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the cell."""
        return {
            "design": self.design.to_dict(),
            "family_size": self.family_size,
            "replications": self.replications,
            "alpha": self.alpha,
            "uncorrected_any": self.uncorrected_any.to_dict(),
            "holm_session": self.holm_session.to_dict(),
            "holm_row": self.holm_row.to_dict(),
            "reality_check": self.reality_check.to_dict(),
            "spa_untruncated": self.spa_untruncated.to_dict(),
            "spa_hansen": self.spa_hansen.to_dict(),
        }


def _superiority_p_values(differential: np.ndarray, *, aggregation: str) -> dict[str, float]:
    """Return one-sided p-values for candidate superiority, keyed by member.

    A two-sided p-value halved is the one-sided p-value only on the side the
    mean falls; on the other side it is one minus that.  Family-wise control
    over claims of *superiority* needs the directional quantity, so it is formed
    explicitly here rather than by halving.
    """
    p_values: dict[str, float] = {}
    for member in range(differential.shape[0]):
        series = (
            differential[member].mean(axis=1)
            if aggregation == "session"
            else differential[member].reshape(-1)
        )
        result = diebold_mariano(
            series,
            candidate=f"member_{member}",
            benchmark="zero_return",
            aggregation=aggregation,
        )
        directional = (
            result.p_value / 2.0 if result.mean_differential < 0.0 else 1.0 - result.p_value / 2.0
        )
        p_values[f"member_{member}"] = float(min(1.0, max(0.0, directional)))
    return p_values


def family_wise_cell(
    design: PanelDesign,
    *,
    family_size: int,
    replications: int,
    seed: int,
    alpha: float = 0.05,
    bootstrap_replications: int = 499,
) -> FamilyWiseCell:
    """Measure the false-positive rate of every correction at one design.

    All ``family_size`` forecasts are drawn at a population out-of-sample
    :math:`R^2` of exactly zero, so any claim that one of them beats the zero
    benchmark is a false positive.  The members share the panel's predictable
    component and differ only in their noise draw, which reproduces the
    correlation structure of a real model family rather than an independent one.
    """
    if family_size < 2:
        raise ValueError("a family-wise experiment needs at least two members")
    if replications < 2:
        raise ValueError("replications must be at least two")
    generator = np.random.default_rng(seed)
    counts = dict.fromkeys(
        ("uncorrected", "holm_session", "holm_row", "reality_check", "spa", "hansen"), 0
    )
    for _ in range(replications):
        panel = simulate_panel(design, generator)
        forecasts = forecast_family(
            panel, population_r_squared=0.0, family_size=family_size, rng=generator
        )
        differential = np.stack(
            [loss_differentials(panel.returns, forecast) for forecast in forecasts]
        )
        session_p = _superiority_p_values(differential, aggregation="session")
        row_p = _superiority_p_values(differential, aggregation="row")
        counts["uncorrected"] += int(min(session_p.values()) <= alpha)
        counts["holm_session"] += int(bool(holm_step_down(session_p, alpha=alpha).rejected))
        counts["holm_row"] += int(bool(holm_step_down(row_p, alpha=alpha).rejected))

        benchmark_loss = np.square(panel.returns).mean(axis=1)
        panel_frame = pd.DataFrame(
            {"zero_return": benchmark_loss}
            | {
                f"member_{member}": np.square(panel.returns - forecasts[member]).mean(axis=1)
                for member in range(family_size)
            }
        )
        snooping = reality_check_and_spa(
            panel_frame,
            benchmark="zero_return",
            replications=bootstrap_replications,
            seed=int(generator.integers(0, 2**31 - 1)),
        )
        counts["reality_check"] += int(snooping.reality_check_p_value <= alpha)
        counts["spa"] += int(snooping.spa_p_value_consistent <= alpha)
        counts["hansen"] += int(snooping.hansen_p_value_consistent <= alpha)
    return FamilyWiseCell(
        design=design,
        family_size=family_size,
        replications=replications,
        alpha=alpha,
        uncorrected_any=rejection_rate(counts["uncorrected"], replications),
        holm_session=rejection_rate(counts["holm_session"], replications),
        holm_row=rejection_rate(counts["holm_row"], replications),
        reality_check=rejection_rate(counts["reality_check"], replications),
        spa_untruncated=rejection_rate(counts["spa"], replications),
        spa_hansen=rejection_rate(counts["hansen"], replications),
    )


@dataclass(frozen=True)
class SearchThresholdCell:
    """How often each search threshold is cleared by a skill-free grid."""

    trial_count: int
    session_count: int
    trial_correlation: float
    replications: int
    alpha: float
    false_strategy_expectation: RejectionRate
    joint_bootstrap_quantile: RejectionRate
    mean_independent_equivalent_trials: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the cell."""
        return {
            "trial_count": self.trial_count,
            "session_count": self.session_count,
            "trial_correlation": self.trial_correlation,
            "replications": self.replications,
            "alpha": self.alpha,
            "false_strategy_expectation": self.false_strategy_expectation.to_dict(),
            "joint_bootstrap_quantile": self.joint_bootstrap_quantile.to_dict(),
            "mean_independent_equivalent_trials": self.mean_independent_equivalent_trials,
        }


def _grid_sharpe(values: np.ndarray) -> np.ndarray:
    """Return per-trial per-period Sharpe ratios of a sessions-by-trials block."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    return np.where(scale > 0.0, mean / np.where(scale > 0.0, scale, 1.0), 0.0)


def search_threshold_cell(
    *,
    trial_count: int,
    session_count: int,
    trial_correlation: float,
    replications: int,
    seed: int,
    alpha: float = 0.05,
    bootstrap_replications: int = 999,
) -> SearchThresholdCell:
    r"""Compare two search thresholds on grids that contain no skill at all.

    A grid of ``trial_count`` configurations is simulated with equicorrelated
    zero-mean session returns.  Two thresholds are then applied to the best
    Sharpe ratio in the grid: the False Strategy Theorem's expected maximum
    :math:`\sqrt V q(N)`, computed from the *realised* cross-trial dispersion as
    the accepted run computes it, and the upper ``1 - alpha`` quantile of the
    maximum under a synchronised stationary-bootstrap resample of the same grid.

    The first is an expectation, so a skill-free grid is expected to clear it
    about half the time; the second is a critical value and should be cleared
    with probability ``alpha``.  Reporting them side by side is the point: it
    shows in numbers what the deflated Sharpe ratio does and does not promise.
    """
    if trial_count < 2:
        raise ValueError("a search experiment needs at least two trials")
    if session_count < 8:
        raise ValueError("a search experiment needs at least eight sessions")
    if not 0.0 <= trial_correlation < 1.0:
        raise ValueError("trial_correlation must lie in [0, 1)")
    generator = np.random.default_rng(seed)
    expectation_hits = 0
    quantile_hits = 0
    equivalents: list[float] = []
    for _ in range(replications):
        common = generator.standard_normal((session_count, 1))
        idiosyncratic = generator.standard_normal((session_count, trial_count))
        values = (
            np.sqrt(trial_correlation) * common + np.sqrt(1.0 - trial_correlation) * idiosyncratic
        )
        sharpe = _grid_sharpe(values)
        best = float(sharpe.max())
        threshold = float(
            np.sqrt(float(np.var(sharpe, ddof=1))) * false_strategy_quantile(float(trial_count))
        )
        expectation_hits += int(best > threshold)

        centred = values - values.mean(axis=0)
        indices = stationary_bootstrap_indices(
            session_count,
            block_length=1.0,
            replications=bootstrap_replications,
            rng=generator,
        )
        resampled = centred[indices]
        block_mean = resampled.mean(axis=1)
        block_scale = resampled.std(axis=1, ddof=1)
        block_sharpe = np.where(block_scale > 0.0, block_mean / block_scale, 0.0)
        maxima = block_sharpe.max(axis=1)
        quantile_hits += int(best > float(np.quantile(maxima, 1.0 - alpha)))
        # The sampling dispersion of one trial across resamples, not the spread
        # across trials within a resample; see ``joint_search`` for why the
        # second reads a highly dependent grid as a highly independent one.
        dispersion = float(np.mean(block_sharpe.std(axis=0, ddof=1)))
        if dispersion > 0.0:
            ratio = float(maxima.mean()) / dispersion
            equivalents.append(_invert_false_strategy(ratio, trial_count))
    return SearchThresholdCell(
        trial_count=trial_count,
        session_count=session_count,
        trial_correlation=trial_correlation,
        replications=replications,
        alpha=alpha,
        false_strategy_expectation=rejection_rate(expectation_hits, replications),
        joint_bootstrap_quantile=rejection_rate(quantile_hits, replications),
        mean_independent_equivalent_trials=(
            float(np.mean(equivalents)) if equivalents else float("nan")
        ),
    )


def _invert_false_strategy(ratio: float, trial_count: int) -> float:
    """Return the independent trial count whose expected maximum matches ``ratio``."""
    low, high = 1.0 + 1e-6, float(max(trial_count, 2))
    if false_strategy_quantile(low) >= ratio:
        return 1.0
    while false_strategy_quantile(high) < ratio and high < 1e6:
        high *= 2.0
    for _ in range(120):
        middle = np.sqrt(low * high)
        if false_strategy_quantile(middle) < ratio:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))
