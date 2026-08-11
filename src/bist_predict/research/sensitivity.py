"""Repeat the accepted experiment across every arbitrary configuration choice.

The accepted run fixes a fold geometry and a portfolio breadth. Neither was
derived from theory, so a single headline number drawn from one of them is one
draw and not an estimate. This module re-runs the whole evaluation over the
grid of defensible alternatives, which supplies both a dispersion measure and
the trial count that the deflated Sharpe ratio requires.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import pandas as pd

from bist_predict.ingest.calendar import OfficialTradingCalendar
from bist_predict.ingest.corporate_actions import CorporateAction
from bist_predict.ingest.types import OHLCVBar
from bist_predict.research.baselines import BaselineBenchmarkResult, run_baseline_benchmark
from bist_predict.research.inference.sharpe import (
    deflated_sharpe_threshold,
    per_period_sharpe_ratio,
)
from bist_predict.research.portfolio_backtest import CostModel, PortfolioBacktester, StrategyConfig
from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.reporting import compute_portfolio_metrics
from bist_predict.research.splits import ExpandingWindowSplitter
from bist_predict.features.manifest import FeatureManifest

__all__ = [
    "SensitivityGrid",
    "SensitivityTrial",
    "configuration_grid",
    "run_configuration_sensitivity",
    "summarise_sensitivity",
]


@dataclass(frozen=True)
class SensitivityTrial:
    """One complete re-run of the accepted evaluation under a varied setting."""

    min_train_dates: int
    validation_dates: int
    step_dates: int
    embargo_dates: int
    top_k: int
    fold_count: int
    evaluated_sample_count: int
    session_count: int
    gross_return: float
    net_return: float
    annualised_return: float
    per_period_sharpe: float
    annualised_sharpe: float
    maximum_drawdown: float
    turnover: float
    trade_count: int
    portfolio_model_zero_mean_r_squared: float
    best_model: str
    best_zero_mean_r_squared: float

    @property
    def trial_id(self) -> str:
        """Return a stable, human-readable identity for the trial."""
        return (
            f"train{self.min_train_dates}"
            f"_val{self.validation_dates}"
            f"_step{self.step_dates}"
            f"_emb{self.embargo_dates}"
            f"_k{self.top_k}"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the trial."""
        return {"trial_id": self.trial_id, **asdict(self)}


@dataclass(frozen=True)
class SensitivityGrid:
    """Every evaluated configuration, with the session returns each produced.

    The summary statistics alone cannot support a joint test across the grid:
    comparing the maximum of 72 numbers to a threshold requires knowing how the
    72 co-move, and that information lives in the session-by-session returns.
    They are carried here so the search correction can be computed from the
    joint distribution rather than assumed.
    """

    trials: tuple[SensitivityTrial, ...]
    session_returns: pd.DataFrame

    def __post_init__(self) -> None:
        expected = {"trial_id", "date", "net_return"}
        missing = sorted(expected.difference(self.session_returns.columns))
        if missing:
            raise ValueError(f"session_returns missing required columns: {', '.join(missing)}")

    def aligned_returns(self) -> pd.DataFrame:
        """Return a dates-by-trials matrix on the sessions every trial evaluated.

        Trials with different fold geometries start and stop on different dates.
        A joint resample has to draw one date index and apply it to all of them,
        so the intersection is the only admissible common support.
        """
        wide = self.session_returns.pivot(index="date", columns="trial_id", values="net_return")
        return wide.dropna(axis=0, how="any").sort_index()


def configuration_grid(
    *,
    min_train_dates: Sequence[int],
    validation_dates: Sequence[int],
    embargo_dates: Sequence[int],
    top_k: Sequence[int],
) -> tuple[dict[str, int], ...]:
    """Return the full cross product with ``step_dates`` tied to the fold width.

    Tying the step to the validation width keeps folds non-overlapping in every
    trial, so the grid varies the fold geometry without also varying how many
    times each session is reused.
    """
    for name, values in (
        ("min_train_dates", min_train_dates),
        ("validation_dates", validation_dates),
        ("embargo_dates", embargo_dates),
        ("top_k", top_k),
    ):
        if not values:
            raise ValueError(f"{name} must contain at least one value")
    return tuple(
        {
            "min_train_dates": int(train),
            "validation_dates": int(validation),
            "step_dates": int(validation),
            "embargo_dates": int(embargo),
            "top_k": int(breadth),
        }
        for train in sorted(set(min_train_dates))
        for validation in sorted(set(validation_dates))
        for embargo in sorted(set(embargo_dates))
        for breadth in sorted(set(top_k))
    )


def _trial(
    setting: dict[str, int],
    benchmark: BaselineBenchmarkResult,
    *,
    bars: tuple[OHLCVBar, ...],
    calendar: OfficialTradingCalendar,
    corporate_actions: tuple[CorporateAction, ...],
    strategy: StrategyConfig,
    costs: CostModel,
    portfolio_model: str,
    starting_equity: float,
) -> tuple[SensitivityTrial, pd.DataFrame] | None:
    result = PortfolioBacktester(
        strategy=StrategyConfig(
            top_k=setting["top_k"],
            decision_cost_rate=strategy.decision_cost_rate,
            max_participation=strategy.max_participation,
            liquidity_lookback_sessions=strategy.liquidity_lookback_sessions,
            min_trade_value=strategy.min_trade_value,
        ),
        costs=costs,
    ).run(
        benchmark.predictions,
        bars,
        model_name=portfolio_model,
        starting_equity=starting_equity,
        corporate_actions=corporate_actions,
        calendar=calendar,
    )
    if not result.daily_snapshots:
        return None
    metrics = compute_portfolio_metrics(result)
    net_returns = np.asarray(
        [snapshot.net_return for snapshot in result.daily_snapshots], dtype=np.float64
    )
    prediction_metrics = recompute_prediction_metrics(benchmark.predictions)
    scored = {
        name: float(values["zero_mean_r_squared"] or 0.0)
        for name, values in prediction_metrics.items()
        if values.get("zero_mean_r_squared") is not None
    }
    best_model = max(sorted(scored), key=lambda name: scored[name])
    trial = SensitivityTrial(
        min_train_dates=setting["min_train_dates"],
        validation_dates=setting["validation_dates"],
        step_dates=setting["step_dates"],
        embargo_dates=setting["embargo_dates"],
        top_k=setting["top_k"],
        fold_count=len(benchmark.folds),
        evaluated_sample_count=int(
            len(benchmark.predictions.loc[benchmark.predictions["model_name"] == portfolio_model])
        ),
        session_count=len(result.daily_snapshots),
        gross_return=float(cast(float, metrics["gross_return"])),
        net_return=float(cast(float, metrics["net_return"])),
        annualised_return=float(cast(float, metrics["annualized_return"])),
        per_period_sharpe=per_period_sharpe_ratio(net_returns) if net_returns.size >= 3 else 0.0,
        annualised_sharpe=float(cast(float, metrics["sharpe"])),
        maximum_drawdown=float(cast(float, metrics["maximum_drawdown"])),
        turnover=float(cast(float, metrics["turnover"])),
        trade_count=int(cast(float, metrics["trade_count"])),
        portfolio_model_zero_mean_r_squared=scored.get(portfolio_model, float("nan")),
        best_model=best_model,
        best_zero_mean_r_squared=scored[best_model],
    )
    returns = pd.DataFrame(
        {
            "trial_id": trial.trial_id,
            "date": [snapshot.date for snapshot in result.daily_snapshots],
            "net_return": net_returns,
        }
    )
    return trial, returns


def run_configuration_sensitivity(
    grid: Iterable[dict[str, int]],
    *,
    panel: pd.DataFrame,
    manifest: FeatureManifest,
    bars: tuple[OHLCVBar, ...],
    calendar: OfficialTradingCalendar,
    corporate_actions: tuple[CorporateAction, ...],
    strategy: StrategyConfig,
    costs: CostModel,
    portfolio_model: str,
    starting_equity: float,
) -> SensitivityGrid:
    """Evaluate every grid point and drop settings that produce no folds.

    Refitting the baselines dominates the cost and depends only on the fold
    geometry, so grid points that share a geometry reuse one fit and differ
    only in the portfolio simulation.
    """
    trials: list[SensitivityTrial] = []
    session_returns: list[pd.DataFrame] = []
    fitted: dict[tuple[int, int, int, int], BaselineBenchmarkResult | None] = {}
    for setting in grid:
        geometry = (
            setting["min_train_dates"],
            setting["validation_dates"],
            setting["step_dates"],
            setting["embargo_dates"],
        )
        if geometry not in fitted:
            splitter = ExpandingWindowSplitter(
                min_train_dates=geometry[0],
                validation_dates=geometry[1],
                step_dates=geometry[2],
                embargo_dates=geometry[3],
            )
            try:
                fitted[geometry] = run_baseline_benchmark(panel, manifest, splitter)
            except ValueError:
                fitted[geometry] = None
        benchmark = fitted[geometry]
        if benchmark is None:
            continue
        trial = _trial(
            setting,
            benchmark,
            bars=bars,
            calendar=calendar,
            corporate_actions=corporate_actions,
            strategy=strategy,
            costs=costs,
            portfolio_model=portfolio_model,
            starting_equity=starting_equity,
        )
        if trial is not None:
            trials.append(trial[0])
            session_returns.append(trial[1])
    if not trials:
        raise ValueError("no configuration in the grid produced an evaluable experiment")
    return SensitivityGrid(
        trials=tuple(trials),
        session_returns=pd.concat(session_returns, ignore_index=True),
    )


def summarise_sensitivity(
    trials: Sequence[SensitivityTrial],
    *,
    reported: SensitivityTrial | None = None,
) -> dict[str, object]:
    """Summarise the grid and expose the inputs the deflated Sharpe needs.

    ``trial_sharpe_variance`` is the variance of the *per-period* Sharpe ratios
    across the grid, which is the quantity Bailey and Lopez de Prado (2014)
    deflate against. Using annualised values here would inflate the threshold
    by the annualisation factor squared.

    ``expected_maximum_sharpe_under_no_skill`` applies their False Strategy
    Theorem to this grid: it is the Sharpe ratio the best of ``N`` genuinely
    skill-free configurations would be *expected* to show. It is an expectation
    and not a critical value, so a skill-free grid exceeds it about half the
    time; ``best_trial_exceeds_expected_maximum`` is therefore a diagnostic and
    not a test. The deflated Sharpe ratio is the test, because it converts the
    threshold into a probability using the winner's own higher moments.
    """
    if not trials:
        raise ValueError("sensitivity summary requires at least one trial")
    per_period = np.asarray([trial.per_period_sharpe for trial in trials], dtype=np.float64)
    net_returns = np.asarray([trial.net_return for trial in trials], dtype=np.float64)
    best = max(trials, key=lambda trial: (trial.per_period_sharpe, trial.trial_id))
    worst = min(trials, key=lambda trial: (trial.per_period_sharpe, trial.trial_id))
    variance = float(np.var(per_period, ddof=1)) if len(trials) > 1 else 0.0
    threshold = deflated_sharpe_threshold(len(trials), variance)
    return {
        "trial_count": len(trials),
        "trial_sharpe_variance": variance,
        "expected_maximum_sharpe_under_no_skill": threshold,
        "best_trial_exceeds_expected_maximum": bool(best.per_period_sharpe > threshold),
        "per_period_sharpe": {
            "minimum": float(per_period.min()),
            "median": float(np.median(per_period)),
            "maximum": float(per_period.max()),
        },
        "net_return": {
            "minimum": float(net_returns.min()),
            "median": float(np.median(net_returns)),
            "maximum": float(net_returns.max()),
            "share_positive": float(np.mean(net_returns > 0.0)),
        },
        "best_trial": best.to_dict(),
        "worst_trial": worst.to_dict(),
        "reported_trial": None if reported is None else reported.to_dict(),
        "reported_rank_by_sharpe": (
            None
            if reported is None
            else int(
                1
                + sum(1 for trial in trials if trial.per_period_sharpe > reported.per_period_sharpe)
            )
        ),
        "trials": [trial.to_dict() for trial in trials],
    }
