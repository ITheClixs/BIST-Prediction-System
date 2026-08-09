"""Assemble every inferential claim the accepted run is allowed to make.

The run already saves point estimates. This module turns them into statements
with a stated null, a stated sign convention, an error bar, and a correction for
the number of models and configurations that were examined.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from bist_predict.research.inference.dependence import cross_sectional_dependence
from bist_predict.research.inference.detectability import detectability_report
from bist_predict.research.inference.forecast_tests import (
    diebold_mariano,
    squared_error_differential,
)
from bist_predict.research.inference.joint_search import joint_search_test
from bist_predict.research.inference.multiplicity import holm_step_down
from bist_predict.research.inference.nested import clark_west, encompassing_adjustment
from bist_predict.research.inference.sharpe import sharpe_inference
from bist_predict.research.inference.snooping import reality_check_and_spa

__all__ = ["build_inference_report", "session_squared_error_panel"]


def session_squared_error_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-session mean squared error with one column per model."""
    required = {"date", "ticker", "model_name", "target", "predicted_return"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {', '.join(missing)}")
    working = predictions.copy()
    working["squared_error"] = np.square(working["target"] - working["predicted_return"])
    panel = working.pivot_table(
        index="date", columns="model_name", values="squared_error", aggfunc="mean"
    )
    if panel.isna().any().any():
        raise ValueError("every model must be evaluated on every session")
    panel.columns = [str(name) for name in panel.columns]
    return panel.sort_index()


def _session_information_coefficient(frame: pd.DataFrame) -> tuple[float, float]:
    """Return the mean per-session cross-sectional IC and its standard error.

    A correlation pooled over stock-sessions also absorbs the common time-series
    component, so a forecast that only tracks the market direction earns one while
    ranking nothing. Selection within a session is a cross-sectional operation, so
    the quantity a top-k feasibility bound needs is the per-session correlation,
    averaged over sessions with its own dispersion as the error bar.
    """
    per_session = (
        frame.groupby("date")
        .apply(
            lambda group: group["predicted_return"].corr(group["target"]),
            include_groups=False,
        )
        .dropna()
    )
    if per_session.empty:
        raise ValueError("no session admits a cross-sectional correlation")
    mean = float(per_session.mean())
    if len(per_session) < 2:
        return mean, float("nan")
    return mean, float(per_session.std(ddof=1) / np.sqrt(len(per_session)))


def _loss_differential_correlation(
    predictions: pd.DataFrame, *, candidate: str, benchmark: str
) -> float:
    """Return the mean within-session correlation of the loss differential.

    ``cross_sectional_dependence`` measures the dependence of the target, which is
    what the panel diagnostic is about. The standard error of a Diebold-Mariano
    test is a function of the dependence of ``d_it`` instead, and the two are not
    equal, so the quantity that governs the test is computed here directly.
    """
    differential = squared_error_differential(
        predictions, candidate=candidate, benchmark=benchmark, aggregation="row"
    )
    # The row-level differential is indexed by the "<date>|<ticker>" sample id the
    # prediction artifact threads through the pipeline.
    keys = differential.index.to_series().str.split("|", n=1, expand=True)
    frame = pd.DataFrame(
        {
            "date": keys[0].to_numpy(),
            "ticker": keys[1].to_numpy(),
            "value": differential.to_numpy(dtype=float),
        }
    )
    return float(cross_sectional_dependence(frame, value_column="value").mean_pairwise_correlation)


def _equal_predictive_accuracy(
    predictions: pd.DataFrame,
    *,
    benchmark: str,
    candidates: Sequence[str],
    alpha: float,
) -> dict[str, object]:
    session_tests: dict[str, dict[str, float | int | str]] = {}
    row_tests: dict[str, dict[str, float | int | str]] = {}
    for candidate in candidates:
        session = diebold_mariano(
            squared_error_differential(predictions, candidate=candidate, benchmark=benchmark),
            candidate=candidate,
            benchmark=benchmark,
        )
        row = diebold_mariano(
            squared_error_differential(
                predictions, candidate=candidate, benchmark=benchmark, aggregation="row"
            ),
            candidate=candidate,
            benchmark=benchmark,
            aggregation="row",
        )
        session_tests[candidate] = session.to_dict()
        row_tests[candidate] = row.to_dict()
    holm = holm_step_down(
        {name: float(test["p_value"]) for name, test in session_tests.items()}, alpha=alpha
    )
    return {
        "benchmark": benchmark,
        "loss": "squared_error",
        "test": "diebold_mariano_harvey_leybourne_newbold",
        "session_aggregated": session_tests,
        "row_level_for_comparison": row_tests,
        "family_wise_correction": holm.to_dict(),
        "any_candidate_survives_correction": bool(holm.rejected),
    }


def _predictive_content(
    predictions: pd.DataFrame,
    *,
    benchmark: str,
    candidates: Sequence[str],
    alpha: float,
) -> dict[str, object]:
    """Return the Clark--West tests that the squared-error comparison cannot make.

    Against a zero-forecast benchmark the two models are nested, and a fitted
    forecast loses the squared-error comparison whenever its estimation variance
    exceeds twice its covariance with the target.  That happens under the null
    itself, so a Diebold--Mariano rejection towards the benchmark carries no
    information about predictive content.  Clark and West (2007) remove the
    offending term; the accompanying simulation study reports the size of both.
    """
    tests: dict[str, dict[str, float | int | str]] = {}
    for candidate in candidates:
        test = clark_west(
            encompassing_adjustment(predictions, candidate=candidate, benchmark=benchmark),
            candidate=candidate,
            benchmark=benchmark,
        )
        tests[candidate] = test.to_dict()
    holm = holm_step_down(
        {name: float(test["p_value"]) for name, test in tests.items()}, alpha=alpha
    )
    return {
        "benchmark": benchmark,
        "test": "clark_west_encompassing",
        "null": "the population forecast is the zero forecast",
        "session_aggregated": tests,
        "family_wise_correction": holm.to_dict(),
        "any_candidate_shows_predictive_content": bool(holm.rejected),
    }


def build_inference_report(
    predictions: pd.DataFrame,
    *,
    net_returns: Sequence[float],
    benchmark_model: str,
    portfolio_model: str,
    periods_per_year: int,
    trial_count: int,
    trial_sharpe_variance: float,
    grid_maximum_sharpe: float,
    round_trip_cost_rate: float,
    universe_size: int,
    selected: int,
    seed: int,
    replications: int = 10_000,
    alpha: float = 0.05,
    trial_session_returns: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Return the complete inferential block persisted into ``metrics.json``."""
    models = sorted(str(name) for name in predictions["model_name"].unique())
    if benchmark_model not in models:
        raise ValueError(f"benchmark model is absent from the predictions: {benchmark_model}")
    if portfolio_model not in models:
        raise ValueError(f"portfolio model is absent from the predictions: {portfolio_model}")
    candidates = [name for name in models if name != benchmark_model]
    if not candidates:
        raise ValueError("inference requires at least one candidate model")

    target_frame = (
        predictions.loc[predictions["model_name"] == benchmark_model, ["date", "ticker", "target"]]
        .drop_duplicates(["date", "ticker"])
        .reset_index(drop=True)
    )
    dependence = cross_sectional_dependence(target_frame, value_column="target")
    loss_panel = session_squared_error_panel(predictions)
    snooping = reality_check_and_spa(
        loss_panel,
        benchmark=benchmark_model,
        replications=replications,
        seed=seed,
    )
    sharpe = sharpe_inference(
        net_returns,
        periods_per_year=periods_per_year,
        trial_count=trial_count,
        trial_sharpe_variance=trial_sharpe_variance,
    )
    accuracy = _equal_predictive_accuracy(
        predictions, benchmark=benchmark_model, candidates=candidates, alpha=alpha
    )
    predictive_content = _predictive_content(
        predictions, benchmark=benchmark_model, candidates=candidates, alpha=alpha
    )
    session_aggregated = cast(dict[str, dict[str, float]], accuracy["session_aggregated"])
    session_errors = {
        name: float(test["standard_error"]) for name, test in session_aggregated.items()
    }
    portfolio_rows = predictions.loc[predictions["model_name"] == portfolio_model]
    pooled_ic = float(portfolio_rows["predicted_return"].corr(portfolio_rows["target"]))
    session_ic, session_ic_error = _session_information_coefficient(portfolio_rows)
    reference_candidate = min(session_errors, key=lambda name: session_errors[name])
    detectability = detectability_report(
        session_standard_errors=session_errors,
        benchmark_mean_squared_error=float(loss_panel[benchmark_model].mean()),
        session_count=int(loss_panel.shape[0]),
        dependence=dependence.to_dict(),
        sharpe=sharpe.to_dict(),
        grid_maximum_sharpe=grid_maximum_sharpe,
        periods_per_year=periods_per_year,
        round_trip_cost_rate=round_trip_cost_rate,
        target_volatility=float(target_frame["target"].std(ddof=1)),
        pooled_information_coefficient=pooled_ic,
        session_information_coefficient=session_ic,
        session_information_coefficient_standard_error=session_ic_error,
        loss_differential_correlation=_loss_differential_correlation(
            predictions, candidate=reference_candidate, benchmark=benchmark_model
        ),
        universe_size=universe_size,
        selected=selected,
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "alpha": alpha,
        "models": models,
        "benchmark_model": benchmark_model,
        "portfolio_model": portfolio_model,
        "cross_sectional_dependence": dependence.to_dict(),
        "equal_predictive_accuracy": accuracy,
        "predictive_content": predictive_content,
        "data_snooping": snooping.to_dict(),
        "portfolio_sharpe": sharpe.to_dict(),
        "detectability": detectability.to_dict(),
    }
    if trial_session_returns is not None:
        # The deflated Sharpe ratio corrects the grid maximum by an assumption
        # about how the trials disperse. Resampling the whole grid on one index
        # draw measures that dispersion instead of assuming it.
        report["joint_search"] = joint_search_test(
            trial_session_returns, replications=replications, seed=seed
        ).to_dict()
    return report
