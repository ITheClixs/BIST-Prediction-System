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
from bist_predict.research.inference.multiplicity import holm_step_down
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
    session_aggregated = cast(dict[str, dict[str, float]], accuracy["session_aggregated"])
    session_errors = {
        name: float(test["standard_error"]) for name, test in session_aggregated.items()
    }
    portfolio_rows = predictions.loc[predictions["model_name"] == portfolio_model]
    realised_ic = float(portfolio_rows["predicted_return"].corr(portfolio_rows["target"]))
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
        realised_information_coefficient=realised_ic,
        universe_size=universe_size,
        selected=selected,
    )
    return {
        "schema_version": 1,
        "alpha": alpha,
        "models": models,
        "benchmark_model": benchmark_model,
        "portfolio_model": portfolio_model,
        "cross_sectional_dependence": dependence.to_dict(),
        "equal_predictive_accuracy": accuracy,
        "data_snooping": snooping.to_dict(),
        "portfolio_sharpe": sharpe.to_dict(),
        "detectability": detectability.to_dict(),
    }
