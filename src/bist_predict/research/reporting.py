"""Ledger-derived portfolio metrics and stratified OOS diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bist_predict.research.portfolio_backtest import PortfolioBacktestResult
from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.predictions import PREDICTION_COLUMNS, validate_predictions


def _annualized_return(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    total = float(np.prod(1.0 + returns))
    if total <= 0.0:
        return -1.0
    return total ** (252.0 / len(returns)) - 1.0


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    volatility = float(np.std(returns, ddof=1))
    return (
        float(np.mean(returns)) / volatility * math.sqrt(252.0)
        if volatility > 0.0
        else 0.0
    )


def compute_portfolio_metrics(
    result: PortfolioBacktestResult,
    *,
    benchmark_returns: Sequence[float] | None = None,
) -> dict[str, object]:
    """Compute portfolio metrics entirely from the persisted event ledgers."""
    snapshots = result.daily_snapshots
    net_returns = np.asarray([item.net_return for item in snapshots], dtype=np.float64)
    gross_returns = np.asarray(
        [item.gross_return for item in snapshots], dtype=np.float64
    )
    annualized = _annualized_return(net_returns)
    annualized_volatility = (
        float(np.std(net_returns, ddof=1) * math.sqrt(252.0))
        if len(net_returns) > 1
        else 0.0
    )
    downside = net_returns[net_returns < 0.0]
    downside_deviation = (
        float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    )
    sortino = (
        float(np.mean(net_returns)) / downside_deviation * math.sqrt(252.0)
        if downside_deviation > 0.0
        else 0.0
    )
    equity = np.asarray(
        [result.portfolio.starting_equity]
        + [snapshot.ending_equity for snapshot in snapshots],
        dtype=np.float64,
    )
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    maximum_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0
    exits = [position for position in result.positions if position.quantity == 0]
    winning_exits = [position for position in exits if position.realized_pnl > 0.0]
    cost_decomposition = {
        "commission": sum(item.commission for item in result.costs),
        "bid_ask_spread": sum(item.bid_ask_spread for item in result.costs),
        "slippage": sum(item.slippage for item in result.costs),
        "market_impact": sum(item.market_impact for item in result.costs),
        "taxes": sum(item.taxes for item in result.costs),
        "total": sum(item.total_cost for item in result.costs),
    }
    benchmark = (
        np.asarray(benchmark_returns, dtype=np.float64)
        if benchmark_returns is not None
        else np.zeros(len(net_returns), dtype=np.float64)
    )
    if len(benchmark) != len(net_returns):
        raise ValueError("benchmark return count must match portfolio sessions")
    benchmark_total = float(np.prod(1.0 + benchmark) - 1.0)
    active_returns = net_returns - benchmark
    active_volatility = (
        float(np.std(active_returns, ddof=1)) if len(active_returns) > 1 else 0.0
    )
    information_ratio = (
        float(np.mean(active_returns)) / active_volatility * math.sqrt(252.0)
        if active_volatility > 0.0
        else 0.0
    )
    net_total = (
        result.portfolio.ending_equity / result.portfolio.starting_equity - 1.0
    )
    return {
        "gross_return": float(np.prod(1.0 + gross_returns) - 1.0),
        "net_return": net_total,
        "annualized_return": annualized,
        "annualized_volatility": annualized_volatility,
        "sharpe": _sharpe(net_returns),
        "sortino": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar": annualized / abs(maximum_drawdown) if maximum_drawdown < 0.0 else 0.0,
        "turnover": sum(item.turnover for item in snapshots),
        "trade_count": len(exits),
        "hit_rate": len(winning_exits) / len(exits) if exits else 0.0,
        "average_holding_period_sessions": 1.0 if exits else 0.0,
        "gross_exposure": float(np.mean([item.gross_exposure for item in snapshots]))
        if snapshots
        else 0.0,
        "net_exposure": float(np.mean([item.net_exposure for item in snapshots]))
        if snapshots
        else 0.0,
        "concentration": float(np.mean([item.concentration for item in snapshots]))
        if snapshots
        else 0.0,
        "cost_decomposition": cost_decomposition,
        "benchmark_return": benchmark_total,
        "benchmark_relative_return": net_total - benchmark_total,
        "information_ratio": information_ratio,
    }


def block_bootstrap_intervals(
    returns: Sequence[float],
    *,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Circular block-bootstrap confidence intervals for principal results."""
    values = np.asarray(returns, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap returns must be finite and non-empty")
    if block_size <= 0 or block_size > len(values):
        raise ValueError("block_size must lie between one and the sample count")
    if iterations < 100:
        raise ValueError("bootstrap requires at least 100 iterations")
    rng = np.random.default_rng(seed)
    annualized_samples: list[float] = []
    sharpe_samples: list[float] = []
    for _ in range(iterations):
        sample_parts: list[np.ndarray] = []
        while sum(len(part) for part in sample_parts) < len(values):
            start = int(rng.integers(0, len(values)))
            indices = (np.arange(start, start + block_size) % len(values)).astype(int)
            sample_parts.append(values[indices])
        sample = np.concatenate(sample_parts)[: len(values)]
        annualized_samples.append(_annualized_return(sample))
        sharpe_samples.append(_sharpe(sample))

    def interval(samples: list[float], estimate: float) -> dict[str, float]:
        lower, upper = np.percentile(samples, [2.5, 97.5])
        return {
            "estimate": estimate,
            "lower": min(float(lower), estimate),
            "upper": max(float(upper), estimate),
            "confidence_level": 0.95,
        }

    return {
        "annualized_return": interval(
            annualized_samples, _annualized_return(values)
        ),
        "sharpe": interval(sharpe_samples, _sharpe(values)),
    }


def grouped_prediction_metrics(
    predictions: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> dict[str, dict[str, dict[str, dict[str, float | int | None]]]]:
    """Stratify OOS metrics by every declared research reporting dimension."""
    validate_predictions(predictions)
    required_metadata = {
        "date",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    }
    missing = sorted(required_metadata.difference(sample_metadata.columns))
    if missing:
        raise ValueError(f"sample metadata missing columns: {', '.join(missing)}")
    if sample_metadata.duplicated(["date", "ticker"]).any():
        raise ValueError("sample metadata must have one row per date and ticker")
    enriched = predictions.merge(
        sample_metadata[list(required_metadata)],
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    if enriched[["sector", "liquidity_bucket", "market_regime"]].isna().any().any():
        raise ValueError("sample metadata does not cover every prediction")
    enriched["year"] = pd.to_datetime(enriched["date"]).dt.year.astype(str)
    dimensions = {
        "fold": "fold_id",
        "year": "year",
        "ticker": "ticker",
        "sector": "sector",
        "liquidity_bucket": "liquidity_bucket",
        "market_regime": "market_regime",
    }
    result: dict[str, dict[str, dict[str, dict[str, float | int | None]]]] = {}
    for label, column in dimensions.items():
        result[label] = {
            str(value): recompute_prediction_metrics(group[list(PREDICTION_COLUMNS)])
            for value, group in enriched.groupby(column, sort=True)
        }
    return result

