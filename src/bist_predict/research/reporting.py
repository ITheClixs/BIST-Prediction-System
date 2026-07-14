"""Ledger-derived portfolio metrics and stratified OOS diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from bist_predict.research.portfolio_backtest import PortfolioBacktestResult
from bist_predict.research.portfolio_backtest import prediction_identifier
from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.predictions import PREDICTION_COLUMNS, validate_predictions


_PORTFOLIO_ATTRIBUTION_METRICS = (
    "gross_pnl",
    "gross_return_contribution",
    "net_pnl",
    "net_return_contribution",
    "commission",
    "bid_ask_spread",
    "slippage",
    "market_impact",
    "taxes",
    "transaction_costs",
    "turnover",
    "trade_count",
    "average_gross_exposure_contribution",
    "average_net_exposure_contribution",
)


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
    return float(np.mean(returns)) / volatility * math.sqrt(252.0) if volatility > 0.0 else 0.0


def compute_portfolio_metrics(
    result: PortfolioBacktestResult,
    *,
    benchmark_returns: Sequence[float] | None = None,
) -> dict[str, object]:
    """Compute portfolio metrics entirely from the persisted event ledgers."""
    snapshots = result.daily_snapshots
    net_returns = np.asarray([item.net_return for item in snapshots], dtype=np.float64)
    gross_returns = np.asarray([item.gross_return for item in snapshots], dtype=np.float64)
    annualized = _annualized_return(net_returns)
    annualized_volatility = (
        float(np.std(net_returns, ddof=1) * math.sqrt(252.0)) if len(net_returns) > 1 else 0.0
    )
    downside = net_returns[net_returns < 0.0]
    downside_deviation = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float(np.mean(net_returns)) / downside_deviation * math.sqrt(252.0)
        if downside_deviation > 0.0
        else 0.0
    )
    equity = np.asarray(
        [result.portfolio.starting_equity] + [snapshot.ending_equity for snapshot in snapshots],
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
    active_volatility = float(np.std(active_returns, ddof=1)) if len(active_returns) > 1 else 0.0
    information_ratio = (
        float(np.mean(active_returns)) / active_volatility * math.sqrt(252.0)
        if active_volatility > 0.0
        else 0.0
    )
    net_total = result.portfolio.ending_equity / result.portfolio.starting_equity - 1.0
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
        "annualized_return": interval(annualized_samples, _annualized_return(values)),
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


def _assert_attribution_close(label: str, actual: float, expected: float) -> None:
    tolerance = 1e-9 * max(1.0, abs(actual), abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"portfolio attribution does not reconcile {label}: "
            f"actual={actual}, expected={expected}"
        )


def _summed_attribution(frame: pd.DataFrame) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        metric: float(frame[metric].sum()) for metric in _PORTFOLIO_ATTRIBUTION_METRICS
    }
    values["trade_count"] = int(values["trade_count"])
    return values


def _sector_status(metadata: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    if "sector" not in metadata.columns:
        return (
            pd.Series("unavailable", index=metadata.index, dtype="object"),
            {
                "status": "unavailable",
                "reason": "sample metadata has no sourced sector field",
            },
        )
    sectors = metadata["sector"].astype("string").str.strip()
    unavailable = sectors.isna() | sectors.str.lower().isin(
        {"", "unavailable", "unclassified", "unknown", "none"}
    )
    normalized = sectors.mask(unavailable, "unavailable").astype(str)
    if unavailable.all():
        status = {
            "status": "unavailable",
            "reason": "sample metadata has no sourced sector values",
        }
    elif unavailable.any():
        status = {
            "status": "partial",
            "reason": "some executed samples have no sourced sector value",
        }
    else:
        status = {"status": "available", "reason": "sourced sector values provided"}
    return normalized, status


def grouped_portfolio_metrics(
    result: PortfolioBacktestResult,
    predictions: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> dict[str, object]:
    """Attribute executed portfolio outcomes once across declared dimensions.

    PnL return contributions use initial capital as a common denominator, so
    groups sum exactly to the portfolio result. Turnover uses each execution
    session's starting equity. Exposure contributions are averaged over every
    portfolio session, including sessions in which a group had no position.
    """
    validate_predictions(predictions)
    required_metadata = {"date", "ticker", "liquidity_bucket", "market_regime"}
    missing_metadata = sorted(required_metadata.difference(sample_metadata.columns))
    if missing_metadata:
        raise ValueError(f"sample metadata missing columns: {', '.join(missing_metadata)}")
    if sample_metadata.duplicated(["date", "ticker"]).any():
        raise ValueError("sample metadata must have one row per date and ticker")
    if any(abs(snapshot.distributions) > 1e-12 for snapshot in result.daily_snapshots):
        raise ValueError(
            "portfolio distributions cannot be attributed without ticker-level action lineage"
        )

    metadata = sample_metadata.copy()
    metadata["date"] = metadata["date"].astype(str)
    metadata["ticker"] = metadata["ticker"].astype(str)
    metadata["sector"], sector_status = _sector_status(metadata)
    if metadata[["liquidity_bucket", "market_regime"]].isna().any().any():
        raise ValueError("sample metadata dimensions must not be missing")
    metadata_lookup = metadata.set_index(["date", "ticker"])

    prediction_lookup: dict[str, pd.Series] = {}
    for _, prediction in predictions.iterrows():
        prediction_id = prediction_identifier(
            str(prediction["fold_id"]),
            str(prediction["model_name"]),
            str(prediction["date"]),
            str(prediction["ticker"]),
        )
        if prediction_id in prediction_lookup:
            raise ValueError(f"ambiguous prediction identity: {prediction_id}")
        prediction_lookup[prediction_id] = prediction

    signals = {signal.signal_id: signal for signal in result.signals}
    if len(signals) != len(result.signals):
        raise ValueError("signal ledger contains duplicate signal IDs")
    orders = {order.order_id: order for order in result.orders}
    if len(orders) != len(result.orders):
        raise ValueError("order ledger contains duplicate order IDs")
    fills = {fill.fill_id: fill for fill in result.fills}
    if len(fills) != len(result.fills):
        raise ValueError("fill ledger contains duplicate fill IDs")
    costs = {cost.fill_id: cost for cost in result.costs}
    if len(costs) != len(result.costs):
        raise ValueError("cost ledger contains duplicate fill IDs")
    if set(costs) != set(fills):
        raise ValueError("every fill must have exactly one cost record")

    snapshots = {snapshot.date: snapshot for snapshot in result.daily_snapshots}
    if len(snapshots) != len(result.daily_snapshots):
        raise ValueError("daily snapshot ledger contains duplicate dates")
    session_count = len(snapshots)
    starting_capital = result.portfolio.starting_equity
    if starting_capital <= 0.0:
        raise ValueError("portfolio starting equity must be positive")

    by_signal: dict[str, dict[str, object]] = {}
    for fill in result.fills:
        order = orders.get(fill.order_id)
        if order is None:
            raise ValueError(f"fill references unknown order: {fill.fill_id}")
        signal = signals.get(order.signal_id)
        if signal is None:
            raise ValueError(f"order references unknown signal: {order.order_id}")
        if order.status != "filled":
            raise ValueError(f"fill references non-filled order: {fill.fill_id}")
        if signal.execution_date is None or signal.execution_date not in snapshots:
            raise ValueError(f"executed signal has no daily snapshot: {signal.signal_id}")
        prediction = prediction_lookup.get(signal.prediction_id)
        if prediction is None:
            raise ValueError(f"signal references unknown prediction: {signal.signal_id}")
        metadata_key = (signal.signal_date, signal.ticker)
        if metadata_key not in metadata_lookup.index:
            raise ValueError(
                f"sample metadata does not cover executed signal: {signal.signal_date} {signal.ticker}"
            )
        sample = metadata_lookup.loc[metadata_key]
        if isinstance(sample, pd.DataFrame):
            raise ValueError("sample metadata must have one row per date and ticker")
        record = by_signal.setdefault(
            signal.signal_id,
            {
                "fold": str(prediction["fold_id"]),
                "year": str(pd.Timestamp(signal.execution_date).year),
                "ticker": signal.ticker,
                "sector": str(sample["sector"]),
                "liquidity_bucket": str(sample["liquidity_bucket"]),
                "market_regime": str(sample["market_regime"]),
                "execution_date": signal.execution_date,
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "commission": 0.0,
                "bid_ask_spread": 0.0,
                "slippage": 0.0,
                "market_impact": 0.0,
                "taxes": 0.0,
                "transaction_costs": 0.0,
            },
        )
        if record["execution_date"] != signal.execution_date:
            raise ValueError(f"signal fills cross execution sessions: {signal.signal_id}")
        if fill.side == "buy":
            record["buy_notional"] = cast(float, record["buy_notional"]) + fill.notional
        elif fill.side == "sell":
            record["sell_notional"] = cast(float, record["sell_notional"]) + fill.notional
        else:
            raise ValueError(f"unsupported fill side for long-only attribution: {fill.side}")
        cost = costs[fill.fill_id]
        for field in (
            "commission",
            "bid_ask_spread",
            "slippage",
            "market_impact",
            "taxes",
        ):
            record[field] = cast(float, record[field]) + float(getattr(cost, field))
        record["transaction_costs"] = cast(float, record["transaction_costs"]) + cost.total_cost

    rows: list[dict[str, object]] = []
    for signal_id, record in sorted(by_signal.items()):
        buy_notional = cast(float, record.pop("buy_notional"))
        sell_notional = cast(float, record.pop("sell_notional"))
        if buy_notional <= 0.0 or sell_notional <= 0.0:
            raise ValueError(f"executed signal is not a complete round trip: {signal_id}")
        execution_date = str(record["execution_date"])
        snapshot = snapshots[execution_date]
        gross_pnl = sell_notional - buy_notional
        transaction_costs = cast(float, record["transaction_costs"])
        net_pnl = gross_pnl - transaction_costs
        rows.append(
            {
                **record,
                "gross_pnl": gross_pnl,
                "gross_return_contribution": gross_pnl / starting_capital,
                "net_pnl": net_pnl,
                "net_return_contribution": net_pnl / starting_capital,
                "turnover": (buy_notional + sell_notional) / snapshot.starting_equity,
                "trade_count": 1,
                "average_gross_exposure_contribution": (
                    buy_notional / snapshot.starting_equity / session_count
                    if session_count
                    else 0.0
                ),
                "average_net_exposure_contribution": (
                    buy_notional / snapshot.starting_equity / session_count
                    if session_count
                    else 0.0
                ),
            }
        )

    columns = [
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
        "execution_date",
        *_PORTFOLIO_ATTRIBUTION_METRICS,
    ]
    attribution = pd.DataFrame.from_records(rows, columns=columns)
    aggregate = _summed_attribution(attribution)

    snapshot_gross_pnl = sum(snapshot.gross_pnl for snapshot in result.daily_snapshots)
    snapshot_costs = sum(snapshot.transaction_costs for snapshot in result.daily_snapshots)
    snapshot_turnover = sum(snapshot.turnover for snapshot in result.daily_snapshots)
    snapshot_gross_exposure = (
        sum(snapshot.gross_exposure for snapshot in result.daily_snapshots) / session_count
        if session_count
        else 0.0
    )
    snapshot_net_exposure = (
        sum(snapshot.net_exposure for snapshot in result.daily_snapshots) / session_count
        if session_count
        else 0.0
    )
    portfolio_net_pnl = result.portfolio.ending_equity - starting_capital
    _assert_attribution_close("gross PnL", float(aggregate["gross_pnl"]), snapshot_gross_pnl)
    _assert_attribution_close(
        "transaction costs", float(aggregate["transaction_costs"]), snapshot_costs
    )
    _assert_attribution_close("net PnL", float(aggregate["net_pnl"]), portfolio_net_pnl)
    _assert_attribution_close("turnover", float(aggregate["turnover"]), snapshot_turnover)
    _assert_attribution_close(
        "gross exposure",
        float(aggregate["average_gross_exposure_contribution"]),
        snapshot_gross_exposure,
    )
    _assert_attribution_close(
        "net exposure",
        float(aggregate["average_net_exposure_contribution"]),
        snapshot_net_exposure,
    )
    closed_positions = [position for position in result.positions if position.quantity == 0]
    _assert_attribution_close(
        "realized position PnL",
        float(aggregate["gross_pnl"]),
        sum(position.realized_pnl for position in closed_positions),
    )
    if int(aggregate["trade_count"]) != len(closed_positions):
        raise ValueError("portfolio attribution trade count does not match closed positions")

    dimensions: dict[str, dict[str, dict[str, float | int]]] = {}
    reconciliation: dict[str, dict[str, object]] = {}
    for dimension in (
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    ):
        grouped: dict[str, dict[str, float | int]] = {}
        if not attribution.empty:
            for value, group in attribution.groupby(dimension, sort=True):
                grouped[str(value)] = _summed_attribution(group)
        dimensions[dimension] = grouped
        deltas = {
            metric: float(sum(group[metric] for group in grouped.values()) - aggregate[metric])
            for metric in _PORTFOLIO_ATTRIBUTION_METRICS
        }
        passed = all(
            abs(delta) <= 1e-9 * max(1.0, abs(float(aggregate[metric])))
            for metric, delta in deltas.items()
        )
        if not passed:
            raise RuntimeError(f"grouped portfolio attribution failed for dimension: {dimension}")
        reconciliation[dimension] = {"passed": True, "deltas": deltas}

    return {
        "schema_version": 1,
        "basis": {
            "year": "execution_date",
            "return_contribution_denominator": "portfolio_starting_equity",
            "turnover_denominator": "session_starting_equity",
            "exposure_basis": "mean_session_contribution",
        },
        "aggregate": aggregate,
        "dimensions": dimensions,
        "dimension_status": {
            "fold": {"status": "available"},
            "year": {"status": "available"},
            "ticker": {"status": "available"},
            "sector": sector_status,
            "liquidity_bucket": {"status": "available"},
            "market_regime": {"status": "available"},
        },
        "reconciliation": reconciliation,
    }
