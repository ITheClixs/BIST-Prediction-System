"""Portfolio metrics, grouped diagnostics, and bootstrap interval tests."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_predict.research.portfolio_backtest import (
    CostRecord,
    DailySnapshot,
    Portfolio,
    PortfolioBacktestResult,
    Position,
)
from bist_predict.research.predictions import PREDICTION_COLUMNS
from bist_predict.research.reporting import (
    block_bootstrap_intervals,
    compute_portfolio_metrics,
    grouped_prediction_metrics,
)


def _result() -> PortfolioBacktestResult:
    snapshots = (
        DailySnapshot("2024-01-03", 100.0, 80.0, -18.0, 0.0, 2.0, -0.18, -0.20, 1.0, 0.8, 0.8, 1.0),
        DailySnapshot("2024-01-04", 80.0, 90.0, 11.0, 0.0, 1.0, 0.1375, 0.125, 1.2, 0.9, 0.9, 0.5),
    )
    positions = (
        Position("2024-01-03T18:00:00+03:00", "THYAO", 0, 100.0, 90.0, 0.0, 0.0, -10.0),
        Position("2024-01-04T18:00:00+03:00", "GARAN", 0, 50.0, 55.0, 0.0, 0.0, 10.0),
    )
    costs = (
        CostRecord("fill-1", 0.5, 0.5, 0.4, 0.3, 0.3, 2.0),
        CostRecord("fill-2", 0.3, 0.2, 0.2, 0.2, 0.1, 1.0),
    )
    return PortfolioBacktestResult(
        signals=(),
        orders=(),
        fills=(),
        positions=positions,
        cash_ledger=(),
        costs=costs,
        daily_snapshots=snapshots,
        portfolio=Portfolio(100.0, 90.0, 90.0, ()),
    )


def _predictions() -> pd.DataFrame:
    rows = []
    for index, (ticker, target, predicted) in enumerate(
        (("THYAO", 0.02, 0.01), ("GARAN", -0.01, -0.005))
    ):
        rows.append(
            {
                "date": f"2024-01-0{index + 3}",
                "ticker": ticker,
                "fold_id": "fold_0001",
                "model_name": "ridge",
                "model_version": "ridge-v1",
                "training_end": "2023-12-29",
                "feature_manifest_hash": "a" * 64,
                "target": target,
                "prediction": int(predicted > 0.0),
                "predicted_probability": 0.7 if predicted > 0.0 else 0.3,
                "predicted_return": predicted,
            }
        )
    return pd.DataFrame.from_records(rows, columns=PREDICTION_COLUMNS)


def test_portfolio_metrics_prepend_initial_capital_and_decompose_costs() -> None:
    metrics = compute_portfolio_metrics(_result(), benchmark_returns=[-0.10, 0.05])

    assert metrics["gross_return"] == pytest.approx((1.0 - 0.18) * (1.0 + 0.1375) - 1.0)
    assert metrics["net_return"] == pytest.approx(-0.10)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.20)
    assert metrics["trade_count"] == 2
    assert metrics["hit_rate"] == 0.5
    assert metrics["average_holding_period_sessions"] == 1.0
    assert metrics["cost_decomposition"] == {
        "commission": pytest.approx(0.8),
        "bid_ask_spread": pytest.approx(0.7),
        "slippage": pytest.approx(0.6),
        "market_impact": pytest.approx(0.5),
        "taxes": pytest.approx(0.4),
        "total": pytest.approx(3.0),
    }


def test_block_bootstrap_intervals_are_seeded_and_ordered() -> None:
    returns = [0.01, -0.005, 0.004, 0.002, -0.003, 0.006] * 5

    first = block_bootstrap_intervals(returns, block_size=5, iterations=200, seed=17)
    second = block_bootstrap_intervals(returns, block_size=5, iterations=200, seed=17)

    assert first == second
    assert first["annualized_return"]["lower"] <= first["annualized_return"]["estimate"]
    assert first["annualized_return"]["estimate"] <= first["annualized_return"]["upper"]


def test_prediction_metrics_group_by_fold_year_ticker_sector_liquidity_and_regime() -> None:
    predictions = _predictions()
    metadata = pd.DataFrame(
        {
            "date": predictions["date"],
            "ticker": predictions["ticker"],
            "sector": ["transport", "banking"],
            "liquidity_bucket": ["high", "high"],
            "market_regime": ["up", "down"],
        }
    )

    grouped = grouped_prediction_metrics(predictions, metadata)

    assert set(grouped) == {
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    }
    assert set(grouped["ticker"]) == {"GARAN", "THYAO"}
    assert grouped["fold"]["fold_0001"]["ridge"]["sample_count"] == 2
