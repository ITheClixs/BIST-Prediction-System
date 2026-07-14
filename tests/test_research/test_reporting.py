"""Portfolio metrics, grouped diagnostics, and bootstrap interval tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bist_predict.ingest.types import OHLCVBar, OpenQuality
from bist_predict.research.portfolio_backtest import (
    CostModel,
    CostRecord,
    DailySnapshot,
    Portfolio,
    PortfolioBacktester,
    PortfolioBacktestResult,
    Position,
    StrategyConfig,
)
from bist_predict.research.predictions import PREDICTION_COLUMNS
from bist_predict.research.reporting import (
    block_bootstrap_intervals,
    compute_portfolio_metrics,
    grouped_portfolio_metrics,
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


def _attribution_predictions() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "date": "2023-12-29",
                "ticker": "GARAN",
                "fold_id": "fold_0001",
                "model_name": "ridge",
                "model_version": "ridge-v1",
                "training_end": "2023-12-28",
                "feature_manifest_hash": "a" * 64,
                "target": 0.10,
                "prediction": 1,
                "predicted_probability": 0.80,
                "predicted_return": 0.10,
            },
            {
                "date": "2024-01-02",
                "ticker": "THYAO",
                "fold_id": "fold_0002",
                "model_name": "ridge",
                "model_version": "ridge-v1",
                "training_end": "2023-12-29",
                "feature_manifest_hash": "a" * 64,
                "target": -0.05,
                "prediction": 1,
                "predicted_probability": 0.80,
                "predicted_return": 0.10,
            },
        ],
        columns=PREDICTION_COLUMNS,
    )


def _attribution_result(predictions: pd.DataFrame) -> PortfolioBacktestResult:
    prices = [
        OHLCVBar(
            ticker="GARAN",
            date=date(2023, 12, 29),
            open=48.0,
            high=51.0,
            low=47.0,
            close=50.0,
            adj_close=50.0,
            volume=1_000_000,
            source="synthetic",
            open_quality=OpenQuality.OBSERVED,
        ),
        OHLCVBar(
            ticker="GARAN",
            date=date(2024, 1, 2),
            open=50.0,
            high=56.0,
            low=49.0,
            close=55.0,
            adj_close=55.0,
            volume=1_000_000,
            source="synthetic",
            open_quality=OpenQuality.OBSERVED,
        ),
        OHLCVBar(
            ticker="THYAO",
            date=date(2024, 1, 2),
            open=98.0,
            high=101.0,
            low=97.0,
            close=100.0,
            adj_close=100.0,
            volume=1_000_000,
            source="synthetic",
            open_quality=OpenQuality.OBSERVED,
        ),
        OHLCVBar(
            ticker="THYAO",
            date=date(2024, 1, 3),
            open=100.0,
            high=101.0,
            low=89.0,
            close=90.0,
            adj_close=90.0,
            volume=1_000_000,
            source="synthetic",
            open_quality=OpenQuality.OBSERVED,
        ),
    ]
    return PortfolioBacktester(
        strategy=StrategyConfig(
            top_k=1,
            decision_cost_rate=0.0,
            max_participation=0.01,
            min_trade_value=0.0,
        ),
        costs=CostModel(
            commission_rate=0.001,
            bid_ask_spread_rate=0.001,
            slippage_rate=0.0005,
            market_impact_coefficient=0.0,
            tax_rate=0.0005,
        ),
    ).run(predictions, prices, model_name="ridge", starting_equity=10_000.0)


def test_portfolio_attribution_reconciles_each_grouping_dimension() -> None:
    predictions = _attribution_predictions()
    result = _attribution_result(predictions)
    metadata = pd.DataFrame(
        {
            "date": predictions["date"],
            "ticker": predictions["ticker"],
            "liquidity_bucket": ["high", "medium"],
            "market_regime": ["up", "down"],
        }
    )

    report = grouped_portfolio_metrics(result, predictions, metadata)

    assert set(report["dimensions"]) == {
        "fold",
        "year",
        "ticker",
        "sector",
        "liquidity_bucket",
        "market_regime",
    }
    assert set(report["dimensions"]["fold"]) == {"fold_0001", "fold_0002"}
    assert set(report["dimensions"]["ticker"]) == {"GARAN", "THYAO"}
    assert set(report["dimensions"]["liquidity_bucket"]) == {"high", "medium"}
    assert set(report["dimensions"]["market_regime"]) == {"down", "up"}
    assert set(report["dimensions"]["sector"]) == {"unavailable"}
    assert report["dimension_status"]["sector"] == {
        "status": "unavailable",
        "reason": "sample metadata has no sourced sector field",
    }

    aggregate = report["aggregate"]
    assert aggregate["gross_pnl"] == pytest.approx(
        sum(snapshot.gross_pnl for snapshot in result.daily_snapshots)
    )
    assert aggregate["net_pnl"] == pytest.approx(
        result.portfolio.ending_equity - result.portfolio.starting_equity
    )
    assert aggregate["transaction_costs"] == pytest.approx(
        sum(cost.total_cost for cost in result.costs)
    )
    assert aggregate["turnover"] == pytest.approx(
        sum(snapshot.turnover for snapshot in result.daily_snapshots)
    )
    assert aggregate["average_gross_exposure_contribution"] == pytest.approx(
        sum(snapshot.gross_exposure for snapshot in result.daily_snapshots)
        / len(result.daily_snapshots)
    )
    assert aggregate["trade_count"] == 2
    for dimension in report["dimensions"]:
        assert report["reconciliation"][dimension]["passed"] is True
        assert all(
            delta == pytest.approx(0.0, abs=1e-10)
            for delta in report["reconciliation"][dimension]["deltas"].values()
        )


def test_portfolio_attribution_rejects_unattributed_distributions() -> None:
    predictions = _attribution_predictions()
    result = _attribution_result(predictions)
    snapshot = result.daily_snapshots[0]
    result = PortfolioBacktestResult(
        signals=result.signals,
        orders=result.orders,
        fills=result.fills,
        positions=result.positions,
        cash_ledger=result.cash_ledger,
        costs=result.costs,
        daily_snapshots=(
            DailySnapshot(
                date=snapshot.date,
                starting_equity=snapshot.starting_equity,
                ending_equity=snapshot.ending_equity,
                gross_pnl=snapshot.gross_pnl,
                distributions=1.0,
                transaction_costs=snapshot.transaction_costs,
                gross_return=snapshot.gross_return,
                net_return=snapshot.net_return,
                turnover=snapshot.turnover,
                gross_exposure=snapshot.gross_exposure,
                net_exposure=snapshot.net_exposure,
                concentration=snapshot.concentration,
            ),
            *result.daily_snapshots[1:],
        ),
        portfolio=result.portfolio,
    )
    metadata = pd.DataFrame(
        {
            "date": predictions["date"],
            "ticker": predictions["ticker"],
            "liquidity_bucket": ["high", "medium"],
            "market_regime": ["up", "down"],
        }
    )

    with pytest.raises(ValueError, match="distributions cannot be attributed"):
        grouped_portfolio_metrics(result, predictions, metadata)
