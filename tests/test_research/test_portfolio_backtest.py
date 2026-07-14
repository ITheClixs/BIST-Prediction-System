"""Event-ledger portfolio accounting and cost invariants."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bist_predict.ingest.types import OHLCVBar, OpenQuality
from bist_predict.research.portfolio_backtest import (
    CostModel,
    PortfolioBacktester,
    StrategyConfig,
)
from bist_predict.research.predictions import PREDICTION_COLUMNS

MANIFEST_HASH = "a" * 64


def _predictions(
    predicted_returns: tuple[float, ...] = (0.03, 0.02),
) -> pd.DataFrame:
    records = []
    for ticker, predicted_return in zip(("GARAN", "THYAO"), predicted_returns):
        records.append(
            {
                "date": "2024-01-02",
                "ticker": ticker,
                "fold_id": "fold_0001",
                "model_name": "ridge",
                "model_version": "ridge-v1",
                "training_end": "2023-12-29",
                "feature_manifest_hash": MANIFEST_HASH,
                "target": 0.01,
                "prediction": int(predicted_return > 0.0),
                "predicted_probability": 0.8 if predicted_return > 0.0 else 0.2,
                "predicted_return": predicted_return,
            }
        )
    return pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)


def _prices(
    *,
    open_quality: OpenQuality = OpenQuality.OBSERVED,
    close_multiplier: float = 1.01,
) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            ticker=ticker,
            date=date(2024, 1, 3),
            open=open_price,
            high=open_price * 1.02,
            low=open_price * 0.99,
            close=open_price * close_multiplier,
            adj_close=open_price * close_multiplier,
            volume=1_000_000,
            source="synthetic",
            open_quality=open_quality,
        )
        for ticker, open_price in (("GARAN", 50.0), ("THYAO", 100.0))
    ]


def _backtester(costs: CostModel | None = None) -> PortfolioBacktester:
    return PortfolioBacktester(
        strategy=StrategyConfig(top_k=2, decision_cost_rate=0.002, max_participation=0.01),
        costs=costs
        or CostModel(
            commission_rate=0.0002,
            bid_ask_spread_rate=0.001,
            slippage_rate=0.0003,
            market_impact_coefficient=0.0001,
            tax_rate=0.0,
        ),
    )


def test_backtest_persists_orders_fills_positions_cash_and_reconciles() -> None:
    result = _backtester().run(
        _predictions(), _prices(), model_name="ridge", starting_equity=100_000.0
    )

    assert len(result.signals) == 2
    assert len(result.orders) == 4
    assert len(result.fills) == 4
    assert len(result.positions) == 4
    assert len(result.cash_ledger) == 8
    assert len(result.costs) == 4
    snapshot = result.daily_snapshots[0]
    assert snapshot.ending_equity == pytest.approx(
        snapshot.starting_equity
        + snapshot.gross_pnl
        + snapshot.distributions
        - snapshot.transaction_costs
    )
    assert snapshot.ending_equity == pytest.approx(result.ending_equity)
    assert all(
        cost.total_cost
        == pytest.approx(
            cost.commission + cost.bid_ask_spread + cost.slippage + cost.market_impact + cost.taxes
        )
        for cost in result.costs
    )


def test_no_position_strategy_has_zero_return_turnover_and_cost() -> None:
    result = _backtester().run(
        _predictions((-0.01, -0.02)),
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
    )

    snapshot = result.daily_snapshots[0]
    assert result.fills == ()
    assert snapshot.gross_return == 0.0
    assert snapshot.net_return == 0.0
    assert snapshot.turnover == 0.0
    assert snapshot.transaction_costs == 0.0


def test_proxy_open_is_rejected_before_order_submission() -> None:
    result = _backtester().run(
        _predictions(),
        _prices(open_quality=OpenQuality.PROXY),
        model_name="ridge",
        starting_equity=100_000.0,
    )

    assert result.orders == ()
    assert {signal.rejection_reason for signal in result.signals} == {"proxy_open"}


def test_higher_realized_costs_cannot_improve_net_performance() -> None:
    low = _backtester(CostModel(0.0001, 0.0005, 0.0001, 0.00005, 0.0)).run(
        _predictions(), _prices(), model_name="ridge", starting_equity=100_000.0
    )
    high = _backtester(CostModel(0.001, 0.005, 0.002, 0.001, 0.001)).run(
        _predictions(), _prices(), model_name="ridge", starting_equity=100_000.0
    )

    assert len(low.fills) == len(high.fills)
    assert high.ending_equity <= low.ending_equity
    assert high.daily_snapshots[0].transaction_costs >= low.daily_snapshots[0].transaction_costs


def test_participation_limit_caps_filled_quantity() -> None:
    result = _backtester().run(
        _predictions(), _prices(), model_name="ridge", starting_equity=10_000_000.0
    )

    buy_fills = [fill for fill in result.fills if fill.side == "buy"]
    assert buy_fills
    assert all(fill.quantity <= 10_000 for fill in buy_fills)
