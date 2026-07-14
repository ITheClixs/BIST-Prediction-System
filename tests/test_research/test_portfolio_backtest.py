"""Event-ledger portfolio accounting and cost invariants."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time

import pandas as pd
import pytest

import bist_predict.research.portfolio_backtest as portfolio_backtest
from bist_predict.ingest.calendar import OfficialTradingCalendar
from bist_predict.ingest.corporate_actions import CorporateAction, CorporateActionType
from bist_predict.ingest.types import OHLCVBar, OpenQuality
from bist_predict.research.portfolio_backtest import (
    CostModel,
    Position,
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
    history = [
        OHLCVBar(
            ticker=ticker,
            date=date(2024, 1, 2),
            open=open_price * 0.98,
            high=open_price,
            low=open_price * 0.97,
            close=open_price * 0.99,
            adj_close=open_price * 0.99,
            volume=800_000,
            source="synthetic",
        )
        for ticker, open_price in (("GARAN", 50.0), ("THYAO", 100.0))
    ]
    execution = [
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
    return history + execution


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
    assert all(entry.balance >= 0.0 for entry in result.cash_ledger)


def test_signal_below_full_round_trip_cost_is_rejected() -> None:
    result = PortfolioBacktester(
        strategy=StrategyConfig(
            top_k=1,
            decision_cost_rate=0.0001,
            max_participation=0.01,
        ),
        costs=CostModel(
            commission_rate=0.0002,
            bid_ask_spread_rate=0.001,
            slippage_rate=0.0003,
            market_impact_coefficient=0.0001,
            tax_rate=0.0,
        ),
    ).run(
        _predictions((0.001, -0.01)),
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
    )

    assert result.fills == ()
    assert result.signals[0].rejection_reason == "non_positive_expected_net_return"


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
    predictions = _predictions()
    selection_costs = CostModel(0.0002, 0.001, 0.0003, 0.0001, 0.0)
    low = PortfolioBacktester(
        strategy=StrategyConfig(top_k=2, decision_cost_rate=0.002, max_participation=0.01),
        costs=CostModel(0.0001, 0.0005, 0.0001, 0.00005, 0.0),
        selection_costs=selection_costs,
    ).run(
        predictions,
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
    )
    high = PortfolioBacktester(
        strategy=StrategyConfig(top_k=2, decision_cost_rate=0.002, max_participation=0.01),
        costs=CostModel(0.001, 0.005, 0.002, 0.001, 0.001),
        selection_costs=selection_costs,
    ).run(
        predictions,
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
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
    assert all(fill.quantity <= 8_000 for fill in buy_fills)


def test_execution_session_volume_cannot_change_orders_costs_or_equity() -> None:
    prices = _prices()
    perturbed = [
        replace(bar, volume=bar.volume * 100) if bar.date == date(2024, 1, 3) else bar
        for bar in prices
    ]

    baseline = _backtester().run(
        _predictions(), prices, model_name="ridge", starting_equity=10_000_000.0
    )
    changed = _backtester().run(
        _predictions(), perturbed, model_name="ridge", starting_equity=10_000_000.0
    )

    assert changed.orders == baseline.orders
    assert changed.fills == baseline.fills
    assert changed.costs == baseline.costs
    assert changed.ending_equity == baseline.ending_equity
    assert {signal.liquidity_reference_volume for signal in baseline.signals} == {800_000.0}
    assert {signal.liquidity_as_of for signal in baseline.signals} == {"2024-01-02"}
    assert {order.liquidity_reference_volume for order in baseline.orders} == {800_000.0}
    assert {order.liquidity_as_of for order in baseline.orders} == {"2024-01-02"}
    assert {fill.liquidity_reference_volume for fill in baseline.fills} == {800_000.0}
    assert {fill.liquidity_as_of for fill in baseline.fills} == {"2024-01-02"}


def test_top_k_ranks_predicted_net_return_without_probability_shrinkage() -> None:
    predictions = _predictions((0.03, 0.02))
    predictions.loc[predictions["ticker"] == "GARAN", "predicted_probability"] = 0.51
    predictions.loc[predictions["ticker"] == "THYAO", "predicted_probability"] = 0.99
    backtester = PortfolioBacktester(
        strategy=StrategyConfig(top_k=1, decision_cost_rate=0.002, max_participation=0.01),
        costs=CostModel(0.0002, 0.001, 0.0003, 0.0001, 0.0),
    )

    result = backtester.run(
        predictions,
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
    )

    assert {fill.ticker for fill in result.fills} == {"GARAN"}
    garan_signal = next(signal for signal in result.signals if signal.ticker == "GARAN")
    explicit_round_trip_cost = 2.0 * (0.0002 + 0.001 / 2.0 + 0.0003 + 0.0001 * 0.1)
    assert garan_signal.expected_net_return == pytest.approx(0.03 - explicit_round_trip_cost)


def test_half_day_exit_uses_official_close_timestamp() -> None:
    predictions = _predictions((0.03, 0.02))
    predictions["date"] = "2026-03-18"
    prices = _prices()
    prices = [
        replace(bar, date=date(2026, 3, 18) if bar.date == date(2024, 1, 2) else date(2026, 3, 19))
        for bar in prices
    ]
    calendar = OfficialTradingCalendar(
        index_name="XIST",
        sessions=(date(2026, 3, 19),),
        source="official-test",
        source_retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        session_close_overrides={date(2026, 3, 19): time(13, 0)},
    )

    result = _backtester().run(
        predictions,
        prices,
        model_name="ridge",
        starting_equity=100_000.0,
        calendar=calendar,
    )

    assert {fill.timestamp for fill in result.fills if fill.side == "sell"} == {
        "2026-03-19T13:00:00+03:00"
    }


def test_open_to_close_position_receives_no_pre_open_dividend_entitlement() -> None:
    dividend = CorporateAction(
        ticker="GARAN",
        effective_date=date(2024, 1, 3),
        action_type=CorporateActionType.CASH_DIVIDEND,
        cash_amount=2.0,
        currency="TRY",
        source="kap",
        source_retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    result = _backtester().run(
        _predictions(),
        _prices(),
        model_name="ridge",
        starting_equity=100_000.0,
        corporate_actions=[dividend],
    )

    assert result.daily_snapshots[0].distributions == 0.0
    assert len(result.corporate_action_ledger) == 1
    action_record = result.corporate_action_ledger[0]
    assert action_record.status == "no_entitlement"
    assert action_record.reason == "no_open_position"
    assert action_record.cash_delta == 0.0
    assert result.artifact_frames()["corporate_action_ledger"].iloc[0]["action_type"] == (
        "cash_dividend"
    )
    assert {entry.category for entry in result.cash_ledger} == {
        "buy_notional",
        "sell_notional",
        "transaction_cost",
    }


def _open_position(ticker: str = "GARAN") -> Position:
    return Position(
        timestamp="2024-01-02T18:00:00+03:00",
        ticker=ticker,
        quantity=100,
        average_entry_price=50.0,
        market_price=60.0,
        market_value=6_000.0,
        unrealized_pnl=1_000.0,
        realized_pnl=0.0,
    )


def test_cash_dividend_credits_only_entitled_shares() -> None:
    action = CorporateAction(
        ticker="GARAN",
        effective_date=date(2024, 1, 3),
        action_type=CorporateActionType.CASH_DIVIDEND,
        cash_amount=2.0,
        currency="TRY",
        source="kap",
        source_retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    result = portfolio_backtest.apply_corporate_actions(
        actions=[action],
        positions=[_open_position()],
        cash=1_000.0,
        timestamp="2024-01-03T10:00:00+03:00",
    )

    assert result.cash == 1_200.0
    assert result.distributions == 200.0
    assert result.positions == (_open_position(),)
    assert result.cash_ledger[0].category == "cash_dividend"
    assert result.action_ledger[0].status == "applied"
    assert result.action_ledger[0].cash_delta == 200.0


def test_split_and_bonus_adjust_quantity_and_basis_without_creating_wealth() -> None:
    actions = [
        CorporateAction(
            ticker="GARAN",
            effective_date=date(2024, 1, 3),
            action_type=CorporateActionType.STOCK_SPLIT,
            ratio=2.0,
            source="kap",
        ),
        CorporateAction(
            ticker="GARAN",
            effective_date=date(2024, 1, 3),
            action_type=CorporateActionType.BONUS_ISSUE,
            ratio=1.5,
            source="kap",
        ),
    ]

    result = portfolio_backtest.apply_corporate_actions(
        actions=actions,
        positions=[_open_position()],
        cash=1_000.0,
        timestamp="2024-01-03T10:00:00+03:00",
    )

    adjusted = result.positions[0]
    assert adjusted.quantity == 300
    assert adjusted.average_entry_price == pytest.approx(50.0 / 3.0)
    assert adjusted.market_price == pytest.approx(20.0)
    assert adjusted.market_value == pytest.approx(6_000.0)
    assert adjusted.unrealized_pnl == pytest.approx(1_000.0)
    assert result.cash == 1_000.0
    assert [record.status for record in result.action_ledger] == ["applied", "applied"]


def test_ticker_change_remaps_an_existing_holding() -> None:
    action = CorporateAction(
        ticker="GARAN",
        effective_date=date(2024, 1, 3),
        action_type=CorporateActionType.TICKER_CHANGE,
        new_ticker="GARAN2",
        source="kap",
    )

    result = portfolio_backtest.apply_corporate_actions(
        actions=[action],
        positions=[_open_position()],
        cash=1_000.0,
        timestamp="2024-01-03T10:00:00+03:00",
    )

    assert result.positions[0].ticker == "GARAN2"
    assert result.action_ledger[0].resulting_ticker == "GARAN2"


def test_rights_issue_with_entitlement_fails_closed_without_exercise_policy() -> None:
    action = CorporateAction(
        ticker="GARAN",
        effective_date=date(2024, 1, 3),
        action_type=CorporateActionType.RIGHTS_ISSUE,
        ratio=0.25,
        subscription_price=20.0,
        source="kap",
    )

    with pytest.raises(ValueError, match="rights issue requires an explicit exercise policy"):
        portfolio_backtest.apply_corporate_actions(
            actions=[action],
            positions=[_open_position()],
            cash=1_000.0,
            timestamp="2024-01-03T10:00:00+03:00",
        )


def test_delisting_requires_price_and_cash_settles_when_price_is_known() -> None:
    missing_price = CorporateAction(
        ticker="GARAN",
        effective_date=date(2024, 1, 3),
        action_type=CorporateActionType.DELISTING,
        source="kap",
    )
    with pytest.raises(ValueError, match="delisting requires an explicit settlement price"):
        portfolio_backtest.apply_corporate_actions(
            actions=[missing_price],
            positions=[_open_position()],
            cash=1_000.0,
            timestamp="2024-01-03T10:00:00+03:00",
        )

    settled = replace(missing_price, delisting_price=55.0)
    result = portfolio_backtest.apply_corporate_actions(
        actions=[settled],
        positions=[_open_position()],
        cash=1_000.0,
        timestamp="2024-01-03T10:00:00+03:00",
    )

    assert result.cash == 6_500.0
    assert result.positions == ()
    assert result.distributions == 0.0
    assert result.cash_ledger[0].category == "delisting_proceeds"
    assert result.action_ledger[0].cash_delta == 5_500.0
