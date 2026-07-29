"""Transaction-cost-aware one-session portfolio backtest with full ledgers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from typing import Any, Iterable

import pandas as pd

from bist_predict.ingest.corporate_actions import CorporateAction, CorporateActionType
from bist_predict.ingest.calendar import OfficialTradingCalendar
from bist_predict.ingest.types import OHLCVBar, OpenQuality, VolumeQuality
from bist_predict.research.predictions import validate_predictions


def _identifier(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]


def prediction_identifier(
    fold_id: str,
    model_name: str,
    signal_date: str,
    ticker: str,
) -> str:
    """Return the stable identity shared by artifacts, signals, and tracking."""
    return _identifier(fold_id, model_name, signal_date, ticker)


@dataclass(frozen=True)
class CostModel:
    """Explicit realized transaction-cost components as decimal rates."""

    commission_rate: float
    bid_ask_spread_rate: float
    slippage_rate: float
    market_impact_coefficient: float
    tax_rate: float

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("transaction cost rates must be non-negative")


def round_trip_cost_rate(costs: CostModel, participation_rate: float) -> float:
    """Return the cost of entering and exiting a position, as a rate on notional.

    Only half the quoted spread is charged per side, since a marketable order
    crosses from the mid to one side of the book rather than across it, and the
    impact term follows the square-root law at the participation rate the
    strategy is allowed to take.  The same rate serves two purposes: it is the
    hurdle a predicted return has to clear before a name is tradeable, and it is
    the cost floor the feasibility bound in the detectability module compares an
    information coefficient against.
    """
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("the participation rate must lie in (0, 1]")
    one_way = (
        costs.commission_rate
        + costs.bid_ask_spread_rate / 2.0
        + costs.slippage_rate
        + costs.market_impact_coefficient * math.sqrt(participation_rate)
    )
    return 2.0 * one_way + costs.tax_rate


@dataclass(frozen=True)
class StrategyConfig:
    """Long-only top-k constraints and declared decision-time cost assumption."""

    top_k: int = 10
    decision_cost_rate: float = 0.003
    max_participation: float = 0.01
    min_trade_value: float = 100.0
    liquidity_lookback_sessions: int = 20

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.decision_cost_rate < 0.0:
            raise ValueError("decision_cost_rate must be non-negative")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must lie in (0, 1]")
        if self.min_trade_value < 0.0:
            raise ValueError("min_trade_value must be non-negative")
        if self.liquidity_lookback_sessions <= 0:
            raise ValueError("liquidity_lookback_sessions must be positive")


@dataclass(frozen=True)
class Signal:
    signal_id: str
    prediction_id: str
    signal_date: str
    execution_date: str | None
    ticker: str
    predicted_return: float
    predicted_probability: float
    expected_net_return: float
    target_weight: float
    eligible: bool
    rejection_reason: str | None
    liquidity_reference_volume: float | None = None
    liquidity_as_of: str | None = None


@dataclass(frozen=True)
class Order:
    order_id: str
    signal_id: str
    timestamp: str
    ticker: str
    side: str
    requested_quantity: int
    reference_price: float
    status: str
    rejection_reason: str | None
    liquidity_reference_volume: float
    liquidity_as_of: str


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    timestamp: str
    ticker: str
    side: str
    quantity: int
    reference_price: float
    fill_price: float
    notional: float
    participation_rate: float
    liquidity_reference_volume: float
    liquidity_as_of: str


@dataclass(frozen=True)
class Position:
    timestamp: str
    ticker: str
    quantity: int
    average_entry_price: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass(frozen=True)
class Portfolio:
    starting_equity: float
    ending_equity: float
    cash: float
    open_positions: tuple[Position, ...]


@dataclass(frozen=True)
class CashLedger:
    ledger_id: str
    timestamp: str
    category: str
    amount: float
    balance: float
    reference_id: str


@dataclass(frozen=True)
class CostRecord:
    fill_id: str
    commission: float
    bid_ask_spread: float
    slippage: float
    market_impact: float
    taxes: float
    total_cost: float


@dataclass(frozen=True)
class DailySnapshot:
    date: str
    starting_equity: float
    ending_equity: float
    gross_pnl: float
    distributions: float
    transaction_costs: float
    gross_return: float
    net_return: float
    turnover: float
    gross_exposure: float
    net_exposure: float
    concentration: float


@dataclass(frozen=True)
class CorporateActionRecord:
    """Auditable result of applying one sourced action to a holding."""

    action_id: str
    timestamp: str
    effective_date: str
    ticker: str
    action_type: str
    status: str
    quantity_before: int
    quantity_after: int
    cash_delta: float
    resulting_ticker: str | None
    reason: str | None
    source: str
    source_retrieved_at: str | None


@dataclass(frozen=True)
class CorporateActionApplication:
    """Portfolio state after a deterministic corporate-action batch."""

    positions: tuple[Position, ...]
    cash: float
    distributions: float
    action_ledger: tuple[CorporateActionRecord, ...]
    cash_ledger: tuple[CashLedger, ...]


def _action_record(
    action: CorporateAction,
    *,
    timestamp: str,
    status: str,
    quantity_before: int,
    quantity_after: int,
    cash_delta: float,
    resulting_ticker: str | None,
    reason: str | None,
) -> CorporateActionRecord:
    action_id = _identifier(
        "corporate-action",
        action.ticker,
        action.effective_date,
        action.action_type.value,
        action.source,
        action.ratio,
        action.cash_amount,
        action.currency,
        action.subscription_price,
        action.new_ticker,
        action.delisting_price,
    )
    return CorporateActionRecord(
        action_id=action_id,
        timestamp=timestamp,
        effective_date=action.effective_date.isoformat(),
        ticker=action.ticker,
        action_type=action.action_type.value,
        status=status,
        quantity_before=quantity_before,
        quantity_after=quantity_after,
        cash_delta=cash_delta,
        resulting_ticker=resulting_ticker,
        reason=reason,
        source=action.source,
        source_retrieved_at=(
            action.source_retrieved_at.isoformat()
            if action.source_retrieved_at is not None
            else None
        ),
    )


def apply_corporate_actions(
    *,
    actions: Iterable[CorporateAction],
    positions: Iterable[Position],
    cash: float,
    timestamp: str,
) -> CorporateActionApplication:
    """Apply sourced actions without inventing rights or delisting policies.

    Splits and bonus issues preserve position wealth while adjusting share count
    and per-share basis. Dividends credit entitled shares. Rights issues fail
    closed because exercising them requires an explicit capital-allocation
    policy. Delistings require a sourced cash settlement price.
    """
    if cash < 0.0:
        raise ValueError("cash cannot be negative")

    holdings: dict[str, Position] = {}
    for position in positions:
        if position.quantity <= 0:
            raise ValueError("corporate actions require open positions")
        if position.ticker in holdings:
            raise ValueError(f"duplicate open position: {position.ticker}")
        holdings[position.ticker] = position

    balance = cash
    distributions = 0.0
    action_ledger: list[CorporateActionRecord] = []
    cash_ledger: list[CashLedger] = []
    ordered_actions = sorted(
        actions,
        key=lambda action: (
            action.effective_date,
            action.ticker,
            action.action_type.value,
        ),
    )
    for action in ordered_actions:
        current_position = holdings.get(action.ticker)
        if current_position is None:
            action_ledger.append(
                _action_record(
                    action,
                    timestamp=timestamp,
                    status="no_entitlement",
                    quantity_before=0,
                    quantity_after=0,
                    cash_delta=0.0,
                    resulting_ticker=action.new_ticker,
                    reason="no_open_position",
                )
            )
            continue

        quantity_before = current_position.quantity
        if action.action_type in {
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.BONUS_ISSUE,
        }:
            assert action.ratio is not None
            adjusted_quantity = quantity_before * action.ratio
            rounded_quantity = round(adjusted_quantity)
            if not math.isclose(adjusted_quantity, rounded_quantity, abs_tol=1e-9):
                raise ValueError("fractional shares require an explicit cash-in-lieu policy")
            adjusted = replace(
                current_position,
                timestamp=timestamp,
                quantity=rounded_quantity,
                average_entry_price=current_position.average_entry_price / action.ratio,
                market_price=current_position.market_price / action.ratio,
            )
            holdings[action.ticker] = adjusted
            action_ledger.append(
                _action_record(
                    action,
                    timestamp=timestamp,
                    status="applied",
                    quantity_before=quantity_before,
                    quantity_after=adjusted.quantity,
                    cash_delta=0.0,
                    resulting_ticker=action.ticker,
                    reason=None,
                )
            )
            continue

        if action.action_type is CorporateActionType.CASH_DIVIDEND:
            assert action.cash_amount is not None
            cash_delta = quantity_before * action.cash_amount
            balance += cash_delta
            distributions += cash_delta
            record = _action_record(
                action,
                timestamp=timestamp,
                status="applied",
                quantity_before=quantity_before,
                quantity_after=quantity_before,
                cash_delta=cash_delta,
                resulting_ticker=action.ticker,
                reason=None,
            )
            action_ledger.append(record)
            cash_ledger.append(
                CashLedger(
                    ledger_id=_identifier(record.action_id, "cash-dividend"),
                    timestamp=timestamp,
                    category="cash_dividend",
                    amount=cash_delta,
                    balance=balance,
                    reference_id=record.action_id,
                )
            )
            continue

        if action.action_type is CorporateActionType.TICKER_CHANGE:
            assert action.new_ticker is not None
            if action.new_ticker in holdings:
                raise ValueError(f"ticker change would collide with {action.new_ticker}")
            remapped = replace(
                current_position,
                timestamp=timestamp,
                ticker=action.new_ticker,
            )
            del holdings[action.ticker]
            holdings[action.new_ticker] = remapped
            action_ledger.append(
                _action_record(
                    action,
                    timestamp=timestamp,
                    status="applied",
                    quantity_before=quantity_before,
                    quantity_after=quantity_before,
                    cash_delta=0.0,
                    resulting_ticker=action.new_ticker,
                    reason=None,
                )
            )
            continue

        if action.action_type is CorporateActionType.RIGHTS_ISSUE:
            raise ValueError("rights issue requires an explicit exercise policy")

        if action.action_type is CorporateActionType.DELISTING:
            if action.delisting_price is None:
                raise ValueError("delisting requires an explicit settlement price")
            cash_delta = quantity_before * action.delisting_price
            balance += cash_delta
            del holdings[action.ticker]
            record = _action_record(
                action,
                timestamp=timestamp,
                status="applied",
                quantity_before=quantity_before,
                quantity_after=0,
                cash_delta=cash_delta,
                resulting_ticker=None,
                reason="cash_settlement",
            )
            action_ledger.append(record)
            cash_ledger.append(
                CashLedger(
                    ledger_id=_identifier(record.action_id, "delisting-proceeds"),
                    timestamp=timestamp,
                    category="delisting_proceeds",
                    amount=cash_delta,
                    balance=balance,
                    reference_id=record.action_id,
                )
            )

    return CorporateActionApplication(
        positions=tuple(holdings[ticker] for ticker in sorted(holdings)),
        cash=balance,
        distributions=distributions,
        action_ledger=tuple(action_ledger),
        cash_ledger=tuple(cash_ledger),
    )


@dataclass(frozen=True)
class PortfolioBacktestResult:
    signals: tuple[Signal, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    positions: tuple[Position, ...]
    cash_ledger: tuple[CashLedger, ...]
    costs: tuple[CostRecord, ...]
    daily_snapshots: tuple[DailySnapshot, ...]
    portfolio: Portfolio
    corporate_action_ledger: tuple[CorporateActionRecord, ...] = ()

    @property
    def ending_equity(self) -> float:
        return self.portfolio.ending_equity

    def artifact_frames(self) -> dict[str, pd.DataFrame]:
        """Return the six required ledger tables with stable field names."""

        def frame(items: tuple[Any, ...], item_type: Any) -> pd.DataFrame:
            return pd.DataFrame.from_records(
                [asdict(item) for item in items],
                columns=[field.name for field in fields(item_type)],
            )

        return {
            "signals": frame(self.signals, Signal),
            "orders": frame(self.orders, Order),
            "fills": frame(self.fills, Fill),
            "positions": frame(self.positions, Position),
            "cash_ledger": frame(self.cash_ledger, CashLedger),
            "daily_equity": frame(self.daily_snapshots, DailySnapshot),
            "costs": frame(self.costs, CostRecord),
            "corporate_action_ledger": frame(self.corporate_action_ledger, CorporateActionRecord),
        }


@dataclass(frozen=True)
class _Candidate:
    prediction: pd.Series
    bar: OHLCVBar | None
    execution_date: str | None
    score: float
    rejection_reason: str | None
    liquidity_reference_volume: float | None
    liquidity_as_of: str | None


class PortfolioBacktester:
    """Execute cost-aware top-k signals at observed next-session opens."""

    def __init__(
        self,
        *,
        strategy: StrategyConfig,
        costs: CostModel,
        selection_costs: CostModel | None = None,
    ) -> None:
        self._strategy = strategy
        self._costs = costs
        self._selection_costs = selection_costs or costs

    @staticmethod
    def _next_bar(
        bars_by_ticker: dict[str, list[OHLCVBar]], ticker: str, signal_date: date
    ) -> OHLCVBar | None:
        return next(
            (bar for bar in bars_by_ticker.get(ticker, ()) if bar.date > signal_date),
            None,
        )

    def _lagged_liquidity(
        self,
        bars_by_ticker: dict[str, list[OHLCVBar]],
        ticker: str,
        signal_date: date,
    ) -> tuple[float | None, str | None]:
        known_bars = [
            bar
            for bar in bars_by_ticker.get(ticker, ())
            if bar.date <= signal_date
            and bar.volume > 0
            and bar.volume_quality is not VolumeQuality.MISSING
        ][-self._strategy.liquidity_lookback_sessions :]
        if not known_bars:
            return None, None
        reference_volume = sum(bar.volume for bar in known_bars) / len(known_bars)
        return reference_volume, known_bars[-1].date.isoformat()

    def _candidate(
        self,
        prediction: pd.Series,
        bars_by_ticker: dict[str, list[OHLCVBar]],
    ) -> _Candidate:
        signal_date = date.fromisoformat(str(prediction["date"]))
        ticker = str(prediction["ticker"])
        bar = self._next_bar(bars_by_ticker, ticker, signal_date)
        liquidity_reference_volume, liquidity_as_of = self._lagged_liquidity(
            bars_by_ticker,
            ticker,
            signal_date,
        )
        estimated_cost_rate = max(
            self._strategy.decision_cost_rate,
            self._estimated_round_trip_cost_rate(self._strategy.max_participation),
        )
        score = float(prediction["predicted_return"]) - estimated_cost_rate
        reason: str | None = None
        if bar is None:
            reason = "missing_execution_price"
        elif bar.open_quality is OpenQuality.PROXY:
            reason = "proxy_open"
        elif bar.open_quality is not OpenQuality.OBSERVED or bar.open <= 0.0:
            reason = "missing_open"
        elif liquidity_reference_volume is None:
            reason = "missing_lagged_liquidity"
        elif score <= 0.0:
            reason = "non_positive_expected_net_return"
        return _Candidate(
            prediction=prediction,
            bar=bar,
            execution_date=bar.date.isoformat() if bar is not None else None,
            score=score,
            rejection_reason=reason,
            liquidity_reference_volume=liquidity_reference_volume,
            liquidity_as_of=liquidity_as_of,
        )

    def _cost_record(
        self,
        fill_id: str,
        side: str,
        notional: float,
        participation_rate: float,
    ) -> CostRecord:
        commission = notional * self._costs.commission_rate
        spread = notional * self._costs.bid_ask_spread_rate / 2.0
        slippage = notional * self._costs.slippage_rate
        impact = notional * self._costs.market_impact_coefficient * math.sqrt(participation_rate)
        taxes = notional * self._costs.tax_rate if side == "sell" else 0.0
        total = commission + spread + slippage + impact + taxes
        return CostRecord(
            fill_id=fill_id,
            commission=commission,
            bid_ask_spread=spread,
            slippage=slippage,
            market_impact=impact,
            taxes=taxes,
            total_cost=total,
        )

    def _estimated_round_trip_cost_rate(self, participation_rate: float) -> float:
        return round_trip_cost_rate(self._selection_costs, participation_rate)

    def _affordable_quantity(
        self,
        bar: OHLCVBar,
        budget: float,
        liquidity_reference_volume: float,
    ) -> int:
        volume_limit = math.floor(liquidity_reference_volume * self._strategy.max_participation)
        cash_limit = math.floor(budget / bar.open)
        low = 0
        high = min(volume_limit, cash_limit)
        while low < high:
            candidate = (low + high + 1) // 2
            notional = candidate * bar.open
            participation = candidate / liquidity_reference_volume
            estimated_cost = self._cost_record(
                "buy-cost-estimate",
                "buy",
                notional,
                participation,
            ).total_cost
            if notional + estimated_cost <= budget:
                low = candidate
            else:
                high = candidate - 1
        return low

    def run(
        self,
        predictions: pd.DataFrame,
        prices: Iterable[OHLCVBar],
        *,
        model_name: str,
        starting_equity: float,
        corporate_actions: Iterable[CorporateAction] = (),
        calendar: OfficialTradingCalendar | None = None,
    ) -> PortfolioBacktestResult:
        """Run the one-session strategy and preserve every state transition."""
        validate_predictions(predictions)
        if starting_equity <= 0.0:
            raise ValueError("starting_equity must be positive")
        selected_predictions = predictions.loc[predictions["model_name"] == model_name].sort_values(
            ["date", "ticker"], kind="stable"
        )
        if selected_predictions.empty:
            raise ValueError(f"no predictions for model: {model_name}")

        bars_by_ticker: dict[str, list[OHLCVBar]] = {}
        price_keys: set[tuple[str, date]] = set()
        for bar in prices:
            price_key = (bar.ticker, bar.date)
            if price_key in price_keys:
                raise ValueError(f"duplicate execution price: {bar.ticker} {bar.date}")
            price_keys.add(price_key)
            bars_by_ticker.setdefault(bar.ticker, []).append(bar)
        for bars in bars_by_ticker.values():
            bars.sort(key=lambda bar: bar.date)
        sourced_actions = tuple(corporate_actions)

        candidates = [
            self._candidate(row, bars_by_ticker) for _, row in selected_predictions.iterrows()
        ]
        by_execution: dict[str, list[_Candidate]] = {}
        for candidate in candidates:
            if candidate.execution_date is not None:
                by_execution.setdefault(candidate.execution_date, []).append(candidate)

        signals: list[Signal] = []
        orders: list[Order] = []
        fills: list[Fill] = []
        positions: list[Position] = []
        cash_ledger: list[CashLedger] = []
        costs: list[CostRecord] = []
        snapshots: list[DailySnapshot] = []
        corporate_action_ledger: list[CorporateActionRecord] = []
        cash = starting_equity

        # This benchmark opens after the effective-date action processing point
        # and liquidates at the same close, so it never carries an entitlement.
        # Still process and persist every supplied event rather than silently
        # discarding it; the public processor above handles actual holdings.
        actions_by_date: dict[date, list[CorporateAction]] = {}
        for action in sourced_actions:
            actions_by_date.setdefault(action.effective_date, []).append(action)
        for effective_date in sorted(actions_by_date):
            if calendar is None:
                action_timestamp = f"{effective_date.isoformat()}T10:00:00+03:00"
            else:
                action_timestamp = calendar.session_bounds(effective_date)[0].isoformat()
            application = apply_corporate_actions(
                actions=actions_by_date[effective_date],
                positions=(),
                cash=cash,
                timestamp=action_timestamp,
            )
            if application.cash != cash or application.distributions != 0.0:
                raise RuntimeError("flat strategy received an unexpected action entitlement")
            corporate_action_ledger.extend(application.action_ledger)

        for execution_date in sorted(by_execution):
            execution_session = date.fromisoformat(execution_date)
            if calendar is None:
                open_timestamp = f"{execution_date}T10:00:00+03:00"
                close_timestamp = f"{execution_date}T18:00:00+03:00"
            else:
                session_open, session_close = calendar.session_bounds(execution_session)
                open_timestamp = session_open.isoformat()
                close_timestamp = session_close.isoformat()
            day_candidates = by_execution[execution_date]
            eligible = sorted(
                (candidate for candidate in day_candidates if candidate.rejection_reason is None),
                key=lambda candidate: (
                    -candidate.score,
                    str(candidate.prediction["ticker"]),
                ),
            )
            selected = eligible[: self._strategy.top_k]
            selected_keys = {
                (str(item.prediction["date"]), str(item.prediction["ticker"])) for item in selected
            }
            selected_count = len(selected)
            target_weight = 1.0 / selected_count if selected_count else 0.0
            candidate_signals: dict[tuple[str, str], Signal] = {}
            for candidate in day_candidates:
                prediction = candidate.prediction
                prediction_key = (str(prediction["date"]), str(prediction["ticker"]))
                reason = candidate.rejection_reason
                if reason is None and prediction_key not in selected_keys:
                    reason = "not_top_k"
                prediction_id = prediction_identifier(
                    str(prediction["fold_id"]),
                    str(prediction["model_name"]),
                    str(prediction["date"]),
                    str(prediction["ticker"]),
                )
                signal = Signal(
                    signal_id=_identifier("signal", prediction_id),
                    prediction_id=prediction_id,
                    signal_date=str(prediction["date"]),
                    execution_date=candidate.execution_date,
                    ticker=str(prediction["ticker"]),
                    predicted_return=float(prediction["predicted_return"]),
                    predicted_probability=float(prediction["predicted_probability"]),
                    expected_net_return=candidate.score,
                    target_weight=target_weight if reason is None else 0.0,
                    eligible=reason is None,
                    rejection_reason=reason,
                    liquidity_reference_volume=candidate.liquidity_reference_volume,
                    liquidity_as_of=candidate.liquidity_as_of,
                )
                signals.append(signal)
                candidate_signals[prediction_key] = signal

            day_starting_equity = cash
            day_gross_pnl = 0.0
            day_distributions = 0.0
            day_costs = 0.0
            day_notional = 0.0
            entry_notionals: list[float] = []

            for candidate in selected:
                assert candidate.bar is not None
                assert candidate.liquidity_reference_volume is not None
                assert candidate.liquidity_as_of is not None
                bar = candidate.bar
                prediction_key = (
                    str(candidate.prediction["date"]),
                    str(candidate.prediction["ticker"]),
                )
                signal = candidate_signals[prediction_key]
                allocation = min(cash, day_starting_equity * target_weight)
                quantity = self._affordable_quantity(
                    bar,
                    allocation,
                    candidate.liquidity_reference_volume,
                )
                buy_order_id = _identifier(signal.signal_id, "buy", execution_date)
                if quantity <= 0 or quantity * bar.open < self._strategy.min_trade_value:
                    orders.append(
                        Order(
                            order_id=buy_order_id,
                            signal_id=signal.signal_id,
                            timestamp=open_timestamp,
                            ticker=bar.ticker,
                            side="buy",
                            requested_quantity=max(quantity, 0),
                            reference_price=bar.open,
                            status="rejected",
                            rejection_reason="below_minimum_trade",
                            liquidity_reference_volume=candidate.liquidity_reference_volume,
                            liquidity_as_of=candidate.liquidity_as_of,
                        )
                    )
                    continue

                orders.append(
                    Order(
                        order_id=buy_order_id,
                        signal_id=signal.signal_id,
                        timestamp=open_timestamp,
                        ticker=bar.ticker,
                        side="buy",
                        requested_quantity=quantity,
                        reference_price=bar.open,
                        status="filled",
                        rejection_reason=None,
                        liquidity_reference_volume=candidate.liquidity_reference_volume,
                        liquidity_as_of=candidate.liquidity_as_of,
                    )
                )
                buy_notional = quantity * bar.open
                participation = quantity / candidate.liquidity_reference_volume
                buy_fill_id = _identifier(buy_order_id, "fill")
                fills.append(
                    Fill(
                        fill_id=buy_fill_id,
                        order_id=buy_order_id,
                        timestamp=open_timestamp,
                        ticker=bar.ticker,
                        side="buy",
                        quantity=quantity,
                        reference_price=bar.open,
                        fill_price=bar.open,
                        notional=buy_notional,
                        participation_rate=participation,
                        liquidity_reference_volume=candidate.liquidity_reference_volume,
                        liquidity_as_of=candidate.liquidity_as_of,
                    )
                )
                buy_cost = self._cost_record(buy_fill_id, "buy", buy_notional, participation)
                costs.append(buy_cost)
                cash -= buy_notional
                cash_ledger.append(
                    CashLedger(
                        _identifier(buy_fill_id, "notional"),
                        open_timestamp,
                        "buy_notional",
                        -buy_notional,
                        cash,
                        buy_fill_id,
                    )
                )
                cash -= buy_cost.total_cost
                cash_ledger.append(
                    CashLedger(
                        _identifier(buy_fill_id, "cost"),
                        open_timestamp,
                        "transaction_cost",
                        -buy_cost.total_cost,
                        cash,
                        buy_fill_id,
                    )
                )
                if cash < -1e-8:
                    raise RuntimeError("buy notional and costs exceeded available cash")
                positions.append(
                    Position(
                        timestamp=open_timestamp,
                        ticker=bar.ticker,
                        quantity=quantity,
                        average_entry_price=bar.open,
                        market_price=bar.open,
                        market_value=buy_notional,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                    )
                )

                sell_order_id = _identifier(signal.signal_id, "sell", execution_date)
                sell_notional = quantity * bar.close
                orders.append(
                    Order(
                        order_id=sell_order_id,
                        signal_id=signal.signal_id,
                        timestamp=close_timestamp,
                        ticker=bar.ticker,
                        side="sell",
                        requested_quantity=quantity,
                        reference_price=bar.close,
                        status="filled",
                        rejection_reason=None,
                        liquidity_reference_volume=candidate.liquidity_reference_volume,
                        liquidity_as_of=candidate.liquidity_as_of,
                    )
                )
                sell_fill_id = _identifier(sell_order_id, "fill")
                fills.append(
                    Fill(
                        fill_id=sell_fill_id,
                        order_id=sell_order_id,
                        timestamp=close_timestamp,
                        ticker=bar.ticker,
                        side="sell",
                        quantity=quantity,
                        reference_price=bar.close,
                        fill_price=bar.close,
                        notional=sell_notional,
                        participation_rate=participation,
                        liquidity_reference_volume=candidate.liquidity_reference_volume,
                        liquidity_as_of=candidate.liquidity_as_of,
                    )
                )
                sell_cost = self._cost_record(sell_fill_id, "sell", sell_notional, participation)
                costs.append(sell_cost)
                cash += sell_notional
                cash_ledger.append(
                    CashLedger(
                        _identifier(sell_fill_id, "notional"),
                        close_timestamp,
                        "sell_notional",
                        sell_notional,
                        cash,
                        sell_fill_id,
                    )
                )
                cash -= sell_cost.total_cost
                cash_ledger.append(
                    CashLedger(
                        _identifier(sell_fill_id, "cost"),
                        close_timestamp,
                        "transaction_cost",
                        -sell_cost.total_cost,
                        cash,
                        sell_fill_id,
                    )
                )
                realized = quantity * (bar.close - bar.open)
                positions.append(
                    Position(
                        timestamp=close_timestamp,
                        ticker=bar.ticker,
                        quantity=0,
                        average_entry_price=bar.open,
                        market_price=bar.close,
                        market_value=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=realized,
                    )
                )
                day_gross_pnl += realized
                day_costs += buy_cost.total_cost + sell_cost.total_cost
                day_notional += buy_notional + sell_notional
                entry_notionals.append(buy_notional)

            expected_equity = day_starting_equity + day_gross_pnl + day_distributions - day_costs
            if not math.isclose(cash, expected_equity, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError("portfolio cash failed accounting reconciliation")
            gross_return = day_gross_pnl / day_starting_equity
            net_return = (cash - day_starting_equity) / day_starting_equity
            entry_total = sum(entry_notionals)
            snapshots.append(
                DailySnapshot(
                    date=execution_date,
                    starting_equity=day_starting_equity,
                    ending_equity=cash,
                    gross_pnl=day_gross_pnl,
                    distributions=day_distributions,
                    transaction_costs=day_costs,
                    gross_return=gross_return,
                    net_return=net_return,
                    turnover=day_notional / day_starting_equity,
                    gross_exposure=entry_total / day_starting_equity,
                    net_exposure=entry_total / day_starting_equity,
                    concentration=(
                        max(entry_notionals) / entry_total if entry_total > 0.0 else 0.0
                    ),
                )
            )

        # Missing execution prices have no session on which to create a daily
        # snapshot, but their rejected signals must still be retained.
        represented = {(signal.signal_date, signal.ticker) for signal in signals}
        for candidate in candidates:
            prediction_key = (
                str(candidate.prediction["date"]),
                str(candidate.prediction["ticker"]),
            )
            if prediction_key in represented:
                continue
            prediction_id = prediction_identifier(
                str(candidate.prediction["fold_id"]),
                str(candidate.prediction["model_name"]),
                *prediction_key,
            )
            signals.append(
                Signal(
                    signal_id=_identifier("signal", prediction_id),
                    prediction_id=prediction_id,
                    signal_date=prediction_key[0],
                    execution_date=None,
                    ticker=prediction_key[1],
                    predicted_return=float(candidate.prediction["predicted_return"]),
                    predicted_probability=float(candidate.prediction["predicted_probability"]),
                    expected_net_return=candidate.score,
                    target_weight=0.0,
                    eligible=False,
                    rejection_reason=candidate.rejection_reason,
                    liquidity_reference_volume=candidate.liquidity_reference_volume,
                    liquidity_as_of=candidate.liquidity_as_of,
                )
            )

        signals.sort(key=lambda signal: (signal.signal_date, signal.ticker))
        portfolio = Portfolio(
            starting_equity=starting_equity,
            ending_equity=cash,
            cash=cash,
            open_positions=(),
        )
        return PortfolioBacktestResult(
            signals=tuple(signals),
            orders=tuple(orders),
            fills=tuple(fills),
            positions=tuple(positions),
            cash_ledger=tuple(cash_ledger),
            costs=tuple(costs),
            daily_snapshots=tuple(snapshots),
            portfolio=portfolio,
            corporate_action_ledger=tuple(corporate_action_ledger),
        )
