"""Transaction-cost-aware one-session portfolio backtest with full ledgers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields
from datetime import date
from typing import Any, Iterable

import pandas as pd

from bist_predict.ingest.corporate_actions import CorporateAction
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


@dataclass(frozen=True)
class StrategyConfig:
    """Long-only top-k constraints and declared decision-time cost assumption."""

    top_k: int = 10
    decision_cost_rate: float = 0.003
    max_participation: float = 0.01
    min_trade_value: float = 100.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.decision_cost_rate < 0.0:
            raise ValueError("decision_cost_rate must be non-negative")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must lie in (0, 1]")
        if self.min_trade_value < 0.0:
            raise ValueError("min_trade_value must be non-negative")


@dataclass(frozen=True)
class Signal:
    signal_id: str
    prediction_id: str
    signal_date: str
    execution_date: str | None
    ticker: str
    predicted_return: float
    predicted_probability: float
    uncertainty_adjusted_return: float
    target_weight: float
    eligible: bool
    rejection_reason: str | None


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
class PortfolioBacktestResult:
    signals: tuple[Signal, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    positions: tuple[Position, ...]
    cash_ledger: tuple[CashLedger, ...]
    costs: tuple[CostRecord, ...]
    daily_snapshots: tuple[DailySnapshot, ...]
    portfolio: Portfolio

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
        }


@dataclass(frozen=True)
class _Candidate:
    prediction: pd.Series
    bar: OHLCVBar | None
    execution_date: str | None
    score: float
    rejection_reason: str | None


class PortfolioBacktester:
    """Execute cost-aware top-k signals at observed next-session opens."""

    def __init__(self, *, strategy: StrategyConfig, costs: CostModel) -> None:
        self._strategy = strategy
        self._costs = costs

    @staticmethod
    def _next_bar(
        bars_by_ticker: dict[str, list[OHLCVBar]], ticker: str, signal_date: date
    ) -> OHLCVBar | None:
        return next(
            (bar for bar in bars_by_ticker.get(ticker, ()) if bar.date > signal_date),
            None,
        )

    def _candidate(
        self,
        prediction: pd.Series,
        bars_by_ticker: dict[str, list[OHLCVBar]],
    ) -> _Candidate:
        signal_date = date.fromisoformat(str(prediction["date"]))
        bar = self._next_bar(bars_by_ticker, str(prediction["ticker"]), signal_date)
        certainty = 2.0 * abs(float(prediction["predicted_probability"]) - 0.5)
        score = (
            float(prediction["predicted_return"]) * certainty - self._strategy.decision_cost_rate
        )
        reason: str | None = None
        if bar is None:
            reason = "missing_execution_price"
        elif bar.open_quality is OpenQuality.PROXY:
            reason = "proxy_open"
        elif bar.open_quality is not OpenQuality.OBSERVED or bar.open <= 0.0:
            reason = "missing_open"
        elif bar.volume_quality is VolumeQuality.MISSING or bar.volume <= 0:
            reason = "missing_volume"
        elif score <= 0.0:
            reason = "non_positive_expected_net_return"
        return _Candidate(
            prediction=prediction,
            bar=bar,
            execution_date=bar.date.isoformat() if bar is not None else None,
            score=score,
            rejection_reason=reason,
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

    def run(
        self,
        predictions: pd.DataFrame,
        prices: Iterable[OHLCVBar],
        *,
        model_name: str,
        starting_equity: float,
        corporate_actions: Iterable[CorporateAction] = (),
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
        action_dates = {action.effective_date for action in corporate_actions}

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
        cash = starting_equity

        for execution_date in sorted(by_execution):
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
                    uncertainty_adjusted_return=candidate.score,
                    target_weight=target_weight if reason is None else 0.0,
                    eligible=reason is None,
                    rejection_reason=reason,
                )
                signals.append(signal)
                candidate_signals[prediction_key] = signal

            day_starting_equity = cash
            day_gross_pnl = 0.0
            day_distributions = 0.0
            day_costs = 0.0
            day_notional = 0.0
            entry_notionals: list[float] = []
            # Corporate actions effective at the open are processed before entry.
            # The benchmark carries no overnight position, so these events cannot
            # create a distribution or alter an existing holding.
            if date.fromisoformat(execution_date) in action_dates:
                day_distributions += 0.0

            for candidate in selected:
                assert candidate.bar is not None
                bar = candidate.bar
                prediction_key = (
                    str(candidate.prediction["date"]),
                    str(candidate.prediction["ticker"]),
                )
                signal = candidate_signals[prediction_key]
                fixed_cost_reserve = 1.0 + self._strategy.decision_cost_rate / 2.0
                allocation = day_starting_equity * target_weight / fixed_cost_reserve
                quantity_by_cash = math.floor(allocation / bar.open)
                quantity_by_volume = math.floor(bar.volume * self._strategy.max_participation)
                quantity = min(quantity_by_cash, quantity_by_volume)
                buy_order_id = _identifier(signal.signal_id, "buy", execution_date)
                open_timestamp = f"{execution_date}T10:00:00+03:00"
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
                    )
                )
                buy_notional = quantity * bar.open
                participation = quantity / bar.volume
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

                close_timestamp = f"{execution_date}T18:00:00+03:00"
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
                    uncertainty_adjusted_return=candidate.score,
                    target_weight=0.0,
                    eligible=False,
                    rejection_reason=candidate.rejection_reason,
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
        )
