"""Corporate-action records and economic return adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Iterable


class CorporateActionType(str, Enum):
    """Corporate actions relevant to point-in-time equity research."""

    STOCK_SPLIT = "stock_split"
    BONUS_ISSUE = "bonus_issue"
    RIGHTS_ISSUE = "rights_issue"
    CASH_DIVIDEND = "cash_dividend"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"


@dataclass(frozen=True)
class CorporateAction:
    """A sourced corporate action effective on one trading date.

    ``ratio`` is the number of post-action shares per pre-action share for stock
    splits and bonus issues. Rights issues use it as new shares offered per existing
    share and require ``subscription_price``.
    """

    ticker: str
    effective_date: date
    action_type: CorporateActionType
    source: str
    ratio: float | None = None
    cash_amount: float | None = None
    currency: str | None = None
    subscription_price: float | None = None
    new_ticker: str | None = None
    delisting_price: float | None = None
    source_retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.action_type in {
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.BONUS_ISSUE,
        } and (self.ratio is None or self.ratio <= 0):
            raise ValueError(f"{self.action_type.value} requires a positive ratio")

        if self.action_type is CorporateActionType.RIGHTS_ISSUE:
            if self.ratio is None or self.ratio <= 0:
                raise ValueError("rights_issue requires a positive ratio")
            if self.subscription_price is None or self.subscription_price < 0:
                raise ValueError("rights_issue requires a non-negative subscription price")

        if self.action_type is CorporateActionType.CASH_DIVIDEND:
            if self.cash_amount is None or self.cash_amount < 0:
                raise ValueError("cash_dividend requires a non-negative cash amount")
            if not self.currency:
                raise ValueError("cash_dividend requires a currency")

        if self.action_type is CorporateActionType.TICKER_CHANGE:
            if not self.new_ticker:
                raise ValueError("ticker_change requires a new ticker")

        if self.delisting_price is not None and self.delisting_price < 0:
            raise ValueError("delisting price cannot be negative")


def calculate_economic_return(
    *,
    start_price: float,
    end_price: float,
    corporate_actions: Iterable[CorporateAction] = (),
) -> float:
    """Return shareholder wealth after splits, bonus issues, and dividends.

    The caller supplies actions inside the measured interval. Rights issues are
    rejected because their return depends on an explicit exercise policy and cash
    flow treatment; silently assuming one would fabricate performance.
    """

    if start_price <= 0:
        raise ValueError("start price must be positive")
    if end_price < 0:
        raise ValueError("end price cannot be negative")

    shares = 1.0
    cash_distributions = 0.0

    for action in sorted(corporate_actions, key=lambda item: item.effective_date):
        if action.action_type in {
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.BONUS_ISSUE,
        }:
            assert action.ratio is not None
            shares *= action.ratio
        elif action.action_type is CorporateActionType.CASH_DIVIDEND:
            assert action.cash_amount is not None
            cash_distributions += shares * action.cash_amount
        elif action.action_type is CorporateActionType.RIGHTS_ISSUE:
            raise ValueError("rights issue return requires an explicit exercise policy")
        elif action.action_type is CorporateActionType.DELISTING:
            if action.delisting_price is not None:
                end_price = action.delisting_price

    ending_wealth = shares * end_price + cash_distributions
    return ending_wealth / start_price - 1.0
