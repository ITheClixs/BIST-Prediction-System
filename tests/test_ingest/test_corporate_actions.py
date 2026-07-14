"""Tests for economic corporate-action handling."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from bist_predict.ingest.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    calculate_economic_return,
)


def test_two_for_one_split_does_not_create_false_negative_return() -> None:
    split = CorporateAction(
        ticker="THYAO",
        effective_date=date(2026, 6, 1),
        action_type=CorporateActionType.STOCK_SPLIT,
        ratio=2.0,
        source="kap",
        source_retrieved_at=datetime(2026, 6, 1, 6, 0, tzinfo=UTC),
    )

    result = calculate_economic_return(
        start_price=100.0,
        end_price=50.0,
        corporate_actions=[split],
    )

    assert result == pytest.approx(0.0)


def test_cash_dividend_is_represented_and_included_in_economic_return() -> None:
    dividend = CorporateAction(
        ticker="GARAN",
        effective_date=date(2026, 4, 15),
        action_type=CorporateActionType.CASH_DIVIDEND,
        cash_amount=2.5,
        currency="TRY",
        source="kap",
    )

    result = calculate_economic_return(
        start_price=100.0,
        end_price=98.0,
        corporate_actions=[dividend],
    )

    assert dividend.cash_amount == 2.5
    assert dividend.currency == "TRY"
    assert result == pytest.approx(0.005)


def test_ticker_change_preserves_old_and_new_identity() -> None:
    ticker_change = CorporateAction(
        ticker="OLD",
        effective_date=date(2026, 5, 2),
        action_type=CorporateActionType.TICKER_CHANGE,
        new_ticker="NEW",
        source="borsa_istanbul",
    )

    assert ticker_change.ticker == "OLD"
    assert ticker_change.new_ticker == "NEW"


def test_split_requires_a_positive_ratio() -> None:
    with pytest.raises(ValueError, match="positive ratio"):
        CorporateAction(
            ticker="THYAO",
            effective_date=date(2026, 6, 1),
            action_type=CorporateActionType.STOCK_SPLIT,
            source="kap",
        )
