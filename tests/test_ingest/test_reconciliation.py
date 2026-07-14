"""Provider overlap comparison and partial-gap repair."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.ingest.reconciliation import reconcile_price_bars
from bist_predict.ingest.types import OHLCVBar


def _bar(source: str, session: date, *, open_price: float, close: float) -> OHLCVBar:
    return OHLCVBar(
        ticker="THYAO",
        date=session,
        open=open_price,
        high=max(open_price, close) + 1.0,
        low=min(open_price, close) - 1.0,
        close=close,
        adj_close=close,
        volume=1_000_000,
        source=source,
    )


def test_reconciliation_uses_fallback_to_repair_partial_intervals() -> None:
    primary = [
        _bar("primary", date(2026, 4, 1), open_price=100.0, close=101.0),
        _bar("primary", date(2026, 4, 3), open_price=102.0, close=103.0),
    ]
    fallback = [
        _bar("fallback", date(2026, 4, 1), open_price=100.1, close=101.1),
        _bar("fallback", date(2026, 4, 2), open_price=101.0, close=102.0),
        _bar("fallback", date(2026, 4, 3), open_price=102.1, close=103.1),
    ]

    bars, report = reconcile_price_bars(primary, fallback)

    assert [(bar.date, bar.source) for bar in bars] == [
        (date(2026, 4, 1), "primary"),
        (date(2026, 4, 2), "fallback"),
        (date(2026, 4, 3), "primary"),
    ]
    assert report.missing_from_primary == ("2026-04-02",)
    assert report.fallback_fill_count == 1
    assert len(report.overlaps) == 2


def test_reconciliation_reports_price_return_volume_and_adjustment_deviations() -> None:
    primary = [
        _bar("primary", date(2026, 4, 1), open_price=100.0, close=101.0),
        _bar("primary", date(2026, 4, 2), open_price=101.0, close=103.0),
    ]
    fallback = [
        _bar("fallback", date(2026, 4, 1), open_price=99.0, close=100.0),
        _bar("fallback", date(2026, 4, 2), open_price=100.0, close=104.0),
    ]

    _, report = reconcile_price_bars(primary, fallback)
    comparison = report.overlaps[1]

    assert comparison.timestamp_aligned is True
    assert comparison.open_absolute_difference == pytest.approx(1.0)
    assert comparison.close_absolute_difference == pytest.approx(1.0)
    assert comparison.return_difference is not None
    assert comparison.volume_absolute_difference == 0
    assert comparison.corporate_action_consistent is True
