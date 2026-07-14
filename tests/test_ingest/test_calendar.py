"""Official-session calendar validation."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from bist_predict.ingest.calendar import (
    OfficialTradingCalendar,
    borsa_istanbul_equity_calendar,
)
from bist_predict.ingest.types import OHLCVBar


def _bar(session: date) -> OHLCVBar:
    return OHLCVBar(
        ticker="THYAO",
        date=session,
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        adj_close=101.0,
        volume=1_000_000,
        source="test",
    )


def _calendar() -> OfficialTradingCalendar:
    return OfficialTradingCalendar(
        index_name="XIST",
        sessions=(date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)),
        source="https://www.borsaistanbul.com/en/market-data/official-holidays",
        source_retrieved_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


def test_calendar_reports_missing_unexpected_and_duplicate_sessions() -> None:
    calendar = _calendar()
    bars = [
        _bar(date(2026, 4, 1)),
        _bar(date(2026, 4, 1)),
        _bar(date(2026, 4, 3)),
        _bar(date(2026, 4, 4)),
    ]

    report = calendar.validate_bars(bars)

    assert report.missing_expected_sessions == ("2026-04-02",)
    assert report.duplicate_sessions == ("2026-04-01",)
    assert report.unexpected_sessions == ("2026-04-04",)
    assert report.unexpected_weekend_rows == ("2026-04-04",)
    assert report.timezone == "Europe/Istanbul"


def test_calendar_exposes_timezone_aware_session_boundaries() -> None:
    session_open, session_close = _calendar().session_bounds(date(2026, 4, 1))

    assert session_open.isoformat() == "2026-04-01T10:00:00+03:00"
    assert session_close.isoformat() == "2026-04-01T18:00:00+03:00"


def test_calendar_refuses_weekends_in_official_snapshot() -> None:
    with pytest.raises(ValueError, match="weekend"):
        OfficialTradingCalendar(
            index_name="XIST",
            sessions=(date(2026, 4, 4),),
            source="official",
            source_retrieved_at=datetime(2026, 3, 1, tzinfo=UTC),
        )


def test_borsa_calendar_uses_official_full_and_half_day_schedule() -> None:
    calendar = borsa_istanbul_equity_calendar(
        date(2025, 6, 4),
        date(2026, 3, 20),
        source_retrieved_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert date(2025, 6, 6) not in calendar.sessions
    assert date(2026, 3, 20) not in calendar.sessions
    assert calendar.session_bounds(date(2025, 6, 5))[1].timetz().replace(tzinfo=None) == time(13, 0)
    assert calendar.session_bounds(date(2026, 3, 19))[1].timetz().replace(tzinfo=None) == time(
        13, 0
    )
    assert "borsaistanbul.com/en/official-holidays" in calendar.source
