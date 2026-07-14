"""Immutable official trading-session snapshots and data validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from bist_predict.ingest.types import OHLCVBar

BORSA_OFFICIAL_HOLIDAY_SOURCE = "https://www.borsaistanbul.com/en/official-holidays"

_FULL_DAY_CLOSURES = frozenset(
    {
        date(2025, 1, 1),
        date(2025, 3, 31),
        date(2025, 4, 1),
        date(2025, 4, 23),
        date(2025, 5, 1),
        date(2025, 5, 19),
        date(2025, 6, 6),
        date(2025, 6, 9),
        date(2025, 7, 15),
        date(2025, 10, 29),
        date(2026, 1, 1),
        date(2026, 3, 20),
        date(2026, 4, 23),
        date(2026, 5, 1),
        date(2026, 5, 19),
        date(2026, 5, 27),
        date(2026, 5, 28),
        date(2026, 5, 29),
        date(2026, 7, 15),
        date(2026, 10, 29),
    }
)
_HALF_DAY_CLOSES = MappingProxyType(
    {
        date(2025, 6, 5): time(13, 0),
        date(2025, 10, 28): time(13, 0),
        date(2026, 3, 19): time(13, 0),
        date(2026, 5, 26): time(13, 0),
        date(2026, 10, 28): time(13, 0),
    }
)


@dataclass(frozen=True)
class CalendarValidation:
    missing_expected_sessions: tuple[str, ...]
    unexpected_sessions: tuple[str, ...]
    unexpected_weekend_rows: tuple[str, ...]
    duplicate_sessions: tuple[str, ...]
    timezone: str


@dataclass(frozen=True)
class OfficialTradingCalendar:
    """A sourced Borsa Istanbul session snapshot used without date inference."""

    index_name: str
    sessions: tuple[date, ...]
    source: str
    source_retrieved_at: datetime
    timezone: str = "Europe/Istanbul"
    session_close_overrides: Mapping[date, time] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sessions = tuple(self.sessions)
        if len(set(sessions)) != len(sessions):
            raise ValueError("official calendar contains duplicate sessions")
        if any(session.weekday() >= 5 for session in sessions):
            raise ValueError("official calendar cannot contain a weekend session")
        if self.source_retrieved_at.tzinfo is None:
            raise ValueError("source_retrieved_at must be timezone-aware")
        if not self.source.strip():
            raise ValueError("official calendar requires a source")
        session_set = set(sessions)
        close_overrides = dict(self.session_close_overrides)
        unknown_overrides = set(close_overrides) - session_set
        if unknown_overrides:
            unknown = ", ".join(session.isoformat() for session in sorted(unknown_overrides))
            raise ValueError(f"session close overrides reference non-sessions: {unknown}")
        if any(close <= time(10, 0) for close in close_overrides.values()):
            raise ValueError("session close overrides must be after the session open")
        object.__setattr__(self, "sessions", tuple(sorted(sessions)))
        object.__setattr__(self, "session_close_overrides", MappingProxyType(close_overrides))

    def validate_bars(self, bars: Sequence[OHLCVBar]) -> CalendarValidation:
        expected = set(self.sessions)
        observed = {bar.date for bar in bars}
        key_counts = Counter((bar.ticker, bar.date) for bar in bars)
        duplicate_dates = {session for (_, session), count in key_counts.items() if count > 1}
        unexpected = observed - expected
        return CalendarValidation(
            missing_expected_sessions=tuple(
                session.isoformat() for session in sorted(expected - observed)
            ),
            unexpected_sessions=tuple(session.isoformat() for session in sorted(unexpected)),
            unexpected_weekend_rows=tuple(
                session.isoformat() for session in sorted(unexpected) if session.weekday() >= 5
            ),
            duplicate_sessions=tuple(session.isoformat() for session in sorted(duplicate_dates)),
            timezone=self.timezone,
        )

    def session_bounds(self, session: date) -> tuple[datetime, datetime]:
        if session not in set(self.sessions):
            raise ValueError(f"date is not an official session: {session}")
        timezone = ZoneInfo(self.timezone)
        session_close = self.session_close_overrides.get(session, time(18, 0))
        return (
            datetime.combine(session, time(10, 0), tzinfo=timezone),
            datetime.combine(session, session_close, tzinfo=timezone),
        )


def borsa_istanbul_equity_calendar(
    start: date,
    end: date,
    *,
    source_retrieved_at: datetime,
) -> OfficialTradingCalendar:
    """Build the checked-in 2025-2026 Borsa Istanbul equity schedule.

    The finite supported range is deliberate: dates are sourced from Borsa
    Istanbul's published holiday table rather than inferred from weekdays.
    """
    if start > end:
        raise ValueError("calendar start must not be after end")
    if start.year < 2025 or end.year > 2026:
        raise ValueError("official Borsa calendar snapshot supports 2025-2026 only")

    sessions: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in _FULL_DAY_CLOSURES:
            sessions.append(current)
        current += timedelta(days=1)

    session_set = set(sessions)
    return OfficialTradingCalendar(
        index_name="XIST",
        sessions=tuple(sessions),
        source=BORSA_OFFICIAL_HOLIDAY_SOURCE,
        source_retrieved_at=source_retrieved_at,
        session_close_overrides={
            session: session_close
            for session, session_close in _HALF_DAY_CLOSES.items()
            if session in session_set
        },
    )
