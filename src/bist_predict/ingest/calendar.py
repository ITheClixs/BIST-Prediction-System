"""Immutable official trading-session snapshots and data validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Sequence
from zoneinfo import ZoneInfo

from bist_predict.ingest.types import OHLCVBar


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
        object.__setattr__(self, "sessions", tuple(sorted(sessions)))

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
            unexpected_sessions=tuple(
                session.isoformat() for session in sorted(unexpected)
            ),
            unexpected_weekend_rows=tuple(
                session.isoformat()
                for session in sorted(unexpected)
                if session.weekday() >= 5
            ),
            duplicate_sessions=tuple(
                session.isoformat() for session in sorted(duplicate_dates)
            ),
            timezone=self.timezone,
        )

    def session_bounds(self, session: date) -> tuple[datetime, datetime]:
        if session not in set(self.sessions):
            raise ValueError(f"date is not an official session: {session}")
        timezone = ZoneInfo(self.timezone)
        return (
            datetime.combine(session, time(10, 0), tzinfo=timezone),
            datetime.combine(session, time(18, 0), tzinfo=timezone),
        )
