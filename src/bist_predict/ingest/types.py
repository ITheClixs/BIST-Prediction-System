"""Data types for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV price bar for one ticker on one date."""

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


@dataclass(frozen=True)
class MacroDataPoint:
    """A single macro-economic data point."""

    indicator: str
    date: date
    value: float
    source: str

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


@dataclass(frozen=True)
class SentimentRecord:
    """A single sentiment observation for a ticker."""

    ticker: str
    date: date
    source: str
    headline: str | None
    sentiment_score: float | None
    raw_text: str | None

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


class PriceCollector(Protocol):
    """Protocol for price data collectors."""

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]: ...


class MacroCollector(Protocol):
    """Protocol for macro data collectors."""

    async def fetch(
        self, indicator: str, start_date: date, end_date: date
    ) -> list[MacroDataPoint]: ...


class SentimentCollector(Protocol):
    """Protocol for sentiment data collectors."""

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]: ...
