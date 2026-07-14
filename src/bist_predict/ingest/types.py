"""Data types for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Protocol


class OpenQuality(str, Enum):
    """Whether an opening price is directly tradable research data."""

    OBSERVED = "observed"
    PROXY = "proxy"
    MISSING = "missing"


class VolumeQuality(str, Enum):
    """How the reported trading volume was obtained."""

    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    MISSING = "missing"


@dataclass(frozen=True)
class PriceRepresentation:
    """OHLCV values under one explicit price-adjustment convention."""

    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class OHLCVBar:
    """A provider price record with explicit adjustment and quality metadata.

    The original scalar fields remain the raw, tradable representation so existing
    collectors and storage call sites remain compatible. Adjusted representations
    are optional because they must not be inferred from equal-looking price values.
    """

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str
    split_adjusted_prices: PriceRepresentation | None = None
    total_return_prices: PriceRepresentation | None = None
    open_quality: OpenQuality = OpenQuality.OBSERVED
    volume_quality: VolumeQuality = VolumeQuality.OBSERVED
    provider_symbol: str | None = None
    provider_record_id: str | None = None
    source_retrieved_at: datetime | None = None
    raw_prices: PriceRepresentation = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_prices",
            PriceRepresentation(
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
            ),
        )

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

    async def fetch(self, ticker: str, start_date: date, end_date: date) -> list[OHLCVBar]: ...


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
