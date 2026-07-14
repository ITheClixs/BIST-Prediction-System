"""Ingestion scheduler — orchestrates data collection from all sources."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import date
from typing import Any, Callable, Coroutine, Sequence

from bist_predict.config import Config
from bist_predict.ingest.quality import ValidationError, validate_bar
from bist_predict.ingest.reconciliation import (
    ReconciliationReport,
    reconcile_price_bars,
)
from bist_predict.ingest.types import (
    MacroDataPoint,
    OHLCVBar,
    PriceRepresentation,
    SentimentRecord,
)
from bist_predict.storage.database import Database

logger = logging.getLogger(__name__)

PriceFetcher = Callable[[str, date, date], Coroutine[Any, Any, list[OHLCVBar]]]


class IngestionScheduler:
    """Orchestrates data fetching from all sources with fallback and storage."""

    def __init__(
        self,
        db: Database,
        config: Config,
        price_primary: PriceFetcher | None = None,
        price_fallback: PriceFetcher | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._price_primary = price_primary
        self._price_fallback = price_fallback
        self._last_reconciliation = ReconciliationReport()

    @property
    def last_reconciliation(self) -> ReconciliationReport:
        """Return diagnostics from the most recent provider merge."""
        return self._last_reconciliation

    async def fetch_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch both providers and use fallback observations to fill partial gaps."""
        primary_bars: list[OHLCVBar] = []
        if self._price_primary is not None:
            try:
                primary_bars = await self._price_primary(ticker, start_date, end_date)
            except Exception as e:
                logger.warning("Primary source failed for %s: %s", ticker, e)

        fallback_bars: list[OHLCVBar] = []
        if self._price_fallback is not None:
            try:
                fallback_bars = await self._price_fallback(ticker, start_date, end_date)
            except Exception as e:
                logger.warning("Fallback source failed for %s: %s", ticker, e)

        bars, report = reconcile_price_bars(primary_bars, fallback_bars)
        self._last_reconciliation = report
        return bars

    async def store_prices(self, bars: Sequence[OHLCVBar]) -> int:
        """Validate and store price bars. Returns count of newly stored bars."""
        stored = 0
        with self._db.connect() as conn:
            for bar in bars:
                try:
                    validate_bar(bar)
                except ValidationError as e:
                    logger.warning("Skipping invalid bar: %s", e)
                    continue

                try:
                    conn.execute(
                        """INSERT INTO raw_prices
                           (ticker, date, open, high, low, close, adj_close, volume,
                            source, open_quality, volume_quality, provider_symbol,
                            provider_record_id, source_retrieved_at,
                            split_adj_open, split_adj_high, split_adj_low,
                            split_adj_close, split_adj_volume,
                            total_return_open, total_return_high, total_return_low,
                            total_return_close, total_return_volume)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                   ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            bar.ticker, bar.date_str, bar.open, bar.high,
                            bar.low, bar.close, bar.adj_close, bar.volume, bar.source,
                            bar.open_quality.value,
                            bar.volume_quality.value,
                            bar.provider_symbol,
                            bar.provider_record_id,
                            (
                                bar.source_retrieved_at.isoformat()
                                if bar.source_retrieved_at
                                else None
                            ),
                            *self._representation_values(bar.split_adjusted_prices),
                            *self._representation_values(bar.total_return_prices),
                        ),
                    )
                    stored += 1
                except sqlite3.IntegrityError:
                    logger.debug("Duplicate bar skipped: %s %s", bar.ticker, bar.date_str)

            conn.commit()
        return stored

    @staticmethod
    def _representation_values(
        representation: PriceRepresentation | None,
    ) -> tuple[object, ...]:
        if representation is None:
            return (None, None, None, None, None)
        return (
            representation.open,
            representation.high,
            representation.low,
            representation.close,
            representation.volume,
        )

    async def store_macro(self, points: Sequence[MacroDataPoint]) -> int:
        """Store macro data points. Returns count of newly stored points."""
        stored = 0
        with self._db.connect() as conn:
            for point in points:
                try:
                    conn.execute(
                        """INSERT INTO macro_data (indicator, date, value, source)
                           VALUES (?, ?, ?, ?)""",
                        (point.indicator, point.date_str, point.value, point.source),
                    )
                    stored += 1
                except sqlite3.IntegrityError:
                    logger.debug(
                        "Duplicate macro point skipped: %s %s", point.indicator, point.date_str
                    )

            conn.commit()
        return stored

    async def store_sentiment(self, records: Sequence[SentimentRecord]) -> int:
        """Store sentiment records. Returns count of newly stored records."""
        stored = 0
        with self._db.connect() as conn:
            for record in records:
                conn.execute(
                    """INSERT INTO sentiment_data
                       (ticker, date, source, headline, sentiment_score, raw_text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        record.ticker, record.date_str, record.source,
                        record.headline, record.sentiment_score, record.raw_text,
                    ),
                )
                stored += 1

            conn.commit()
        return stored
