"""Ingestion scheduler — orchestrates data collection from all sources."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import date
from typing import Any, Callable, Coroutine, Sequence

from bist_predict.config import Config
from bist_predict.ingest.quality import ValidationError, validate_bar
from bist_predict.ingest.types import MacroDataPoint, OHLCVBar, SentimentRecord
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

    async def fetch_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch price data with fallback. Tries primary source first."""
        if self._price_primary is not None:
            try:
                bars = await self._price_primary(ticker, start_date, end_date)
                if bars:
                    return bars
            except Exception as e:
                logger.warning("Primary source failed for %s: %s", ticker, e)

        if self._price_fallback is not None:
            try:
                return await self._price_fallback(ticker, start_date, end_date)
            except Exception as e:
                logger.warning("Fallback source failed for %s: %s", ticker, e)

        return []

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
                           (ticker, date, open, high, low, close, adj_close, volume, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            bar.ticker, bar.date_str, bar.open, bar.high,
                            bar.low, bar.close, bar.adj_close, bar.volume, bar.source,
                        ),
                    )
                    stored += 1
                except sqlite3.IntegrityError:
                    logger.debug("Duplicate bar skipped: %s %s", bar.ticker, bar.date_str)

            conn.commit()
        return stored

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
