"""Tests for the data ingestion scheduler."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bist_predict.config import Config, DataConfig
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.types import OHLCVBar, MacroDataPoint, SentimentRecord
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


def _make_bar(ticker: str = "THYAO", d: date = date(2026, 4, 1)) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker, date=d, open=310.0, high=315.0,
        low=308.0, close=312.5, adj_close=312.5,
        volume=1_000_000, source="isyatirim",
    )


def _make_macro(d: date = date(2026, 4, 1)) -> MacroDataPoint:
    return MacroDataPoint(indicator="USD_TRY", date=d, value=38.45, source="tcmb")


def _make_sentiment(ticker: str = "THYAO", d: date = date(2026, 4, 1)) -> SentimentRecord:
    return SentimentRecord(
        ticker=ticker, date=d, source="google_news",
        headline="Test headline", sentiment_score=None, raw_text="Test text",
    )


class TestIngestionScheduler:
    @pytest.mark.asyncio
    async def test_store_price_bars(self, db: Database, config: Config) -> None:
        mock_primary = AsyncMock(return_value=[_make_bar()])
        mock_fallback = AsyncMock(return_value=[])

        scheduler = IngestionScheduler(
            db=db,
            config=config,
            price_primary=mock_primary,
            price_fallback=mock_fallback,
        )
        stored = await scheduler.store_prices([_make_bar()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT ticker, close FROM raw_prices").fetchone()
            assert row[0] == "THYAO"
            assert row[1] == 312.5

    @pytest.mark.asyncio
    async def test_store_macro_data(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_macro([_make_macro()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT indicator, value FROM macro_data").fetchone()
            assert row[0] == "USD_TRY"
            assert row[1] == 38.45

    @pytest.mark.asyncio
    async def test_store_sentiment(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_sentiment([_make_sentiment()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT ticker, source FROM sentiment_data").fetchone()
            assert row[0] == "THYAO"
            assert row[1] == "google_news"

    @pytest.mark.asyncio
    async def test_fetch_with_fallback(self, db: Database, config: Config) -> None:
        mock_primary = AsyncMock(side_effect=Exception("API down"))
        mock_fallback = AsyncMock(return_value=[_make_bar(d=date(2026, 4, 1))])

        scheduler = IngestionScheduler(
            db=db,
            config=config,
            price_primary=mock_primary,
            price_fallback=mock_fallback,
        )
        bars = await scheduler.fetch_prices("THYAO", date(2026, 4, 1), date(2026, 4, 1))

        assert len(bars) == 1
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_bars_ignored(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        bar = _make_bar()
        await scheduler.store_prices([bar])
        stored = await scheduler.store_prices([bar])  # duplicate
        assert stored == 0

        with db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
            assert count == 1
