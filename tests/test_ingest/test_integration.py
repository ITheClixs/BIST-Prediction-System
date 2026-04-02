"""Integration test for the full fetch pipeline using mocked HTTP."""

from __future__ import annotations

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


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_fetch_store_query_cycle(self, db: Database, config: Config) -> None:
        """Test: fetch → validate → store → query roundtrip."""
        bars = [
            OHLCVBar("THYAO", date(2026, 4, 1), 310.0, 315.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),
            OHLCVBar("THYAO", date(2026, 3, 31), 305.0, 311.0, 303.0, 310.0, 310.0, 900_000, "isyatirim"),
            OHLCVBar("GARAN", date(2026, 4, 1), 85.0, 87.0, 84.5, 86.5, 86.5, 5_000_000, "isyatirim"),
        ]

        mock_primary = AsyncMock(return_value=bars)
        scheduler = IngestionScheduler(
            db=db, config=config,
            price_primary=mock_primary,
        )

        fetched = await scheduler.fetch_prices("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert len(fetched) == 3

        stored = await scheduler.store_prices(fetched)
        assert stored == 3

        latest = db.get_latest_date("THYAO")
        assert latest == "2026-04-01"

        latest_garan = db.get_latest_date("GARAN")
        assert latest_garan == "2026-04-01"

        stored_again = await scheduler.store_prices(fetched)
        assert stored_again == 0

    @pytest.mark.asyncio
    async def test_invalid_bars_filtered_out(self, db: Database, config: Config) -> None:
        """Invalid bars should be skipped, valid ones stored."""
        bars = [
            OHLCVBar("THYAO", date(2026, 4, 1), 310.0, 315.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),
            OHLCVBar("BAD", date(2026, 4, 1), 310.0, 300.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),
        ]

        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_prices(bars)
        assert stored == 1

    @pytest.mark.asyncio
    async def test_macro_and_sentiment_storage(self, db: Database, config: Config) -> None:
        """Test macro and sentiment data storage."""
        scheduler = IngestionScheduler(db=db, config=config)

        macros = [
            MacroDataPoint("USD_TRY", date(2026, 4, 1), 38.45, "tcmb"),
            MacroDataPoint("EUR_TRY", date(2026, 4, 1), 41.20, "tcmb"),
        ]
        stored_macro = await scheduler.store_macro(macros)
        assert stored_macro == 2

        sentiments = [
            SentimentRecord("THYAO", date(2026, 4, 1), "google_news", "THY yükseldi", None, "text"),
        ]
        stored_sent = await scheduler.store_sentiment(sentiments)
        assert stored_sent == 1

        with db.connect() as conn:
            macro_count = conn.execute("SELECT COUNT(*) FROM macro_data").fetchone()[0]
            sent_count = conn.execute("SELECT COUNT(*) FROM sentiment_data").fetchone()[0]
            assert macro_count == 2
            assert sent_count == 1
