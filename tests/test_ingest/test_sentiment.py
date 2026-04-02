"""Tests for sentiment data collectors."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.sentiment import GoogleNewsSentiment
from bist_predict.ingest.types import SentimentRecord

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>THYAO - Google News</title>
    <item>
      <title>THY hisseleri güçlü yükseldi</title>
      <pubDate>Tue, 01 Apr 2026 10:00:00 GMT</pubDate>
      <description>Türk Hava Yolları hisseleri bugün güçlü yükseliş gösterdi.</description>
    </item>
    <item>
      <title>THYAO bilançosu beklentilerin üzerinde</title>
      <pubDate>Mon, 31 Mar 2026 14:00:00 GMT</pubDate>
      <description>THY bilançosu beklentilerin üzerinde geldi.</description>
    </item>
  </channel>
</rss>"""


class TestGoogleNewsSentiment:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_sentiment_records(self) -> None:
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(200, text=SAMPLE_RSS)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(records) == 2
        assert all(isinstance(r, SentimentRecord) for r in records)
        assert records[0].ticker == "THYAO"
        assert records[0].source == "google_news"
        assert records[0].headline is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_feed(self) -> None:
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><title>Empty</title></channel></rss>"""
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(200, text=empty_rss)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert records == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_http_error_returns_empty(self) -> None:
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(429)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert records == []
