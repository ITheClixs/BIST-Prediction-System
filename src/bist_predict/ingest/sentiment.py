"""Sentiment data collectors — Google News RSS, Turkish finance RSS, etc."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from bist_predict.ingest.types import SentimentRecord

logger = logging.getLogger(__name__)


class GoogleNewsSentiment:
    """Fetches news headlines from Google News RSS for sentiment analysis."""

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]:
        """Fetch news headlines for a BIST ticker from Google News RSS."""
        query = f"{ticker} borsa hisse"
        url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": "tr", "gl": "TR", "ceid": "TR:tr"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Google News fetch failed for %s: %s", ticker, e)
            return []

        feed = feedparser.parse(response.text)
        records: list[SentimentRecord] = []

        for entry in feed.entries:
            pub_date = _parse_rss_date(entry.get("published", ""))
            if pub_date is None:
                continue

            if not (start_date <= pub_date <= end_date):
                continue

            record = SentimentRecord(
                ticker=ticker,
                date=pub_date,
                source="google_news",
                headline=entry.get("title"),
                sentiment_score=None,
                raw_text=entry.get("description"),
            )
            records.append(record)

        return records


class TurkishFinanceRSS:
    """Fetches headlines from Turkish financial news RSS feeds."""

    FEEDS = [
        ("https://www.bloomberght.com/rss", "bloomberght"),
        ("https://bigpara.hurriyet.com.tr/rss/", "bigpara"),
    ]

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]:
        """Fetch headlines mentioning the ticker from Turkish finance RSS feeds."""
        records: list[SentimentRecord] = []
        ticker_lower = ticker.lower()

        for feed_url, source_name in self.FEEDS:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(feed_url)
                    response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("RSS fetch failed for %s from %s: %s", ticker, source_name, e)
                continue

            feed = feedparser.parse(response.text)
            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("description", "")
                combined = f"{title} {description}".lower()

                if ticker_lower not in combined:
                    continue

                pub_date = _parse_rss_date(entry.get("published", ""))
                if pub_date is None:
                    continue

                if not (start_date <= pub_date <= end_date):
                    continue

                record = SentimentRecord(
                    ticker=ticker,
                    date=pub_date,
                    source=source_name,
                    headline=title,
                    sentiment_score=None,
                    raw_text=description,
                )
                records.append(record)

        return records


def _parse_rss_date(date_str: str) -> date | None:
    """Parse an RSS pubDate string to a date. Returns None on failure."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).date()
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(date_str).date()
    except (ValueError, TypeError):
        return None
