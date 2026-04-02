"""Tests for data types and validation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.ingest.types import OHLCVBar, MacroDataPoint, SentimentRecord


class TestOHLCVBar:
    def test_create_valid_bar(self) -> None:
        bar = OHLCVBar(
            ticker="THYAO",
            date=date(2026, 4, 1),
            open=310.0,
            high=315.0,
            low=308.0,
            close=312.5,
            adj_close=312.5,
            volume=1_000_000,
            source="isyatirim",
        )
        assert bar.ticker == "THYAO"
        assert bar.close == 312.5

    def test_date_str(self) -> None:
        bar = OHLCVBar(
            ticker="THYAO",
            date=date(2026, 4, 1),
            open=310.0,
            high=315.0,
            low=308.0,
            close=312.5,
            adj_close=312.5,
            volume=1_000_000,
            source="isyatirim",
        )
        assert bar.date_str == "2026-04-01"


class TestMacroDataPoint:
    def test_create_macro_point(self) -> None:
        point = MacroDataPoint(
            indicator="USD_TRY",
            date=date(2026, 4, 1),
            value=38.45,
            source="tcmb",
        )
        assert point.indicator == "USD_TRY"
        assert point.value == 38.45


class TestSentimentRecord:
    def test_create_sentiment_record(self) -> None:
        record = SentimentRecord(
            ticker="THYAO",
            date=date(2026, 4, 1),
            source="google_news",
            headline="THY hisseleri yükseldi",
            sentiment_score=0.72,
            raw_text="THY hisseleri yükseldi",
        )
        assert record.sentiment_score == 0.72
        assert record.source == "google_news"
