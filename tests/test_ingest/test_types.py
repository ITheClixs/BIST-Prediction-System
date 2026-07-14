"""Tests for data types and validation."""

from __future__ import annotations

from datetime import UTC, date, datetime

from bist_predict.ingest.types import (
    OHLCVBar,
    MacroDataPoint,
    OpenQuality,
    PriceRepresentation,
    SentimentRecord,
    VolumeQuality,
)


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

    def test_legacy_constructor_exposes_raw_prices_without_inventing_adjustments(
        self,
    ) -> None:
        bar = OHLCVBar(
            "THYAO",
            date(2026, 4, 1),
            310.0,
            315.0,
            308.0,
            312.5,
            311.0,
            1_000_000,
            "legacy_provider",
        )

        assert bar.raw_prices == PriceRepresentation(
            open=310.0,
            high=315.0,
            low=308.0,
            close=312.5,
            volume=1_000_000,
        )
        assert bar.split_adjusted_prices is None
        assert bar.total_return_prices is None

    def test_preserves_explicit_price_representations_and_provider_provenance(
        self,
    ) -> None:
        retrieved_at = datetime(2026, 4, 2, 7, 30, tzinfo=UTC)
        raw = PriceRepresentation(100.0, 104.0, 99.0, 102.0, 20_000)
        split_adjusted = PriceRepresentation(50.0, 52.0, 49.5, 51.0, 40_000)
        total_return = PriceRepresentation(49.8, 51.8, 49.3, 51.4, 40_000)

        bar = OHLCVBar(
            ticker="THYAO",
            date=date(2026, 4, 1),
            open=100.0,
            high=104.0,
            low=99.0,
            close=102.0,
            adj_close=51.4,
            volume=20_000,
            source="provider_a",
            split_adjusted_prices=split_adjusted,
            total_return_prices=total_return,
            open_quality=OpenQuality.OBSERVED,
            volume_quality=VolumeQuality.RECONSTRUCTED,
            provider_symbol="THYAO.IS",
            provider_record_id="provider-a:THYAO:2026-04-01",
            source_retrieved_at=retrieved_at,
        )

        assert bar.raw_prices == raw
        assert bar.split_adjusted_prices is split_adjusted
        assert bar.total_return_prices is total_return
        assert bar.open_quality is OpenQuality.OBSERVED
        assert bar.volume_quality is VolumeQuality.RECONSTRUCTED
        assert bar.provider_symbol == "THYAO.IS"
        assert bar.provider_record_id == "provider-a:THYAO:2026-04-01"
        assert bar.source_retrieved_at == retrieved_at

    def test_quality_defaults_keep_existing_collectors_backward_compatible(
        self,
    ) -> None:
        bar = OHLCVBar(
            "THYAO",
            date(2026, 4, 1),
            310.0,
            315.0,
            308.0,
            312.5,
            312.5,
            1_000_000,
            "isyatirim",
        )

        assert bar.open_quality is OpenQuality.OBSERVED
        assert bar.volume_quality is VolumeQuality.OBSERVED

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
