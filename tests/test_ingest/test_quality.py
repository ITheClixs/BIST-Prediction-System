"""Tests for OHLCV data quality validation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.ingest.quality import validate_bar, ValidationError
from bist_predict.ingest.types import OHLCVBar


def make_bar(**overrides) -> OHLCVBar:
    """Helper to create a bar with defaults."""
    defaults = dict(
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
    defaults.update(overrides)
    return OHLCVBar(**defaults)


class TestValidateBar:
    def test_valid_bar_passes(self) -> None:
        bar = make_bar()
        assert validate_bar(bar) is True

    def test_high_below_low_fails(self) -> None:
        bar = make_bar(high=300.0, low=308.0)
        with pytest.raises(ValidationError, match="high .* below low"):
            validate_bar(bar)

    def test_open_above_high_fails(self) -> None:
        bar = make_bar(open=320.0, high=315.0)
        with pytest.raises(ValidationError, match="open .* above high"):
            validate_bar(bar)

    def test_close_above_high_fails(self) -> None:
        bar = make_bar(close=320.0, high=315.0)
        with pytest.raises(ValidationError, match="close .* above high"):
            validate_bar(bar)

    def test_open_below_low_fails(self) -> None:
        bar = make_bar(open=305.0, low=308.0)
        with pytest.raises(ValidationError, match="open .* below low"):
            validate_bar(bar)

    def test_close_below_low_fails(self) -> None:
        bar = make_bar(close=305.0, low=308.0)
        with pytest.raises(ValidationError, match="close .* below low"):
            validate_bar(bar)

    def test_negative_volume_fails(self) -> None:
        bar = make_bar(volume=-100)
        with pytest.raises(ValidationError, match="volume"):
            validate_bar(bar)

    def test_zero_volume_passes(self) -> None:
        bar = make_bar(volume=0)
        assert validate_bar(bar) is True

    def test_negative_price_fails(self) -> None:
        bar = make_bar(close=-1.0, low=-2.0, open=-1.5, high=0.5)
        with pytest.raises(ValidationError, match="negative"):
            validate_bar(bar)
