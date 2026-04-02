"""OHLCV data quality validation rules."""

from __future__ import annotations

from bist_predict.ingest.types import OHLCVBar


class ValidationError(Exception):
    """Raised when a data quality check fails."""


def validate_bar(bar: OHLCVBar) -> bool:
    """Validate an OHLCV bar. Returns True if valid, raises ValidationError otherwise."""
    for field_name, value in [("open", bar.open), ("high", bar.high), ("low", bar.low), ("close", bar.close)]:
        if value < 0:
            raise ValidationError(f"{bar.ticker} {bar.date_str}: {field_name} is negative ({value})")

    if bar.high < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: high ({bar.high}) below low ({bar.low})"
        )

    if bar.open > bar.high:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: open ({bar.open}) above high ({bar.high})"
        )
    if bar.open < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: open ({bar.open}) below low ({bar.low})"
        )

    if bar.close > bar.high:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: close ({bar.close}) above high ({bar.high})"
        )
    if bar.close < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: close ({bar.close}) below low ({bar.low})"
        )

    if bar.volume < 0:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: volume is negative ({bar.volume})"
        )

    return True
