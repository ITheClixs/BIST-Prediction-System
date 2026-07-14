"""Scheduled provider contract validation tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from bist_predict.ingest.provider_smoke import validate_provider_bars
from bist_predict.ingest.types import OHLCVBar


def _bar(ticker: str = "THYAO") -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=date(2026, 7, 13),
        open=300.0,
        high=305.0,
        low=299.0,
        close=303.0,
        adj_close=303.0,
        volume=1_000_000,
        source="yahoo",
        provider_symbol=f"{ticker}.IS",
        provider_record_id=f"yahoo:{ticker}.IS:2026-07-13",
        source_retrieved_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def test_provider_smoke_requires_nonempty_observed_provenanced_weekday_bars() -> None:
    assert validate_provider_bars([_bar()], ticker="THYAO") == {
        "ticker": "THYAO",
        "row_count": 1,
        "start": "2026-07-13",
        "end": "2026-07-13",
        "sources": ["yahoo"],
    }

    with pytest.raises(ValueError, match="no rows"):
        validate_provider_bars([], ticker="THYAO")
    with pytest.raises(ValueError, match="unexpected ticker"):
        validate_provider_bars([_bar("GARAN")], ticker="THYAO")
