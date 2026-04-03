"""Tests for Is Yatirim API client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.isyatirim import IsYatirimClient
from bist_predict.ingest.types import OHLCVBar


SAMPLE_RESPONSE = {
    "value": [
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "2026-04-01T00:00:00",
            "HGDG_ACILIS": 310.0,
            "HGDG_KAPANIS": 312.5,
            "HGDG_BIRINCISI": 315.0,
            "HGDG_SONUNCUSU": 308.0,
            "HGDG_HACIMLOT": 1000000,
        },
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "2026-03-31T00:00:00",
            "HGDG_ACILIS": 305.0,
            "HGDG_KAPANIS": 310.0,
            "HGDG_BIRINCISI": 311.0,
            "HGDG_SONUNCUSU": 303.0,
            "HGDG_HACIMLOT": 900000,
        },
    ]
}


class TestIsYatirimClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_ohlcv_bars(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(bars) == 2
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].ticker == "THYAO"
        assert bars[0].source == "isyatirim"

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_parses_prices_correctly(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        apr1 = bars[0]
        assert apr1.open == 310.0
        assert apr1.high == 315.0
        assert apr1.low == 308.0
        assert apr1.close == 312.5
        assert apr1.volume == 1_000_000

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json={"value": []})
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert bars == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_http_error_raises(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(500)
        )

        client = IsYatirimClient()
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
