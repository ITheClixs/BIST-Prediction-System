"""Tests for Is Yatirim API client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.isyatirim import BASE_URL, IsYatirimClient
from bist_predict.ingest.types import OHLCVBar


SAMPLE_RESPONSE = {
    "ok": True,
    "errorCode": None,
    "errorDescription": None,
    "value": [
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "01-04-2026",
            "HGDG_KAPANIS": 312.5,
            "HGDG_MAX": 315.0,
            "HGDG_MIN": 308.0,
            "HGDG_AOF": 310.0,
            "HGDG_HACIM": 310000000.0,
        },
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "31-03-2026",
            "HGDG_KAPANIS": 310.0,
            "HGDG_MAX": 311.0,
            "HGDG_MIN": 303.0,
            "HGDG_AOF": 305.0,
            "HGDG_HACIM": 274500000.0,
        },
    ],
}


class TestIsYatirimClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_ohlcv_bars(self) -> None:
        respx.get(BASE_URL).mock(
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
        respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        apr1 = bars[0]
        assert apr1.open == 310.0  # AOF (weighted average) used as open proxy
        assert apr1.high == 315.0
        assert apr1.low == 308.0
        assert apr1.close == 312.5
        assert apr1.volume == 1_000_000  # 310M TRY / 310 AOF

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self) -> None:
        respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json={"ok": True, "value": []})
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert bars == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_http_error_raises(self) -> None:
        respx.get(BASE_URL).mock(
            return_value=httpx.Response(500)
        )

        client = IsYatirimClient()
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_api_error_raises(self) -> None:
        respx.get(BASE_URL).mock(
            return_value=httpx.Response(200, json={
                "ok": False,
                "errorDescription": "Invalid ticker",
                "value": [],
            })
        )

        client = IsYatirimClient()
        with pytest.raises(ValueError, match="IsYatirim API error"):
            await client.fetch("INVALID", date(2026, 3, 31), date(2026, 4, 1))
