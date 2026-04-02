"""Tests for TCMB EVDS macro data client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.tcmb import TcmbClient, INDICATORS
from bist_predict.ingest.types import MacroDataPoint


SAMPLE_EVDS_RESPONSE = {
    "items": [
        {"Tarih": "01-04-2026", "TP_DK_USD_A_YTL": "38.4500"},
        {"Tarih": "31-03-2026", "TP_DK_USD_A_YTL": "38.3200"},
    ],
    "totalCount": 2,
}


class TestTcmbClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_macro_points(self) -> None:
        respx.get("https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A.YTL").mock(
            return_value=httpx.Response(200, json=SAMPLE_EVDS_RESPONSE)
        )

        client = TcmbClient(api_key="test-key")
        points = await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))

        assert len(points) == 2
        assert all(isinstance(p, MacroDataPoint) for p in points)
        assert points[0].indicator == "USD_TRY"
        assert points[0].source == "tcmb"
        assert points[0].value == 38.45

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self) -> None:
        respx.get("https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A.YTL").mock(
            return_value=httpx.Response(200, json={"items": [], "totalCount": 0})
        )

        client = TcmbClient(api_key="test-key")
        points = await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))
        assert points == []

    def test_indicators_mapping_exists(self) -> None:
        assert "USD_TRY" in INDICATORS
        assert "EUR_TRY" in INDICATORS
        assert "GOLD_TRY" in INDICATORS
        assert "POLICY_RATE" in INDICATORS

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_missing_api_key_raises(self) -> None:
        client = TcmbClient(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))
