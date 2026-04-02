"""Is Yatirim API client for BIST historical price data."""

from __future__ import annotations

from datetime import date, datetime

import httpx

from bist_predict.ingest.types import OHLCVBar

BASE_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/"
    "Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/"
    "HisseSenetleriGecmisVeriler"
)


class IsYatirimClient:
    """Fetches historical OHLCV data from Is Yatirim's public API."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a ticker between start_date and end_date."""
        params = {
            "hession": ticker,
            "startdate": start_date.strftime("%d-%m-%Y"),
            "enddate": end_date.strftime("%d-%m-%Y"),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()

        data = response.json()
        rows = data.get("value", [])

        bars: list[OHLCVBar] = []
        for row in rows:
            dt = datetime.fromisoformat(row["HGDG_TARIH"]).date()
            bar = OHLCVBar(
                ticker=ticker,
                date=dt,
                open=float(row["HGDG_ACILIS"]),
                high=float(row["HGDG_BIRINCISI"]),
                low=float(row["HGDG_SONUNCUSU"]),
                close=float(row["HGDG_KAPANIS"]),
                adj_close=float(row["HGDG_KAPANIS"]),
                volume=int(row["HGDG_HACIMLOT"]),
                source="isyatirim",
            )
            bars.append(bar)

        return bars
