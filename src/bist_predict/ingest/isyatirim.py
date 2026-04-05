"""Is Yatirim API client for BIST historical price data."""

from __future__ import annotations

from datetime import date, datetime

import httpx

from bist_predict.ingest.types import OHLCVBar

BASE_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/"
    "Common/Data.aspx/HisseTekil"
)

HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": (
        "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/"
        "Sayfalar/Tarihsel-Fiyat-Bilgileri.aspx"
    ),
}


class IsYatirimClient:
    """Fetches historical OHLCV data from Is Yatirim's public API."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a ticker between start_date and end_date."""
        params = {
            "hisse": ticker,
            "startdate": start_date.strftime("%d-%m-%Y"),
            "enddate": end_date.strftime("%d-%m-%Y"),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(BASE_URL, params=params, headers=HEADERS)
            response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            desc = data.get("errorDescription", "unknown error")
            raise ValueError(f"IsYatirim API error: {desc}")

        rows = data.get("value", [])

        bars: list[OHLCVBar] = []
        for row in rows:
            dt = datetime.strptime(row["HGDG_TARIH"], "%d-%m-%Y").date()
            close = float(row["HGDG_KAPANIS"])
            high = float(row["HGDG_MAX"])
            low = float(row["HGDG_MIN"])
            # API does not provide an open price; use weighted average as proxy
            open_price = float(row["HGDG_AOF"])
            # Volume is in TRY; derive approximate lot volume from TRY volume / avg price
            volume_try = float(row["HGDG_HACIM"])
            avg_price = float(row["HGDG_AOF"])
            volume_lots = int(volume_try / avg_price) if avg_price > 0 else 0

            bar = OHLCVBar(
                ticker=ticker,
                date=dt,
                open=open_price,
                high=high,
                low=low,
                close=close,
                adj_close=close,
                volume=volume_lots,
                source="isyatirim",
            )
            bars.append(bar)

        return bars
