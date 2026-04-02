"""TCMB EVDS client for Turkish macro-economic data."""

from __future__ import annotations

from datetime import date, datetime

import httpx

from bist_predict.ingest.types import MacroDataPoint

BASE_URL = "https://evds2.tcmb.gov.tr/service/evds"

INDICATORS: dict[str, tuple[str, str]] = {
    "USD_TRY": ("TP.DK.USD.A.YTL", "TP_DK_USD_A_YTL"),
    "EUR_TRY": ("TP.DK.EUR.A.YTL", "TP_DK_EUR_A_YTL"),
    "GOLD_TRY": ("TP.DK.ALT.A.YTL", "TP_DK_ALT_A_YTL"),
    "POLICY_RATE": ("TP.PO.FAIZ.ON", "TP_PO_FAIZ_ON"),
    "CPI": ("TP.FG.J0", "TP_FG_J0"),
    "BOND_2Y": ("TP.GS.DT02", "TP_GS_DT02"),
}


class TcmbClient:
    """Fetches macro-economic data from TCMB EVDS API."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def fetch(
        self, indicator: str, start_date: date, end_date: date
    ) -> list[MacroDataPoint]:
        """Fetch macro data for an indicator between start_date and end_date."""
        if not self._api_key:
            raise ValueError("TCMB EVDS API key is required. Register free at evds2.tcmb.gov.tr")

        series_code, field_name = INDICATORS[indicator]

        params = {
            "series": series_code,
            "startDate": start_date.strftime("%d-%m-%Y"),
            "endDate": end_date.strftime("%d-%m-%Y"),
            "type": "json",
            "key": self._api_key,
        }

        url = f"{BASE_URL}/series={series_code}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        points: list[MacroDataPoint] = []
        for item in items:
            raw_value = item.get(field_name)
            if raw_value is None or raw_value == "":
                continue

            dt = datetime.strptime(item["Tarih"], "%d-%m-%Y").date()
            point = MacroDataPoint(
                indicator=indicator,
                date=dt,
                value=float(raw_value),
                source="tcmb",
            )
            points.append(point)

        return points
