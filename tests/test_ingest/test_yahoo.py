"""Tests for Yahoo Finance fallback client."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from bist_predict.ingest.yahoo import YahooFinanceClient
from bist_predict.ingest.types import OHLCVBar


class TestYahooFinanceClient:
    def test_ticker_suffix(self) -> None:
        client = YahooFinanceClient()
        assert client._bist_ticker("THYAO") == "THYAO.IS"
        assert client._bist_ticker("GARAN") == "GARAN.IS"

    @patch("bist_predict.ingest.yahoo.yf.download")
    def test_fetch_returns_ohlcv_bars(self, mock_download: MagicMock) -> None:
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "Open": [310.0, 305.0],
                "High": [315.0, 311.0],
                "Low": [308.0, 303.0],
                "Close": [312.5, 310.0],
                "Adj Close": [312.5, 310.0],
                "Volume": [1_000_000, 900_000],
            },
            index=pd.DatetimeIndex([
                pd.Timestamp("2026-04-01"),
                pd.Timestamp("2026-03-31"),
            ], name="Date"),
        )
        mock_download.return_value = mock_df

        client = YahooFinanceClient()
        bars = client.fetch_sync("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(bars) == 2
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].ticker == "THYAO"
        assert bars[0].source == "yahoo"

    @patch("bist_predict.ingest.yahoo.yf.download")
    def test_fetch_empty_dataframe(self, mock_download: MagicMock) -> None:
        import pandas as pd

        mock_download.return_value = pd.DataFrame()

        client = YahooFinanceClient()
        bars = client.fetch_sync("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert bars == []
