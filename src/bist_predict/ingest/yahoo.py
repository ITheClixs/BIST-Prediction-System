"""Yahoo Finance fallback client for BIST price data."""

from __future__ import annotations

import asyncio
from datetime import date

import yfinance as yf

from bist_predict.ingest.types import OHLCVBar


class YahooFinanceClient:
    """Fetches historical OHLCV data from Yahoo Finance as a fallback source."""

    def _bist_ticker(self, ticker: str) -> str:
        """Convert a BIST ticker to Yahoo Finance format."""
        return f"{ticker}.IS"

    def fetch_sync(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Synchronous fetch — wraps yfinance which is sync-only."""
        yahoo_ticker = self._bist_ticker(ticker)
        df = yf.download(
            yahoo_ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )

        if df.empty:
            return []

        bars: list[OHLCVBar] = []
        for idx, row in df.iterrows():
            bar = OHLCVBar(
                ticker=ticker,
                date=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                adj_close=float(row["Adj Close"]),
                volume=int(row["Volume"]),
                source="yahoo",
            )
            bars.append(bar)

        return bars

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Async wrapper — runs yfinance in a thread to avoid blocking."""
        return await asyncio.to_thread(self.fetch_sync, ticker, start_date, end_date)
