"""CLI entry point for bist-predict."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import click

from bist_predict.config import load_config
from bist_predict.ingest.isyatirim import IsYatirimClient
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.sentiment import GoogleNewsSentiment, TurkishFinanceRSS
from bist_predict.ingest.tcmb import TcmbClient, INDICATORS
from bist_predict.ingest.yahoo import YahooFinanceClient
from bist_predict.storage.database import Database

BIST_100_SAMPLE = [
    "THYAO", "GARAN", "AKBNK", "EREGL", "SISE", "TUPRS", "TCELL", "TOASO",
    "VESTL", "SAHOL", "KCHOL", "HEKTS", "BIMAS", "ASELS", "SASA", "KOZAL",
    "PETKM", "DOHOL", "FROTO", "ENKAI", "ARCLK", "ISCTR", "YKBNK", "VAKBN",
    "HALKB", "TAVHL", "TTKOM", "EKGYO", "PGSUS", "MGROS",
]


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """BIST-100 Stock Market Prediction System."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.command()
@click.option("--days", default=30, help="Number of days of history to fetch")
@click.option("--ticker", default=None, help="Fetch a single ticker instead of all BIST-100")
def fetch(days: int, ticker: str | None) -> None:
    """Fetch latest market data from all sources."""
    asyncio.run(_fetch(days, ticker))


async def _fetch(days: int, ticker: str | None) -> None:
    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    is_client = IsYatirimClient()
    yahoo_client = YahooFinanceClient()

    scheduler = IngestionScheduler(
        db=db,
        config=config,
        price_primary=is_client.fetch,
        price_fallback=yahoo_client.fetch,
    )

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    tickers = [ticker] if ticker else BIST_100_SAMPLE

    total_bars = 0
    for t in tickers:
        latest = db.get_latest_date(t)
        fetch_start = start_date
        if latest:
            fetch_start = max(start_date, date.fromisoformat(latest) + timedelta(days=1))
            if fetch_start > end_date:
                click.echo(f"  {t}: up to date")
                continue

        click.echo(f"  {t}: fetching {fetch_start} → {end_date}...")
        bars = await scheduler.fetch_prices(t, fetch_start, end_date)
        stored = await scheduler.store_prices(bars)
        total_bars += stored

        await asyncio.sleep(config.data.rate_limit_delay)

    click.echo(f"\nStored {total_bars} new price bars.")

    if config.data.tcmb_api_key:
        tcmb = TcmbClient(api_key=config.data.tcmb_api_key)
        total_macro = 0
        for indicator in INDICATORS:
            click.echo(f"  Macro: {indicator}...")
            try:
                points = await tcmb.fetch(indicator, start_date, end_date)
                stored = await scheduler.store_macro(points)
                total_macro += stored
            except Exception as e:
                click.echo(f"    Warning: {e}")
        click.echo(f"Stored {total_macro} new macro data points.")
    else:
        click.echo("Skipping macro data (no TCMB API key in config.toml)")

    google_news = GoogleNewsSentiment()
    total_sentiment = 0
    for t in tickers[:10]:
        click.echo(f"  Sentiment: {t}...")
        records = await google_news.fetch(t, start_date, end_date)
        stored = await scheduler.store_sentiment(records)
        total_sentiment += stored
        await asyncio.sleep(config.data.rate_limit_delay)

    click.echo(f"Stored {total_sentiment} new sentiment records.")
    click.echo("\nFetch complete.")


@main.command()
def stocks() -> None:
    """List tracked BIST-100 stocks."""
    click.echo("BIST-100 Tracked Stocks:")
    click.echo("=" * 40)
    for i, ticker in enumerate(BIST_100_SAMPLE, 1):
        click.echo(f"  {i:3d}. {ticker}")
    click.echo(f"\nTotal: {len(BIST_100_SAMPLE)} stocks")


@main.command()
def config() -> None:
    """Show current configuration."""
    cfg = load_config()
    click.echo("Current Configuration:")
    click.echo("=" * 40)
    click.echo(f"  Database: {cfg.db_path}")
    click.echo(f"  TCMB API key: {'set' if cfg.data.tcmb_api_key else 'not set'}")
    click.echo(f"  Fetch retries: {cfg.data.fetch_retries}")
    click.echo(f"  Rate limit delay: {cfg.data.rate_limit_delay}s")
    click.echo(f"  Min confidence: {cfg.signals.min_confidence}")
    click.echo(f"  Backtest commission: {cfg.backtest.commission}")
    click.echo(f"  Backtest slippage: {cfg.backtest.slippage}")
