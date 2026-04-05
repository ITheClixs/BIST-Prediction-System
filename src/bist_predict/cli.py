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
@click.option("--ticker", default=None, help="Compute features for a single ticker")
@click.option("--date", "target_date", default=None, help="Target date (YYYY-MM-DD), defaults to latest")
def features(ticker: str | None, target_date: str | None) -> None:
    """Compute features for latest data."""
    from bist_predict.features.engine import FeatureEngine

    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    engine = FeatureEngine(db)

    if target_date is None:
        target_date = date.today().isoformat()

    tickers = [ticker] if ticker else BIST_100_SAMPLE

    total_features = 0
    for t in tickers:
        latest = db.get_latest_date(t)
        if latest is None:
            click.echo(f"  {t}: no price data, skipping")
            continue

        click.echo(f"  {t}: computing features for {target_date}...")
        feats = engine.compute_and_store(t, target_date)
        total_features += len(feats)
        click.echo(f"    → {len(feats)} features computed")

    click.echo(f"\nTotal: {total_features} features computed and stored.")


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


@main.command()
@click.option("--ticker", default=None, help="Train for a single ticker")
def train(ticker: str | None) -> None:
    """Train or retrain prediction models."""
    from bist_predict.models.xgboost_model import XGBoostModel
    from bist_predict.models.lightgbm_model import LightGBMModel
    from bist_predict.models.registry import ModelRegistry
    from bist_predict.models.types import build_tabular_dataset

    import numpy as np

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    registry = ModelRegistry(db)

    tickers = [ticker] if ticker else BIST_100_SAMPLE
    all_X, all_y_dir, all_y_pct = [], [], []

    for t in tickers:
        X, y_dir, y_pct, _ = build_tabular_dataset(db, t)
        if X.shape[0] > 0:
            all_X.append(X)
            all_y_dir.append(y_dir)
            all_y_pct.append(y_pct)
            click.echo(f"  {t}: {X.shape[0]} samples, {X.shape[1]} features")

    if not all_X:
        click.echo("No training data available. Run 'fetch' and 'features' first.")
        return

    X = np.vstack(all_X)
    y_dir = np.concatenate(all_y_dir)
    y_pct = np.concatenate(all_y_pct)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
    y_pct_train, y_pct_val = y_pct[:split], y_pct[split:]

    click.echo(f"\nTraining on {split} samples, validating on {len(X) - split}...")

    for ModelClass in [XGBoostModel, LightGBMModel]:
        model = ModelClass()
        click.echo(f"\n  Training {model.name}...")
        metrics = model.train(X_train, y_dir_train, y_pct_train, X_val, y_dir_val, y_pct_val)
        click.echo(f"    Accuracy: {metrics.get('val_accuracy', 0):.3f}")
        click.echo(f"    MAE: {metrics.get('val_mae', 0):.5f}")

        version = date.today().isoformat()
        model_path = str(config.db_path.parent / "models" / model.name / version)
        model.save(model_path)
        registry.register(model.name, version, model_path, metrics)
        registry.activate(model.name, version)
        click.echo(f"    Saved and activated: {model.name} {version}")

    click.echo("\nTraining complete.")


@main.command()
@click.option("--ticker", default=None, help="Get signal for a single ticker")
@click.option("--detail", is_flag=True, help="Show detailed signal breakdown")
def signals(ticker: str | None, detail: bool) -> None:
    """Get today's trading signals."""
    from bist_predict.models.xgboost_model import XGBoostModel
    from bist_predict.models.lightgbm_model import LightGBMModel
    from bist_predict.models.registry import ModelRegistry
    from bist_predict.models.types import Prediction, build_tabular_dataset

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    registry = ModelRegistry(db)

    tickers = [ticker] if ticker else BIST_100_SAMPLE
    predictions: list[Prediction] = []

    for ModelClass in [XGBoostModel, LightGBMModel]:
        model = ModelClass()
        active = registry.get_active(model.name)
        if active is None:
            click.echo(f"  No active {model.name} model. Run 'train' first.")
            continue
        model.load(active["model_path"])

        for t in tickers:
            X, _, _, dates = build_tabular_dataset(db, t)
            if X.shape[0] == 0:
                continue
            latest_X = X[-1:].copy()
            probs, pct = model.predict(latest_X)
            direction = "UP" if probs[0] > 0.5 else "DOWN"
            confidence = probs[0] if direction == "UP" else 1 - probs[0]
            predictions.append(Prediction(
                ticker=t, direction=direction, confidence=float(confidence),
                predicted_pct_move=float(pct[0]), model_name=model.name,
            ))

    for tier in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        tier_preds = [p for p in predictions if p.signal_tier == tier]
        if tier_preds:
            click.echo(f"\n{'=' * 40}")
            click.echo(f"  {tier}")
            click.echo(f"{'=' * 40}")
            for p in sorted(tier_preds, key=lambda x: -x.confidence):
                click.echo(f"  {p.ticker:8s} {p.confidence:5.1%} conf  {p.predicted_pct_move:+.2f}% target  ({p.model_name})")

    if not predictions:
        click.echo("No signals. Run 'train' first.")


@main.command()
def backtest() -> None:
    """Run walk-forward backtest."""
    click.echo("Backtesting not yet wired -- models + evaluation complete, integration pending.")


@main.command()
@click.option("--ticker", default=None, help="Show accuracy for a single ticker")
def accuracy(ticker: str | None) -> None:
    """Show prediction accuracy history."""
    from bist_predict.evaluation.tracker import AccuracyTracker

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    tracker = AccuracyTracker(db)

    tickers = [ticker] if ticker else BIST_100_SAMPLE[:5]

    for t in tickers:
        acc_30 = tracker.rolling_accuracy(t, window=30)
        acc_90 = tracker.rolling_accuracy(t, window=90)
        click.echo(f"  {t}: 30d={acc_30:.1%}  90d={acc_90:.1%}")

    if ticker:
        buckets = tracker.confidence_buckets(ticker)
        if buckets:
            click.echo(f"\nConfidence Bucket Analysis for {ticker}:")
            for label, data in sorted(buckets.items()):
                click.echo(f"  {label}%: {data['accuracy']:.1%} accuracy ({int(data['count'])} predictions)")
