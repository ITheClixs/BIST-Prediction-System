"""CLI entry point for bist-predict."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import click

from bist_predict.config import load_config
from bist_predict.ingest.isyatirim import IsYatirimClient
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.sentiment import GoogleNewsSentiment
from bist_predict.ingest.tcmb import TcmbClient, INDICATORS
from bist_predict.ingest.yahoo import YahooFinanceClient
from bist_predict.storage.database import Database


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Forecasting research for a fixed BIST large-cap prototype universe."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _resolve_tickers(db: Database, ticker: str | None) -> list[str]:
    """Resolve command ticker scope from DB state."""
    if ticker:
        db.upsert_tracked_stock(ticker, source="manual")
        return [ticker]
    return db.list_tracked_stocks()


def _get_price_dates(db: Database, ticker: str) -> list[str]:
    """Return all stored raw price dates for a ticker."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT date FROM raw_prices
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()
    return [row[0] for row in rows]


def _get_stored_feature_dates(db: Database, ticker: str) -> set[str]:
    """Return raw feature dates already stored for a ticker."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT date FROM features
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()
    return {row[0] for row in rows}


def _get_feature_dates_to_compute(db: Database, ticker: str, target_date: str | None) -> list[str]:
    """Return feature dates to compute for a ticker.

    If target_date is omitted, backfill all missing dates from raw_prices so newly
    fetched tickers become trainable without a manual per-date loop.
    """
    if target_date is not None:
        return [target_date]

    price_dates = _get_price_dates(db, ticker)
    if not price_dates:
        return []

    stored_feature_dates = _get_stored_feature_dates(db, ticker)
    return [d for d in price_dates if d not in stored_feature_dates]


@main.command()
@click.option("--days", default=30, help="Number of days of history to fetch")
@click.option("--ticker", default=None, help="Fetch one ticker instead of the configured universe")
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

    tickers = _resolve_tickers(db, ticker)

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
    """List tracked stocks from the persistent DB universe."""
    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    tickers = db.list_tracked_stocks(active_only=False)

    click.echo("Tracked Stocks:")
    click.echo("=" * 40)
    for i, ticker in enumerate(tickers, 1):
        latest = db.get_latest_date(ticker)
        status = latest if latest else "no price data"
        click.echo(f"  {i:3d}. {ticker:8s} latest={status}")
    click.echo(f"\nTotal: {len(tickers)} stocks")


@main.command()
@click.option("--ticker", default=None, help="Compute features for a single ticker")
@click.option(
    "--date", "target_date", default=None, help="Target date (YYYY-MM-DD), defaults to latest"
)
def features(ticker: str | None, target_date: str | None) -> None:
    """Compute features for latest data."""
    _compute_features(ticker, target_date)


def _compute_features(ticker: str | None, target_date: str | None) -> None:
    """Compute features for one date or backfill missing history."""
    from bist_predict.features.engine import FeatureEngine

    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    engine = FeatureEngine(db)
    tickers = _resolve_tickers(db, ticker)

    total_features = 0
    for t in tickers:
        price_dates = _get_price_dates(db, t)
        if not price_dates:
            click.echo(f"  {t}: no price data, skipping")
            continue

        dates_to_compute = _get_feature_dates_to_compute(db, t, target_date)
        if not dates_to_compute:
            click.echo(f"  {t}: features up to date")
            continue

        if len(dates_to_compute) == 1:
            click.echo(f"  {t}: computing features for {dates_to_compute[0]}...")
        else:
            click.echo(
                f"  {t}: computing features for {len(dates_to_compute)} dates "
                f"({dates_to_compute[0]} -> {dates_to_compute[-1]})..."
            )

        ticker_feature_count = 0
        for feature_date in dates_to_compute:
            feats = engine.compute_and_store(t, feature_date)
            ticker_feature_count += len(feats)

        total_features += ticker_feature_count
        click.echo(f"    → {ticker_feature_count} features computed")

    click.echo(f"\nTotal: {total_features} features computed and stored.")


@main.command()
def config() -> None:
    """Show current configuration."""
    cfg = load_config()
    click.echo("Current Configuration:")
    click.echo("=" * 40)
    click.echo(f"  Database: {cfg.db_path}")
    click.echo(f"  Experiment scope: {cfg.research.experiment_scope}")
    click.echo(f"  Accepted models: {','.join(cfg.research.accepted_models)}")
    click.echo(f"  TCMB API key: {'set' if cfg.data.tcmb_api_key else 'not set'}")
    click.echo(f"  Fetch retries: {cfg.data.fetch_retries}")
    click.echo(f"  Rate limit delay: {cfg.data.rate_limit_delay}s")
    click.echo(f"  Min confidence: {cfg.signals.min_confidence}")
    click.echo(f"  Active models: {cfg.models.active_models}")
    click.echo(f"  Include neural: {cfg.models.include_neural}")
    click.echo(f"  Sequence length: {cfg.models.seq_len}")
    click.echo(f"  Validation fraction: {cfg.models.validation_fraction}")
    click.echo(f"  Backtest commission: {cfg.backtest.commission}")
    click.echo(f"  Backtest slippage: {cfg.backtest.slippage}")


@main.command("reproduce-smoke")
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=Path("runs"),
    show_default=True,
)
def reproduce_smoke(runs_root: Path) -> None:
    """Run the deterministic synthetic methodology smoke benchmark."""
    from bist_predict.research.accepted_benchmark import (
        AcceptedBenchmarkConfig,
        generate_synthetic_prices,
        run_accepted_benchmark,
    )

    bundle = run_accepted_benchmark(
        generate_synthetic_prices(),
        runs_root=runs_root,
        config=AcceptedBenchmarkConfig.synthetic_smoke(),
        command="make reproduce-smoke",
    )
    click.echo(f"Created {bundle.run_id}")
    click.echo(f"Artifacts: {bundle.path}")
    click.echo("Scope: synthetic methodology smoke; not market-performance evidence.")


@main.command()
@click.option(
    "--prices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Provider-quality input_prices.parquet with observed opens and provenance.",
)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=Path("runs"),
    show_default=True,
)
@click.option(
    "--corporate-actions",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Sourced corporate_actions.parquet covering the input interval.",
)
@click.option(
    "--corporate-action-coverage",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Sourced per-ticker action-query coverage for the input interval.",
)
@click.option(
    "--prediction-store",
    type=click.Path(path_type=Path),
    default=Path("prediction_tracking"),
    show_default=True,
    help="Create-only store for actionable signal-time prediction records.",
)
def benchmark(
    prices: Path,
    runs_root: Path,
    corporate_actions: Path,
    corporate_action_coverage: Path,
    prediction_store: Path,
) -> None:
    """Run the accepted fixed-universe baseline benchmark on explicit inputs."""
    from bist_predict.research.accepted_benchmark import (
        AcceptedBenchmarkConfig,
        load_corporate_action_coverage_artifact,
        load_corporate_action_artifact,
        load_price_artifact,
        run_accepted_benchmark,
    )
    from bist_predict.research.prediction_tracking import (
        ImmutablePredictionStore,
        persist_run_signal_predictions,
    )

    bundle = run_accepted_benchmark(
        load_price_artifact(prices),
        corporate_actions=load_corporate_action_artifact(corporate_actions),
        corporate_action_coverage=load_corporate_action_coverage_artifact(
            corporate_action_coverage
        ),
        runs_root=runs_root,
        config=AcceptedBenchmarkConfig(),
        command=(
            f"bist-predict benchmark --prices {prices} "
            f"--corporate-actions {corporate_actions} "
            f"--corporate-action-coverage {corporate_action_coverage}"
        ),
    )
    tracked = persist_run_signal_predictions(
        bundle.path,
        ImmutablePredictionStore(prediction_store),
    )
    click.echo(f"Created {bundle.run_id}")
    click.echo(f"Artifacts: {bundle.path}")
    click.echo(f"Persisted {len(tracked)} actionable prediction records: {prediction_store}")


@main.command()
@click.argument("run_id")
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("runs"),
    show_default=True,
)
def reproduce(run_id: str, runs_root: Path) -> None:
    """Replay bundled inputs and require exact scientific artifact hashes."""
    from bist_predict.research.accepted_benchmark import reproduce_run

    run_path = runs_root / run_id
    if not run_path.is_dir():
        raise click.ClickException(f"run does not exist: {run_path}")
    with tempfile.TemporaryDirectory(prefix="bist-reproduce-") as scratch:
        failures = reproduce_run(run_path, scratch_root=Path(scratch))
    if failures:
        details = ", ".join(f"{name}={reason}" for name, reason in failures.items())
        raise click.ClickException(f"artifact reproduction failed: {details}")
    click.echo(
        f"Reproduced {run_id}: all scientific artifact hashes match; "
        "original environment provenance remains immutable."
    )


@main.command()
@click.option("--ticker", default=None, help="Train for a single ticker")
@click.option("--models", "model_names", default=None, help="Comma-separated base models")
@click.option("--include-neural", is_flag=True, help="Also train LSTM and Transformer")
@click.option("--seq-len", default=None, type=int, help="Sequence length for neural models")
def train(
    ticker: str | None,
    model_names: str | None,
    include_neural: bool,
    seq_len: int | None,
) -> None:
    """EXPERIMENTAL: train the unaccepted legacy ensemble path."""
    _train_models(ticker, model_names, include_neural, seq_len)


def _train_models(
    ticker: str | None,
    model_names: str | None = None,
    include_neural: bool = False,
    seq_len: int | None = None,
) -> None:
    """Train and activate models for the selected tickers."""
    from bist_predict.models.factory import parse_model_names
    from bist_predict.research.ensemble_pipeline import (
        EnsembleTrainConfig,
        train_calibrated_ensemble,
    )

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    tickers = _resolve_tickers(db, ticker)

    selected_models = parse_model_names(model_names or config.models.active_models)
    neural_enabled = include_neural or config.models.include_neural
    training_config = EnsembleTrainConfig(
        model_names=selected_models,
        include_neural=neural_enabled,
        seq_len=seq_len or config.models.seq_len,
        validation_fraction=config.models.validation_fraction,
        min_confidence=config.signals.min_confidence,
    )

    click.echo(f"Training calibrated ensemble with: {', '.join(selected_models)}")
    if neural_enabled:
        click.echo(f"  Neural models enabled with seq_len={training_config.seq_len}")

    try:
        report = train_calibrated_ensemble(db, training_config, tickers=tickers)
    except ValueError as e:
        click.echo(f"No ensemble trained: {e}")
        click.echo("Run 'fetch' and 'features' first, or reduce the validation/data requirements.")
        return

    click.echo(f"\nTraining complete: ensemble {report.version}")
    click.echo(f"  Base models: {', '.join(report.base_models)}")
    click.echo(f"  Validation samples: {report.validation_samples}")
    click.echo(f"  Calibration: {report.calibration_status}")
    click.echo(f"  Accuracy: {report.metrics['accuracy']:.3f}")
    click.echo(f"  MAE: {report.metrics['mae']:.5f}")


@main.command()
@click.option("--ticker", default=None, help="Get signal for a single ticker")
@click.option("--detail", is_flag=True, help="Show detailed signal breakdown")
def signals(ticker: str | None, detail: bool) -> None:
    """EXPERIMENTAL: emit unaccepted legacy model signals."""
    _generate_signals(ticker, detail)


def _generate_signals(ticker: str | None, detail: bool) -> None:
    """Load active models and emit predictions."""
    import json

    from bist_predict.models.calibration import PlattCalibrator
    from bist_predict.models.ensemble import EnsembleCombiner
    from bist_predict.models.factory import create_model, parse_model_names
    from bist_predict.models.registry import ModelRegistry
    from bist_predict.models.types import (
        Prediction,
        build_inference_row,
        build_sequence_inference_row,
    )
    from bist_predict.research.ensemble_pipeline import SEQUENCE_MODELS

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    registry = ModelRegistry(db)

    tickers = _resolve_tickers(db, ticker)
    predictions: list[Prediction] = []

    active_ensemble = registry.get_active("ensemble")
    if active_ensemble is not None:
        try:
            ensemble_path = active_ensemble["model_path"]
            with open(f"{ensemble_path}/ensemble_metadata.json") as f:
                metadata = json.load(f)
            combiner = EnsembleCombiner()
            combiner.load(ensemble_path)
            calibrator = PlattCalibrator()
            calibrator.load(ensemble_path)
            base_models = metadata["base_models"]
            seq_len = int(metadata["seq_len"])

            for t in tickers:
                model_predictions = {}
                for model_name in base_models:
                    active = registry.get_active(model_name)
                    if active is None:
                        continue
                    model = create_model(model_name)
                    model.load(active["model_path"])
                    if model_name in SEQUENCE_MODELS:
                        inference = build_sequence_inference_row(db, t, seq_len=seq_len)
                    else:
                        inference = build_inference_row(db, t)
                    if inference is None:
                        continue
                    latest_X, _ = inference
                    expected_features = getattr(model, "n_features", None)
                    actual_features = latest_X.shape[-1]
                    if expected_features is not None and actual_features != expected_features:
                        continue
                    model_predictions[model_name] = model.predict(latest_X)

                if len(model_predictions) < 2:
                    continue
                probs, pct = combiner.predict(model_predictions)
                if calibrator.is_fitted:
                    probs = calibrator.transform(probs)
                direction = "UP" if probs[0] > 0.5 else "DOWN"
                confidence = probs[0] if direction == "UP" else 1 - probs[0]
                predictions.append(
                    Prediction(
                        ticker=t,
                        direction=direction,
                        confidence=float(confidence),
                        predicted_pct_move=float(pct[0]),
                        model_name="ensemble",
                    )
                )
        except Exception as e:
            raise click.ClickException(f"registered ensemble is unusable: {e}") from e

    if not predictions:
        base_models = parse_model_names(config.models.active_models)
        for model_name in base_models:
            model = create_model(model_name)
            if model_name in SEQUENCE_MODELS:
                continue
            active = registry.get_active(model.name)
            if active is None:
                click.echo(f"  No active {model.name} model. Run 'train' first.")
                continue
            model.load(active["model_path"])

            expected_features = model.n_features

            for t in tickers:
                inference = build_inference_row(db, t)
                if inference is None:
                    continue
                latest_X, _ = inference
                if expected_features is not None and latest_X.shape[1] != expected_features:
                    continue
                probs, pct = model.predict(latest_X)
                direction = "UP" if probs[0] > 0.5 else "DOWN"
                confidence = probs[0] if direction == "UP" else 1 - probs[0]
                predictions.append(
                    Prediction(
                        ticker=t,
                        direction=direction,
                        confidence=float(confidence),
                        predicted_pct_move=float(pct[0]),
                        model_name=model.name,
                    )
                )

    _print_predictions(predictions, detail)


def _print_predictions(predictions: list[object], detail: bool) -> None:
    """Print grouped predictions."""
    for tier in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        tier_preds = [p for p in predictions if p.signal_tier == tier]
        if tier_preds:
            click.echo(f"\n{'=' * 40}")
            click.echo(f"  {tier}")
            click.echo(f"{'=' * 40}")
            for p in sorted(tier_preds, key=lambda x: -x.confidence):
                click.echo(
                    f"  {p.ticker:8s} {p.confidence:5.1%} conf  {p.predicted_pct_move:+.2f}% target  ({p.model_name})"
                )

    if detail:
        hold_preds = [p for p in predictions if p.signal_tier == "HOLD"]
        if hold_preds:
            click.echo(f"\n{'=' * 40}")
            click.echo("  HOLD")
            click.echo(f"{'=' * 40}")
            for p in sorted(hold_preds, key=lambda x: -x.confidence):
                click.echo(
                    f"  {p.ticker:8s} {p.confidence:5.1%} conf  {p.predicted_pct_move:+.2f}% target  ({p.model_name})"
                )

    if not predictions:
        click.echo("No signals. Run 'train' first.")


@main.command()
@click.option("--days", default=365, help="Number of days of history to fetch before training")
@click.option("--ticker", default=None, help="Run the pipeline for a single ticker")
@click.option("--detail", is_flag=True, help="Show detailed signal breakdown")
def pipeline(days: int, ticker: str | None, detail: bool) -> None:
    """EXPERIMENTAL: run the unaccepted legacy live-model path."""
    click.echo("Step 1/4: Fetching latest data...")
    asyncio.run(_fetch(days, ticker))

    click.echo("\nStep 2/4: Computing feature history...")
    _compute_features(ticker, target_date=None)

    click.echo("\nStep 3/4: Training models...")
    _train_models(ticker)

    click.echo("\nStep 4/4: Generating signals...")
    _generate_signals(ticker, detail)


@main.command()
@click.option("--ticker", default=None, help="Backtest a single ticker")
@click.option("--models", "model_names", default=None, help="Comma-separated base models")
@click.option("--include-neural", is_flag=True, help="Also include LSTM and Transformer")
@click.option("--seq-len", default=None, type=int, help="Sequence length for neural models")
@click.option("--train-window", default=252, help="Training window in samples")
@click.option("--val-window", default=63, help="Validation window in samples")
@click.option("--step-size", default=21, help="Walk-forward step size")
def backtest(
    ticker: str | None,
    model_names: str | None,
    include_neural: bool,
    seq_len: int | None,
    train_window: int,
    val_window: int,
    step_size: int,
) -> None:
    """EXPERIMENTAL: run the unaccepted legacy ensemble backtest."""
    from bist_predict.models.factory import parse_model_names
    from bist_predict.research.ensemble_pipeline import (
        EnsembleBacktestConfig,
        run_walk_forward_backtest,
    )

    config = load_config()
    db = Database(config.db_path)
    db.initialize()
    tickers = _resolve_tickers(db, ticker)
    selected_models = parse_model_names(model_names or config.models.active_models)
    backtest_config = EnsembleBacktestConfig(
        model_names=selected_models,
        include_neural=include_neural or config.models.include_neural,
        seq_len=seq_len or config.models.seq_len,
        validation_fraction=config.models.validation_fraction,
        min_confidence=config.signals.min_confidence,
        train_window=train_window,
        val_window=val_window,
        step_size=step_size,
        commission=config.backtest.commission,
        slippage=config.backtest.slippage,
    )

    try:
        report = run_walk_forward_backtest(db, backtest_config, tickers=tickers)
    except ValueError as e:
        click.echo(f"No backtest run: {e}")
        return

    click.echo("Walk-forward backtest")
    click.echo("=" * 40)
    click.echo(f"  Folds: {report.folds}")
    click.echo(f"  Trades: {report.trade_count}")
    if not report.prediction_metrics:
        click.echo("  Insufficient data for the configured windows.")
        return
    click.echo(f"  Accuracy: {report.prediction_metrics['accuracy']:.3f}")
    click.echo(f"  F1: {report.prediction_metrics['f1']:.3f}")
    click.echo(f"  Brier: {report.prediction_metrics['brier_score']:.5f}")
    click.echo(f"  MAE: {report.prediction_metrics['mae']:.5f}")
    click.echo(f"  Sharpe: {report.trading_metrics['sharpe_ratio']:.3f}")
    click.echo(f"  Max DD: {report.trading_metrics['max_drawdown']:.2%}")
    click.echo(f"  Total return: {report.trading_metrics['total_return']:.2%}")


@main.command("mature-predictions")
@click.option(
    "--store",
    type=click.Path(path_type=Path),
    default=Path("prediction_tracking"),
    show_default=True,
)
@click.option(
    "--prices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Provider-quality input_prices.parquet containing completed targets.",
)
@click.option(
    "--as-of",
    required=True,
    help="Timezone-aware ISO timestamp after the target interval, for example 2026-04-03T18:10:00+03:00.",
)
def mature_predictions(store: Path, prices: Path, as_of: str) -> None:
    """Freeze realized outcomes without retraining or replacing predictions."""
    from bist_predict.ingest.calendar import borsa_istanbul_equity_calendar
    from bist_predict.research.accepted_benchmark import load_price_artifact
    from bist_predict.research.prediction_tracking import ImmutablePredictionStore

    try:
        as_of_timestamp = datetime.fromisoformat(as_of)
    except ValueError as error:
        raise click.ClickException("--as-of must be a valid ISO timestamp") from error
    if as_of_timestamp.tzinfo is None:
        raise click.ClickException("--as-of must include a timezone offset")
    bars = load_price_artifact(prices)
    if not bars:
        raise click.ClickException("price artifact contains no rows")
    calendar = borsa_istanbul_equity_calendar(
        min(bar.date for bar in bars),
        max(bar.date for bar in bars),
        source_retrieved_at=as_of_timestamp,
    )
    outcomes = ImmutablePredictionStore(store).mature(
        as_of=as_of_timestamp,
        prices=bars,
        calendar=calendar,
    )
    click.echo(f"Matured predictions: {len(outcomes)}")


@main.command()
@click.option("--ticker", default=None, help="Limit immutable outcomes to one ticker")
@click.option(
    "--store",
    type=click.Path(path_type=Path),
    default=Path("prediction_tracking"),
    show_default=True,
)
def accuracy(ticker: str | None, store: Path) -> None:
    """Report accuracy from frozen signal-time records and outcomes only."""
    from bist_predict.research.prediction_tracking import ImmutablePredictionStore

    metrics = ImmutablePredictionStore(store).accuracy_metrics(ticker=ticker)
    click.echo(f"Resolved predictions: {metrics['resolved_predictions']}")
    click.echo(f"Directional accuracy: {metrics['directional_accuracy']:.1%}")
    click.echo(f"MAE: {metrics['mae']:.6f}")
