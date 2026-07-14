"""Executable accepted baseline benchmark and exact bundled-input replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from bist_predict.ingest.types import (
    OHLCVBar,
    OpenQuality,
    PriceRepresentation,
    VolumeQuality,
)
from bist_predict.research.baselines import ACCEPTED_BASELINES, run_baseline_benchmark
from bist_predict.research.panel import build_canonical_panel, panel_to_frame
from bist_predict.research.portfolio_backtest import (
    CostModel,
    PortfolioBacktester,
    StrategyConfig,
)
from bist_predict.research.reporting import compute_portfolio_metrics
from bist_predict.research.run_artifacts import RunBundle, RunBundleWriter
from bist_predict.research.splits import ExpandingWindowSplitter
from bist_predict.research.stationary_features import (
    STATIONARY_FEATURE_MANIFEST,
    build_stationary_snapshots,
)


@dataclass(frozen=True)
class AcceptedBenchmarkConfig:
    """Complete declared choices for the accepted baseline experiment."""

    experiment_scope: str = "fixed_bist_large_cap_prototype"
    methodology_version: str = "accepted-baseline-v1"
    min_train_dates: int = 24
    validation_dates: int = 10
    step_dates: int = 10
    embargo_dates: int = 1
    portfolio_model: str = "ridge"
    top_k: int = 3
    starting_equity: float = 100_000.0
    decision_cost_rate: float = 0.0001
    max_participation: float = 0.01
    min_trade_value: float = 100.0
    commission_rate: float = 0.0002
    bid_ask_spread_rate: float = 0.001
    slippage_rate: float = 0.0003
    market_impact_coefficient: float = 0.0001
    tax_rate: float = 0.0
    seed: int = 42

    @classmethod
    def synthetic_smoke(cls) -> AcceptedBenchmarkConfig:
        """Return the bounded non-market configuration used by CI."""
        return cls(
            experiment_scope="synthetic_methodology_smoke",
            top_k=2,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> AcceptedBenchmarkConfig:
        fields = {name for name in cls.__dataclass_fields__}
        unknown = sorted(set(payload).difference(fields))
        if unknown:
            raise ValueError(f"unknown benchmark config fields: {', '.join(unknown)}")
        return cls(**payload)  # type: ignore[arg-type]


def _price_frame(prices: Iterable[OHLCVBar]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for bar in prices:
        split = bar.split_adjusted_prices
        total = bar.total_return_prices
        records.append(
            {
                "ticker": bar.ticker,
                "date": bar.date.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "adj_close": bar.adj_close,
                "volume": bar.volume,
                "source": bar.source,
                "open_quality": bar.open_quality.value,
                "volume_quality": bar.volume_quality.value,
                "provider_symbol": bar.provider_symbol,
                "provider_record_id": bar.provider_record_id,
                "source_retrieved_at": (
                    bar.source_retrieved_at.isoformat()
                    if bar.source_retrieved_at is not None
                    else None
                ),
                "split_adjusted_open": split.open if split else None,
                "split_adjusted_high": split.high if split else None,
                "split_adjusted_low": split.low if split else None,
                "split_adjusted_close": split.close if split else None,
                "split_adjusted_volume": split.volume if split else None,
                "total_return_open": total.open if total else None,
                "total_return_high": total.high if total else None,
                "total_return_low": total.low if total else None,
                "total_return_close": total.close if total else None,
                "total_return_volume": total.volume if total else None,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["date", "ticker"], kind="stable", ignore_index=True
    )


def _optional_representation(row: pd.Series, prefix: str) -> PriceRepresentation | None:
    columns = [
        f"{prefix}_open",
        f"{prefix}_high",
        f"{prefix}_low",
        f"{prefix}_close",
        f"{prefix}_volume",
    ]
    if row[columns].isna().all():
        return None
    if row[columns].isna().any():
        raise ValueError(f"partial {prefix} representation for {row['ticker']} {row['date']}")
    return PriceRepresentation(
        open=float(row[columns[0]]),
        high=float(row[columns[1]]),
        low=float(row[columns[2]]),
        close=float(row[columns[3]]),
        volume=int(row[columns[4]]),
    )


def _prices_from_frame(frame: pd.DataFrame) -> tuple[OHLCVBar, ...]:
    bars: list[OHLCVBar] = []
    for _, row in frame.iterrows():
        retrieved = row["source_retrieved_at"]
        bars.append(
            OHLCVBar(
                ticker=str(row["ticker"]),
                date=date.fromisoformat(str(row["date"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_close=float(row["adj_close"]),
                volume=int(row["volume"]),
                source=str(row["source"]),
                split_adjusted_prices=_optional_representation(
                    row, "split_adjusted"
                ),
                total_return_prices=_optional_representation(row, "total_return"),
                open_quality=OpenQuality(str(row["open_quality"])),
                volume_quality=VolumeQuality(str(row["volume_quality"])),
                provider_symbol=(
                    None if pd.isna(row["provider_symbol"]) else str(row["provider_symbol"])
                ),
                provider_record_id=(
                    None
                    if pd.isna(row["provider_record_id"])
                    else str(row["provider_record_id"])
                ),
                source_retrieved_at=(
                    None
                    if pd.isna(retrieved)
                    else datetime.fromisoformat(str(retrieved))
                ),
            )
        )
    return tuple(bars)


def load_price_artifact(path: Path) -> tuple[OHLCVBar, ...]:
    """Load the explicit provider-quality price schema from Parquet."""
    return _prices_from_frame(pd.read_parquet(path))


def _validate_prices(prices: tuple[OHLCVBar, ...]) -> None:
    if not prices:
        raise ValueError("accepted benchmark requires price observations")
    sessions_by_ticker: dict[str, set[date]] = {}
    for bar in prices:
        if bar.open_quality is not OpenQuality.OBSERVED:
            raise ValueError("accepted benchmark requires observed opening prices")
        if bar.volume_quality is not VolumeQuality.OBSERVED:
            raise ValueError("accepted benchmark requires observed volume")
        if not bar.source or bar.source_retrieved_at is None:
            raise ValueError("accepted benchmark requires source provenance and retrieval time")
        sessions_by_ticker.setdefault(bar.ticker, set()).add(bar.date)
    if len(sessions_by_ticker) < 2:
        raise ValueError("pooled benchmark requires at least two tickers")
    session_sets = list(sessions_by_ticker.values())
    if any(sessions != session_sets[0] for sessions in session_sets[1:]):
        raise ValueError("all accepted-universe tickers must share identical sessions")


def _canonical_frame_hash(frame: pd.DataFrame) -> str:
    encoded = frame.to_json(orient="records", date_format="iso", double_precision=15)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sample_metadata(panel: pd.DataFrame, prices: tuple[OHLCVBar, ...]) -> pd.DataFrame:
    average_volume: dict[str, float] = {}
    for ticker in sorted({bar.ticker for bar in prices}):
        average_volume[ticker] = float(
            np.mean([bar.volume for bar in prices if bar.ticker == ticker])
        )
    ordered = sorted(average_volume, key=lambda ticker: (average_volume[ticker], ticker))
    buckets = {
        ticker: ("low" if index < len(ordered) / 3 else "medium" if index < 2 * len(ordered) / 3 else "high")
        for index, ticker in enumerate(ordered)
    }
    regimes = (
        panel.groupby("date", sort=True)["target_return"]
        .mean()
        .map(lambda value: "up" if value > 0.0 else "down")
    )
    metadata = panel[["date", "ticker"]].copy()
    metadata["sector"] = "unclassified"
    metadata["liquidity_bucket"] = metadata["ticker"].map(buckets)
    metadata["market_regime"] = metadata["date"].map(regimes)
    return metadata


def _equal_weight_returns(panel: pd.DataFrame, execution_dates: list[str]) -> list[float]:
    working = panel.copy()
    working["execution_date"] = pd.to_datetime(working["target_start"]).dt.date.astype(str)
    by_date = working.groupby("execution_date")["target_return"].mean()
    return [float(by_date.get(session, 0.0)) for session in execution_dates]


def _cost_model(config: AcceptedBenchmarkConfig, multiplier: float = 1.0) -> CostModel:
    return CostModel(
        commission_rate=config.commission_rate * multiplier,
        bid_ask_spread_rate=config.bid_ask_spread_rate * multiplier,
        slippage_rate=config.slippage_rate * multiplier,
        market_impact_coefficient=config.market_impact_coefficient * multiplier,
        tax_rate=config.tax_rate * multiplier,
    )


def _strategy(config: AcceptedBenchmarkConfig) -> StrategyConfig:
    return StrategyConfig(
        top_k=config.top_k,
        decision_cost_rate=config.decision_cost_rate,
        max_participation=config.max_participation,
        min_trade_value=config.min_trade_value,
    )


def run_accepted_benchmark(
    prices: Iterable[OHLCVBar],
    *,
    runs_root: Path,
    config: AcceptedBenchmarkConfig,
    now: datetime | None = None,
    git_sha: str | None = None,
    dirty_working_tree: bool | None = None,
    command: str = "bist-predict benchmark",
) -> RunBundle:
    """Run all accepted baselines, one cost-aware portfolio, and persist evidence."""
    price_frame = _price_frame(tuple(prices))
    bars = _prices_from_frame(price_frame)
    _validate_prices(bars)
    snapshots = build_stationary_snapshots(bars)
    panel_rows = build_canonical_panel(
        snapshots, bars, STATIONARY_FEATURE_MANIFEST
    )
    panel = panel_to_frame(panel_rows, STATIONARY_FEATURE_MANIFEST)
    splitter = ExpandingWindowSplitter(
        min_train_dates=config.min_train_dates,
        validation_dates=config.validation_dates,
        step_dates=config.step_dates,
        embargo_dates=config.embargo_dates,
    )
    benchmark = run_baseline_benchmark(
        panel, STATIONARY_FEATURE_MANIFEST, splitter
    )
    backtester = PortfolioBacktester(
        strategy=_strategy(config), costs=_cost_model(config)
    )
    portfolio = backtester.run(
        benchmark.predictions,
        bars,
        model_name=config.portfolio_model,
        starting_equity=config.starting_equity,
    )
    session_dates = [snapshot.date for snapshot in portfolio.daily_snapshots]
    equal_weight_returns = _equal_weight_returns(panel, session_dates)
    sensitivity: dict[str, object] = {}
    for multiplier in (0.0, 1.0, 2.0):
        result = PortfolioBacktester(
            strategy=_strategy(config), costs=_cost_model(config, multiplier)
        ).run(
            benchmark.predictions,
            bars,
            model_name=config.portfolio_model,
            starting_equity=config.starting_equity,
        )
        sensitivity[f"{multiplier:.1f}x"] = {
            "cost_multiplier": multiplier,
            "metrics": compute_portfolio_metrics(
                result, benchmark_returns=equal_weight_returns
            ),
        }

    effective_now = now or datetime.now(UTC)
    tickers = sorted({bar.ticker for bar in bars})
    data_manifest = {
        "dataset_id": f"{config.experiment_scope}-{_canonical_frame_hash(price_frame)[:12]}",
        "sources": sorted({bar.source for bar in bars}),
        "universe_version": config.experiment_scope,
        "start": min(bar.date for bar in bars).isoformat(),
        "end": max(bar.date for bar in bars).isoformat(),
        "row_count": len(bars),
        "sha256": _canonical_frame_hash(price_frame),
        "created_at": effective_now.astimezone(UTC).isoformat(),
        "missing_sessions": [],
        "quality_summary": {
            "open_quality": {"observed": len(bars)},
            "volume_quality": {"observed": len(bars)},
            "identical_ticker_sessions": True,
        },
    }
    universe_manifest = {
        "universe_version": config.experiment_scope,
        "membership_type": "fixed_prototype_not_historical_index_membership",
        "tickers": tickers,
        "start": data_manifest["start"],
        "end": data_manifest["end"],
    }
    metadata = _sample_metadata(panel, bars)
    writer = RunBundleWriter(
        runs_root,
        git_sha=git_sha,
        dirty_working_tree=dirty_working_tree,
        now=effective_now,
    )
    return writer.write(
        config=config.to_dict(),
        data_manifest=data_manifest,
        universe_manifest=universe_manifest,
        feature_manifest=STATIONARY_FEATURE_MANIFEST,
        folds=[fold.to_dict() for fold in benchmark.folds],
        predictions=benchmark.predictions,
        portfolio=portfolio,
        model_artifact={
            "methodology_version": config.methodology_version,
            "accepted_models": list(ACCEPTED_BASELINES),
            "portfolio_model": config.portfolio_model,
            "fit_scope": "per-fold train rows only",
        },
        trials=(),
        seeds=(config.seed,),
        command=command,
        input_frames={
            "input_prices": price_frame,
            "panel": panel,
            "sample_metadata": metadata,
        },
        sample_metadata=metadata,
        benchmark_returns=equal_weight_returns,
        additional_metrics={
            "benchmarks": {
                "cash": {"total_return": 0.0},
                "equal_weight_eligible_universe": {
                    "total_return": float(np.prod(1.0 + np.asarray(equal_weight_returns)) - 1.0)
                },
                "relevant_bist_index": {"status": "not_available_in_input_dataset"},
            },
            "cost_sensitivity": sensitivity,
        },
    )


def reproduce_run(run_path: Path, *, scratch_root: Path) -> dict[str, str]:
    """Replay a run from its bundled prices and report byte-level hash drift."""
    expected = json.loads((run_path / "artifact_hashes.json").read_text())
    run_manifest = json.loads((run_path / "run_manifest.json").read_text())
    config = AcceptedBenchmarkConfig.from_mapping(
        json.loads((run_path / "config.yaml").read_text())
    )
    prices = _prices_from_frame(pd.read_parquet(run_path / "input_prices.parquet"))
    replay = run_accepted_benchmark(
        prices,
        runs_root=scratch_root,
        config=config,
        now=datetime.fromisoformat(run_manifest["created_at"]),
        git_sha=str(run_manifest["git_sha"]),
        dirty_working_tree=bool(run_manifest["dirty_working_tree"]),
        command=str(run_manifest["training_command"]),
    )
    actual = json.loads((replay.path / "artifact_hashes.json").read_text())
    failures: dict[str, str] = {}
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            failures[name] = "missing_from_replay"
        elif name not in expected:
            failures[name] = "unexpected_in_replay"
        elif expected[name] != actual[name]:
            failures[name] = "sha256_mismatch"
    return failures


def generate_synthetic_prices() -> tuple[OHLCVBar, ...]:
    """Generate deterministic observed-quality prices for methodology CI only."""
    rng = np.random.default_rng(42)
    sessions = [stamp.date() for stamp in pd.bdate_range("2023-01-02", periods=170)]
    retrieved_at = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    bars: list[OHLCVBar] = []
    for ticker_index, ticker in enumerate(("GARAN", "ISCTR", "KCHOL", "THYAO")):
        previous_close = 30.0 + 15.0 * ticker_index
        for session_index, session in enumerate(sessions):
            market_cycle = 0.0025 * np.sin(session_index / 7.0)
            overnight = 0.0004 * np.cos(session_index / 5.0 + ticker_index)
            noise = float(rng.normal(0.0, 0.002))
            intraday = market_cycle + 0.0007 * ticker_index + noise
            open_price = previous_close * (1.0 + overnight)
            close_price = open_price * (1.0 + intraday)
            high = max(open_price, close_price) * 1.004
            low = min(open_price, close_price) * 0.996
            volume = 1_000_000 + ticker_index * 250_000 + (session_index % 20) * 10_000
            representation = PriceRepresentation(
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            )
            bars.append(
                OHLCVBar(
                    ticker=ticker,
                    date=session,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    adj_close=close_price,
                    volume=volume,
                    source="synthetic_methodology_smoke",
                    split_adjusted_prices=representation,
                    total_return_prices=representation,
                    open_quality=OpenQuality.OBSERVED,
                    volume_quality=VolumeQuality.OBSERVED,
                    provider_symbol=f"{ticker}.IS",
                    provider_record_id=f"{ticker}-{session.isoformat()}",
                    source_retrieved_at=retrieved_at,
                )
            )
            previous_close = close_price
    return tuple(bars)
