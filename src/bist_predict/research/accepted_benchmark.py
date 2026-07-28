"""Executable accepted baseline benchmark and exact bundled-input replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable, Mapping, cast

import numpy as np
import pandas as pd

from bist_predict.ingest.calendar import (
    OfficialTradingCalendar,
    borsa_istanbul_equity_calendar,
)
from bist_predict.ingest.corporate_actions import CorporateAction, CorporateActionType
from bist_predict.ingest.types import (
    OHLCVBar,
    OpenQuality,
    PriceRepresentation,
    VolumeQuality,
)
from bist_predict.features.lineage import FeatureArtifactLineage
from bist_predict.research.baselines import ACCEPTED_BASELINES, run_baseline_benchmark
from bist_predict.research.missingness import build_missingness_report
from bist_predict.research.panel import build_canonical_panel, panel_to_frame
from bist_predict.research.portfolio_backtest import (
    CostModel,
    PortfolioBacktester,
    StrategyConfig,
)
from bist_predict.research.inference_report import build_inference_report
from bist_predict.research.reporting import (
    TRADING_SESSIONS_PER_YEAR,
    compute_portfolio_metrics,
    grouped_portfolio_metrics,
)
from bist_predict.research.sensitivity import (
    configuration_grid,
    run_configuration_sensitivity,
    summarise_sensitivity,
)
from bist_predict.research.run_artifacts import RunBundle, RunBundleWriter
from bist_predict.research.splits import ExpandingWindowSplitter
from bist_predict.research.stationary_features import (
    STATIONARY_FEATURE_MANIFEST,
    build_stationary_snapshots,
)


_GRID_AXES = {"train": "min_train_dates", "val": "validation_dates", "embargo": "embargo_dates"}


def parse_block_sizes(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated block-length list into positive integers."""
    try:
        values = tuple(int(item) for item in spec.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"bootstrap_block_sizes must be integers: {spec}") from error
    if not values or any(value < 1 for value in values):
        raise ValueError(f"bootstrap_block_sizes must be positive integers: {spec}")
    return tuple(sorted(set(values)))


def parse_sensitivity_grid(spec: str) -> dict[str, tuple[int, ...]]:
    """Parse ``train=..|val=..|embargo=..|topk=..`` into declared axis values.

    The grid lives in the configuration as one string rather than as a nested
    structure so that ``config.yaml`` round-trips through JSON without changing
    the configuration hash that names the run.
    """
    axes: dict[str, tuple[int, ...]] = {}
    for section in spec.split("|"):
        if "=" not in section:
            raise ValueError(f"sensitivity_grid section must be name=values: {section}")
        name, _, raw = section.partition("=")
        key = _GRID_AXES.get(name.strip(), "top_k" if name.strip() == "topk" else None)
        if key is None:
            raise ValueError(f"unknown sensitivity_grid axis: {name}")
        if key in axes:
            raise ValueError(f"duplicate sensitivity_grid axis: {name}")
        try:
            values = tuple(int(item) for item in raw.split(",") if item.strip())
        except ValueError as error:
            raise ValueError(f"sensitivity_grid values must be integers: {section}") from error
        if not values or any(value < 1 for value in values):
            raise ValueError(f"sensitivity_grid values must be positive integers: {section}")
        axes[key] = tuple(sorted(set(values)))
    missing = sorted({"min_train_dates", "validation_dates", "embargo_dates", "top_k"} - set(axes))
    if missing:
        raise ValueError(f"sensitivity_grid is missing axes: {', '.join(missing)}")
    return axes


@dataclass(frozen=True)
class AcceptedBenchmarkConfig:
    """Complete declared choices for the accepted baseline experiment."""

    experiment_scope: str = "fixed_bist_large_cap_prototype"
    methodology_version: str = "accepted-baseline-v2"
    min_train_dates: int = 24
    validation_dates: int = 10
    step_dates: int = 10
    embargo_dates: int = 1
    portfolio_model: str = "ridge"
    top_k: int = 3
    starting_equity: float = 100_000.0
    decision_cost_rate: float = 0.0001
    max_participation: float = 0.01
    liquidity_lookback_sessions: int = 20
    min_trade_value: float = 100.0
    commission_rate: float = 0.0002
    bid_ask_spread_rate: float = 0.001
    slippage_rate: float = 0.0003
    market_impact_coefficient: float = 0.0001
    tax_rate: float = 0.0
    seed: int = 42
    bootstrap_iterations: int = 10_000
    bootstrap_block_sizes: str = "1,2,3,5,8,13,21"
    sensitivity_grid: str = "train=24,36,48|val=5,10,20|embargo=1,2|topk=1,2,3,4"

    def __post_init__(self) -> None:
        parse_sensitivity_grid(self.sensitivity_grid)
        parse_block_sizes(self.bootstrap_block_sizes)
        if self.bootstrap_iterations < 1_000:
            raise ValueError("bootstrap_iterations must be at least one thousand")

    @classmethod
    def synthetic_smoke(cls) -> AcceptedBenchmarkConfig:
        """Return the bounded non-market configuration used by CI."""
        return cls(
            experiment_scope="synthetic_methodology_smoke",
            top_k=2,
            bootstrap_iterations=2_000,
            bootstrap_block_sizes="1,5",
            sensitivity_grid="train=24,36|val=10|embargo=1|topk=1,2",
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


@dataclass(frozen=True)
class CorporateActionCoverage:
    """Sourced assertion that one ticker/date interval was checked for actions."""

    ticker: str
    start: date
    end: date
    source: str
    source_retrieved_at: datetime

    def __post_init__(self) -> None:
        if not self.ticker or not self.source:
            raise ValueError("corporate-action coverage requires ticker and source")
        if self.start > self.end:
            raise ValueError("corporate-action coverage start must not exceed end")
        if self.source_retrieved_at.tzinfo is None:
            raise ValueError("corporate-action coverage retrieval time must be timezone-aware")


_CORPORATE_ACTION_COLUMNS = (
    "ticker",
    "effective_date",
    "action_type",
    "source",
    "ratio",
    "cash_amount",
    "currency",
    "subscription_price",
    "new_ticker",
    "delisting_price",
    "source_retrieved_at",
)
_CORPORATE_ACTION_COVERAGE_COLUMNS = (
    "ticker",
    "start",
    "end",
    "source",
    "source_retrieved_at",
)


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
                split_adjusted_prices=_optional_representation(row, "split_adjusted"),
                total_return_prices=_optional_representation(row, "total_return"),
                open_quality=OpenQuality(str(row["open_quality"])),
                volume_quality=VolumeQuality(str(row["volume_quality"])),
                provider_symbol=(
                    None if pd.isna(row["provider_symbol"]) else str(row["provider_symbol"])
                ),
                provider_record_id=(
                    None if pd.isna(row["provider_record_id"]) else str(row["provider_record_id"])
                ),
                source_retrieved_at=(
                    None if pd.isna(retrieved) else datetime.fromisoformat(str(retrieved))
                ),
            )
        )
    return tuple(bars)


def load_price_artifact(path: Path) -> tuple[OHLCVBar, ...]:
    """Load the explicit provider-quality price schema from Parquet."""
    return _prices_from_frame(pd.read_parquet(path))


def _corporate_action_frame(actions: Iterable[CorporateAction]) -> pd.DataFrame:
    records = [
        {
            "ticker": action.ticker,
            "effective_date": action.effective_date.isoformat(),
            "action_type": action.action_type.value,
            "source": action.source,
            "ratio": action.ratio,
            "cash_amount": action.cash_amount,
            "currency": action.currency,
            "subscription_price": action.subscription_price,
            "new_ticker": action.new_ticker,
            "delisting_price": action.delisting_price,
            "source_retrieved_at": (
                action.source_retrieved_at.isoformat()
                if action.source_retrieved_at is not None
                else None
            ),
        }
        for action in actions
    ]
    return pd.DataFrame.from_records(records, columns=_CORPORATE_ACTION_COLUMNS).sort_values(
        ["effective_date", "ticker", "action_type"],
        kind="stable",
        ignore_index=True,
    )


def _corporate_actions_from_frame(frame: pd.DataFrame) -> tuple[CorporateAction, ...]:
    if tuple(frame.columns) != _CORPORATE_ACTION_COLUMNS:
        raise ValueError("corporate action artifact schema mismatch")
    actions: list[CorporateAction] = []
    for row in frame.to_dict(orient="records"):
        retrieved = row["source_retrieved_at"]
        actions.append(
            CorporateAction(
                ticker=str(row["ticker"]),
                effective_date=date.fromisoformat(str(row["effective_date"])),
                action_type=CorporateActionType(str(row["action_type"])),
                source=str(row["source"]),
                ratio=None if pd.isna(row["ratio"]) else float(row["ratio"]),
                cash_amount=(None if pd.isna(row["cash_amount"]) else float(row["cash_amount"])),
                currency=None if pd.isna(row["currency"]) else str(row["currency"]),
                subscription_price=(
                    None if pd.isna(row["subscription_price"]) else float(row["subscription_price"])
                ),
                new_ticker=(None if pd.isna(row["new_ticker"]) else str(row["new_ticker"])),
                delisting_price=(
                    None if pd.isna(row["delisting_price"]) else float(row["delisting_price"])
                ),
                source_retrieved_at=(
                    None if pd.isna(retrieved) else datetime.fromisoformat(str(retrieved))
                ),
            )
        )
    return tuple(actions)


def load_corporate_action_artifact(path: Path) -> tuple[CorporateAction, ...]:
    """Load the explicit sourced corporate-action snapshot from Parquet."""
    return _corporate_actions_from_frame(pd.read_parquet(path))


def _corporate_action_coverage_frame(
    coverage: Iterable[CorporateActionCoverage],
) -> pd.DataFrame:
    records = [
        {
            "ticker": item.ticker,
            "start": item.start.isoformat(),
            "end": item.end.isoformat(),
            "source": item.source,
            "source_retrieved_at": item.source_retrieved_at.isoformat(),
        }
        for item in coverage
    ]
    return pd.DataFrame.from_records(
        records,
        columns=_CORPORATE_ACTION_COVERAGE_COLUMNS,
    ).sort_values(["ticker"], kind="stable", ignore_index=True)


def _corporate_action_coverage_from_frame(
    frame: pd.DataFrame,
) -> tuple[CorporateActionCoverage, ...]:
    if tuple(frame.columns) != _CORPORATE_ACTION_COVERAGE_COLUMNS:
        raise ValueError("corporate-action coverage artifact schema mismatch")
    return tuple(
        CorporateActionCoverage(
            ticker=str(row["ticker"]),
            start=date.fromisoformat(str(row["start"])),
            end=date.fromisoformat(str(row["end"])),
            source=str(row["source"]),
            source_retrieved_at=datetime.fromisoformat(str(row["source_retrieved_at"])),
        )
        for row in frame.to_dict(orient="records")
    )


def load_corporate_action_coverage_artifact(
    path: Path,
) -> tuple[CorporateActionCoverage, ...]:
    """Load sourced no-event/event coverage for every accepted ticker."""
    return _corporate_action_coverage_from_frame(pd.read_parquet(path))


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
        if bar.source_retrieved_at.tzinfo is None:
            raise ValueError("accepted benchmark requires timezone-aware retrieval time")
        if not bar.provider_symbol or not bar.provider_record_id:
            raise ValueError("accepted benchmark requires provider symbol and stable record ID")
        if bar.split_adjusted_prices is None or bar.total_return_prices is None:
            raise ValueError(
                "accepted benchmark requires explicit split-adjusted and total-return prices"
            )
        sessions_by_ticker.setdefault(bar.ticker, set()).add(bar.date)
    if len(sessions_by_ticker) < 2:
        raise ValueError("pooled benchmark requires at least two tickers")
    session_sets = list(sessions_by_ticker.values())
    if any(sessions != session_sets[0] for sessions in session_sets[1:]):
        raise ValueError("all accepted-universe tickers must share identical sessions")


def _validate_corporate_actions(
    actions: tuple[CorporateAction, ...] | None,
    bars: tuple[OHLCVBar, ...],
    config: AcceptedBenchmarkConfig,
) -> tuple[CorporateAction, ...]:
    if actions is None:
        if config.experiment_scope == "synthetic_methodology_smoke":
            return ()
        raise ValueError("accepted market benchmark requires a corporate-action snapshot")
    tickers = {bar.ticker for bar in bars}
    start = min(bar.date for bar in bars)
    end = max(bar.date for bar in bars)
    identities: set[tuple[str, date, CorporateActionType]] = set()
    for action in actions:
        identity = (action.ticker, action.effective_date, action.action_type)
        if identity in identities:
            raise ValueError(f"duplicate corporate action: {identity}")
        identities.add(identity)
        if action.ticker not in tickers:
            raise ValueError(
                f"corporate action ticker is outside the accepted universe: {action.ticker}"
            )
        if not start <= action.effective_date <= end:
            raise ValueError("corporate action lies outside the dataset interval")
        if not action.source or action.source_retrieved_at is None:
            raise ValueError("accepted corporate actions require source provenance")
        if action.source_retrieved_at.tzinfo is None:
            raise ValueError("corporate action retrieval time must be timezone-aware")
    return tuple(sorted(actions, key=lambda item: (item.effective_date, item.ticker)))


def _validate_corporate_action_coverage(
    coverage: tuple[CorporateActionCoverage, ...] | None,
    bars: tuple[OHLCVBar, ...],
    config: AcceptedBenchmarkConfig,
    *,
    source_retrieved_at: datetime,
) -> tuple[CorporateActionCoverage, ...]:
    tickers = sorted({bar.ticker for bar in bars})
    start = min(bar.date for bar in bars)
    end = max(bar.date for bar in bars)
    if coverage is None and config.experiment_scope == "synthetic_methodology_smoke":
        return tuple(
            CorporateActionCoverage(
                ticker=ticker,
                start=start,
                end=end,
                source="synthetic_methodology_smoke_actions",
                source_retrieved_at=source_retrieved_at,
            )
            for ticker in tickers
        )
    if coverage is None:
        raise ValueError("accepted market benchmark requires corporate-action coverage")
    by_ticker = {item.ticker: item for item in coverage}
    if len(by_ticker) != len(coverage):
        raise ValueError("duplicate corporate-action coverage ticker")
    if sorted(by_ticker) != tickers:
        raise ValueError("corporate-action coverage must match the accepted universe")
    for item in coverage:
        if item.start > start or item.end < end:
            raise ValueError("corporate-action coverage does not span the dataset interval")
    return tuple(sorted(coverage, key=lambda item: item.ticker))


def _calendar_for_run(
    bars: tuple[OHLCVBar, ...],
    config: AcceptedBenchmarkConfig,
    *,
    source_retrieved_at: datetime,
) -> OfficialTradingCalendar:
    start = min(bar.date for bar in bars)
    end = max(bar.date for bar in bars)
    if config.experiment_scope == "synthetic_methodology_smoke":
        sessions = tuple(sorted({bar.date for bar in bars}))
        return OfficialTradingCalendar(
            index_name="SYNTHETIC",
            sessions=sessions,
            source="synthetic_methodology_smoke_calendar",
            source_retrieved_at=source_retrieved_at,
        )
    return borsa_istanbul_equity_calendar(
        start,
        end,
        source_retrieved_at=source_retrieved_at,
    )


def _calendar_frame(calendar: OfficialTradingCalendar) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for session in calendar.sessions:
        session_open, session_close = calendar.session_bounds(session)
        records.append(
            {
                "date": session.isoformat(),
                "session_open": session_open,
                "session_close": session_close,
                "is_half_day": session in calendar.session_close_overrides,
                "index_name": calendar.index_name,
                "source": calendar.source,
                "source_retrieved_at": calendar.source_retrieved_at,
            }
        )
    return pd.DataFrame.from_records(records)


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
        ticker: (
            "low"
            if index < len(ordered) / 3
            else "medium"
            if index < 2 * len(ordered) / 3
            else "high"
        )
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
    missing_dates = [session for session in execution_dates if session not in by_date.index]
    if missing_dates:
        raise ValueError(f"equal-weight benchmark is missing execution dates: {missing_dates}")
    return [float(by_date.loc[session]) for session in execution_dates]


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
        liquidity_lookback_sessions=config.liquidity_lookback_sessions,
        min_trade_value=config.min_trade_value,
    )


def _feature_evidence(
    panel: pd.DataFrame,
    bars: tuple[OHLCVBar, ...],
    folds: tuple[object, ...],
    *,
    data_manifest: Mapping[str, object],
    git_sha: str,
    calculation_timestamp: datetime,
    code_version: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[tuple[FeatureArtifactLineage, list[dict[str, object]]], ...],
]:
    """Build fold-aware missingness and content-addressed feature evidence."""
    source_by_key = {(bar.date.isoformat(), bar.ticker): bar.source for bar in bars}
    validation_fold_by_sample: dict[str, str] = {}
    for fold in folds:
        fold_id = str(getattr(fold, "fold_id"))
        for sample_id in getattr(fold, "validation_indices"):
            validation_fold_by_sample[str(sample_id)] = f"{fold_id}_validation"
    source_by_sample: dict[object, str] = {}
    fold_by_sample: dict[object, str] = {}
    for sample_index, sample in panel.iterrows():
        key = (str(sample["date"]), str(sample["ticker"]))
        if key not in source_by_key:
            raise ValueError(f"feature source is missing for sample: {key}")
        source_by_sample[sample_index] = source_by_key[key]
        sample_id = f"{key[0]}|{key[1]}"
        fold_by_sample[sample_index] = validation_fold_by_sample.get(
            sample_id, "pre_validation_history"
        )
    missingness = build_missingness_report(
        panel,
        STATIONARY_FEATURE_MANIFEST,
        source_by_sample=source_by_sample,
        fold_by_sample=fold_by_sample,
    )
    data_manifest_hash = hashlib.sha256(
        json.dumps(data_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lineages: list[FeatureArtifactLineage] = []
    artifacts: list[tuple[FeatureArtifactLineage, list[dict[str, object]]]] = []
    feature_columns = [
        "date",
        "ticker",
        *STATIONARY_FEATURE_MANIFEST.ordered_feature_names,
        *(f"{name}__missing_reason" for name in STATIONARY_FEATURE_MANIFEST.ordered_feature_names),
    ]
    for ticker, ticker_panel in panel.groupby("ticker", sort=True):
        dates = pd.to_datetime(ticker_panel["date"]).dt.date
        lineage = FeatureArtifactLineage(
            feature_manifest_hash=STATIONARY_FEATURE_MANIFEST.manifest_hash,
            git_commit=git_sha,
            input_data_manifest_hash=data_manifest_hash,
            calculation_timestamp=calculation_timestamp,
            code_version=code_version,
            ticker=str(ticker),
            start_date=min(dates),
            end_date=max(dates),
        )
        rows = (
            ticker_panel.loc[:, feature_columns]
            .where(pd.notna(ticker_panel.loc[:, feature_columns]), None)
            .to_dict(orient="records")
        )
        lineages.append(lineage)
        artifacts.append((lineage, rows))
    lineage_frame = pd.DataFrame.from_records(
        [{**lineage.to_dict(), "artifact_id": lineage.artifact_id} for lineage in lineages]
    )
    return missingness, lineage_frame, tuple(artifacts)


def run_accepted_benchmark(
    prices: Iterable[OHLCVBar],
    *,
    corporate_actions: Iterable[CorporateAction] | None = None,
    corporate_action_coverage: Iterable[CorporateActionCoverage] | None = None,
    runs_root: Path,
    config: AcceptedBenchmarkConfig,
    now: datetime | None = None,
    git_sha: str | None = None,
    dirty_working_tree: bool | None = None,
    command: str = "bist-predict benchmark",
) -> RunBundle:
    """Run all accepted baselines, one cost-aware portfolio, and persist evidence."""
    effective_now = now or datetime.now(UTC)
    price_frame = _price_frame(tuple(prices))
    bars = _prices_from_frame(price_frame)
    _validate_prices(bars)
    supplied_actions = None if corporate_actions is None else tuple(corporate_actions)
    action_records = _validate_corporate_actions(supplied_actions, bars, config)
    action_frame = _corporate_action_frame(action_records)
    supplied_coverage = (
        None if corporate_action_coverage is None else tuple(corporate_action_coverage)
    )
    action_coverage = _validate_corporate_action_coverage(
        supplied_coverage,
        bars,
        config,
        source_retrieved_at=effective_now,
    )
    action_coverage_frame = _corporate_action_coverage_frame(action_coverage)
    calendar = _calendar_for_run(bars, config, source_retrieved_at=effective_now)
    calendar_validation = calendar.validate_bars(bars)
    calendar_issues = {
        "missing_expected_sessions": list(calendar_validation.missing_expected_sessions),
        "unexpected_sessions": list(calendar_validation.unexpected_sessions),
        "unexpected_weekend_rows": list(calendar_validation.unexpected_weekend_rows),
        "duplicate_sessions": list(calendar_validation.duplicate_sessions),
    }
    if any(calendar_issues.values()):
        raise ValueError(f"price data violates official trading calendar: {calendar_issues}")
    calendar_frame = _calendar_frame(calendar)
    snapshots = build_stationary_snapshots(bars)
    panel_rows = build_canonical_panel(
        snapshots,
        bars,
        STATIONARY_FEATURE_MANIFEST,
        calendar=calendar,
    )
    panel = panel_to_frame(panel_rows, STATIONARY_FEATURE_MANIFEST)
    splitter = ExpandingWindowSplitter(
        min_train_dates=config.min_train_dates,
        validation_dates=config.validation_dates,
        step_dates=config.step_dates,
        embargo_dates=config.embargo_dates,
    )
    benchmark = run_baseline_benchmark(panel, STATIONARY_FEATURE_MANIFEST, splitter)
    backtester = PortfolioBacktester(strategy=_strategy(config), costs=_cost_model(config))
    portfolio = backtester.run(
        benchmark.predictions,
        bars,
        model_name=config.portfolio_model,
        starting_equity=config.starting_equity,
        corporate_actions=action_records,
        calendar=calendar,
    )
    session_dates = [snapshot.date for snapshot in portfolio.daily_snapshots]
    equal_weight_returns = _equal_weight_returns(panel, session_dates)
    cost_sensitivity: dict[str, object] = {}
    selection_costs = _cost_model(config)
    for multiplier in (0.0, 1.0, 2.0):
        result = PortfolioBacktester(
            strategy=_strategy(config),
            costs=_cost_model(config, multiplier),
            selection_costs=selection_costs,
        ).run(
            benchmark.predictions,
            bars,
            model_name=config.portfolio_model,
            starting_equity=config.starting_equity,
            corporate_actions=action_records,
            calendar=calendar,
        )
        cost_sensitivity[f"{multiplier:.1f}x"] = {
            "cost_multiplier": multiplier,
            "metrics": compute_portfolio_metrics(result, benchmark_returns=equal_weight_returns),
        }

    grid_axes = parse_sensitivity_grid(config.sensitivity_grid)
    sensitivity_trials = run_configuration_sensitivity(
        configuration_grid(
            min_train_dates=grid_axes["min_train_dates"],
            validation_dates=grid_axes["validation_dates"],
            embargo_dates=grid_axes["embargo_dates"],
            top_k=grid_axes["top_k"],
        ),
        panel=panel,
        manifest=STATIONARY_FEATURE_MANIFEST,
        bars=bars,
        calendar=calendar,
        corporate_actions=action_records,
        strategy=_strategy(config),
        costs=_cost_model(config),
        portfolio_model=config.portfolio_model,
        starting_equity=config.starting_equity,
    )
    reported_trial_id = (
        f"train{config.min_train_dates}"
        f"_val{config.validation_dates}"
        f"_step{config.step_dates}"
        f"_emb{config.embargo_dates}"
        f"_k{config.top_k}"
    )
    reported_trial = next(
        (trial for trial in sensitivity_trials if trial.trial_id == reported_trial_id), None
    )
    if reported_trial is None:
        raise ValueError(
            "the accepted configuration must appear in its own sensitivity grid: "
            f"{reported_trial_id}"
        )
    sensitivity = summarise_sensitivity(sensitivity_trials, reported=reported_trial)
    inference = build_inference_report(
        benchmark.predictions,
        net_returns=[snapshot.net_return for snapshot in portfolio.daily_snapshots],
        benchmark_model="zero_return",
        portfolio_model=config.portfolio_model,
        periods_per_year=TRADING_SESSIONS_PER_YEAR,
        trial_count=int(cast(int, sensitivity["trial_count"])),
        trial_sharpe_variance=float(cast(float, sensitivity["trial_sharpe_variance"])),
        seed=config.seed,
        replications=config.bootstrap_iterations,
    )

    tickers = sorted({bar.ticker for bar in bars})
    data_manifest = {
        "dataset_id": f"{config.experiment_scope}-{_canonical_frame_hash(price_frame)[:12]}",
        "sources": sorted(
            {bar.source for bar in bars}
            | {calendar.source}
            | {action.source for action in action_records}
            | {item.source for item in action_coverage}
        ),
        "universe_version": config.experiment_scope,
        "start": min(bar.date for bar in bars).isoformat(),
        "end": max(bar.date for bar in bars).isoformat(),
        "row_count": len(bars),
        "sha256": _canonical_frame_hash(price_frame),
        "created_at": effective_now.astimezone(UTC).isoformat(),
        "missing_sessions": list(calendar_validation.missing_expected_sessions),
        "quality_summary": {
            "open_quality": {"observed": len(bars)},
            "volume_quality": {"observed": len(bars)},
            "identical_ticker_sessions": True,
            "calendar_source": calendar.source,
            "calendar_sha256": _canonical_frame_hash(calendar_frame),
            "calendar_validation": calendar_issues,
            "corporate_action_count": len(action_records),
            "corporate_action_types": sorted(
                {action.action_type.value for action in action_records}
            ),
            "corporate_action_coverage_tickers": [item.ticker for item in action_coverage],
            "corporate_action_policy": (
                "positions open after effective-open actions; no pre-open entitlement is credited"
            ),
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
    portfolio_grouped = grouped_portfolio_metrics(
        portfolio,
        benchmark.predictions,
        metadata,
    )
    writer = RunBundleWriter(
        runs_root,
        git_sha=git_sha,
        dirty_working_tree=dirty_working_tree,
        now=effective_now,
    )
    missingness, feature_lineage, feature_artifacts = _feature_evidence(
        panel,
        bars,
        benchmark.folds,
        data_manifest=data_manifest,
        git_sha=writer.git_sha,
        calculation_timestamp=effective_now,
        code_version=config.methodology_version,
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
            "schema_version": 2,
            "methodology_version": config.methodology_version,
            "accepted_models": list(ACCEPTED_BASELINES),
            "portfolio_model": config.portfolio_model,
            "fit_scope": "per-fold train rows only",
            "fitted_model_states": list(benchmark.fitted_model_states),
            "corporate_action_policy": (
                "execution uses post-action observed opens and closes; positions carry no overnight entitlement"
            ),
        },
        trials=(),
        seeds=(config.seed,),
        command=command,
        input_frames={
            "corporate_action_coverage": action_coverage_frame,
            "corporate_actions": action_frame,
            "input_prices": price_frame,
            "configuration_sensitivity": pd.DataFrame.from_records(
                [trial.to_dict() for trial in sensitivity_trials]
            ),
            "official_calendar": calendar_frame,
            "feature_lineage": feature_lineage,
            "missingness": missingness,
            "panel": panel,
            "sample_metadata": metadata,
        },
        feature_artifacts=feature_artifacts,
        sample_metadata=metadata,
        benchmark_returns=equal_weight_returns,
        bootstrap_iterations=config.bootstrap_iterations,
        bootstrap_block_sizes=parse_block_sizes(config.bootstrap_block_sizes),
        additional_metrics={
            "benchmarks": {
                "cash": {"total_return": 0.0},
                "equal_weight_eligible_universe": {
                    "total_return": float(np.prod(1.0 + np.asarray(equal_weight_returns)) - 1.0)
                },
                "relevant_bist_index": {"status": "not_available_in_input_dataset"},
            },
            "cost_sensitivity": cost_sensitivity,
            "portfolio_grouped": portfolio_grouped,
            "configuration_sensitivity": sensitivity,
            "inference": inference,
        },
    )


def reproduce_run(run_path: Path, *, scratch_root: Path) -> dict[str, str]:
    """Replay bundled inputs and report byte-level scientific artifact drift.

    ``environment.json`` and ``run_manifest.json`` preserve the environment in
    which a run was created.  They are provenance, not scientific outputs, so a
    replay on another checkout, Python patch release, or operating system is
    allowed to differ in those two files.  All other recorded artifacts remain
    byte-for-byte replay requirements.
    """
    expected = json.loads((run_path / "artifact_hashes.json").read_text())
    run_manifest = json.loads((run_path / "run_manifest.json").read_text())
    config = AcceptedBenchmarkConfig.from_mapping(
        json.loads((run_path / "config.yaml").read_text())
    )
    prices = _prices_from_frame(pd.read_parquet(run_path / "input_prices.parquet"))
    corporate_actions = _corporate_actions_from_frame(
        pd.read_parquet(run_path / "corporate_actions.parquet")
    )
    corporate_action_coverage = _corporate_action_coverage_from_frame(
        pd.read_parquet(run_path / "corporate_action_coverage.parquet")
    )
    replay = run_accepted_benchmark(
        prices,
        corporate_actions=corporate_actions,
        corporate_action_coverage=corporate_action_coverage,
        runs_root=scratch_root,
        config=config,
        now=datetime.fromisoformat(run_manifest["created_at"]),
        git_sha=str(run_manifest["git_sha"]),
        dirty_working_tree=bool(run_manifest["dirty_working_tree"]),
        command=str(run_manifest["training_command"]),
    )
    actual = json.loads((replay.path / "artifact_hashes.json").read_text())
    failures: dict[str, str] = {}
    provenance_artifacts = {"environment.json", "run_manifest.json"}
    expected_scientific = set(expected).difference(provenance_artifacts)
    actual_scientific = set(actual).difference(provenance_artifacts)
    for name in sorted(expected_scientific | actual_scientific):
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
