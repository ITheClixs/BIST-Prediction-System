"""Canonical point-in-time panel with executable next-session targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from bist_predict.features.manifest import (
    FeatureManifest,
    FeatureSchemaError,
)
from bist_predict.ingest.calendar import OfficialTradingCalendar
from bist_predict.ingest.types import OHLCVBar, OpenQuality

ISTANBUL = ZoneInfo("Europe/Istanbul")
SESSION_OPEN = time(10, 0)
SESSION_CLOSE = time(18, 0)


class MissingReason(str, Enum):
    """Why an expected feature is unavailable without conflating it with zero."""

    MISSING_OBSERVATION = "missing_observation"
    INSUFFICIENT_LOOKBACK = "insufficient_lookback"
    CALCULATION_FAILURE = "calculation_failure"
    NOT_APPLICABLE = "not_applicable"
    STALE_SOURCE = "stale_source"


class PanelBuildError(ValueError):
    """Raised when source records cannot satisfy the canonical panel contract."""


@dataclass(frozen=True)
class FeatureSnapshot:
    """Point-in-time feature values and their immutable schema provenance."""

    date: date
    ticker: str
    feature_available_at: datetime
    values: Mapping[str, float | None]
    missing_reasons: Mapping[str, MissingReason]
    feature_manifest_hash: str


@dataclass(frozen=True)
class CanonicalPanelRow:
    """One supervised sample ordered by trading date and ticker."""

    date: date
    ticker: str
    feature_available_at: datetime
    signal_generated_at: datetime
    execution_timestamp: datetime
    target_start: datetime
    target_end: datetime
    target_return: float
    target_direction: int
    feature_values: tuple[float | None, ...]
    missing_reasons: tuple[MissingReason | None, ...]
    feature_manifest_hash: str


def _session_timestamp(session: date, at: time) -> datetime:
    return datetime.combine(session, at, tzinfo=ISTANBUL)


def _normalized_features(
    snapshot: FeatureSnapshot,
    manifest: FeatureManifest,
) -> tuple[tuple[float | None, ...], tuple[MissingReason | None, ...]]:
    try:
        manifest.validate_matrix_schema(
            tuple(snapshot.values),
            manifest_hash=snapshot.feature_manifest_hash,
        )
    except FeatureSchemaError as error:
        raise PanelBuildError(str(error)) from error

    values: list[float | None] = []
    reasons: list[MissingReason | None] = []
    for name in manifest.ordered_feature_names:
        raw_value = snapshot.values[name]
        reason = snapshot.missing_reasons.get(name)
        is_missing = raw_value is None
        if raw_value is not None:
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise PanelBuildError(f"non-numeric feature value: {name}") from error
            is_missing = not math.isfinite(value)
        else:
            value = math.nan

        if is_missing and reason is None:
            raise PanelBuildError(f"missing reason required for feature: {name}")
        if not is_missing and reason is not None:
            raise PanelBuildError(f"missing reason supplied for observed feature: {name}")

        values.append(None if is_missing else value)
        reasons.append(reason)

    unknown_reasons = set(snapshot.missing_reasons) - set(manifest.ordered_feature_names)
    if unknown_reasons:
        raise PanelBuildError(
            f"missing reasons reference unknown features: {', '.join(sorted(unknown_reasons))}"
        )
    return tuple(values), tuple(reasons)


def build_canonical_panel(
    snapshots: Iterable[FeatureSnapshot],
    prices: Iterable[OHLCVBar],
    manifest: FeatureManifest,
    *,
    calendar: OfficialTradingCalendar | None = None,
) -> tuple[CanonicalPanelRow, ...]:
    """Build a sorted panel using observed next-session opens and closes.

    The target is the one-session return available to a signal formed after the
    feature date close: ``close(t+1) / open(t+1) - 1``. Proxy opens are rejected
    because they are not executable prices.
    """

    prices_by_ticker: dict[str, list[OHLCVBar]] = {}
    price_keys: set[tuple[str, date]] = set()
    for bar in prices:
        key = (bar.ticker, bar.date)
        if key in price_keys:
            raise PanelBuildError(f"duplicate price record: {bar.ticker} {bar.date}")
        price_keys.add(key)
        prices_by_ticker.setdefault(bar.ticker, []).append(bar)
    for bars in prices_by_ticker.values():
        bars.sort(key=lambda bar: bar.date)

    rows: list[CanonicalPanelRow] = []
    snapshot_keys: set[tuple[str, date]] = set()
    for snapshot in snapshots:
        key = (snapshot.ticker, snapshot.date)
        if key in snapshot_keys:
            raise PanelBuildError(f"duplicate feature snapshot: {snapshot.ticker} {snapshot.date}")
        snapshot_keys.add(key)
        if snapshot.feature_available_at.tzinfo is None:
            raise PanelBuildError("feature_available_at must be timezone-aware")

        feature_values, missing_reasons = _normalized_features(snapshot, manifest)
        target_bar = next(
            (bar for bar in prices_by_ticker.get(snapshot.ticker, ()) if bar.date > snapshot.date),
            None,
        )
        if target_bar is None:
            raise PanelBuildError(f"no future target session for {snapshot.ticker} {snapshot.date}")
        if target_bar.open_quality is not OpenQuality.OBSERVED:
            raise PanelBuildError(
                f"execution requires an observed open: {snapshot.ticker} {target_bar.date}"
            )
        if target_bar.open <= 0.0:
            raise PanelBuildError("execution open must be positive")

        if calendar is None:
            target_start = _session_timestamp(target_bar.date, SESSION_OPEN)
            target_end = _session_timestamp(target_bar.date, SESSION_CLOSE)
        else:
            try:
                target_start, target_end = calendar.session_bounds(target_bar.date)
            except ValueError as error:
                raise PanelBuildError(str(error)) from error
        signal_generated_at = snapshot.feature_available_at + timedelta(minutes=1)
        if not snapshot.feature_available_at < signal_generated_at < target_start:
            raise PanelBuildError("feature and signal timestamps must precede execution")

        target_return = target_bar.close / target_bar.open - 1.0
        rows.append(
            CanonicalPanelRow(
                date=snapshot.date,
                ticker=snapshot.ticker,
                feature_available_at=snapshot.feature_available_at,
                signal_generated_at=signal_generated_at,
                execution_timestamp=target_start,
                target_start=target_start,
                target_end=target_end,
                target_return=target_return,
                target_direction=int(target_return > 0.0),
                feature_values=feature_values,
                missing_reasons=missing_reasons,
                feature_manifest_hash=manifest.manifest_hash,
            )
        )

    return tuple(sorted(rows, key=lambda row: (row.date, row.ticker)))


def panel_to_frame(
    rows: Iterable[CanonicalPanelRow],
    manifest: FeatureManifest,
) -> pd.DataFrame:
    """Convert immutable panel rows into the explicit tabular contract."""
    feature_names = manifest.ordered_feature_names
    base_columns = [
        "date",
        "ticker",
        "feature_available_at",
        "signal_generated_at",
        "execution_timestamp",
        "target_start",
        "target_end",
        "target_return",
        "target_direction",
        "feature_manifest_hash",
    ]
    missing_columns = [f"{name}__missing_reason" for name in feature_names]
    records: list[dict[str, object]] = []
    for row in rows:
        if row.feature_manifest_hash != manifest.manifest_hash:
            raise PanelBuildError("panel row manifest hash mismatch")
        if len(row.feature_values) != len(feature_names):
            raise PanelBuildError("panel row feature width differs from manifest")
        record: dict[str, object] = {
            "date": row.date.isoformat(),
            "ticker": row.ticker,
            "feature_available_at": row.feature_available_at,
            "signal_generated_at": row.signal_generated_at,
            "execution_timestamp": row.execution_timestamp,
            "target_start": row.target_start,
            "target_end": row.target_end,
            "target_return": row.target_return,
            "target_direction": row.target_direction,
            "feature_manifest_hash": row.feature_manifest_hash,
        }
        record.update(zip(feature_names, row.feature_values))
        record.update(
            {
                column: reason.value if reason is not None else None
                for column, reason in zip(missing_columns, row.missing_reasons)
            }
        )
        records.append(record)
    columns = base_columns + list(feature_names) + missing_columns
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["date", "ticker"], kind="stable")
        .reset_index(drop=True)
    )
