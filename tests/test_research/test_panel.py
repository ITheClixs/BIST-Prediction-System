"""Canonical panel and executable-target invariants."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.ingest.types import OHLCVBar, OpenQuality, VolumeQuality
from bist_predict.research.panel import (
    FeatureSnapshot,
    MissingReason,
    PanelBuildError,
    build_canonical_panel,
    panel_to_frame,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")


@pytest.fixture
def manifest() -> FeatureManifest:
    return FeatureManifest(
        schema_version="1.0.0",
        features=(
            FeatureSpec(
                name="log_return_1d",
                formula="log(adj_close_t / adj_close_t_minus_1)",
                formula_version="1",
                lookback=2,
                availability_rule="after_session_close",
                missing_value_policy="preserve",
                normalization_policy="none",
            ),
            FeatureSpec(
                name="atr_over_close",
                formula="atr_14 / adj_close",
                formula_version="1",
                lookback=15,
                availability_rule="after_session_close",
                missing_value_policy="preserve_with_reason",
                normalization_policy="none",
            ),
        ),
    )


def _bar(
    ticker: str,
    session: date,
    *,
    open_price: float,
    close: float,
    open_quality: OpenQuality = OpenQuality.OBSERVED,
) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker,
        date=session,
        open=open_price,
        high=max(open_price, close) + 1.0,
        low=min(open_price, close) - 1.0,
        close=close,
        adj_close=close,
        volume=10_000,
        source="test",
        open_quality=open_quality,
        volume_quality=VolumeQuality.OBSERVED,
    )


def _snapshot(
    ticker: str,
    session: date,
    manifest: FeatureManifest,
    *,
    atr: float | None = 0.02,
    reasons: dict[str, MissingReason] | None = None,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        date=session,
        ticker=ticker,
        feature_available_at=datetime(
            session.year, session.month, session.day, 18, 10, tzinfo=ISTANBUL
        ),
        values={"log_return_1d": 0.01, "atr_over_close": atr},
        missing_reasons=reasons or {},
        feature_manifest_hash=manifest.manifest_hash,
    )


def test_panel_is_sorted_and_target_matches_next_open_to_close(
    manifest: FeatureManifest,
) -> None:
    feature_date = date(2026, 1, 5)
    target_date = date(2026, 1, 6)
    snapshots = [
        _snapshot("THYAO", feature_date, manifest),
        _snapshot("GARAN", feature_date, manifest),
    ]
    prices = [
        _bar("THYAO", target_date, open_price=100.0, close=102.0),
        _bar("GARAN", target_date, open_price=50.0, close=49.0),
    ]

    rows = build_canonical_panel(reversed(snapshots), reversed(prices), manifest)

    assert [(row.date, row.ticker) for row in rows] == [
        (feature_date, "GARAN"),
        (feature_date, "THYAO"),
    ]
    assert rows[0].target_return == pytest.approx(-0.02)
    assert rows[0].target_direction == 0
    assert rows[1].target_return == pytest.approx(0.02)
    assert rows[1].target_direction == 1
    assert all(row.execution_timestamp == row.target_start for row in rows)
    assert all(row.feature_available_at < row.execution_timestamp for row in rows)
    assert all(row.execution_timestamp <= row.target_start for row in rows)


def test_panel_preserves_missing_value_and_requires_reason(
    manifest: FeatureManifest,
) -> None:
    feature_date = date(2026, 1, 5)
    target_date = date(2026, 1, 6)
    prices = [_bar("THYAO", target_date, open_price=100.0, close=101.0)]

    with pytest.raises(PanelBuildError, match="missing reason"):
        build_canonical_panel(
            [_snapshot("THYAO", feature_date, manifest, atr=None)], prices, manifest
        )

    rows = build_canonical_panel(
        [
            _snapshot(
                "THYAO",
                feature_date,
                manifest,
                atr=None,
                reasons={"atr_over_close": MissingReason.INSUFFICIENT_LOOKBACK},
            )
        ],
        prices,
        manifest,
    )

    assert rows[0].feature_values == (0.01, None)
    assert rows[0].missing_reasons == (None, MissingReason.INSUFFICIENT_LOOKBACK)


def test_panel_rejects_proxy_open_for_execution(manifest: FeatureManifest) -> None:
    feature_date = date(2026, 1, 5)
    target_date = date(2026, 1, 6)
    snapshot = _snapshot("THYAO", feature_date, manifest)
    proxy_bar = _bar(
        "THYAO",
        target_date,
        open_price=100.0,
        close=101.0,
        open_quality=OpenQuality.PROXY,
    )

    with pytest.raises(PanelBuildError, match="observed open"):
        build_canonical_panel([snapshot], [proxy_bar], manifest)


def test_panel_frame_keeps_timestamps_targets_and_manifest_order(
    manifest: FeatureManifest,
) -> None:
    feature_date = date(2026, 1, 5)
    target_date = date(2026, 1, 6)
    rows = build_canonical_panel(
        [_snapshot("THYAO", feature_date, manifest)],
        [_bar("THYAO", target_date, open_price=100.0, close=102.0)],
        manifest,
    )

    frame = panel_to_frame(rows, manifest)

    assert list(frame.columns) == [
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
        "log_return_1d",
        "atr_over_close",
        "log_return_1d__missing_reason",
        "atr_over_close__missing_reason",
    ]
    assert frame.loc[0, "target_return"] == pytest.approx(0.02)
    assert frame.loc[0, "feature_manifest_hash"] == manifest.manifest_hash
