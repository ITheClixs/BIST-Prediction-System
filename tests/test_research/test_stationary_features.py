"""Point-in-time and scale-invariant accepted feature tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from bist_predict.ingest.types import OHLCVBar
from bist_predict.research.stationary_features import (
    FeatureHistoryError,
    STATIONARY_FEATURE_MANIFEST,
    build_stationary_snapshots,
)


def _bars(*, sessions: int = 130, price_scale: float = 1.0) -> list[OHLCVBar]:
    start = date(2025, 1, 1)
    bars: list[OHLCVBar] = []
    for ticker_index, ticker in enumerate(("GARAN", "THYAO")):
        for offset in range(sessions):
            trend = 80.0 + ticker_index * 40.0 + offset * (0.15 + ticker_index * 0.03)
            close = trend + ((offset % 7) - 3) * 0.2
            open_price = close * (1.0 + ((offset % 3) - 1) * 0.001)
            bars.append(
                OHLCVBar(
                    ticker=ticker,
                    date=start + timedelta(days=offset),
                    open=open_price * price_scale,
                    high=max(open_price, close) * 1.01 * price_scale,
                    low=min(open_price, close) * 0.99 * price_scale,
                    close=close * price_scale,
                    adj_close=close * price_scale,
                    volume=1_000_000 + ticker_index * 50_000 + offset * 1_000,
                    source="synthetic",
                )
            )
    return bars


def _snapshot_map(bars: list[OHLCVBar]) -> dict[tuple[date, str], tuple[float | None, ...]]:
    return {
        (snapshot.date, snapshot.ticker): tuple(snapshot.values.values())
        for snapshot in build_stationary_snapshots(bars)
    }


def test_future_price_perturbation_cannot_change_prior_features() -> None:
    bars = _bars()
    cutoff = date(2025, 4, 25)
    perturbed = [
        replace(
            bar,
            open=bar.open * 100.0,
            high=bar.high * 100.0,
            low=bar.low * 100.0,
            close=bar.close * 100.0,
            adj_close=bar.adj_close * 100.0,
            volume=bar.volume * 100,
        )
        if bar.date > cutoff
        else bar
        for bar in bars
    ]

    original = _snapshot_map(bars)
    changed = _snapshot_map(perturbed)

    prior_keys = [key for key in original if key[0] <= cutoff]
    assert prior_keys
    assert all(changed[key] == pytest.approx(original[key]) for key in prior_keys)


def test_nominal_price_scale_cannot_identify_a_ticker() -> None:
    original = build_stationary_snapshots(_bars())
    scaled = build_stationary_snapshots(_bars(price_scale=10.0))

    assert len(original) == len(scaled)
    for left, right in zip(original, scaled, strict=True):
        assert (left.date, left.ticker) == (right.date, right.ticker)
        assert tuple(right.values.values()) == pytest.approx(tuple(left.values.values()), abs=1e-12)


def test_every_enabled_feature_is_observed_on_eligible_samples() -> None:
    snapshots = build_stationary_snapshots(_bars())

    assert snapshots
    assert tuple(snapshots[0].values) == STATIONARY_FEATURE_MANIFEST.ordered_feature_names
    for name in STATIONARY_FEATURE_MANIFEST.ordered_feature_names:
        assert any(snapshot.values[name] is not None for snapshot in snapshots), name
    assert not {
        "close",
        "sma_20",
        "ema_20",
        "atr_14",
        "vwap",
        "volume",
        "obv",
    }.intersection(STATIONARY_FEATURE_MANIFEST.ordered_feature_names)


def test_requested_lookback_and_target_horizon_must_fit_each_ticker() -> None:
    with pytest.raises(FeatureHistoryError, match="THYAO.*101"):
        build_stationary_snapshots(
            [bar for bar in _bars(sessions=100) if bar.ticker == "THYAO"],
            target_horizon_sessions=1,
        )
