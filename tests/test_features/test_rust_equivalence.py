"""Numerical equivalence between Rust indicators and independent references."""

from __future__ import annotations

import numpy as np
import pytest

bist_features = pytest.importorskip(
    "bist_features",
    reason=(
        "the Rust indicator extension is optional; build it with "
        "'cd rust/bist_features && uv run --project ../.. maturin develop --release'"
    ),
)
from bist_predict.features.indicator_reference import (  # noqa: E402
    atr_reference,
    ema_reference,
    obv_reference,
    rsi_reference,
    sma_reference,
    vwap_reference,
)


def _market_arrays(size: int = 256) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260714)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size)))
    spread = rng.uniform(0.001, 0.02, size)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    volume = rng.integers(10_000, 5_000_000, size).astype(np.float64)
    return high, low, close.astype(np.float64), volume


@pytest.mark.parametrize(
    ("rust_result", "reference_result"),
    [
        (
            lambda high, low, close, volume: bist_features.compute_sma(close, 14),
            lambda high, low, close, volume: sma_reference(close, 14),
        ),
        (
            lambda high, low, close, volume: bist_features.compute_ema(close, 14),
            lambda high, low, close, volume: ema_reference(close, 14),
        ),
        (
            lambda high, low, close, volume: bist_features.compute_rsi(close, period=14),
            lambda high, low, close, volume: rsi_reference(close, 14),
        ),
        (
            lambda high, low, close, volume: bist_features.compute_atr(high, low, close, period=14),
            lambda high, low, close, volume: atr_reference(high, low, close, 14),
        ),
        (
            lambda high, low, close, volume: bist_features.compute_obv(close, volume),
            lambda high, low, close, volume: obv_reference(close, volume),
        ),
        (
            lambda high, low, close, volume: bist_features.compute_vwap(high, low, close, volume),
            lambda high, low, close, volume: vwap_reference(high, low, close, volume),
        ),
    ],
)
def test_rust_matches_independent_reference_on_market_data(
    rust_result: object, reference_result: object
) -> None:
    high, low, close, volume = _market_arrays()
    actual = rust_result(high, low, close, volume)  # type: ignore[operator]
    expected = reference_result(high, low, close, volume)  # type: ignore[operator]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)


def test_rust_matches_reference_on_constant_and_zero_volume_inputs() -> None:
    close = np.full(64, 100.0, dtype=np.float64)
    high = close.copy()
    low = close.copy()
    volume = np.zeros(64, dtype=np.float64)

    np.testing.assert_allclose(
        bist_features.compute_rsi(close, period=14),
        rsi_reference(close, 14),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bist_features.compute_atr(high, low, close, period=14),
        atr_reference(high, low, close, 14),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bist_features.compute_vwap(high, low, close, volume),
        vwap_reference(high, low, close, volume),
        equal_nan=True,
    )


@pytest.mark.parametrize("size", [0, 1, 13, 14])
def test_short_inputs_match_reference_without_fabricating_values(size: int) -> None:
    close = np.arange(1, size + 1, dtype=np.float64)

    np.testing.assert_allclose(
        bist_features.compute_sma(close, 14),
        sma_reference(close, 14),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bist_features.compute_ema(close, 14),
        ema_reference(close, 14),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bist_features.compute_rsi(close, period=14),
        rsi_reference(close, 14),
        equal_nan=True,
    )
