"""Independent NumPy references used only for Rust verification and benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _values(data: FloatArray) -> FloatArray:
    return np.asarray(data, dtype=np.float64)


def sma_reference(data: FloatArray, period: int) -> FloatArray:
    """Reference simple moving average matching the Rust initialization."""
    values = _values(data)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if period <= 0 or len(values) < period:
        return result
    rolling_sum = float(np.sum(values[:period]))
    result[period - 1] = rolling_sum / period
    for index in range(period, len(values)):
        rolling_sum += values[index] - values[index - period]
        result[index] = rolling_sum / period
    return result


def ema_reference(data: FloatArray, period: int) -> FloatArray:
    """Reference EMA seeded with a simple average after leading NaNs."""
    values = _values(data)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if period <= 0:
        return result
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) == 0:
        return result
    first = int(valid[0])
    if first + period > len(values):
        return result
    seed_index = first + period - 1
    result[seed_index] = float(np.sum(values[first : first + period])) / period
    multiplier = 2.0 / (period + 1.0)
    for index in range(seed_index + 1, len(values)):
        result[index] = values[index] * multiplier + result[index - 1] * (1.0 - multiplier)
    return result


def rsi_reference(close: FloatArray, period: int = 14) -> FloatArray:
    """Reference Wilder RSI with the same flat-series convention as Rust."""
    values = _values(close)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if period <= 0 or len(values) <= period:
        return result
    changes = np.diff(values, prepend=values[0])
    gains = np.where(changes > 0.0, changes, 0.0)
    losses = np.where(changes < 0.0, -changes, 0.0)
    average_gain = float(np.sum(gains[1 : period + 1])) / period
    average_loss = float(np.sum(losses[1 : period + 1])) / period
    result[period] = (
        100.0 if average_loss == 0.0 else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    )
    for index in range(period + 1, len(values)):
        average_gain = (average_gain * (period - 1.0) + gains[index]) / period
        average_loss = (average_loss * (period - 1.0) + losses[index]) / period
        result[index] = (
            100.0 if average_loss == 0.0 else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
        )
    return result


def atr_reference(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    period: int = 14,
) -> FloatArray:
    """Reference Wilder ATR matching the Rust seed window."""
    highs = _values(high)
    lows = _values(low)
    closes = _values(close)
    result = np.full(len(closes), np.nan, dtype=np.float64)
    if period <= 0 or len(closes) <= period:
        return result
    true_range = np.zeros(len(closes), dtype=np.float64)
    true_range[0] = highs[0] - lows[0]
    for index in range(1, len(closes)):
        true_range[index] = max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
    average = float(np.sum(true_range[1 : period + 1])) / period
    result[period] = average
    for index in range(period + 1, len(closes)):
        average = (average * (period - 1.0) + true_range[index]) / period
        result[index] = average
    return result


def obv_reference(close: FloatArray, volume: FloatArray) -> FloatArray:
    """Reference on-balance volume."""
    closes = _values(close)
    volumes = _values(volume)
    result = np.zeros(len(closes), dtype=np.float64)
    if len(closes) == 0:
        return result
    result[0] = volumes[0]
    for index in range(1, len(closes)):
        if closes[index] > closes[index - 1]:
            result[index] = result[index - 1] + volumes[index]
        elif closes[index] < closes[index - 1]:
            result[index] = result[index - 1] - volumes[index]
        else:
            result[index] = result[index - 1]
    return result


def vwap_reference(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    volume: FloatArray,
) -> FloatArray:
    """Reference cumulative VWAP with the Rust zero-volume convention."""
    highs = _values(high)
    lows = _values(low)
    closes = _values(close)
    volumes = _values(volume)
    result = np.zeros(len(closes), dtype=np.float64)
    cumulative_volume = 0.0
    cumulative_value = 0.0
    for index in range(len(closes)):
        typical_price = (highs[index] + lows[index] + closes[index]) / 3.0
        cumulative_volume += volumes[index]
        cumulative_value += typical_price * volumes[index]
        result[index] = (
            cumulative_value / cumulative_volume if cumulative_volume > 0.0 else typical_price
        )
    return result
