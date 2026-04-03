"""Tests for Rust technical indicators — verified against known reference values."""

from __future__ import annotations

import numpy as np
import pytest

import bist_features


# Reference price data (20 days of synthetic THYAO-like prices)
CLOSE = np.array([
    300.0, 302.5, 301.0, 305.0, 308.0, 306.5, 310.0, 312.0, 309.5, 311.0,
    315.0, 313.5, 316.0, 318.0, 317.0, 320.0, 322.5, 321.0, 319.5, 323.0,
], dtype=np.float64)

HIGH = np.array([
    302.0, 304.0, 303.0, 306.5, 309.0, 308.0, 311.5, 313.5, 312.0, 313.0,
    316.5, 315.0, 317.5, 319.5, 318.5, 321.5, 324.0, 323.0, 321.0, 325.0,
], dtype=np.float64)

LOW = np.array([
    298.0, 300.5, 299.0, 303.0, 306.0, 304.5, 308.0, 310.0, 307.5, 309.0,
    313.0, 311.5, 314.0, 316.0, 315.0, 318.0, 320.5, 319.0, 317.5, 321.0,
], dtype=np.float64)

VOLUME = np.array([
    1000000, 1100000, 950000, 1200000, 1300000, 1050000, 1400000, 1500000,
    1100000, 1250000, 1600000, 1150000, 1350000, 1450000, 1200000, 1550000,
    1700000, 1300000, 1050000, 1650000,
], dtype=np.float64)


class TestRSI:
    def test_output_length_matches_input(self) -> None:
        rsi = bist_features.compute_rsi(CLOSE, period=14)
        assert len(rsi) == len(CLOSE)

    def test_first_values_are_nan(self) -> None:
        rsi = bist_features.compute_rsi(CLOSE, period=14)
        for i in range(14):
            assert np.isnan(rsi[i])

    def test_values_between_0_and_100(self) -> None:
        rsi = bist_features.compute_rsi(CLOSE, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_uptrend_rsi_above_50(self) -> None:
        rsi = bist_features.compute_rsi(CLOSE, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.mean(valid) > 50.0


class TestSMA:
    def test_output_length(self) -> None:
        sma = bist_features.compute_sma(CLOSE, 5)
        assert len(sma) == len(CLOSE)

    def test_first_values_are_nan(self) -> None:
        sma = bist_features.compute_sma(CLOSE, 5)
        for i in range(4):
            assert np.isnan(sma[i])

    def test_known_value(self) -> None:
        sma = bist_features.compute_sma(CLOSE, 5)
        expected = np.mean(CLOSE[:5])
        assert abs(sma[4] - expected) < 0.001


class TestEMA:
    def test_output_length(self) -> None:
        ema = bist_features.compute_ema(CLOSE, 5)
        assert len(ema) == len(CLOSE)

    def test_first_value_equals_sma(self) -> None:
        ema = bist_features.compute_ema(CLOSE, 5)
        expected_sma = np.mean(CLOSE[:5])
        assert abs(ema[4] - expected_sma) < 0.001


class TestMACD:
    def test_returns_three_arrays(self) -> None:
        macd, signal, hist = bist_features.compute_macd(CLOSE)
        assert len(macd) == len(CLOSE)
        assert len(signal) == len(CLOSE)
        assert len(hist) == len(CLOSE)

    def test_histogram_equals_macd_minus_signal(self) -> None:
        macd, signal, hist = bist_features.compute_macd(CLOSE)
        for i in range(len(CLOSE)):
            if not np.isnan(macd[i]) and not np.isnan(signal[i]) and not np.isnan(hist[i]):
                assert abs(hist[i] - (macd[i] - signal[i])) < 0.001


class TestBollingerBands:
    def test_returns_three_arrays(self) -> None:
        upper, middle, lower = bist_features.compute_bollinger_bands(CLOSE, period=20)
        assert len(upper) == len(CLOSE)

    def test_upper_above_lower(self) -> None:
        upper, middle, lower = bist_features.compute_bollinger_bands(CLOSE, period=10)
        for i in range(len(CLOSE)):
            if not np.isnan(upper[i]):
                assert upper[i] >= lower[i]

    def test_middle_is_sma(self) -> None:
        upper, middle, lower = bist_features.compute_bollinger_bands(CLOSE, period=5)
        sma = bist_features.compute_sma(CLOSE, 5)
        for i in range(len(CLOSE)):
            if not np.isnan(middle[i]):
                assert abs(middle[i] - sma[i]) < 0.001


class TestStochastic:
    def test_returns_two_arrays(self) -> None:
        k, d = bist_features.compute_stochastic(HIGH, LOW, CLOSE)
        assert len(k) == len(CLOSE)
        assert len(d) == len(CLOSE)

    def test_k_between_0_and_100(self) -> None:
        k, d = bist_features.compute_stochastic(HIGH, LOW, CLOSE)
        valid = k[~np.isnan(k)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)


class TestATR:
    def test_output_length(self) -> None:
        atr = bist_features.compute_atr(HIGH, LOW, CLOSE, period=14)
        assert len(atr) == len(CLOSE)

    def test_atr_positive(self) -> None:
        atr = bist_features.compute_atr(HIGH, LOW, CLOSE, period=14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid > 0.0)


class TestOBV:
    def test_output_length(self) -> None:
        obv = bist_features.compute_obv(CLOSE, VOLUME)
        assert len(obv) == len(CLOSE)

    def test_first_value_equals_first_volume(self) -> None:
        obv = bist_features.compute_obv(CLOSE, VOLUME)
        assert obv[0] == VOLUME[0]

    def test_up_day_adds_volume(self) -> None:
        obv = bist_features.compute_obv(CLOSE, VOLUME)
        assert obv[1] == VOLUME[0] + VOLUME[1]


class TestVWAP:
    def test_output_length(self) -> None:
        vwap = bist_features.compute_vwap(HIGH, LOW, CLOSE, VOLUME)
        assert len(vwap) == len(CLOSE)

    def test_first_value(self) -> None:
        vwap = bist_features.compute_vwap(HIGH, LOW, CLOSE, VOLUME)
        expected_tp = (HIGH[0] + LOW[0] + CLOSE[0]) / 3.0
        assert abs(vwap[0] - expected_tp) < 0.001


class TestCCI:
    def test_output_length(self) -> None:
        cci = bist_features.compute_cci(HIGH, LOW, CLOSE, period=14)
        assert len(cci) == len(CLOSE)


class TestMFI:
    def test_output_length(self) -> None:
        mfi = bist_features.compute_mfi(HIGH, LOW, CLOSE, VOLUME, period=14)
        assert len(mfi) == len(CLOSE)

    def test_values_between_0_and_100(self) -> None:
        mfi = bist_features.compute_mfi(HIGH, LOW, CLOSE, VOLUME, period=14)
        valid = mfi[~np.isnan(mfi)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)


class TestWilliamsR:
    def test_output_length(self) -> None:
        wr = bist_features.compute_williams_r(HIGH, LOW, CLOSE, period=14)
        assert len(wr) == len(CLOSE)

    def test_values_between_neg100_and_0(self) -> None:
        wr = bist_features.compute_williams_r(HIGH, LOW, CLOSE, period=14)
        valid = wr[~np.isnan(wr)]
        assert np.all(valid >= -100.0)
        assert np.all(valid <= 0.0)
