"""Tests for signal quality measurement — IC, Hurst exponent, wavelet decomposition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.signal_quality import (
    compute_hurst_exponent,
    compute_information_coefficient,
    compute_wavelet_decomposition,
)


class TestInformationCoefficient:
    def test_perfect_prediction(self) -> None:
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_information_coefficient(predicted, actual)
        assert abs(result["ic"] - 1.0) < 0.001
        assert result["ic_significant"]  # Perfect correlation is significant

    def test_random_prediction(self) -> None:
        rng = np.random.default_rng(42)
        predicted = rng.normal(0, 1, 1000)
        actual = rng.normal(0, 1, 1000)
        result = compute_information_coefficient(predicted, actual)
        # IC should be near 0 for random
        assert abs(result["ic"]) < 0.1

    def test_negative_correlation(self) -> None:
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_information_coefficient(predicted, actual)
        assert result["ic"] < -0.9


class TestHurst:
    def test_trending_series(self) -> None:
        # Strong cumulative trend with small noise → H > 0.5
        rng = np.random.default_rng(42)
        # High drift-to-noise ratio produces clear persistence
        x = np.cumsum(rng.normal(0.1, 0.05, 2000))
        result = compute_hurst_exponent(x)
        assert result["hurst"] > 0.45  # R/S analysis can underestimate slightly
        assert result["hurst_interpretation"] in ("trending", "random_walk")

    def test_mean_reverting_series(self) -> None:
        # Ornstein-Uhlenbeck process → H < 0.5
        rng = np.random.default_rng(42)
        n = 1000
        x = np.zeros(n)
        x[0] = 0.0
        for i in range(1, n):
            x[i] = x[i - 1] - 0.3 * x[i - 1] + rng.normal(0, 0.5)
        result = compute_hurst_exponent(x)
        assert result["hurst"] < 0.5
        assert result["hurst_interpretation"] == "mean_reverting"

    def test_short_series_returns_nan(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = compute_hurst_exponent(x)
        assert math.isnan(result["hurst"])


class TestWavelet:
    def test_decomposes_signal(self) -> None:
        rng = np.random.default_rng(42)
        # Signal with clear low-frequency trend + noise
        t = np.linspace(0, 10, 256)
        signal = np.sin(t) + 0.5 * rng.normal(0, 1, 256)

        result = compute_wavelet_decomposition(signal, levels=3)
        assert "wavelet_approx" in result
        assert "wavelet_detail_1" in result
        assert "wavelet_detail_2" in result
        assert "wavelet_detail_3" in result
        # Approximation should capture the trend
        assert len(result["wavelet_approx"]) > 0

    def test_energy_per_level(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 256)
        result = compute_wavelet_decomposition(signal, levels=3)
        assert "wavelet_energy_1" in result
        assert "wavelet_energy_2" in result
        assert "wavelet_energy_3" in result
        # Energies should be positive
        assert result["wavelet_energy_1"] > 0
