"""Tests for Rust candlestick pattern detection."""

from __future__ import annotations

import numpy as np
import pytest

import bist_features


class TestDetectPatterns:
    def test_returns_expected_patterns(self) -> None:
        open = np.array([100.0, 102.0, 101.0, 105.0], dtype=np.float64)
        high = np.array([103.0, 104.0, 103.5, 106.0], dtype=np.float64)
        low = np.array([99.0, 100.5, 99.5, 103.0], dtype=np.float64)
        close = np.array([102.0, 101.0, 103.0, 105.5], dtype=np.float64)

        patterns = bist_features.detect_patterns(open, high, low, close)
        pattern_names = [name for name, _ in patterns]
        assert "doji" in pattern_names
        assert "hammer" in pattern_names
        assert "engulfing" in pattern_names
        assert "morning_star" in pattern_names

    def test_doji_detection(self) -> None:
        open = np.array([100.0, 100.0], dtype=np.float64)
        high = np.array([100.0, 105.0], dtype=np.float64)
        low = np.array([100.0, 95.0], dtype=np.float64)
        close = np.array([100.0, 100.5], dtype=np.float64)

        patterns = dict(bist_features.detect_patterns(open, high, low, close))
        assert patterns["doji"][1] == 1

    def test_output_lengths_match(self) -> None:
        n = 10
        o = np.random.uniform(100, 110, n).astype(np.float64)
        h = (o + np.random.uniform(0, 5, n)).astype(np.float64)
        l = (o - np.random.uniform(0, 5, n)).astype(np.float64)
        c = np.random.uniform(l, h).astype(np.float64)

        patterns = bist_features.detect_patterns(o, h, l, c)
        for name, arr in patterns:
            assert len(arr) == n
