"""Rust benchmark report contract."""

from __future__ import annotations

from bist_predict.research.rust_benchmark import run_benchmark


def test_benchmark_reports_timing_memory_accuracy_and_boundary_cost() -> None:
    report = run_benchmark(sizes=(1_000,), repetitions=1)

    assert report["schema_version"] == 1
    assert report["sizes"] == [1_000]
    assert report["indicators"] == ["sma", "ema", "rsi", "atr", "obv", "vwap"]
    assert report["pyo3_boundary_overhead_upper_bound_ns"] >= 0
    assert report["break_even_input_size"] in (None, 1_000)
    result = report["results"][0]
    assert result["size"] == 1_000
    assert result["rust"]["wall_clock_seconds"] > 0.0
    assert result["numpy_backed_python_reference"]["wall_clock_seconds"] > 0.0
    assert result["rust"]["peak_memory_bytes"] > 0
    assert result["numpy_backed_python_reference"]["peak_memory_bytes"] > 0
    assert result["maximum_absolute_difference"] < 1e-8
