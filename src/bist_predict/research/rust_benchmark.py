"""Reproducible Rust versus NumPy indicator benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import bist_features
from bist_predict.features.indicator_reference import (
    atr_reference,
    ema_reference,
    obv_reference,
    rsi_reference,
    sma_reference,
    vwap_reference,
)

FloatArray = NDArray[np.float64]
IndicatorSuite = tuple[FloatArray, ...]
SEED = 20260714
INDICATORS = ("sma", "ema", "rsi", "atr", "obv", "vwap")


def _market_arrays(size: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(SEED + size)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size)))
    spread = rng.uniform(0.001, 0.02, size)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    volume = rng.integers(10_000, 5_000_000, size).astype(np.float64)
    return high, low, close.astype(np.float64), volume


def _rust_suite(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    volume: FloatArray,
) -> IndicatorSuite:
    return (
        np.asarray(bist_features.compute_sma(close, 14)),
        np.asarray(bist_features.compute_ema(close, 14)),
        np.asarray(bist_features.compute_rsi(close, period=14)),
        np.asarray(bist_features.compute_atr(high, low, close, period=14)),
        np.asarray(bist_features.compute_obv(close, volume)),
        np.asarray(bist_features.compute_vwap(high, low, close, volume)),
    )


def _numpy_suite(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    volume: FloatArray,
) -> IndicatorSuite:
    return (
        sma_reference(close, 14),
        ema_reference(close, 14),
        rsi_reference(close, 14),
        atr_reference(high, low, close, 14),
        obv_reference(close, volume),
        vwap_reference(high, low, close, volume),
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _measure_worker(implementation: str, size: int, repetitions: int) -> dict[str, object]:
    arrays = _market_arrays(size)
    suite: Callable[..., IndicatorSuite]
    if implementation == "rust":
        suite = _rust_suite
    elif implementation == "numpy_backed_python_reference":
        suite = _numpy_suite
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    durations: list[float] = []
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        outputs = suite(*arrays)
        durations.append(time.perf_counter() - started)
        if len(outputs) != len(INDICATORS):
            raise RuntimeError("indicator suite returned an incomplete result")
    return {
        "wall_clock_seconds": median(durations),
        "peak_memory_bytes": _peak_rss_bytes(),
        "memory_measurement": "fresh_process_peak_rss_including_inputs",
        "repetitions": repetitions,
    }


def _subprocess_measure(
    implementation: str, size: int, repetitions: int
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bist_predict.research.rust_benchmark",
            "--worker",
            implementation,
            str(size),
            str(repetitions),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _maximum_difference(size: int) -> float:
    arrays = _market_arrays(size)
    rust_outputs = _rust_suite(*arrays)
    numpy_outputs = _numpy_suite(*arrays)
    maximum = 0.0
    for rust_output, numpy_output in zip(rust_outputs, numpy_outputs, strict=True):
        finite = np.isfinite(rust_output) & np.isfinite(numpy_output)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(rust_output[finite] - numpy_output[finite]))),
            )
        if not np.array_equal(np.isnan(rust_output), np.isnan(numpy_output)):
            raise RuntimeError("Rust and NumPy references disagree on missing positions")
    return maximum


def _boundary_overhead_upper_bound_ns(repetitions: int = 500) -> int:
    one = np.array([100.0], dtype=np.float64)
    rust_durations: list[int] = []
    baseline_durations: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        bist_features.compute_sma(one, 1)
        rust_durations.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        one.copy()
        baseline_durations.append(time.perf_counter_ns() - started)
    return max(0, int(median(rust_durations) - median(baseline_durations)))


def _git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def run_benchmark(
    *,
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000),
    repetitions: int = 3,
) -> dict[str, object]:
    """Run isolated measurements and return a JSON-serializable report."""
    if repetitions <= 0 or not sizes or any(size <= 0 for size in sizes):
        raise ValueError("benchmark sizes and repetitions must be positive")
    commit, dirty = _git_state()
    results: list[dict[str, object]] = []
    break_even: int | None = None
    for size in sizes:
        rust_result = _subprocess_measure("rust", size, repetitions)
        numpy_result = _subprocess_measure(
            "numpy_backed_python_reference", size, repetitions
        )
        rust_seconds = float(rust_result["wall_clock_seconds"])
        numpy_seconds = float(numpy_result["wall_clock_seconds"])
        if break_even is None and rust_seconds <= numpy_seconds:
            break_even = size
        results.append(
            {
                "size": size,
                "rust": rust_result,
                "numpy_backed_python_reference": numpy_result,
                "speedup": numpy_seconds / rust_seconds,
                "maximum_absolute_difference": _maximum_difference(size),
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "dirty_working_tree": dirty,
        "seed": SEED,
        "sizes": list(sizes),
        "repetitions": repetitions,
        "indicators": list(INDICATORS),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "pyo3_boundary_overhead_upper_bound_ns": _boundary_overhead_upper_bound_ns(),
        "pyo3_boundary_note": "One-element Rust SMA call minus NumPy copy; includes Rust output allocation and is an upper bound.",
        "break_even_input_size": break_even,
        "results": results,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", nargs=3, metavar=("IMPLEMENTATION", "SIZE", "REPETITIONS"))
    args = parser.parse_args()
    if args.worker:
        implementation, size, repetitions = args.worker
        print(json.dumps(_measure_worker(implementation, int(size), int(repetitions))))
        return
    report = run_benchmark(repetitions=args.repetitions)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    _main()
