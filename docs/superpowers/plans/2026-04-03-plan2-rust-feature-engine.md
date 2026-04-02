# Plan 2: Rust Feature Engine & Python Feature Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the high-performance Rust feature computation engine (via PyO3) for technical indicators, candlestick patterns, and cross-stock analysis, plus the Python feature layer for macro, sentiment, and temporal features, and a feature store for persistence.

**Architecture:** Rust crate `bist_features` compiled as a Python module via PyO3/maturin. Exposes functions that accept numpy arrays and return computed indicator arrays. Python layer handles I/O-bound features (macro deltas, sentiment aggregation, calendar effects) and orchestrates feature computation. Feature store persists computed features in SQLite keyed by (ticker, date).

**Tech Stack:** Rust, PyO3, maturin, numpy, Python 3.12+, SQLite, pytest

**Design spec:** `docs/superpowers/specs/2026-04-02-bist-predictor-design.md` (Section 2)

---

## File Structure

```
rust/
└── bist_features/
    ├── Cargo.toml
    └── src/
        ├── lib.rs              # PyO3 module entry, exposes Python API
        ├── indicators.rs       # Technical indicators (RSI, MACD, BB, etc.)
        ├── patterns.rs         # Candlestick pattern detection
        └── correlations.rs     # Cross-stock correlations, beta, sector momentum

src/bist_predict/
    ├── features/
    │   ├── __init__.py
    │   ├── engine.py           # Orchestrates all feature computation
    │   ├── macro_features.py   # Macro indicator deltas/trends
    │   ├── sentiment_features.py # Sentiment aggregation per stock
    │   ├── temporal_features.py  # Calendar/temporal features
    │   └── store.py            # Feature store (read/write to SQLite)

tests/
    ├── test_features/
    │   ├── __init__.py
    │   ├── test_rust_indicators.py
    │   ├── test_rust_patterns.py
    │   ├── test_rust_correlations.py
    │   ├── test_macro_features.py
    │   ├── test_sentiment_features.py
    │   ├── test_temporal_features.py
    │   ├── test_store.py
    │   └── test_engine.py
```

---

### Task 1: Rust Crate Scaffolding

**Files:**
- Create: `rust/bist_features/Cargo.toml`
- Create: `rust/bist_features/src/lib.rs`
- Modify: `Cargo.toml` (workspace — already has placeholder)
- Modify: `pyproject.toml` (add maturin build dependency)

- [ ] **Step 1: Create Rust crate directory**

```bash
mkdir -p rust/bist_features/src
```

- [ ] **Step 2: Create rust/bist_features/Cargo.toml**

```toml
[package]
name = "bist_features"
version = "0.1.0"
edition = "2021"

[lib]
name = "bist_features"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
```

- [ ] **Step 3: Create rust/bist_features/src/lib.rs**

```rust
use pyo3::prelude::*;

mod indicators;
mod patterns;
mod correlations;

/// BIST Features — high-performance technical analysis engine
#[pymodule]
fn bist_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(indicators::compute_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_sma, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_ema, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_macd, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_atr, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_obv, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_adx, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_cci, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_mfi, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::compute_williams_r, m)?)?;
    m.add_function(wrap_pyfunction!(patterns::detect_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(correlations::compute_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(correlations::compute_beta, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Create stub files for indicators.rs, patterns.rs, correlations.rs**

`rust/bist_features/src/indicators.rs`:

```rust
use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute Relative Strength Index (RSI) with given period.
/// Formula: RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
#[pyfunction]
#[pyo3(signature = (close, period=14))]
pub fn compute_rsi(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let close = close.as_array();
    let len = close.len();
    let mut rsi = vec![f64::NAN; len];

    if len <= period {
        return PyArray1::from_vec(py, rsi).into();
    }

    // Calculate initial average gain and loss
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=period {
        let change = close[i] - close[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    if avg_loss == 0.0 {
        rsi[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Smoothed RSI for remaining values
    for i in (period + 1)..len {
        let change = close[i] - close[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    PyArray1::from_vec(py, rsi).into()
}

/// Compute Simple Moving Average.
#[pyfunction]
pub fn compute_sma(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let close = close.as_array();
    let len = close.len();
    let mut sma = vec![f64::NAN; len];

    if len < period {
        return PyArray1::from_vec(py, sma).into();
    }

    let mut sum: f64 = close.iter().take(period).sum();
    sma[period - 1] = sum / period as f64;

    for i in period..len {
        sum += close[i] - close[i - period];
        sma[i] = sum / period as f64;
    }

    PyArray1::from_vec(py, sma).into()
}

/// Compute Exponential Moving Average.
#[pyfunction]
pub fn compute_ema(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let close = close.as_array();
    let len = close.len();
    let mut ema = vec![f64::NAN; len];

    if len < period {
        return PyArray1::from_vec(py, ema).into();
    }

    // Seed EMA with SMA
    let sma: f64 = close.iter().take(period).sum::<f64>() / period as f64;
    ema[period - 1] = sma;

    let multiplier = 2.0 / (period as f64 + 1.0);
    for i in period..len {
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    PyArray1::from_vec(py, ema).into()
}

/// Compute MACD (Moving Average Convergence Divergence).
/// Returns (macd_line, signal_line, histogram) as three arrays.
#[pyfunction]
#[pyo3(signature = (close, fast_period=12, slow_period=26, signal_period=9))]
pub fn compute_macd(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let close = close.as_array();
    let len = close.len();

    let fast_ema = compute_ema_raw(&close, fast_period);
    let slow_ema = compute_ema_raw(&close, slow_period);

    let mut macd_line = vec![f64::NAN; len];
    for i in 0..len {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Compute signal line as EMA of MACD line (skipping NaNs)
    let valid_macd: Vec<f64> = macd_line.iter().copied().filter(|v| !v.is_nan()).collect();
    let signal_raw = compute_ema_raw_vec(&valid_macd, signal_period);

    let mut signal_line = vec![f64::NAN; len];
    let mut j = 0;
    for i in 0..len {
        if !macd_line[i].is_nan() {
            if j < signal_raw.len() {
                signal_line[i] = signal_raw[j];
            }
            j += 1;
        }
    }

    let mut histogram = vec![f64::NAN; len];
    for i in 0..len {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    (
        PyArray1::from_vec(py, macd_line).into(),
        PyArray1::from_vec(py, signal_line).into(),
        PyArray1::from_vec(py, histogram).into(),
    )
}

/// Compute Bollinger Bands.
/// Returns (upper, middle, lower) bands.
#[pyfunction]
#[pyo3(signature = (close, period=20, num_std=2.0))]
pub fn compute_bollinger_bands(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    period: usize,
    num_std: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let close = close.as_array();
    let len = close.len();
    let mut upper = vec![f64::NAN; len];
    let mut middle = vec![f64::NAN; len];
    let mut lower = vec![f64::NAN; len];

    if len < period {
        return (
            PyArray1::from_vec(py, upper).into(),
            PyArray1::from_vec(py, middle).into(),
            PyArray1::from_vec(py, lower).into(),
        );
    }

    for i in (period - 1)..len {
        let window = &close.as_slice().unwrap()[i + 1 - period..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();

        middle[i] = mean;
        upper[i] = mean + num_std * std_dev;
        lower[i] = mean - num_std * std_dev;
    }

    (
        PyArray1::from_vec(py, upper).into(),
        PyArray1::from_vec(py, middle).into(),
        PyArray1::from_vec(py, lower).into(),
    )
}

/// Compute Stochastic Oscillator (%K and %D).
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period=14, d_period=3))]
pub fn compute_stochastic(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    k_period: usize,
    d_period: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = close.len();

    let mut k = vec![f64::NAN; len];

    for i in (k_period - 1)..len {
        let h_slice = &high.as_slice().unwrap()[i + 1 - k_period..=i];
        let l_slice = &low.as_slice().unwrap()[i + 1 - k_period..=i];
        let highest = h_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = l_slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = highest - lowest;
        if range > 0.0 {
            k[i] = ((close[i] - lowest) / range) * 100.0;
        } else {
            k[i] = 50.0;
        }
    }

    // %D is SMA of %K
    let d = sma_with_nan(&k, d_period);

    (
        PyArray1::from_vec(py, k).into(),
        PyArray1::from_vec(py, d).into(),
    )
}

/// Compute Average True Range (ATR).
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_atr(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = close.len();
    let mut atr = vec![f64::NAN; len];

    if len <= period {
        return PyArray1::from_vec(py, atr).into();
    }

    // True Range
    let mut tr = vec![0.0; len];
    tr[0] = high[0] - low[0];
    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Initial ATR as average of first `period` TRs
    let initial: f64 = tr[1..=period].iter().sum::<f64>() / period as f64;
    atr[period] = initial;

    // Smoothed ATR
    for i in (period + 1)..len {
        atr[i] = (atr[i - 1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }

    PyArray1::from_vec(py, atr).into()
}

/// Compute On-Balance Volume (OBV).
#[pyfunction]
pub fn compute_obv(py: Python<'_>, close: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let close = close.as_array();
    let volume = volume.as_array();
    let len = close.len();
    let mut obv = vec![0.0; len];

    if len == 0 {
        return PyArray1::from_vec(py, obv).into();
    }

    obv[0] = volume[0];
    for i in 1..len {
        if close[i] > close[i - 1] {
            obv[i] = obv[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            obv[i] = obv[i - 1] - volume[i];
        } else {
            obv[i] = obv[i - 1];
        }
    }

    PyArray1::from_vec(py, obv).into()
}

/// Compute Volume Weighted Average Price (VWAP).
/// Resets daily — for daily bars this is cumulative VWAP.
#[pyfunction]
pub fn compute_vwap(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    let len = close.len();
    let mut vwap = vec![f64::NAN; len];

    let mut cum_vol = 0.0;
    let mut cum_tp_vol = 0.0;

    for i in 0..len {
        let typical_price = (high[i] + low[i] + close[i]) / 3.0;
        cum_vol += volume[i];
        cum_tp_vol += typical_price * volume[i];
        if cum_vol > 0.0 {
            vwap[i] = cum_tp_vol / cum_vol;
        }
    }

    PyArray1::from_vec(py, vwap).into()
}

/// Compute Average Directional Index (ADX).
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_adx(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = close.len();
    let mut adx = vec![f64::NAN; len];

    if len <= 2 * period {
        return PyArray1::from_vec(py, adx).into();
    }

    // +DM, -DM, TR
    let mut plus_dm = vec![0.0; len];
    let mut minus_dm = vec![0.0; len];
    let mut tr = vec![0.0; len];

    for i in 1..len {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];

        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };

        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Smoothed +DM, -DM, TR using Wilder's smoothing
    let mut smooth_plus_dm = vec![0.0; len];
    let mut smooth_minus_dm = vec![0.0; len];
    let mut smooth_tr = vec![0.0; len];

    smooth_plus_dm[period] = plus_dm[1..=period].iter().sum();
    smooth_minus_dm[period] = minus_dm[1..=period].iter().sum();
    smooth_tr[period] = tr[1..=period].iter().sum();

    for i in (period + 1)..len {
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - smooth_plus_dm[i - 1] / period as f64 + plus_dm[i];
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - smooth_minus_dm[i - 1] / period as f64 + minus_dm[i];
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period as f64 + tr[i];
    }

    // +DI, -DI, DX
    let mut dx = vec![f64::NAN; len];
    for i in period..len {
        if smooth_tr[i] > 0.0 {
            let plus_di = 100.0 * smooth_plus_dm[i] / smooth_tr[i];
            let minus_di = 100.0 * smooth_minus_dm[i] / smooth_tr[i];
            let di_sum = plus_di + minus_di;
            if di_sum > 0.0 {
                dx[i] = 100.0 * (plus_di - minus_di).abs() / di_sum;
            }
        }
    }

    // ADX as smoothed DX
    let first_adx_idx = 2 * period;
    if first_adx_idx < len {
        let mut sum = 0.0;
        let mut count = 0;
        for i in period..=first_adx_idx {
            if !dx[i].is_nan() {
                sum += dx[i];
                count += 1;
            }
        }
        if count > 0 {
            adx[first_adx_idx] = sum / count as f64;
            for i in (first_adx_idx + 1)..len {
                if !dx[i].is_nan() {
                    adx[i] = (adx[i - 1] * (period as f64 - 1.0) + dx[i]) / period as f64;
                }
            }
        }
    }

    PyArray1::from_vec(py, adx).into()
}

/// Compute Commodity Channel Index (CCI).
#[pyfunction]
#[pyo3(signature = (high, low, close, period=20))]
pub fn compute_cci(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = close.len();
    let mut cci = vec![f64::NAN; len];

    if len < period {
        return PyArray1::from_vec(py, cci).into();
    }

    // Typical prices
    let tp: Vec<f64> = (0..len).map(|i| (high[i] + low[i] + close[i]) / 3.0).collect();

    for i in (period - 1)..len {
        let window = &tp[i + 1 - period..=i];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let mean_dev: f64 = window.iter().map(|x| (x - mean).abs()).sum::<f64>() / period as f64;
        if mean_dev > 0.0 {
            cci[i] = (tp[i] - mean) / (0.015 * mean_dev);
        } else {
            cci[i] = 0.0;
        }
    }

    PyArray1::from_vec(py, cci).into()
}

/// Compute Money Flow Index (MFI).
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period=14))]
pub fn compute_mfi(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    let len = close.len();
    let mut mfi = vec![f64::NAN; len];

    if len <= period {
        return PyArray1::from_vec(py, mfi).into();
    }

    let tp: Vec<f64> = (0..len).map(|i| (high[i] + low[i] + close[i]) / 3.0).collect();
    let raw_mf: Vec<f64> = (0..len).map(|i| tp[i] * volume[i]).collect();

    for i in period..len {
        let mut pos_flow = 0.0;
        let mut neg_flow = 0.0;
        for j in (i + 1 - period)..=i {
            if j > 0 && tp[j] > tp[j - 1] {
                pos_flow += raw_mf[j];
            } else if j > 0 {
                neg_flow += raw_mf[j];
            }
        }
        if neg_flow > 0.0 {
            let ratio = pos_flow / neg_flow;
            mfi[i] = 100.0 - (100.0 / (1.0 + ratio));
        } else {
            mfi[i] = 100.0;
        }
    }

    PyArray1::from_vec(py, mfi).into()
}

/// Compute Williams %R.
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_williams_r(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = close.len();
    let mut wr = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let h_slice = &high.as_slice().unwrap()[i + 1 - period..=i];
        let l_slice = &low.as_slice().unwrap()[i + 1 - period..=i];
        let highest = h_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = l_slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = highest - lowest;
        if range > 0.0 {
            wr[i] = -100.0 * (highest - close[i]) / range;
        } else {
            wr[i] = -50.0;
        }
    }

    PyArray1::from_vec(py, wr).into()
}

// --- Internal helpers ---

fn compute_ema_raw(data: &ArrayView1<f64>, period: usize) -> Vec<f64> {
    let len = data.len();
    let mut ema = vec![f64::NAN; len];
    if len < period {
        return ema;
    }
    let sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;
    ema[period - 1] = sma;
    let mult = 2.0 / (period as f64 + 1.0);
    for i in period..len {
        ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1];
    }
    ema
}

fn compute_ema_raw_vec(data: &[f64], period: usize) -> Vec<f64> {
    let len = data.len();
    let mut ema = vec![f64::NAN; len];
    if len < period {
        return ema;
    }
    let sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;
    ema[period - 1] = sma;
    let mult = 2.0 / (period as f64 + 1.0);
    for i in period..len {
        ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1];
    }
    ema
}

fn sma_with_nan(data: &[f64], period: usize) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![f64::NAN; len];
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..len {
        if !data[i].is_nan() {
            sum += data[i];
            count += 1;
            if count >= period {
                if count > period {
                    // Find the value to subtract
                    let mut back = i - 1;
                    let mut skip = period - 1;
                    while skip > 0 {
                        if !data[back].is_nan() {
                            skip -= 1;
                        }
                        if skip > 0 {
                            back -= 1;
                        }
                    }
                    // Simple approach: just use a running window of non-NaN values
                }
                result[i] = sum / count.min(period) as f64;
            }
        }
    }
    // Simpler approach: compute SMA over non-NaN values with a fixed window
    let valid: Vec<(usize, f64)> = data.iter().enumerate().filter(|(_, v)| !v.is_nan()).map(|(i, v)| (i, *v)).collect();
    let mut result = vec![f64::NAN; len];
    for j in (period - 1)..valid.len() {
        let s: f64 = valid[j + 1 - period..=j].iter().map(|(_, v)| v).sum();
        result[valid[j].0] = s / period as f64;
    }
    result
}
```

- [ ] **Step 5: Update pyproject.toml to add numpy dependency**

Add `"numpy>=1.26"` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 6: Build the Rust crate**

```bash
cd rust/bist_features
maturin develop --release
```

If maturin is not installed:
```bash
uv pip install maturin
maturin develop --release
```

- [ ] **Step 7: Verify import works**

```bash
uv run python -c "import bist_features; print(dir(bist_features))"
```

- [ ] **Step 8: Commit**

```bash
git add rust/ pyproject.toml Cargo.toml
git commit -m "feat: Rust feature engine with 13 technical indicators via PyO3"
```

---

### Task 2: Rust Technical Indicators Tests

**Files:**
- Create: `tests/test_features/__init__.py`
- Create: `tests/test_features/test_rust_indicators.py`

- [ ] **Step 1: Write tests for all indicators**

`tests/test_features/__init__.py`: empty file.

`tests/test_features/test_rust_indicators.py`:

```python
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
        # Mostly uptrending data should have RSI > 50
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
        # SMA(5) at index 4 = mean of first 5 values
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
        # CLOSE[1] > CLOSE[0], so OBV[1] = OBV[0] + VOLUME[1]
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
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_features/test_rust_indicators.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/
git commit -m "test: comprehensive tests for Rust technical indicators"
```

---

### Task 3: Rust Candlestick Pattern Detection

**Files:**
- Create: `rust/bist_features/src/patterns.rs`
- Create: `tests/test_features/test_rust_patterns.py`

- [ ] **Step 1: Implement patterns.rs**

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Detect candlestick patterns.
/// Returns a dict mapping pattern name -> array of i32 (1 = bullish, -1 = bearish, 0 = none).
#[pyfunction]
pub fn detect_patterns(
    py: Python<'_>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
) -> PyResult<Vec<(&str, Py<PyArray1<i32>>)>> {
    let open = open.as_array();
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = open.len();

    let mut doji = vec![0i32; len];
    let mut hammer = vec![0i32; len];
    let mut engulfing = vec![0i32; len];
    let mut morning_star = vec![0i32; len];

    for i in 0..len {
        let body = (close[i] - open[i]).abs();
        let range = high[i] - low[i];
        let upper_shadow = high[i] - close[i].max(open[i]);
        let lower_shadow = close[i].min(open[i]) - low[i];

        // Doji: body < 10% of range
        if range > 0.0 && body / range < 0.1 {
            doji[i] = 1;
        }

        // Hammer: small body at top, long lower shadow (>= 2x body)
        if body > 0.0 && lower_shadow >= 2.0 * body && upper_shadow <= body * 0.5 {
            hammer[i] = 1; // bullish
        }

        // Engulfing (needs previous bar)
        if i > 0 {
            let prev_body = (close[i - 1] - open[i - 1]).abs();
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bullish = close[i] > open[i];

            // Bullish engulfing: previous bearish, current bullish, current body engulfs previous
            if !prev_bullish && curr_bullish && body > prev_body
                && open[i] <= close[i - 1] && close[i] >= open[i - 1]
            {
                engulfing[i] = 1;
            }
            // Bearish engulfing
            if prev_bullish && !curr_bullish && body > prev_body
                && open[i] >= close[i - 1] && close[i] <= open[i - 1]
            {
                engulfing[i] = -1;
            }
        }

        // Morning star (needs 2 previous bars)
        if i >= 2 {
            let bar0_body = (close[i - 2] - open[i - 2]).abs();
            let bar1_body = (close[i - 1] - open[i - 1]).abs();
            let bar0_bearish = close[i - 2] < open[i - 2];
            let bar2_bullish = close[i] > open[i];
            let bar0_range = high[i - 2] - low[i - 2];

            // Morning star: large bearish, small body (star), large bullish
            if bar0_bearish && bar2_bullish
                && bar0_range > 0.0 && bar0_body / bar0_range > 0.5
                && bar1_body < bar0_body * 0.3
                && body > bar0_body * 0.5
            {
                morning_star[i] = 1;
            }
        }
    }

    Ok(vec![
        ("doji", PyArray1::from_vec(py, doji).into()),
        ("hammer", PyArray1::from_vec(py, hammer).into()),
        ("engulfing", PyArray1::from_vec(py, engulfing).into()),
        ("morning_star", PyArray1::from_vec(py, morning_star).into()),
    ])
}
```

- [ ] **Step 2: Write tests**

`tests/test_features/test_rust_patterns.py`:

```python
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
        # Create a clear doji: open ≈ close, with high/low range
        open = np.array([100.0, 100.0], dtype=np.float64)
        high = np.array([100.0, 105.0], dtype=np.float64)
        low = np.array([100.0, 95.0], dtype=np.float64)
        close = np.array([100.0, 100.5], dtype=np.float64)

        patterns = dict(bist_features.detect_patterns(open, high, low, close))
        assert patterns["doji"][1] == 1  # second bar is doji

    def test_output_lengths_match(self) -> None:
        n = 10
        o = np.random.uniform(100, 110, n)
        h = o + np.random.uniform(0, 5, n)
        l = o - np.random.uniform(0, 5, n)
        c = np.random.uniform(l, h)

        patterns = bist_features.detect_patterns(o, h, l, c)
        for name, arr in patterns:
            assert len(arr) == n
```

- [ ] **Step 3: Rebuild and run tests**

```bash
cd rust/bist_features && maturin develop --release && cd ../..
uv run pytest tests/test_features/test_rust_patterns.py -v
```

- [ ] **Step 4: Commit**

```bash
git add rust/bist_features/src/patterns.rs tests/test_features/test_rust_patterns.py
git commit -m "feat: Rust candlestick pattern detection (doji, hammer, engulfing, morning star)"
```

---

### Task 4: Rust Cross-Stock Correlations

**Files:**
- Create: `rust/bist_features/src/correlations.rs`
- Create: `tests/test_features/test_rust_correlations.py`

- [ ] **Step 1: Implement correlations.rs**

```rust
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute Pearson correlation matrix for a 2D array of returns.
/// Input: rows = stocks, columns = time periods.
/// Returns: NxN correlation matrix.
#[pyfunction]
pub fn compute_correlation_matrix(
    py: Python<'_>,
    returns: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let returns = returns.as_array();
    let n_stocks = returns.nrows();
    let n_periods = returns.ncols();

    let mut corr = vec![vec![0.0f64; n_stocks]; n_stocks];

    // Compute means and stds
    let mut means = vec![0.0; n_stocks];
    let mut stds = vec![0.0; n_stocks];

    for i in 0..n_stocks {
        let row = returns.row(i);
        let mean: f64 = row.iter().sum::<f64>() / n_periods as f64;
        means[i] = mean;
        let variance: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_periods as f64;
        stds[i] = variance.sqrt();
    }

    for i in 0..n_stocks {
        for j in 0..n_stocks {
            if i == j {
                corr[i][j] = 1.0;
            } else if stds[i] > 0.0 && stds[j] > 0.0 {
                let cov: f64 = (0..n_periods)
                    .map(|k| (returns[[i, k]] - means[i]) * (returns[[j, k]] - means[j]))
                    .sum::<f64>() / n_periods as f64;
                corr[i][j] = cov / (stds[i] * stds[j]);
            }
        }
    }

    let flat: Vec<f64> = corr.into_iter().flatten().collect();
    PyArray2::from_vec2(py, &flat.chunks(n_stocks).map(|c| c.to_vec()).collect::<Vec<_>>())
        .unwrap()
        .into()
}

/// Compute beta of a stock relative to a market index.
/// Beta = Cov(stock, market) / Var(market)
#[pyfunction]
pub fn compute_beta(
    py: Python<'_>,
    stock_returns: PyReadonlyArray1<f64>,
    market_returns: PyReadonlyArray1<f64>,
) -> f64 {
    let stock = stock_returns.as_array();
    let market = market_returns.as_array();
    let n = stock.len().min(market.len());

    if n == 0 {
        return f64::NAN;
    }

    let stock_mean: f64 = stock.iter().take(n).sum::<f64>() / n as f64;
    let market_mean: f64 = market.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_market = 0.0;

    for i in 0..n {
        let s_diff = stock[i] - stock_mean;
        let m_diff = market[i] - market_mean;
        cov += s_diff * m_diff;
        var_market += m_diff * m_diff;
    }

    if var_market == 0.0 {
        return f64::NAN;
    }

    cov / var_market
}
```

- [ ] **Step 2: Write tests**

`tests/test_features/test_rust_correlations.py`:

```python
"""Tests for Rust cross-stock correlation functions."""

from __future__ import annotations

import numpy as np
import pytest

import bist_features


class TestCorrelationMatrix:
    def test_diagonal_is_one(self) -> None:
        returns = np.random.randn(5, 100)
        corr = bist_features.compute_correlation_matrix(returns)
        for i in range(5):
            assert abs(corr[i, i] - 1.0) < 0.001

    def test_symmetric(self) -> None:
        returns = np.random.randn(4, 50)
        corr = bist_features.compute_correlation_matrix(returns)
        for i in range(4):
            for j in range(4):
                assert abs(corr[i, j] - corr[j, i]) < 0.001

    def test_perfect_correlation(self) -> None:
        base = np.random.randn(1, 100)
        returns = np.vstack([base, base])  # identical series
        corr = bist_features.compute_correlation_matrix(returns)
        assert abs(corr[0, 1] - 1.0) < 0.001

    def test_values_between_neg1_and_1(self) -> None:
        returns = np.random.randn(5, 100)
        corr = bist_features.compute_correlation_matrix(returns)
        assert np.all(corr >= -1.001)
        assert np.all(corr <= 1.001)


class TestBeta:
    def test_beta_of_market_is_one(self) -> None:
        market = np.random.randn(100)
        beta = bist_features.compute_beta(market, market)
        assert abs(beta - 1.0) < 0.001

    def test_beta_positive_for_correlated(self) -> None:
        market = np.random.randn(100)
        stock = market * 1.5 + np.random.randn(100) * 0.1  # beta ≈ 1.5
        beta = bist_features.compute_beta(stock, market)
        assert beta > 1.0

    def test_empty_returns_nan(self) -> None:
        beta = bist_features.compute_beta(np.array([]), np.array([]))
        assert np.isnan(beta)
```

- [ ] **Step 3: Rebuild and run tests**

```bash
cd rust/bist_features && maturin develop --release && cd ../..
uv run pytest tests/test_features/test_rust_correlations.py -v
```

- [ ] **Step 4: Commit**

```bash
git add rust/bist_features/src/correlations.rs tests/test_features/test_rust_correlations.py
git commit -m "feat: Rust cross-stock correlation matrix and beta computation"
```

---

### Task 5: Python Feature Store

**Files:**
- Create: `src/bist_predict/features/__init__.py`
- Create: `src/bist_predict/features/store.py`
- Create: `tests/test_features/test_store.py`

- [ ] **Step 1: Create features package init**

`src/bist_predict/features/__init__.py`:

```python
"""Feature computation and storage module."""
```

- [ ] **Step 2: Write failing tests**

`tests/test_features/test_store.py`:

```python
"""Tests for the feature store."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


class TestFeatureStore:
    def test_save_and_load_features(self, db: Database) -> None:
        store = FeatureStore(db)
        features = {"rsi_14": 65.3, "sma_20": 312.5, "macd": 1.25}
        store.save("THYAO", "2026-04-01", features)

        loaded = store.load("THYAO", "2026-04-01")
        assert loaded["rsi_14"] == 65.3
        assert loaded["sma_20"] == 312.5
        assert loaded["macd"] == 1.25

    def test_load_nonexistent_returns_empty(self, db: Database) -> None:
        store = FeatureStore(db)
        loaded = store.load("THYAO", "2026-04-01")
        assert loaded == {}

    def test_save_overwrites_existing(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("THYAO", "2026-04-01", {"rsi_14": 70.0})

        loaded = store.load("THYAO", "2026-04-01")
        assert loaded["rsi_14"] == 70.0

    def test_load_multiple_tickers(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("GARAN", "2026-04-01", {"rsi_14": 55.0})

        thyao = store.load("THYAO", "2026-04-01")
        garan = store.load("GARAN", "2026-04-01")
        assert thyao["rsi_14"] == 65.3
        assert garan["rsi_14"] == 55.0

    def test_load_date_range(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-03-31", {"rsi_14": 60.0})
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})
        store.save("THYAO", "2026-04-02", {"rsi_14": 70.0})

        features = store.load_range("THYAO", "2026-03-31", "2026-04-02")
        assert len(features) == 3
        assert features["2026-03-31"]["rsi_14"] == 60.0
        assert features["2026-04-02"]["rsi_14"] == 70.0

    def test_get_latest_feature_date(self, db: Database) -> None:
        store = FeatureStore(db)
        store.save("THYAO", "2026-03-31", {"rsi_14": 60.0})
        store.save("THYAO", "2026-04-01", {"rsi_14": 65.3})

        latest = store.get_latest_date("THYAO")
        assert latest == "2026-04-01"

    def test_get_latest_date_no_data(self, db: Database) -> None:
        store = FeatureStore(db)
        assert store.get_latest_date("THYAO") is None
```

- [ ] **Step 3: Run tests to verify they fail**

- [ ] **Step 4: Implement store.py**

`src/bist_predict/features/store.py`:

```python
"""Feature store — persist computed features in SQLite."""

from __future__ import annotations

from bist_predict.storage.database import Database


class FeatureStore:
    """Read/write features keyed by (ticker, date, feature_name)."""

    def __init__(self, db: Database, version: int = 1) -> None:
        self._db = db
        self._version = version

    def save(self, ticker: str, date: str, features: dict[str, float]) -> None:
        """Save features for a ticker on a date. Overwrites existing values."""
        with self._db.connect() as conn:
            for name, value in features.items():
                conn.execute(
                    """INSERT INTO features (ticker, date, feature_name, value, version)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ticker, date, feature_name, version)
                       DO UPDATE SET value = excluded.value""",
                    (ticker, date, name, value, self._version),
                )
            conn.commit()

    def load(self, ticker: str, date: str) -> dict[str, float]:
        """Load all features for a ticker on a date."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT feature_name, value FROM features
                   WHERE ticker = ? AND date = ? AND version = ?""",
                (ticker, date, self._version),
            ).fetchall()
        return {name: value for name, value in rows}

    def load_range(
        self, ticker: str, start_date: str, end_date: str
    ) -> dict[str, dict[str, float]]:
        """Load features for a ticker across a date range. Returns {date: {name: value}}."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT date, feature_name, value FROM features
                   WHERE ticker = ? AND date >= ? AND date <= ? AND version = ?
                   ORDER BY date""",
                (ticker, start_date, end_date, self._version),
            ).fetchall()

        result: dict[str, dict[str, float]] = {}
        for date, name, value in rows:
            if date not in result:
                result[date] = {}
            result[date][name] = value
        return result

    def get_latest_date(self, ticker: str) -> str | None:
        """Return the most recent date with features for a ticker."""
        with self._db.connect() as conn:
            row = conn.execute(
                """SELECT MAX(date) FROM features
                   WHERE ticker = ? AND version = ?""",
                (ticker, self._version),
            ).fetchone()
        return row[0] if row and row[0] else None
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_features/test_store.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/bist_predict/features/ tests/test_features/test_store.py
git commit -m "feat: feature store for persisting computed features in SQLite"
```

---

### Task 6: Python Macro Features

**Files:**
- Create: `src/bist_predict/features/macro_features.py`
- Create: `tests/test_features/test_macro_features.py`

- [ ] **Step 1: Write failing tests**

`tests/test_features/test_macro_features.py`:

```python
"""Tests for macro feature computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.macro_features import compute_macro_features
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    # Seed macro data
    with db.connect() as conn:
        macro_data = [
            ("USD_TRY", "2026-03-28", 37.80, "tcmb"),
            ("USD_TRY", "2026-03-31", 38.10, "tcmb"),
            ("USD_TRY", "2026-04-01", 38.45, "tcmb"),
            ("EUR_TRY", "2026-03-28", 40.50, "tcmb"),
            ("EUR_TRY", "2026-03-31", 40.80, "tcmb"),
            ("EUR_TRY", "2026-04-01", 41.20, "tcmb"),
            ("GOLD_TRY", "2026-03-28", 2800.0, "tcmb"),
            ("GOLD_TRY", "2026-03-31", 2850.0, "tcmb"),
            ("GOLD_TRY", "2026-04-01", 2830.0, "tcmb"),
        ]
        for indicator, date, value, source in macro_data:
            conn.execute(
                "INSERT INTO macro_data (indicator, date, value, source) VALUES (?, ?, ?, ?)",
                (indicator, date, value, source),
            )
        conn.commit()
    return db


class TestMacroFeatures:
    def test_computes_daily_deltas(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "usd_try_delta" in features
        assert abs(features["usd_try_delta"] - 0.35) < 0.01  # 38.45 - 38.10

    def test_computes_percentage_change(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "usd_try_pct" in features
        expected_pct = (38.45 - 38.10) / 38.10
        assert abs(features["usd_try_pct"] - expected_pct) < 0.001

    def test_includes_multiple_indicators(self, db: Database) -> None:
        features = compute_macro_features(db, "2026-04-01")
        assert "eur_try_delta" in features
        assert "gold_try_delta" in features

    def test_missing_data_returns_nan(self, db: Database) -> None:
        import math
        features = compute_macro_features(db, "2026-01-01")  # No data
        assert math.isnan(features.get("usd_try_delta", float("nan")))
```

- [ ] **Step 2: Implement macro_features.py**

`src/bist_predict/features/macro_features.py`:

```python
"""Compute macro-economic features from stored TCMB data."""

from __future__ import annotations

import math

from bist_predict.storage.database import Database

MACRO_INDICATORS = ["USD_TRY", "EUR_TRY", "GOLD_TRY", "POLICY_RATE", "CPI", "BOND_2Y"]


def compute_macro_features(db: Database, date: str) -> dict[str, float]:
    """Compute macro feature deltas and percentage changes for a given date.

    For each indicator, computes:
    - {indicator}_value: current value
    - {indicator}_delta: change from previous available value
    - {indicator}_pct: percentage change from previous
    """
    features: dict[str, float] = {}

    with db.connect() as conn:
        for indicator in MACRO_INDICATORS:
            key = indicator.lower()

            # Get current value
            row = conn.execute(
                "SELECT value FROM macro_data WHERE indicator = ? AND date = ?",
                (indicator, date),
            ).fetchone()

            if row is None:
                features[f"{key}_value"] = math.nan
                features[f"{key}_delta"] = math.nan
                features[f"{key}_pct"] = math.nan
                continue

            current = row[0]
            features[f"{key}_value"] = current

            # Get previous value
            prev_row = conn.execute(
                """SELECT value FROM macro_data
                   WHERE indicator = ? AND date < ?
                   ORDER BY date DESC LIMIT 1""",
                (indicator, date),
            ).fetchone()

            if prev_row is None:
                features[f"{key}_delta"] = math.nan
                features[f"{key}_pct"] = math.nan
            else:
                prev = prev_row[0]
                features[f"{key}_delta"] = current - prev
                features[f"{key}_pct"] = (current - prev) / prev if prev != 0 else math.nan

    return features
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_features/test_macro_features.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/features/macro_features.py tests/test_features/test_macro_features.py
git commit -m "feat: macro feature computation (deltas, pct changes for TCMB indicators)"
```

---

### Task 7: Python Sentiment Features

**Files:**
- Create: `src/bist_predict/features/sentiment_features.py`
- Create: `tests/test_features/test_sentiment_features.py`

- [ ] **Step 1: Write failing tests**

`tests/test_features/test_sentiment_features.py`:

```python
"""Tests for sentiment feature aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bist_predict.features.sentiment_features import compute_sentiment_features
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    with db.connect() as conn:
        sentiments = [
            ("THYAO", "2026-04-01", "google_news", "THY yükseldi", 0.8, "text1"),
            ("THYAO", "2026-04-01", "google_news", "THY güçlendi", 0.6, "text2"),
            ("THYAO", "2026-04-01", "bloomberght", "THY bilançosu", -0.2, "text3"),
            ("THYAO", "2026-03-31", "google_news", "THY haberleri", 0.5, "text4"),
        ]
        for ticker, date, source, headline, score, raw in sentiments:
            conn.execute(
                """INSERT INTO sentiment_data
                   (ticker, date, source, headline, sentiment_score, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, date, source, headline, score, raw),
            )
        conn.commit()
    return db


class TestSentimentFeatures:
    def test_computes_average_sentiment(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert "sentiment_mean" in features
        # Mean of 0.8, 0.6, -0.2 = 0.4
        assert abs(features["sentiment_mean"] - 0.4) < 0.01

    def test_computes_sentiment_count(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert features["sentiment_count"] == 3

    def test_computes_positive_ratio(self, db: Database) -> None:
        features = compute_sentiment_features(db, "THYAO", "2026-04-01")
        assert "sentiment_positive_ratio" in features
        # 2 out of 3 are positive
        assert abs(features["sentiment_positive_ratio"] - 2 / 3) < 0.01

    def test_no_data_returns_defaults(self, db: Database) -> None:
        import math
        features = compute_sentiment_features(db, "GARAN", "2026-04-01")
        assert features["sentiment_count"] == 0
        assert math.isnan(features["sentiment_mean"])
```

- [ ] **Step 2: Implement sentiment_features.py**

`src/bist_predict/features/sentiment_features.py`:

```python
"""Compute aggregated sentiment features from stored sentiment data."""

from __future__ import annotations

import math

from bist_predict.storage.database import Database


def compute_sentiment_features(
    db: Database, ticker: str, date: str
) -> dict[str, float]:
    """Compute aggregated sentiment features for a ticker on a date.

    Returns:
    - sentiment_mean: average sentiment score
    - sentiment_count: number of sentiment records
    - sentiment_positive_ratio: fraction of positive (>0) scores
    - sentiment_max: maximum sentiment score
    - sentiment_min: minimum sentiment score
    """
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT sentiment_score FROM sentiment_data
               WHERE ticker = ? AND date = ? AND sentiment_score IS NOT NULL""",
            (ticker, date),
        ).fetchall()

    if not rows:
        return {
            "sentiment_mean": math.nan,
            "sentiment_count": 0,
            "sentiment_positive_ratio": math.nan,
            "sentiment_max": math.nan,
            "sentiment_min": math.nan,
        }

    scores = [row[0] for row in rows]
    positive_count = sum(1 for s in scores if s > 0)

    return {
        "sentiment_mean": sum(scores) / len(scores),
        "sentiment_count": len(scores),
        "sentiment_positive_ratio": positive_count / len(scores),
        "sentiment_max": max(scores),
        "sentiment_min": min(scores),
    }
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_features/test_sentiment_features.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/features/sentiment_features.py tests/test_features/test_sentiment_features.py
git commit -m "feat: sentiment feature aggregation (mean, count, positive ratio)"
```

---

### Task 8: Python Temporal Features

**Files:**
- Create: `src/bist_predict/features/temporal_features.py`
- Create: `tests/test_features/test_temporal_features.py`

- [ ] **Step 1: Write failing tests**

`tests/test_features/test_temporal_features.py`:

```python
"""Tests for temporal/calendar feature computation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.features.temporal_features import compute_temporal_features


class TestTemporalFeatures:
    def test_day_of_week(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))  # Wednesday
        assert features["day_of_week"] == 2  # 0=Monday

    def test_month(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["month"] == 4

    def test_is_monday(self) -> None:
        features = compute_temporal_features(date(2026, 3, 30))  # Monday
        assert features["is_monday"] == 1

    def test_is_friday(self) -> None:
        features = compute_temporal_features(date(2026, 4, 3))  # Friday
        assert features["is_friday"] == 1

    def test_day_of_month(self) -> None:
        features = compute_temporal_features(date(2026, 4, 15))
        assert features["day_of_month"] == 15

    def test_is_month_start(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["is_month_start"] == 1
        features2 = compute_temporal_features(date(2026, 4, 15))
        assert features2["is_month_start"] == 0

    def test_is_month_end(self) -> None:
        features = compute_temporal_features(date(2026, 4, 30))
        assert features["is_month_end"] == 1

    def test_quarter(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert features["quarter"] == 2

    def test_week_of_year(self) -> None:
        features = compute_temporal_features(date(2026, 4, 1))
        assert "week_of_year" in features
```

- [ ] **Step 2: Implement temporal_features.py**

`src/bist_predict/features/temporal_features.py`:

```python
"""Compute calendar/temporal features from a date."""

from __future__ import annotations

import calendar
from datetime import date


def compute_temporal_features(d: date) -> dict[str, float]:
    """Compute temporal features for a given date.

    Returns day-of-week effects, month seasonality, and calendar position features.
    """
    day_of_week = d.weekday()  # 0=Monday, 4=Friday
    _, days_in_month = calendar.monthrange(d.year, d.month)

    return {
        "day_of_week": float(day_of_week),
        "month": float(d.month),
        "quarter": float((d.month - 1) // 3 + 1),
        "day_of_month": float(d.day),
        "week_of_year": float(d.isocalendar()[1]),
        "is_monday": float(day_of_week == 0),
        "is_friday": float(day_of_week == 4),
        "is_month_start": float(d.day <= 3),
        "is_month_end": float(d.day >= days_in_month - 2),
        "is_quarter_start": float(d.month in (1, 4, 7, 10) and d.day <= 5),
        "is_quarter_end": float(d.month in (3, 6, 9, 12) and d.day >= days_in_month - 4),
        "is_january": float(d.month == 1),  # January effect
    }
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_features/test_temporal_features.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/features/temporal_features.py tests/test_features/test_temporal_features.py
git commit -m "feat: temporal/calendar feature computation"
```

---

### Task 9: Feature Engine Orchestrator

**Files:**
- Create: `src/bist_predict/features/engine.py`
- Create: `tests/test_features/test_engine.py`

- [ ] **Step 1: Write failing tests**

`tests/test_features/test_engine.py`:

```python
"""Tests for the feature computation engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bist_predict.features.engine import FeatureEngine
from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    # Seed with 30 days of price data for THYAO
    with db.connect() as conn:
        for i in range(30):
            d = date(2026, 3, 3 + i)  # Skip weekends in a real scenario, but fine for tests
            price = 300.0 + i * 0.5 + (i % 3 - 1) * 2  # Slightly uptrending with noise
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "THYAO", d.isoformat(),
                    price - 1.0, price + 2.0, price - 2.0, price, price,
                    1000000 + i * 10000, "isyatirim",
                ),
            )
        conn.commit()
    return db


class TestFeatureEngine:
    def test_compute_features_for_ticker(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_includes_technical_indicators(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert "rsi_14" in features
        assert "sma_20" in features
        assert "ema_10" in features

    def test_includes_temporal_features(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        assert "day_of_week" in features
        assert "month" in features

    def test_compute_and_store(self, db: Database) -> None:
        engine = FeatureEngine(db)
        store = FeatureStore(db)

        engine.compute_and_store("THYAO", "2026-04-01")

        loaded = store.load("THYAO", "2026-04-01")
        assert len(loaded) > 0
        assert "rsi_14" in loaded

    def test_no_price_data_returns_empty(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("NONEXISTENT", "2026-04-01")
        assert features == {}
```

- [ ] **Step 2: Implement engine.py**

`src/bist_predict/features/engine.py`:

```python
"""Feature computation engine — orchestrates Rust and Python feature computation."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np

from bist_predict.features.macro_features import compute_macro_features
from bist_predict.features.sentiment_features import compute_sentiment_features
from bist_predict.features.store import FeatureStore
from bist_predict.features.temporal_features import compute_temporal_features
from bist_predict.storage.database import Database

logger = logging.getLogger(__name__)

# Import Rust module — may not be available if not compiled
try:
    import bist_features as rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    logger.warning("Rust bist_features module not available — using Python fallback")

# Moving average periods
SMA_PERIODS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [5, 10, 20, 50, 100, 200]


class FeatureEngine:
    """Orchestrates feature computation from raw price data."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._store = FeatureStore(db)

    def compute_for_ticker(self, ticker: str, target_date: str) -> dict[str, float]:
        """Compute all features for a ticker on a given date.

        Loads historical price data, computes technical indicators (via Rust),
        macro features, sentiment features, and temporal features.
        """
        # Load price history
        prices = self._load_price_history(ticker, target_date, lookback=252)
        if len(prices) == 0:
            return {}

        features: dict[str, float] = {}

        # Technical indicators (Rust)
        close = np.array([p[5] for p in prices], dtype=np.float64)  # close
        high = np.array([p[3] for p in prices], dtype=np.float64)
        low = np.array([p[4] for p in prices], dtype=np.float64)
        open_ = np.array([p[2] for p in prices], dtype=np.float64)
        volume = np.array([p[7] for p in prices], dtype=np.float64)

        if HAS_RUST and len(close) > 0:
            features.update(self._compute_rust_features(open_, high, low, close, volume))

        # Temporal features
        try:
            dt = date.fromisoformat(target_date)
            features.update(compute_temporal_features(dt))
        except ValueError:
            pass

        # Macro features
        features.update(compute_macro_features(self._db, target_date))

        # Sentiment features
        features.update(compute_sentiment_features(self._db, ticker, target_date))

        return features

    def compute_and_store(self, ticker: str, target_date: str) -> dict[str, float]:
        """Compute features and persist them in the feature store."""
        features = self.compute_for_ticker(ticker, target_date)
        if features:
            self._store.save(ticker, target_date, features)
        return features

    def _load_price_history(
        self, ticker: str, end_date: str, lookback: int = 252
    ) -> list[tuple]:
        """Load up to `lookback` days of price history ending at end_date."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT id, ticker, open, high, low, close, adj_close, volume
                   FROM raw_prices
                   WHERE ticker = ? AND date <= ?
                   ORDER BY date ASC""",
                (ticker, end_date),
            ).fetchall()
        # Return last `lookback` rows
        return rows[-lookback:] if len(rows) > lookback else rows

    def _compute_rust_features(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> dict[str, float]:
        """Compute all Rust-based technical indicators. Returns feature dict with latest values."""
        features: dict[str, float] = {}
        last = len(close) - 1

        # RSI
        rsi = rust.compute_rsi(close, period=14)
        features["rsi_14"] = float(rsi[last])

        # SMAs
        for period in SMA_PERIODS:
            if len(close) >= period:
                sma = rust.compute_sma(close, period)
                features[f"sma_{period}"] = float(sma[last])

        # EMAs
        for period in EMA_PERIODS:
            if len(close) >= period:
                ema = rust.compute_ema(close, period)
                features[f"ema_{period}"] = float(ema[last])

        # MACD
        macd, signal, hist = rust.compute_macd(close)
        features["macd"] = float(macd[last])
        features["macd_signal"] = float(signal[last])
        features["macd_hist"] = float(hist[last])

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = rust.compute_bollinger_bands(close, period=20)
        features["bb_upper"] = float(bb_upper[last])
        features["bb_middle"] = float(bb_middle[last])
        features["bb_lower"] = float(bb_lower[last])
        if not np.isnan(bb_upper[last]) and not np.isnan(bb_lower[last]):
            bb_width = bb_upper[last] - bb_lower[last]
            features["bb_width"] = float(bb_width)
            if bb_width > 0:
                features["bb_position"] = float((close[last] - bb_lower[last]) / bb_width)

        # Stochastic
        k, d = rust.compute_stochastic(high, low, close)
        features["stoch_k"] = float(k[last])
        features["stoch_d"] = float(d[last])

        # ATR
        atr = rust.compute_atr(high, low, close, period=14)
        features["atr_14"] = float(atr[last])

        # OBV
        obv = rust.compute_obv(close, volume)
        features["obv"] = float(obv[last])

        # VWAP
        vwap = rust.compute_vwap(high, low, close, volume)
        features["vwap"] = float(vwap[last])

        # ADX
        adx = rust.compute_adx(high, low, close, period=14)
        features["adx_14"] = float(adx[last])

        # CCI
        cci = rust.compute_cci(high, low, close, period=20)
        features["cci_20"] = float(cci[last])

        # MFI
        mfi = rust.compute_mfi(high, low, close, volume, period=14)
        features["mfi_14"] = float(mfi[last])

        # Williams %R
        wr = rust.compute_williams_r(high, low, close, period=14)
        features["williams_r_14"] = float(wr[last])

        # Price-derived features
        features["close"] = float(close[last])
        features["volume"] = float(volume[last])
        if len(close) >= 2:
            features["return_1d"] = float((close[last] - close[last - 1]) / close[last - 1])
        if len(close) >= 6:
            features["return_5d"] = float((close[last] - close[last - 5]) / close[last - 5])
        if len(close) >= 21:
            features["return_20d"] = float((close[last] - close[last - 20]) / close[last - 20])

        # Volume features
        if len(volume) >= 20:
            vol_sma = np.mean(volume[-20:])
            features["volume_ratio_20d"] = float(volume[last] / vol_sma) if vol_sma > 0 else 1.0

        return features
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_features/test_engine.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/features/engine.py tests/test_features/test_engine.py
git commit -m "feat: feature engine orchestrator with Rust indicators and Python features"
```

---

### Task 10: CLI features Command

**Files:**
- Modify: `src/bist_predict/cli.py`

- [ ] **Step 1: Add features command to cli.py**

Add this command to `cli.py` after the existing `fetch` command:

```python
@main.command()
@click.option("--ticker", default=None, help="Compute features for a single ticker")
@click.option("--date", "target_date", default=None, help="Target date (YYYY-MM-DD), defaults to latest")
def features(ticker: str | None, target_date: str | None) -> None:
    """Compute features for latest data."""
    from bist_predict.features.engine import FeatureEngine

    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    engine = FeatureEngine(db)

    if target_date is None:
        target_date = date.today().isoformat()

    tickers = [ticker] if ticker else BIST_100_SAMPLE

    total_features = 0
    for t in tickers:
        latest = db.get_latest_date(t)
        if latest is None:
            click.echo(f"  {t}: no price data, skipping")
            continue

        click.echo(f"  {t}: computing features for {target_date}...")
        feats = engine.compute_and_store(t, target_date)
        total_features += len(feats)
        click.echo(f"    → {len(feats)} features computed")

    click.echo(f"\nTotal: {total_features} features computed and stored.")
```

- [ ] **Step 2: Verify it works**

```bash
uv run bist-predict features --help
```

- [ ] **Step 3: Commit**

```bash
git add src/bist_predict/cli.py
git commit -m "feat: add 'features' command to CLI"
```

---

## Plan Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Rust crate scaffolding + 13 indicators | - |
| 2 | Rust indicator tests | ~25 tests |
| 3 | Candlestick pattern detection | ~3 tests |
| 4 | Cross-stock correlations + beta | ~7 tests |
| 5 | Feature store (SQLite persistence) | 7 tests |
| 6 | Macro features | 4 tests |
| 7 | Sentiment features | 4 tests |
| 8 | Temporal features | 9 tests |
| 9 | Feature engine orchestrator | 5 tests |
| 10 | CLI features command | manual verify |

**Total: 10 tasks, ~64 tests**

**Subsequent plans:**
- Plan 3: Quantitative Alpha Layer
- Plan 4: Model Layer (Ensemble ML)
- Plan 5: Evaluation & Backtesting
- Plan 6: CLI Polish
