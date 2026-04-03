use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn nan_vec(n: usize) -> Vec<f64> {
    vec![f64::NAN; n]
}

fn sma_slice(data: &[f64], period: usize) -> f64 {
    data.iter().sum::<f64>() / period as f64
}

/// Compute SMA on a slice that may contain NaN leading values.
/// Returns a Vec of the same length with NaN where insufficient data.
fn sma_series(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    // Find first non-NaN index
    let first_valid = data.iter().position(|v| !v.is_nan());
    if first_valid.is_none() {
        return out;
    }
    let first_valid = first_valid.unwrap();
    // Need `period` valid values starting from first_valid
    if first_valid + period > n {
        return out;
    }
    // rolling sum
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in first_valid..n {
        sum += data[i];
        count += 1;
        if count >= period {
            out[i] = sum / period as f64;
            sum -= data[i + 1 - period];
        }
    }
    out
}

/// EMA seeded with SMA, skipping leading NaNs.
fn ema_series(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    let first_valid = match data.iter().position(|v| !v.is_nan()) {
        Some(i) => i,
        None => return out,
    };
    if first_valid + period > n {
        return out;
    }
    let seed = sma_slice(&data[first_valid..first_valid + period], period);
    let k = 2.0 / (period as f64 + 1.0);
    let seed_idx = first_valid + period - 1;
    out[seed_idx] = seed;
    for i in (seed_idx + 1)..n {
        let prev = out[i - 1];
        out[i] = data[i] * k + prev * (1.0 - k);
    }
    out
}

// ---------------------------------------------------------------------------
// 1. RSI  (Wilder's smoothing)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, period=14))]
pub fn compute_rsi(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);
    if n <= period {
        return PyArray1::from_vec_bound(py, out).into();
    }

    // Price changes
    let mut gains = vec![0.0f64; n];
    let mut losses = vec![0.0f64; n];
    for i in 1..n {
        let diff = c[i] - c[i - 1];
        if diff > 0.0 {
            gains[i] = diff;
        } else {
            losses[i] = -diff;
        }
    }

    // Initial average (simple mean of first `period` changes, indices 1..=period)
    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        out[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        out[period] = 100.0 - 100.0 / (1.0 + rs);
    }

    for i in (period + 1)..n {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        if avg_loss == 0.0 {
            out[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            out[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 2. SMA
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn compute_sma(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);
    if n < period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }
    let mut sum: f64 = c.iter().take(period).sum();
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += c[i] - c[i - period];
        out[i] = sum / period as f64;
    }
    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 3. EMA
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn compute_ema(py: Python<'_>, close: PyReadonlyArray1<f64>, period: usize) -> Py<PyArray1<f64>> {
    let c = close.as_array();
    let data: Vec<f64> = c.iter().copied().collect();
    let out = ema_series(&data, period);
    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 4. MACD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, fast=12, slow=26, signal=9))]
pub fn compute_macd(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let c = close.as_array();
    let data: Vec<f64> = c.iter().copied().collect();
    let n = data.len();

    let ema_fast = ema_series(&data, fast);
    let ema_slow = ema_series(&data, slow);

    // MACD line = fast EMA - slow EMA
    let mut macd_line = nan_vec(n);
    for i in 0..n {
        if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
    }

    // Signal line = EMA of MACD line (skipping NaN)
    let signal_line = ema_series(&macd_line, signal);

    // Histogram
    let mut histogram = nan_vec(n);
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    (
        PyArray1::from_vec_bound(py, macd_line).into(),
        PyArray1::from_vec_bound(py, signal_line).into(),
        PyArray1::from_vec_bound(py, histogram).into(),
    )
}

// ---------------------------------------------------------------------------
// 5. Bollinger Bands
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, period=20, num_std=2.0))]
pub fn compute_bollinger_bands(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    period: usize,
    num_std: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let c = close.as_array();
    let n = c.len();
    let mut upper = nan_vec(n);
    let mut middle = nan_vec(n);
    let mut lower = nan_vec(n);

    if n >= period && period > 0 {
        for i in (period - 1)..n {
            let window = &c.as_slice().unwrap()[i + 1 - period..=i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let var = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
            let std = var.sqrt();
            middle[i] = mean;
            upper[i] = mean + num_std * std;
            lower[i] = mean - num_std * std;
        }
    }

    (
        PyArray1::from_vec_bound(py, upper).into(),
        PyArray1::from_vec_bound(py, middle).into(),
        PyArray1::from_vec_bound(py, lower).into(),
    )
}

// ---------------------------------------------------------------------------
// 6. Stochastic Oscillator
// ---------------------------------------------------------------------------

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
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let n = c.len();
    let mut k_vals = nan_vec(n);

    if n >= k_period && k_period > 0 {
        for i in (k_period - 1)..n {
            let hs = &h.as_slice().unwrap()[i + 1 - k_period..=i];
            let ls = &l.as_slice().unwrap()[i + 1 - k_period..=i];
            let hh = hs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ll = ls.iter().cloned().fold(f64::INFINITY, f64::min);
            if (hh - ll).abs() < 1e-15 {
                k_vals[i] = 100.0;
            } else {
                k_vals[i] = (c[i] - ll) / (hh - ll) * 100.0;
            }
        }
    }

    let d_vals = sma_series(&k_vals, d_period);

    (
        PyArray1::from_vec_bound(py, k_vals).into(),
        PyArray1::from_vec_bound(py, d_vals).into(),
    )
}

// ---------------------------------------------------------------------------
// 7. ATR (Wilder's smoothing)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_atr(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);

    if n <= period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    // True Range
    let mut tr = vec![0.0f64; n];
    tr[0] = h[0] - l[0];
    for i in 1..n {
        let hl = h[i] - l[i];
        let hc = (h[i] - c[i - 1]).abs();
        let lc = (l[i] - c[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Initial ATR = simple mean of first `period` TR values (indices 1..=period)
    let mut atr: f64 = tr[1..=period].iter().sum::<f64>() / period as f64;
    out[period] = atr;
    for i in (period + 1)..n {
        atr = (atr * (period as f64 - 1.0) + tr[i]) / period as f64;
        out[i] = atr;
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 8. OBV
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn compute_obv(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> Py<PyArray1<f64>> {
    let c = close.as_array();
    let v = volume.as_array();
    let n = c.len();
    let mut out = vec![0.0f64; n];

    if n == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    out[0] = v[0];
    for i in 1..n {
        if c[i] > c[i - 1] {
            out[i] = out[i - 1] + v[i];
        } else if c[i] < c[i - 1] {
            out[i] = out[i - 1] - v[i];
        } else {
            out[i] = out[i - 1];
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 9. VWAP  (cumulative intraday)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn compute_vwap(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> Py<PyArray1<f64>> {
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let v = volume.as_array();
    let n = c.len();
    let mut out = vec![0.0f64; n];

    let mut cum_vol = 0.0f64;
    let mut cum_tp_vol = 0.0f64;
    for i in 0..n {
        let tp = (h[i] + l[i] + c[i]) / 3.0;
        cum_vol += v[i];
        cum_tp_vol += tp * v[i];
        if cum_vol > 0.0 {
            out[i] = cum_tp_vol / cum_vol;
        } else {
            out[i] = tp;
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 10. ADX (Wilder's smoothing)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_adx(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);

    if n <= 2 * period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    // True Range, +DM, -DM
    let mut tr = vec![0.0f64; n];
    let mut plus_dm = vec![0.0f64; n];
    let mut minus_dm = vec![0.0f64; n];

    for i in 1..n {
        let hl = h[i] - l[i];
        let hc = (h[i] - c[i - 1]).abs();
        let lc = (l[i] - c[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);

        let up = h[i] - h[i - 1];
        let down = l[i - 1] - l[i];
        if up > down && up > 0.0 {
            plus_dm[i] = up;
        }
        if down > up && down > 0.0 {
            minus_dm[i] = down;
        }
    }

    // Wilder smoothed TR, +DM, -DM
    let mut atr: f64 = tr[1..=period].iter().sum::<f64>();
    let mut apdm: f64 = plus_dm[1..=period].iter().sum::<f64>();
    let mut amdm: f64 = minus_dm[1..=period].iter().sum::<f64>();

    let mut dx = nan_vec(n);

    let calc_dx = |atr: f64, apdm: f64, amdm: f64| -> f64 {
        if atr == 0.0 {
            return 0.0;
        }
        let pdi = 100.0 * apdm / atr;
        let mdi = 100.0 * amdm / atr;
        let sum = pdi + mdi;
        if sum == 0.0 {
            0.0
        } else {
            100.0 * (pdi - mdi).abs() / sum
        }
    };

    dx[period] = calc_dx(atr, apdm, amdm);

    for i in (period + 1)..n {
        atr = atr - atr / period as f64 + tr[i];
        apdm = apdm - apdm / period as f64 + plus_dm[i];
        amdm = amdm - amdm / period as f64 + minus_dm[i];
        dx[i] = calc_dx(atr, apdm, amdm);
    }

    // ADX = Wilder-smoothed DX over `period`
    // First ADX = simple mean of first `period` DX values (indices period..2*period)
    let first_adx: f64 = dx[period..2 * period].iter().sum::<f64>() / period as f64;
    out[2 * period - 1] = first_adx;
    let mut adx = first_adx;
    for i in (2 * period)..n {
        adx = (adx * (period as f64 - 1.0) + dx[i]) / period as f64;
        out[i] = adx;
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 11. CCI
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, period=20))]
pub fn compute_cci(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);

    if n < period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    // Typical price
    let tp: Vec<f64> = (0..n).map(|i| (h[i] + l[i] + c[i]) / 3.0).collect();

    for i in (period - 1)..n {
        let window = &tp[i + 1 - period..=i];
        let mean = window.iter().sum::<f64>() / period as f64;
        let mad = window.iter().map(|v| (v - mean).abs()).sum::<f64>() / period as f64;
        if mad.abs() < 1e-15 {
            out[i] = 0.0;
        } else {
            out[i] = (tp[i] - mean) / (0.015 * mad);
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 12. MFI
// ---------------------------------------------------------------------------

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
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let v = volume.as_array();
    let n = c.len();
    let mut out = nan_vec(n);

    if n <= period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    let tp: Vec<f64> = (0..n).map(|i| (h[i] + l[i] + c[i]) / 3.0).collect();
    let mf: Vec<f64> = (0..n).map(|i| tp[i] * v[i]).collect();

    for i in period..n {
        let mut pos_mf = 0.0f64;
        let mut neg_mf = 0.0f64;
        for j in (i + 1 - period)..=i {
            if j == 0 {
                continue;
            }
            if tp[j] > tp[j - 1] {
                pos_mf += mf[j];
            } else if tp[j] < tp[j - 1] {
                neg_mf += mf[j];
            }
        }
        if neg_mf.abs() < 1e-15 {
            out[i] = 100.0;
        } else {
            let ratio = pos_mf / neg_mf;
            out[i] = 100.0 - 100.0 / (1.0 + ratio);
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}

// ---------------------------------------------------------------------------
// 13. Williams %R
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn compute_williams_r(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> Py<PyArray1<f64>> {
    let h = high.as_array();
    let l = low.as_array();
    let c = close.as_array();
    let n = c.len();
    let mut out = nan_vec(n);

    if n < period || period == 0 {
        return PyArray1::from_vec_bound(py, out).into();
    }

    for i in (period - 1)..n {
        let hs = &h.as_slice().unwrap()[i + 1 - period..=i];
        let ls = &l.as_slice().unwrap()[i + 1 - period..=i];
        let hh = hs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ll = ls.iter().cloned().fold(f64::INFINITY, f64::min);
        if (hh - ll).abs() < 1e-15 {
            out[i] = 0.0;
        } else {
            out[i] = -100.0 * (hh - c[i]) / (hh - ll);
        }
    }

    PyArray1::from_vec_bound(py, out).into()
}
