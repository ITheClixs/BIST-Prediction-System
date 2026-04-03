use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn detect_patterns(
    py: Python<'_>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
) -> PyResult<Vec<(String, Py<PyArray1<i32>>)>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let n = o.len();

    let mut doji = vec![0i32; n];
    let mut hammer = vec![0i32; n];
    let mut engulfing = vec![0i32; n];
    let mut morning_star = vec![0i32; n];

    for i in 0..n {
        let body = (c[i] - o[i]).abs();
        let range = h[i] - l[i];
        let upper_shadow = h[i] - c[i].max(o[i]);
        let lower_shadow = c[i].min(o[i]) - l[i];

        // Doji: body < 10% of range
        if range > 0.0 && body < 0.1 * range {
            doji[i] = 1;
        }

        // Hammer: small body at top, lower shadow >= 2x body, upper shadow <= 0.5x body
        if body > 0.0 && lower_shadow >= 2.0 * body && upper_shadow <= 0.5 * body {
            hammer[i] = 1;
        }

        // Engulfing: current body engulfs previous body, opposite color
        if i > 0 {
            let prev_body_start = o[i - 1].min(c[i - 1]);
            let prev_body_end = o[i - 1].max(c[i - 1]);
            let curr_body_start = o[i].min(c[i]);
            let curr_body_end = o[i].max(c[i]);
            let prev_bullish = c[i - 1] > o[i - 1];
            let curr_bullish = c[i] > o[i];

            if prev_bullish != curr_bullish
                && curr_body_start <= prev_body_start
                && curr_body_end >= prev_body_end
                && (c[i] - o[i]).abs() > 0.0
                && (c[i - 1] - o[i - 1]).abs() > 0.0
            {
                engulfing[i] = if curr_bullish { 1 } else { -1 };
            }
        }

        // Morning star: 3-bar pattern
        if i >= 2 {
            let bar0_body = (c[i - 2] - o[i - 2]).abs();
            let bar0_range = h[i - 2] - l[i - 2];
            let bar1_body = (c[i - 1] - o[i - 1]).abs();
            let bar1_range = h[i - 1] - l[i - 1];
            let bar2_body = (c[i] - o[i]).abs();
            let bar2_range = h[i] - l[i];

            let bar0_bearish = c[i - 2] < o[i - 2];
            let bar0_large = bar0_range > 0.0 && bar0_body > 0.5 * bar0_range;
            let bar1_small = bar1_range > 0.0 && bar1_body < 0.3 * bar1_range;
            let bar2_bullish = c[i] > o[i];
            let bar2_large = bar2_range > 0.0 && bar2_body > 0.5 * bar2_range;

            if bar0_bearish && bar0_large && bar1_small && bar2_bullish && bar2_large {
                morning_star[i] = 1;
            }
        }
    }

    let results = vec![
        (
            "doji".to_string(),
            PyArray1::from_vec_bound(py, doji).into(),
        ),
        (
            "hammer".to_string(),
            PyArray1::from_vec_bound(py, hammer).into(),
        ),
        (
            "engulfing".to_string(),
            PyArray1::from_vec_bound(py, engulfing).into(),
        ),
        (
            "morning_star".to_string(),
            PyArray1::from_vec_bound(py, morning_star).into(),
        ),
    ];

    Ok(results)
}
