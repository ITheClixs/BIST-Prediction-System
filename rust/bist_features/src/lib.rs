use pyo3::prelude::*;

mod correlations;
mod indicators;
mod patterns;

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
