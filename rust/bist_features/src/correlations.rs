use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn compute_correlation_matrix(
    py: Python<'_>,
    returns: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let n = returns.as_array().nrows();
    let flat = vec![0.0f64; n * n];
    numpy::PyArray2::from_vec2_bound(py, &flat.chunks(n).map(|c| c.to_vec()).collect::<Vec<_>>())
        .unwrap()
        .into()
}

#[pyfunction]
pub fn compute_beta(
    _py: Python<'_>,
    _stock_returns: PyReadonlyArray1<f64>,
    _market_returns: PyReadonlyArray1<f64>,
) -> f64 {
    f64::NAN
}
