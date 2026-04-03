use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn detect_patterns(
    _py: Python<'_>,
    _open: PyReadonlyArray1<f64>,
    _high: PyReadonlyArray1<f64>,
    _low: PyReadonlyArray1<f64>,
    _close: PyReadonlyArray1<f64>,
) -> PyResult<Vec<(&str, Py<PyArray1<i32>>)>> {
    Ok(vec![])
}
