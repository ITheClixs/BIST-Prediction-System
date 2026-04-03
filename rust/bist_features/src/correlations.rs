use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    let mx = mean(x);
    let my = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return if var_x == 0.0 && var_y == 0.0 {
            1.0
        } else {
            0.0
        };
    }
    cov / denom
}

#[pyfunction]
pub fn compute_correlation_matrix(
    py: Python<'_>,
    returns: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let arr = returns.as_array();
    let n = arr.nrows();
    let cols = arr.ncols();

    // Extract rows as Vec<Vec<f64>> for easier access
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..cols).map(|j| arr[[i, j]]).collect())
        .collect();

    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let corr = pearson(&rows[i], &rows[j]);
            matrix[i][j] = corr;
            matrix[j][i] = corr;
        }
    }

    PyArray2::from_vec2_bound(py, &matrix).unwrap().into()
}

#[pyfunction]
pub fn compute_beta(
    _py: Python<'_>,
    stock_returns: PyReadonlyArray1<f64>,
    market_returns: PyReadonlyArray1<f64>,
) -> f64 {
    let stock = match stock_returns.as_slice() {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    let market = match market_returns.as_slice() {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };

    if stock.is_empty() || market.is_empty() {
        return f64::NAN;
    }

    let n = stock.len();
    let market_mean = mean(market);
    let stock_mean = mean(stock);

    let mut cov = 0.0;
    let mut var_market = 0.0;

    for i in 0..n {
        let dm = market[i] - market_mean;
        let ds = stock[i] - stock_mean;
        cov += dm * ds;
        var_market += dm * dm;
    }

    if var_market == 0.0 {
        return f64::NAN;
    }

    cov / var_market
}
