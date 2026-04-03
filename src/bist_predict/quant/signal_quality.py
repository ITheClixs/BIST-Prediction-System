"""Signal quality measurement — Information Coefficient, Hurst exponent, wavelet decomposition."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def compute_information_coefficient(
    predicted: NDArray[np.float64],
    actual: NDArray[np.float64],
) -> dict[str, float | bool]:
    """Compute Information Coefficient (rank correlation between predictions and actuals).

    IC = Spearman rank correlation. IC > 0.05 is generally considered meaningful.

    Args:
        predicted: predicted return/signal values.
        actual: actual realized returns.

    Returns:
        ic: Spearman rank correlation coefficient.
        ic_pvalue: p-value of the correlation test.
        ic_significant: True if IC is statistically significant (p < 0.05) and |IC| > 0.05.
    """
    if len(predicted) < 5 or len(actual) < 5:
        return {"ic": math.nan, "ic_pvalue": math.nan, "ic_significant": False}

    corr, pvalue = stats.spearmanr(predicted, actual)

    return {
        "ic": float(corr),
        "ic_pvalue": float(pvalue),
        "ic_significant": bool(pvalue < 0.05 and abs(corr) > 0.05),
    }


def compute_hurst_exponent(
    prices: NDArray[np.float64],
    min_observations: int = 100,
) -> dict[str, float | str]:
    """Compute Hurst exponent via R/S (rescaled range) analysis.

    H > 0.5 → trending (persistent, trust momentum)
    H < 0.5 → mean-reverting (anti-persistent, trust O-U/pairs)
    H ≈ 0.5 → random walk (reduce confidence)

    Args:
        prices: 1D price series.
        min_observations: minimum data points needed.

    Returns:
        hurst: Hurst exponent value.
        hurst_interpretation: "trending", "mean_reverting", or "random_walk".
    """
    if len(prices) < min_observations:
        return {"hurst": math.nan, "hurst_interpretation": "insufficient_data"}

    returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(returns)

    # R/S analysis across multiple sub-period sizes
    max_k = int(n / 4)
    sizes = []
    rs_values = []

    for k in range(10, max_k + 1, max(1, max_k // 20)):
        n_blocks = n // k
        if n_blocks < 1:
            continue

        rs_block = []
        for i in range(n_blocks):
            block = returns[i * k : (i + 1) * k]
            mean = np.mean(block)
            deviations = np.cumsum(block - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(block, ddof=1)
            if s > 0:
                rs_block.append(r / s)

        if rs_block:
            sizes.append(k)
            rs_values.append(np.mean(rs_block))

    if len(sizes) < 3:
        return {"hurst": math.nan, "hurst_interpretation": "insufficient_data"}

    # log-log regression: log(R/S) = H * log(n) + c
    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = stats.linregress(log_sizes, log_rs)

    hurst = float(slope)

    if hurst > 0.55:
        interpretation = "trending"
    elif hurst < 0.45:
        interpretation = "mean_reverting"
    else:
        interpretation = "random_walk"

    return {"hurst": hurst, "hurst_interpretation": interpretation}


def compute_wavelet_decomposition(
    prices: NDArray[np.float64],
    levels: int = 3,
    wavelet: str = "db4",
) -> dict[str, float | NDArray[np.float64]]:
    """Discrete wavelet transform to separate price into frequency bands.

    Decomposes into approximation (trend) + detail coefficients (noise levels).

    Args:
        prices: 1D price/return series.
        levels: number of decomposition levels.
        wavelet: wavelet family (default: Daubechies-4).

    Returns:
        wavelet_approx: approximation (low-freq trend) coefficients.
        wavelet_detail_{i}: detail coefficients at level i (1=highest freq).
        wavelet_energy_{i}: energy (sum of squares) at each detail level.
    """
    import pywt

    coeffs = pywt.wavedec(prices, wavelet, level=levels)

    result: dict[str, float | NDArray[np.float64]] = {}
    result["wavelet_approx"] = coeffs[0]

    for i, detail in enumerate(coeffs[1:], 1):
        result[f"wavelet_detail_{i}"] = detail
        result[f"wavelet_energy_{i}"] = float(np.sum(detail ** 2))

    # Energy ratio: high-frequency vs low-frequency
    total_energy = sum(np.sum(c ** 2) for c in coeffs)
    if total_energy > 0:
        result["wavelet_noise_ratio"] = float(np.sum(coeffs[-1] ** 2) / total_energy)
    else:
        result["wavelet_noise_ratio"] = math.nan

    return result
