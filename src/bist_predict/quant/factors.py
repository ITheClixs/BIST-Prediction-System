"""Factor models — Fama-French, cross-sectional/time-series momentum, mean reversion."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def compute_cross_sectional_momentum(
    returns: NDArray[np.float64],
    periods: list[int] | None = None, # default: [63, 126, 252] ≈ 3/6/12 months
) -> dict[str, NDArray[np.float64]]:
    """Rank stocks by trailing cumulative returns over given periods.

    Args:
        returns: (n_days, n_stocks) daily return matrix.
        periods: lookback periods in trading days (default: [63, 126, 252] ≈ 3/6/12 months).

    Returns:
        Dict with keys like "momentum_rank_63" → array of percentile ranks [0, 1] per stock.
    """
    if periods is None:
        periods = [63, 126, 252]

    n_days, n_stocks = returns.shape # Assumes returns are ordered oldest to newest (rows = time, cols = stocks)
    result: dict[str, NDArray[np.float64]] = {}

    for period in periods:
        if n_days < period:
            result[f"momentum_rank_{period}"] = np.full(n_stocks, math.nan)
            continue

        # Cumulative return over trailing period
        trailing = returns[-period:]
        cum_returns = np.prod(1.0 + trailing, axis=0) - 1.0

        # Percentile rank: 0 = worst, 1 = best
        sorted_indices = np.argsort(cum_returns)
        ranks = np.empty(n_stocks, dtype=np.float64)
        ranks[sorted_indices] = np.linspace(0, 1, n_stocks)
        result[f"momentum_rank_{period}"] = ranks

    return result


def compute_time_series_momentum(
    prices: NDArray[np.float64],
    period: int = 252,
) -> dict[str, float]:
    """Compute time-series momentum signal for a single stock.

    Per Moskowitz, Ooi, Pedersen (2012): if trailing excess return > 0, go long.

    Args:
        prices: 1D array of closing prices.
        period: lookback period in trading days.

    Returns:
        Dict with tsmom_signal (+1 or -1) and tsmom_magnitude (trailing return).
    """
    if len(prices) < period + 1:
        return {"tsmom_signal": math.nan, "tsmom_magnitude": math.nan}

    trailing_return = (prices[-1] - prices[-period - 1]) / prices[-period - 1]
    signal = 1.0 if trailing_return > 0 else -1.0

    return {"tsmom_signal": signal, "tsmom_magnitude": trailing_return}


def compute_mean_reversion_ou(
    prices: NDArray[np.float64],
) -> dict[str, float]:
    """Fit Ornstein-Uhlenbeck process to estimate mean-reversion parameters.

    Model: dX = θ(μ - X)dt + σdW
    Estimated via OLS regression: X(t+1) - X(t) = a + b*X(t) + ε
    where θ = -b, μ = -a/b

    Args:
        prices: 1D array of price levels.

    Returns:
        ou_theta: mean-reversion speed (higher = faster reversion)
        ou_mu: long-term mean
        ou_sigma: volatility of mean-reversion process
        ou_deviation: current standardized deviation from mean (z-score)
        ou_signal: mean reversion signal strength (deviation * theta)
    """
    if len(prices) < 30:
        return {
            "ou_theta": math.nan,
            "ou_mu": math.nan,
            "ou_sigma": math.nan,
            "ou_deviation": math.nan,
            "ou_signal": math.nan,
        }

    dx = np.diff(prices)
    x = prices[:-1]

    # OLS: dx = a + b * x
    x_mean = np.mean(x)
    dx_mean = np.mean(dx)
    b = np.sum((x - x_mean) * (dx - dx_mean)) / np.sum((x - x_mean) ** 2)
    a = dx_mean - b * x_mean

    theta = -b
    if theta <= 0:
        # Not mean-reverting
        return {
            "ou_theta": 0.0,
            "ou_mu": np.mean(prices),
            "ou_sigma": float(np.std(dx)),
            "ou_deviation": 0.0,
            "ou_signal": 0.0,
        }

    mu = -a / b
    residuals = dx - (a + b * x)
    sigma = float(np.std(residuals))

    # Current deviation from mean in standard deviations
    deviation = (prices[-1] - mu) / sigma if sigma > 0 else 0.0

    return {
        "ou_theta": float(theta),
        "ou_mu": float(mu),
        "ou_sigma": sigma,
        "ou_deviation": float(deviation),
        "ou_signal": float(-deviation * theta),  # Negative: buy when below mean
    }


def compute_fama_french_factors(
    returns: NDArray[np.float64],
    market_caps: NDArray[np.float64],
    book_to_market: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Compute Fama-French SMB and HML factors adapted for BIST.

    Args:
        returns: (n_days, n_stocks) daily return matrix.
        market_caps: (n_stocks,) market capitalizations.
        book_to_market: (n_stocks,) book-to-market ratios.

    Returns:
        smb: (n_days,) Small-Minus-Big factor returns.
        hml: (n_days,) High-Minus-Low (value) factor returns.
        market_premium: (n_days,) equal-weighted market return.
        factor_exposures: (n_stocks, 3) regression betas [market, smb, hml] per stock.
    """
    n_days, n_stocks = returns.shape

    # Sort by market cap — bottom 50% = small, top 50% = big
    cap_median = np.median(market_caps)
    small_mask = market_caps <= cap_median
    big_mask = market_caps > cap_median

    # Sort by book-to-market — top 30% = high (value), bottom 30% = low (growth)
    btm_30 = np.percentile(book_to_market, 30)
    btm_70 = np.percentile(book_to_market, 70)
    high_mask = book_to_market >= btm_70
    low_mask = book_to_market <= btm_30

    # SMB = avg(small stock returns) - avg(big stock returns)
    smb = np.mean(returns[:, small_mask], axis=1) - np.mean(returns[:, big_mask], axis=1)

    # HML = avg(high B/M returns) - avg(low B/M returns)
    hml = np.mean(returns[:, high_mask], axis=1) - np.mean(returns[:, low_mask], axis=1)

    # Market premium = equal-weighted average return
    market_premium = np.mean(returns, axis=1)

    # Factor exposures via OLS regression per stock
    # Y = alpha + beta_mkt * Mkt + beta_smb * SMB + beta_hml * HML + epsilon
    factors = np.column_stack([market_premium, smb, hml])  # (n_days, 3)
    # Add intercept
    X = np.column_stack([np.ones(n_days), factors])  # (n_days, 4)
    exposures = np.zeros((n_stocks, 3), dtype=np.float64)

    for i in range(n_stocks):
        y = returns[:, i]
        try:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            exposures[i] = betas[1:]  # Skip intercept
        except np.linalg.LinAlgError:
            exposures[i] = [math.nan, math.nan, math.nan]

    return {
        "smb": smb,
        "hml": hml,
        "market_premium": market_premium,
        "factor_exposures": exposures,
    }
