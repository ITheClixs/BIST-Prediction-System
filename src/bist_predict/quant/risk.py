"""Risk and position sizing — Kelly criterion, Ledoit-Wolf covariance, PCA."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def compute_kelly_fraction(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
) -> dict[str, float]:
    """Compute Kelly criterion optimal position fraction.

    f* = (p * b - q) / b
    where p = win probability, b = win/loss ratio, q = 1 - p.

    Args:
        win_prob: probability of winning (0-1).
        win_loss_ratio: average win / average loss.
        fraction: fractional Kelly multiplier (0.25-0.5 for safety).

    Returns:
        kelly_full: full Kelly fraction.
        kelly_fraction: fractional Kelly (full * fraction).
    """
    q = 1.0 - win_prob
    kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    kelly = max(kelly, 0.0)  # Never recommend negative sizing

    return {
        "kelly_full": kelly,
        "kelly_fraction": kelly * fraction,
    }


def compute_ledoit_wolf_covariance(
    returns: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | float]:
    """Compute Ledoit-Wolf shrinkage covariance matrix.

    Robust covariance estimation that prevents overfitting to noisy
    sample correlations by shrinking toward a structured target.

    Args:
        returns: (n_days, n_stocks) daily return matrix.

    Returns:
        covariance: (n_stocks, n_stocks) shrunk covariance matrix.
        shrinkage_coefficient: amount of shrinkage applied (0-1).
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    lw.fit(returns)

    return {
        "covariance": lw.covariance_,
        "shrinkage_coefficient": float(lw.shrinkage_),
    }


def compute_pca_factors(
    returns: NDArray[np.float64],
    n_components: int = 5,
) -> dict[str, NDArray[np.float64]]:
    """Extract principal component factors from return matrix.

    These are latent market drivers (e.g., "banking factor", "export factor").

    Args:
        returns: (n_days, n_stocks) daily return matrix.
        n_components: number of principal components to extract.

    Returns:
        pca_components: (n_days, n_components) factor time series.
        explained_variance_ratio: fraction of variance explained per component.
        loadings: (n_components, n_stocks) factor loading matrix.
    """
    from sklearn.decomposition import PCA

    n_days, n_stocks = returns.shape
    actual_components = min(n_components, n_days, n_stocks)

    pca = PCA(n_components=actual_components)
    components = pca.fit_transform(returns)

    return {
        "pca_components": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "loadings": pca.components_,
    }
