"""Statistical methods — Kalman filter, GARCH, HMM, cointegration."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray


def compute_kalman_trend(
    prices: NDArray[np.float64],
    transition_cov: float = 0.01,
    observation_cov: float = 1.0,
) -> dict[str, float]:
    """Kalman filter to estimate hidden trend state from noisy prices.

    Simple 1D Kalman filter with constant velocity model:
    - State: [level, velocity]
    - Observation: price = level + noise

    Args:
        prices: 1D array of closing prices.
        transition_cov: process noise variance (how much trend can change per step).
        observation_cov: measurement noise variance.

    Returns:
        kalman_trend: filtered trend estimate at final timestep.
        kalman_velocity: estimated daily price velocity.
        kalman_variance: estimation uncertainty (prediction error variance).
    """
    n = len(prices)
    if n < 2:
        return {
            "kalman_trend": float(prices[-1]) if n > 0 else math.nan,
            "kalman_velocity": 0.0,
            "kalman_variance": math.nan,
        }

    # State: [level, velocity]
    state = np.array([prices[0], 0.0])
    # State covariance
    P = np.eye(2) * 100.0

    # Transition matrix: level(t+1) = level(t) + velocity(t); velocity(t+1) = velocity(t)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Observation matrix: we observe level
    H = np.array([[1.0, 0.0]])
    # Process noise
    Q = np.eye(2) * transition_cov
    # Observation noise
    R = np.array([[observation_cov]])

    for i in range(1, n):
        # Predict
        state = F @ state
        P = F @ P @ F.T + Q

        # Update
        y = prices[i] - H @ state  # Innovation
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

        state = state + (K @ y).flatten()
        P = (np.eye(2) - K @ H) @ P

    return {
        "kalman_trend": float(state[0]),
        "kalman_velocity": float(state[1]),
        "kalman_variance": float(P[0, 0]),
    }


def compute_garch_volatility(
    returns: NDArray[np.float64],
    min_observations: int = 100,
) -> dict[str, float]:
    """Fit GARCH(1,1) model and forecast next-period volatility.

    σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

    Args:
        returns: 1D array of daily returns (e.g., log returns or percentage returns).
        min_observations: minimum data points needed to fit.

    Returns:
        garch_vol_forecast: forecasted next-period volatility (annualized std dev).
        garch_vol_surprise: ratio of last realized vol to forecasted vol.
        garch_omega, garch_alpha, garch_beta: fitted GARCH parameters.
    """
    if len(returns) < min_observations:
        return {
            "garch_vol_forecast": math.nan,
            "garch_vol_surprise": math.nan,
            "garch_omega": math.nan,
            "garch_alpha": math.nan,
            "garch_beta": math.nan,
        }

    try:
        from arch import arch_model

        # Scale returns to percentage for numerical stability
        scaled = returns * 100.0
        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp="off", show_warning=False)

        # Forecast next period
        forecast = fit.forecast(horizon=1)
        var_forecast = forecast.variance.values[-1, 0]
        vol_forecast = math.sqrt(var_forecast) / 100.0  # Convert back from pct

        # Annualize
        vol_annualized = vol_forecast * math.sqrt(252)

        # Volatility surprise: actual last-period vol vs predicted
        last_realized = abs(returns[-1])
        vol_surprise = last_realized / vol_forecast if vol_forecast > 0 else math.nan

        params = fit.params
        return {
            "garch_vol_forecast": vol_annualized,
            "garch_vol_surprise": vol_surprise,
            "garch_omega": float(params.get("omega", math.nan)),
            "garch_alpha": float(params.get("alpha[1]", math.nan)),
            "garch_beta": float(params.get("beta[1]", math.nan)),
        }
    except Exception:
        return {
            "garch_vol_forecast": math.nan,
            "garch_vol_surprise": math.nan,
            "garch_omega": math.nan,
            "garch_alpha": math.nan,
            "garch_beta": math.nan,
        }
