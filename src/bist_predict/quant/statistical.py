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


def compute_hmm_regime(
    returns: NDArray[np.float64],
    n_states: int = 3,
    min_observations: int = 100,
) -> dict[str, float]:
    """Fit Hidden Markov Model to detect market regime.

    3-state model: bull (positive return, low vol), bear (negative return, high vol),
    sideways (near-zero return, low vol).

    Args:
        returns: 1D array of daily returns.
        n_states: number of hidden states.
        min_observations: minimum data points needed.

    Returns:
        regime_current: index of most likely current regime.
        regime_bull_prob: probability of bull state.
        regime_bear_prob: probability of bear state.
        regime_sideways_prob: probability of sideways state.
    """
    if len(returns) < min_observations:
        return {
            "regime_current": math.nan,
            "regime_bull_prob": math.nan,
            "regime_bear_prob": math.nan,
            "regime_sideways_prob": math.nan,
        }

    try:
        from hmmlearn.hmm import GaussianHMM

        # Features: return and squared return (proxy for volatility)
        X = np.column_stack([returns, returns ** 2])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        # Get state probabilities for last observation
        log_probs = model.score_samples(X)
        _, state_sequence = model.decode(X)

        # Posterior probabilities for last observation
        posteriors = model.predict_proba(X)
        last_probs = posteriors[-1]

        # Label states by mean return: highest = bull, lowest = bear, middle = sideways
        state_means = model.means_[:, 0]  # Mean return per state
        sorted_states = np.argsort(state_means)
        bear_idx, sideways_idx, bull_idx = sorted_states[0], sorted_states[1], sorted_states[2]

        return {
            "regime_current": float(state_sequence[-1]),
            "regime_bull_prob": float(last_probs[bull_idx]),
            "regime_bear_prob": float(last_probs[bear_idx]),
            "regime_sideways_prob": float(last_probs[sideways_idx]),
        }
    except Exception:
        return {
            "regime_current": math.nan,
            "regime_bull_prob": math.nan,
            "regime_bear_prob": math.nan,
            "regime_sideways_prob": math.nan,
        }


def compute_cointegration(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    min_observations: int = 30,
) -> dict[str, float]:
    """Test for cointegration between two price series (Engle-Granger method).

    Args:
        x: 1D price series for stock A.
        y: 1D price series for stock B.
        min_observations: minimum length required.

    Returns:
        coint_pvalue: p-value of ADF test on residuals (< 0.05 = cointegrated).
        spread_zscore: current z-score of the spread.
        spread_halflife: half-life of mean reversion in the spread (days).
        hedge_ratio: OLS hedge ratio (beta).
    """
    if len(x) < min_observations or len(y) < min_observations:
        return {
            "coint_pvalue": math.nan,
            "spread_zscore": math.nan,
            "spread_halflife": math.nan,
            "hedge_ratio": math.nan,
        }

    try:
        from statsmodels.tsa.stattools import adfuller

        # OLS regression: y = alpha + beta * x + epsilon
        x_with_const = np.column_stack([np.ones(len(x)), x])
        betas, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
        alpha, hedge_ratio = betas

        # Spread = y - beta * x - alpha
        spread = y - hedge_ratio * x - alpha

        # ADF test on spread
        adf_result = adfuller(spread, maxlag=1, regression="c", autolag=None)
        p_value = adf_result[1]

        # Z-score of current spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        z_score = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0.0

        # Half-life via AR(1) on spread: spread(t) = phi * spread(t-1) + eps
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        phi = np.sum(spread_lag * spread_diff) / np.sum(spread_lag ** 2)
        half_life = -math.log(2) / phi if phi < 0 else math.nan

        return {
            "coint_pvalue": float(p_value),
            "spread_zscore": float(z_score),
            "spread_halflife": float(half_life),
            "hedge_ratio": float(hedge_ratio),
        }
    except Exception:
        return {
            "coint_pvalue": math.nan,
            "spread_zscore": math.nan,
            "spread_halflife": math.nan,
            "hedge_ratio": math.nan,
        }
