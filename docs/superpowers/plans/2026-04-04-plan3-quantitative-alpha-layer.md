# Plan 3: Quantitative Alpha Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Quantitative Alpha Layer that sits between the Feature Engine and Model Layer — providing factor models, statistical methods, risk/position sizing, signal quality measurement, and regime-aware routing for the BIST-100 prediction system.

**Architecture:** The quant layer is a Python package (`src/bist_predict/quant/`) with five modules matching the spec sections: factors (Fama-French, momentum, mean reversion), statistical (Kalman, HMM, GARCH, cointegration), risk (Kelly, Ledoit-Wolf, PCA), signal quality (IC, Hurst, wavelets), and regime routing. Each module exposes pure functions or small classes that accept numpy arrays / DataFrames of price/return data and return dicts of computed features or signal values. The quant layer reads from the database (raw_prices) and writes features back via the FeatureStore.

**Tech Stack:** Python 3.12+, numpy, scipy, statsmodels, hmmlearn, arch (GARCH), pywt (wavelets), scikit-learn (Ledoit-Wolf, PCA), pytest

**Design spec:** `docs/superpowers/specs/2026-04-02-bist-predictor-design.md` (Section 3)

---

## File Structure

```
src/bist_predict/
    ├── quant/
    │   ├── __init__.py             # Package init
    │   ├── factors.py              # Fama-French, cross-sectional momentum, time-series momentum, mean reversion (O-U)
    │   ├── statistical.py          # Kalman filter, HMM, GARCH, cointegration
    │   ├── risk.py                 # Kelly criterion, Ledoit-Wolf covariance, PCA factor extraction
    │   ├── signal_quality.py       # Information Coefficient, Hurst exponent, wavelet decomposition
    │   └── regime.py               # Regime-aware routing (consumes HMM output, adjusts weights)

tests/
    ├── test_quant/
    │   ├── __init__.py
    │   ├── test_factors.py
    │   ├── test_statistical.py
    │   ├── test_risk.py
    │   ├── test_signal_quality.py
    │   └── test_regime.py
```

---

## Dependencies

Before starting, add the required quant libraries to `pyproject.toml`:

```bash
uv add scipy statsmodels hmmlearn arch pywt scikit-learn
```

New dependencies section in `pyproject.toml`:
```toml
dependencies = [
    "click>=8.1",
    "httpx>=0.27",
    "yfinance>=0.2",
    "feedparser>=6.0",
    "tomli>=2.0; python_version < '3.11'",
    "numpy>=1.26",
    "scipy>=1.12",
    "statsmodels>=0.14",
    "hmmlearn>=0.3",
    "arch>=7.0",
    "PyWavelets>=1.5",
    "scikit-learn>=1.4",
]
```

---

### Task 1: Add dependencies and create quant package

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bist_predict/quant/__init__.py`
- Create: `tests/test_quant/__init__.py`

- [ ] **Step 1: Add quant dependencies to pyproject.toml**

Update the dependencies list in `pyproject.toml`:

```toml
dependencies = [
    "click>=8.1",
    "httpx>=0.27",
    "yfinance>=0.2",
    "feedparser>=6.0",
    "tomli>=2.0; python_version < '3.11'",
    "numpy>=1.26",
    "scipy>=1.12",
    "statsmodels>=0.14",
    "hmmlearn>=0.3",
    "arch>=7.0",
    "PyWavelets>=1.5",
    "scikit-learn>=1.4",
]
```

- [ ] **Step 2: Create quant package init**

`src/bist_predict/quant/__init__.py`:

```python
"""Quantitative alpha layer — factor models, statistical methods, risk, and regime detection."""
```

- [ ] **Step 3: Create test package init**

`tests/test_quant/__init__.py`:

```python
```

- [ ] **Step 4: Install dependencies and verify**

```bash
uv sync
uv run python -c "import scipy, statsmodels, hmmlearn, arch, pywt, sklearn; print('All quant deps OK')"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/bist_predict/quant/__init__.py tests/test_quant/__init__.py uv.lock
git commit -m "chore: add quant dependencies and create quant package"
```

---

### Task 2: Factor Models — Momentum & Mean Reversion

**Files:**
- Create: `src/bist_predict/quant/factors.py`
- Create: `tests/test_quant/test_factors.py`

- [ ] **Step 1: Write failing tests**

`tests/test_quant/test_factors.py`:

```python
"""Tests for factor models — momentum rankings and mean reversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.factors import (
    compute_cross_sectional_momentum,
    compute_fama_french_factors,
    compute_mean_reversion_ou,
    compute_time_series_momentum,
)


class TestCrossSectionalMomentum:
    def test_ranks_by_trailing_returns(self) -> None:
        # 5 stocks, 252 days of returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (252, 5))
        # Make stock 0 a clear winner (high returns), stock 4 a clear loser
        returns[:, 0] += 0.005
        returns[:, 4] -= 0.005

        result = compute_cross_sectional_momentum(returns, periods=[63, 126, 252])
        assert "momentum_rank_63" in result
        assert "momentum_rank_126" in result
        assert "momentum_rank_252" in result
        # Each rank array should have 5 values (one per stock)
        assert len(result["momentum_rank_63"]) == 5

    def test_ranks_are_percentiles(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 5))
        result = compute_cross_sectional_momentum(returns, periods=[252])
        ranks = result["momentum_rank_252"]
        # Percentile ranks should be between 0 and 1
        assert all(0.0 <= r <= 1.0 for r in ranks)

    def test_insufficient_data_returns_nan(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.02, (10, 3))
        result = compute_cross_sectional_momentum(returns, periods=[252])
        ranks = result["momentum_rank_252"]
        assert all(math.isnan(r) for r in ranks)


class TestTimeSeriesMomentum:
    def test_positive_excess_returns_long_signal(self) -> None:
        # 252 days of positive returns → long signal
        prices = np.cumsum(np.ones(252) * 0.5) + 100.0
        result = compute_time_series_momentum(prices, period=252)
        assert result["tsmom_signal"] == 1.0
        assert result["tsmom_magnitude"] > 0

    def test_negative_excess_returns_short_signal(self) -> None:
        # 252 days of negative returns → short signal
        prices = 100.0 - np.cumsum(np.ones(252) * 0.5)
        result = compute_time_series_momentum(prices, period=252)
        assert result["tsmom_signal"] == -1.0
        assert result["tsmom_magnitude"] < 0

    def test_insufficient_data(self) -> None:
        prices = np.array([100.0, 101.0, 102.0])
        result = compute_time_series_momentum(prices, period=252)
        assert math.isnan(result["tsmom_signal"])


class TestMeanReversionOU:
    def test_estimates_ou_parameters(self) -> None:
        # Generate mean-reverting series: X(t+1) = X(t) + theta*(mu - X(t)) + noise
        rng = np.random.default_rng(42)
        n = 500
        theta_true = 0.1
        mu_true = 100.0
        sigma_true = 1.0
        x = np.zeros(n)
        x[0] = 95.0
        for i in range(1, n):
            x[i] = x[i - 1] + theta_true * (mu_true - x[i - 1]) + sigma_true * rng.normal()

        result = compute_mean_reversion_ou(x)
        assert "ou_theta" in result
        assert "ou_mu" in result
        assert "ou_deviation" in result
        assert "ou_signal" in result
        # theta should be positive (mean-reverting)
        assert result["ou_theta"] > 0
        # mu should be close to 100
        assert abs(result["ou_mu"] - mu_true) < 10.0

    def test_constant_series(self) -> None:
        x = np.ones(100) * 50.0
        result = compute_mean_reversion_ou(x)
        assert abs(result["ou_deviation"]) < 0.01


class TestFamaFrench:
    def test_computes_smb_hml(self) -> None:
        rng = np.random.default_rng(42)
        n_stocks = 20
        n_days = 252
        returns = rng.normal(0.001, 0.02, (n_days, n_stocks))
        market_caps = rng.uniform(1e9, 50e9, n_stocks)
        book_to_market = rng.uniform(0.3, 2.0, n_stocks)

        result = compute_fama_french_factors(returns, market_caps, book_to_market)
        assert "smb" in result
        assert "hml" in result
        assert "market_premium" in result
        # SMB and HML should be arrays of daily factor returns
        assert len(result["smb"]) == n_days
        assert len(result["hml"]) == n_days

    def test_factor_exposures(self) -> None:
        rng = np.random.default_rng(42)
        n_stocks = 20
        n_days = 252
        returns = rng.normal(0.001, 0.02, (n_days, n_stocks))
        market_caps = rng.uniform(1e9, 50e9, n_stocks)
        book_to_market = rng.uniform(0.3, 2.0, n_stocks)

        result = compute_fama_french_factors(returns, market_caps, book_to_market)
        assert "factor_exposures" in result
        # One exposure per stock
        assert result["factor_exposures"].shape == (n_stocks, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_quant/test_factors.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bist_predict.quant.factors'`

- [ ] **Step 3: Implement factors.py**

`src/bist_predict/quant/factors.py`:

```python
"""Factor models — Fama-French, cross-sectional/time-series momentum, mean reversion."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def compute_cross_sectional_momentum(
    returns: NDArray[np.float64],
    periods: list[int] | None = None,
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

    n_days, n_stocks = returns.shape
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
    n = len(x)
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_quant/test_factors.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/quant/factors.py tests/test_quant/test_factors.py
git commit -m "feat: factor models — Fama-French, momentum rankings, O-U mean reversion"
```

---

### Task 3: Statistical Methods — Kalman Filter & GARCH

**Files:**
- Create: `src/bist_predict/quant/statistical.py`
- Create: `tests/test_quant/test_statistical.py`

This task implements Kalman filter and GARCH. HMM and cointegration are added in Task 4 (same file, separate commit for reviewability).

- [ ] **Step 1: Write failing tests**

`tests/test_quant/test_statistical.py`:

```python
"""Tests for statistical methods — Kalman, GARCH, HMM, cointegration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.statistical import (
    compute_garch_volatility,
    compute_kalman_trend,
)


class TestKalmanTrend:
    def test_filters_noisy_uptrend(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        trend = np.linspace(100, 120, n)
        noisy = trend + rng.normal(0, 2, n)

        result = compute_kalman_trend(noisy)
        assert "kalman_trend" in result
        assert "kalman_variance" in result
        # Filtered trend at end should be close to true trend
        assert abs(result["kalman_trend"] - 120.0) < 5.0
        # Variance should be positive
        assert result["kalman_variance"] > 0

    def test_adapts_to_level_shift(self) -> None:
        rng = np.random.default_rng(42)
        # Flat at 100, then jumps to 120
        prices = np.concatenate([
            100 + rng.normal(0, 1, 100),
            120 + rng.normal(0, 1, 100),
        ])
        result = compute_kalman_trend(prices)
        # After shift, trend should be near 120
        assert result["kalman_trend"] > 115.0

    def test_short_series(self) -> None:
        prices = np.array([100.0, 101.0])
        result = compute_kalman_trend(prices)
        assert "kalman_trend" in result


class TestGARCH:
    def test_computes_volatility_forecast(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        # Add volatility cluster
        returns[200:250] *= 3.0

        result = compute_garch_volatility(returns)
        assert "garch_vol_forecast" in result
        assert "garch_vol_surprise" in result
        assert result["garch_vol_forecast"] > 0

    def test_insufficient_data_returns_nan(self) -> None:
        returns = np.array([0.01, -0.01, 0.02])
        result = compute_garch_volatility(returns)
        assert math.isnan(result["garch_vol_forecast"])

    def test_constant_returns(self) -> None:
        returns = np.zeros(200)
        result = compute_garch_volatility(returns)
        # Should not crash on zero-variance input
        assert "garch_vol_forecast" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_quant/test_statistical.py -v
```

- [ ] **Step 3: Implement Kalman and GARCH in statistical.py**

`src/bist_predict/quant/statistical.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_quant/test_statistical.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/quant/statistical.py tests/test_quant/test_statistical.py
git commit -m "feat: Kalman filter trend estimation and GARCH(1,1) volatility forecasting"
```

---

### Task 4: Statistical Methods — HMM & Cointegration

**Files:**
- Modify: `src/bist_predict/quant/statistical.py`
- Modify: `tests/test_quant/test_statistical.py`

- [ ] **Step 1: Add HMM and cointegration tests**

Append to `tests/test_quant/test_statistical.py`:

```python
from bist_predict.quant.statistical import (
    compute_cointegration,
    compute_hmm_regime,
)


class TestHMMRegime:
    def test_detects_regimes(self) -> None:
        rng = np.random.default_rng(42)
        # Bull regime (positive returns, low vol) then bear (negative, high vol)
        bull = rng.normal(0.001, 0.01, 200)
        bear = rng.normal(-0.002, 0.03, 100)
        sideways = rng.normal(0.0, 0.005, 100)
        returns = np.concatenate([bull, bear, sideways])

        result = compute_hmm_regime(returns, n_states=3)
        assert "regime_current" in result
        assert "regime_bull_prob" in result
        assert "regime_bear_prob" in result
        assert "regime_sideways_prob" in result
        # Current regime should be one of 0, 1, 2
        assert result["regime_current"] in (0, 1, 2)
        # Probabilities should sum to ~1
        prob_sum = result["regime_bull_prob"] + result["regime_bear_prob"] + result["regime_sideways_prob"]
        assert abs(prob_sum - 1.0) < 0.01

    def test_insufficient_data(self) -> None:
        returns = np.array([0.01, -0.01])
        result = compute_hmm_regime(returns, n_states=3)
        assert math.isnan(result["regime_bull_prob"])


class TestCointegration:
    def test_cointegrated_pair(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        # Random walk
        x = np.cumsum(rng.normal(0, 1, n)) + 100.0
        # Cointegrated partner: y = 2*x + mean-reverting noise
        noise = np.zeros(n)
        noise[0] = rng.normal()
        for i in range(1, n):
            noise[i] = 0.8 * noise[i - 1] + rng.normal()
        y = 2 * x + noise + 50.0

        result = compute_cointegration(x, y)
        assert "coint_pvalue" in result
        assert "spread_zscore" in result
        assert "spread_halflife" in result
        # Should detect cointegration (p < 0.05)
        assert result["coint_pvalue"] < 0.05

    def test_non_cointegrated_pair(self) -> None:
        rng = np.random.default_rng(42)
        # Two independent random walks
        x = np.cumsum(rng.normal(0, 1, 500)) + 100.0
        y = np.cumsum(rng.normal(0, 1, 500)) + 100.0

        result = compute_cointegration(x, y)
        # Should NOT detect cointegration (p > 0.05)
        assert result["coint_pvalue"] > 0.05

    def test_short_series(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        result = compute_cointegration(x, y)
        assert "coint_pvalue" in result
```

- [ ] **Step 2: Implement HMM and cointegration**

Append to `src/bist_predict/quant/statistical.py`:

```python
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
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_quant/test_statistical.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/quant/statistical.py tests/test_quant/test_statistical.py
git commit -m "feat: HMM regime detection and Engle-Granger cointegration testing"
```

---

### Task 5: Risk & Position Sizing — Kelly, Ledoit-Wolf, PCA

**Files:**
- Create: `src/bist_predict/quant/risk.py`
- Create: `tests/test_quant/test_risk.py`

- [ ] **Step 1: Write failing tests**

`tests/test_quant/test_risk.py`:

```python
"""Tests for risk and position sizing — Kelly, Ledoit-Wolf covariance, PCA."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.risk import (
    compute_kelly_fraction,
    compute_ledoit_wolf_covariance,
    compute_pca_factors,
)


class TestKelly:
    def test_positive_edge(self) -> None:
        # 60% win rate, 1:1 win/loss ratio → f* = (0.6*1 - 0.4)/1 = 0.2
        result = compute_kelly_fraction(win_prob=0.6, win_loss_ratio=1.0, fraction=1.0)
        assert abs(result["kelly_full"] - 0.2) < 0.001
        assert abs(result["kelly_fraction"] - 0.2) < 0.001

    def test_fractional_kelly(self) -> None:
        result = compute_kelly_fraction(win_prob=0.6, win_loss_ratio=1.0, fraction=0.25)
        assert abs(result["kelly_fraction"] - 0.05) < 0.001

    def test_no_edge(self) -> None:
        # 50/50 with 1:1 → f* = 0
        result = compute_kelly_fraction(win_prob=0.5, win_loss_ratio=1.0, fraction=0.25)
        assert result["kelly_full"] == 0.0

    def test_negative_edge_clamps_to_zero(self) -> None:
        result = compute_kelly_fraction(win_prob=0.3, win_loss_ratio=1.0, fraction=0.25)
        assert result["kelly_fraction"] == 0.0

    def test_high_win_loss_ratio(self) -> None:
        # 50% win rate but 3:1 payout → f* = (0.5*3 - 0.5)/3 = 0.333
        result = compute_kelly_fraction(win_prob=0.5, win_loss_ratio=3.0, fraction=1.0)
        assert abs(result["kelly_full"] - 1.0 / 3) < 0.01


class TestLedoitWolf:
    def test_shrinks_covariance(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 10))

        result = compute_ledoit_wolf_covariance(returns)
        assert "covariance" in result
        assert "shrinkage_coefficient" in result
        assert result["covariance"].shape == (10, 10)
        # Shrinkage coefficient between 0 and 1
        assert 0.0 <= result["shrinkage_coefficient"] <= 1.0
        # Covariance matrix should be symmetric
        assert np.allclose(result["covariance"], result["covariance"].T)

    def test_positive_definite(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 10))
        result = compute_ledoit_wolf_covariance(returns)
        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(result["covariance"])
        assert all(ev > 0 for ev in eigenvalues)


class TestPCA:
    def test_extracts_components(self) -> None:
        rng = np.random.default_rng(42)
        # 20 stocks, 252 days, with a strong common factor
        common = rng.normal(0, 0.02, (252, 1))
        idiosyncratic = rng.normal(0, 0.01, (252, 20))
        returns = common + idiosyncratic

        result = compute_pca_factors(returns, n_components=5)
        assert "pca_components" in result
        assert "explained_variance_ratio" in result
        assert result["pca_components"].shape == (252, 5)
        assert len(result["explained_variance_ratio"]) == 5
        # First component should explain most variance (strong common factor)
        assert result["explained_variance_ratio"][0] > 0.2

    def test_loadings_shape(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (252, 20))
        result = compute_pca_factors(returns, n_components=5)
        assert "loadings" in result
        assert result["loadings"].shape == (5, 20)

    def test_insufficient_data(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.02, (3, 20))
        result = compute_pca_factors(returns, n_components=5)
        assert result["pca_components"].shape[1] <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_quant/test_risk.py -v
```

- [ ] **Step 3: Implement risk.py**

`src/bist_predict/quant/risk.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_quant/test_risk.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/quant/risk.py tests/test_quant/test_risk.py
git commit -m "feat: Kelly criterion, Ledoit-Wolf covariance, PCA factor extraction"
```

---

### Task 6: Signal Quality — IC, Hurst, Wavelets

**Files:**
- Create: `src/bist_predict/quant/signal_quality.py`
- Create: `tests/test_quant/test_signal_quality.py`

- [ ] **Step 1: Write failing tests**

`tests/test_quant/test_signal_quality.py`:

```python
"""Tests for signal quality measurement — IC, Hurst exponent, wavelet decomposition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bist_predict.quant.signal_quality import (
    compute_hurst_exponent,
    compute_information_coefficient,
    compute_wavelet_decomposition,
)


class TestInformationCoefficient:
    def test_perfect_prediction(self) -> None:
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_information_coefficient(predicted, actual)
        assert abs(result["ic"] - 1.0) < 0.001
        assert result["ic_significant"]  # Perfect correlation is significant

    def test_random_prediction(self) -> None:
        rng = np.random.default_rng(42)
        predicted = rng.normal(0, 1, 1000)
        actual = rng.normal(0, 1, 1000)
        result = compute_information_coefficient(predicted, actual)
        # IC should be near 0 for random
        assert abs(result["ic"]) < 0.1

    def test_negative_correlation(self) -> None:
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_information_coefficient(predicted, actual)
        assert result["ic"] < -0.9


class TestHurst:
    def test_trending_series(self) -> None:
        # Cumulative sum of biased random walk → H > 0.5
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(0.01, 0.1, 1000))
        result = compute_hurst_exponent(x)
        assert result["hurst"] > 0.5
        assert result["hurst_interpretation"] == "trending"

    def test_mean_reverting_series(self) -> None:
        # Ornstein-Uhlenbeck process → H < 0.5
        rng = np.random.default_rng(42)
        n = 1000
        x = np.zeros(n)
        x[0] = 0.0
        for i in range(1, n):
            x[i] = x[i - 1] - 0.3 * x[i - 1] + rng.normal(0, 0.5)
        result = compute_hurst_exponent(x)
        assert result["hurst"] < 0.5
        assert result["hurst_interpretation"] == "mean_reverting"

    def test_short_series_returns_nan(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = compute_hurst_exponent(x)
        assert math.isnan(result["hurst"])


class TestWavelet:
    def test_decomposes_signal(self) -> None:
        rng = np.random.default_rng(42)
        # Signal with clear low-frequency trend + noise
        t = np.linspace(0, 10, 256)
        signal = np.sin(t) + 0.5 * rng.normal(0, 1, 256)

        result = compute_wavelet_decomposition(signal, levels=3)
        assert "wavelet_approx" in result
        assert "wavelet_detail_1" in result
        assert "wavelet_detail_2" in result
        assert "wavelet_detail_3" in result
        # Approximation should capture the trend
        assert len(result["wavelet_approx"]) > 0

    def test_energy_per_level(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 256)
        result = compute_wavelet_decomposition(signal, levels=3)
        assert "wavelet_energy_1" in result
        assert "wavelet_energy_2" in result
        assert "wavelet_energy_3" in result
        # Energies should be positive
        assert result["wavelet_energy_1"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_quant/test_signal_quality.py -v
```

- [ ] **Step 3: Implement signal_quality.py**

`src/bist_predict/quant/signal_quality.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_quant/test_signal_quality.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/quant/signal_quality.py tests/test_quant/test_signal_quality.py
git commit -m "feat: signal quality — IC, Hurst exponent, wavelet decomposition"
```

---

### Task 7: Regime-Aware Routing

**Files:**
- Create: `src/bist_predict/quant/regime.py`
- Create: `tests/test_quant/test_regime.py`

- [ ] **Step 1: Write failing tests**

`tests/test_quant/test_regime.py`:

```python
"""Tests for regime-aware routing — adjusts ensemble weights based on HMM regime."""

from __future__ import annotations

import pytest

from bist_predict.quant.regime import RegimeRouter


class TestRegimeRouter:
    def test_bull_regime_favors_momentum(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.9,
            regime_bear_prob=0.05,
            regime_sideways_prob=0.05,
        )
        assert weights["momentum_weight"] > weights["mean_reversion_weight"]
        assert weights["kelly_fraction"] == 0.5

    def test_bear_regime_favors_mean_reversion(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.05,
            regime_bear_prob=0.9,
            regime_sideways_prob=0.05,
        )
        assert weights["mean_reversion_weight"] > weights["momentum_weight"]
        assert weights["kelly_fraction"] == 0.25

    def test_sideways_regime_favors_pairs(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.05,
            regime_bear_prob=0.05,
            regime_sideways_prob=0.9,
        )
        assert weights["pairs_weight"] > weights["momentum_weight"]
        assert weights["kelly_fraction"] == 0.25

    def test_uncertain_regime_blends(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.33,
            regime_bear_prob=0.33,
            regime_sideways_prob=0.34,
        )
        # No extreme weighting when uncertain
        assert abs(weights["momentum_weight"] - weights["mean_reversion_weight"]) < 0.2

    def test_weights_sum_to_one(self) -> None:
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=0.6,
            regime_bear_prob=0.3,
            regime_sideways_prob=0.1,
        )
        total = weights["momentum_weight"] + weights["mean_reversion_weight"] + weights["pairs_weight"]
        assert abs(total - 1.0) < 0.01

    def test_nan_regime_returns_defaults(self) -> None:
        import math
        router = RegimeRouter()
        weights = router.get_weights(
            regime_bull_prob=math.nan,
            regime_bear_prob=math.nan,
            regime_sideways_prob=math.nan,
        )
        assert abs(weights["momentum_weight"] - 1 / 3) < 0.01
        assert weights["kelly_fraction"] == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_quant/test_regime.py -v
```

- [ ] **Step 3: Implement regime.py**

`src/bist_predict/quant/regime.py`:

```python
"""Regime-aware routing — adjusts model ensemble weights based on HMM regime detection."""

from __future__ import annotations

import math


class RegimeRouter:
    """Map HMM regime probabilities to ensemble weight adjustments.

    Bull regime → increase momentum weight, Kelly at 0.5×
    Bear regime → increase mean-reversion weight, Kelly at 0.25×
    Sideways regime → increase pairs trading weight, Kelly at 0.25×
    """

    # Weight profiles per regime (momentum, mean_reversion, pairs)
    BULL_WEIGHTS = (0.50, 0.20, 0.30)
    BEAR_WEIGHTS = (0.20, 0.50, 0.30)
    SIDEWAYS_WEIGHTS = (0.15, 0.25, 0.60)
    DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

    KELLY_BULL = 0.5
    KELLY_BEAR = 0.25
    KELLY_SIDEWAYS = 0.25

    def get_weights(
        self,
        regime_bull_prob: float,
        regime_bear_prob: float,
        regime_sideways_prob: float,
    ) -> dict[str, float]:
        """Compute blended strategy weights from regime probabilities.

        Args:
            regime_bull_prob: probability of bull regime (0-1).
            regime_bear_prob: probability of bear regime (0-1).
            regime_sideways_prob: probability of sideways regime (0-1).

        Returns:
            momentum_weight: weight for momentum-based signals.
            mean_reversion_weight: weight for mean-reversion signals.
            pairs_weight: weight for pairs/cointegration signals.
            kelly_fraction: recommended fractional Kelly multiplier.
        """
        # Handle NaN inputs → return equal weights
        if (
            math.isnan(regime_bull_prob)
            or math.isnan(regime_bear_prob)
            or math.isnan(regime_sideways_prob)
        ):
            return {
                "momentum_weight": self.DEFAULT_WEIGHTS[0],
                "mean_reversion_weight": self.DEFAULT_WEIGHTS[1],
                "pairs_weight": self.DEFAULT_WEIGHTS[2],
                "kelly_fraction": 0.25,
            }

        # Blend weight profiles by regime probability
        momentum = (
            regime_bull_prob * self.BULL_WEIGHTS[0]
            + regime_bear_prob * self.BEAR_WEIGHTS[0]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[0]
        )
        mean_rev = (
            regime_bull_prob * self.BULL_WEIGHTS[1]
            + regime_bear_prob * self.BEAR_WEIGHTS[1]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[1]
        )
        pairs = (
            regime_bull_prob * self.BULL_WEIGHTS[2]
            + regime_bear_prob * self.BEAR_WEIGHTS[2]
            + regime_sideways_prob * self.SIDEWAYS_WEIGHTS[2]
        )

        # Normalize to sum to 1
        total = momentum + mean_rev + pairs
        if total > 0:
            momentum /= total
            mean_rev /= total
            pairs /= total

        # Blend Kelly fraction by regime
        kelly = (
            regime_bull_prob * self.KELLY_BULL
            + regime_bear_prob * self.KELLY_BEAR
            + regime_sideways_prob * self.KELLY_SIDEWAYS
        )

        return {
            "momentum_weight": momentum,
            "mean_reversion_weight": mean_rev,
            "pairs_weight": pairs,
            "kelly_fraction": kelly,
        }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_quant/test_regime.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/quant/regime.py tests/test_quant/test_regime.py
git commit -m "feat: regime-aware routing — blends ensemble weights from HMM regime probabilities"
```

---

### Task 8: Integration — Wire quant layer into feature engine

**Files:**
- Modify: `src/bist_predict/features/engine.py`
- Modify: `tests/test_features/test_engine.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_features/test_engine.py`:

```python
class TestFeatureEngineWithQuant:
    def test_includes_temporal_and_macro_features(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        # Temporal features should always be present
        assert "day_of_week" in features
        assert "month" in features

    def test_compute_quant_features(self, db: Database) -> None:
        engine = FeatureEngine(db)
        features = engine.compute_for_ticker("THYAO", "2026-04-01")
        # Quant features should be present when enough price data exists
        # At minimum we should get Kalman and O-U features (need ~30 data points, we seed 30)
        assert "kalman_trend" in features or "ou_theta" in features
```

- [ ] **Step 2: Add quant feature computation to engine.py**

In `src/bist_predict/features/engine.py`, add the quant imports and computation after the technical indicators section. Add these imports at the top:

```python
from bist_predict.quant.factors import (
    compute_mean_reversion_ou,
    compute_time_series_momentum,
)
from bist_predict.quant.statistical import (
    compute_garch_volatility,
    compute_kalman_trend,
)
from bist_predict.quant.signal_quality import compute_hurst_exponent
```

Add this block inside `compute_for_ticker`, after the Rust features section and before temporal features:

```python
        # Quant alpha features
        if len(close) >= 30:
            features.update(compute_kalman_trend(close))
            features.update(compute_mean_reversion_ou(close))

        if len(close) >= 100:
            daily_returns = np.diff(close) / close[:-1]
            features.update(compute_garch_volatility(daily_returns))
            features.update(compute_hurst_exponent(close))

        if len(close) >= 253:
            tsmom = compute_time_series_momentum(close, period=252)
            features.update(tsmom)
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_features/test_engine.py tests/test_quant/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/bist_predict/features/engine.py tests/test_features/test_engine.py
git commit -m "feat: integrate quant alpha features into feature engine"
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Task | Status |
|---|---|---|
| 3.1 Fama-French factors | Task 2 (compute_fama_french_factors) | Covered |
| 3.1 Cross-sectional momentum | Task 2 (compute_cross_sectional_momentum) | Covered |
| 3.1 Time-series momentum | Task 2 (compute_time_series_momentum) | Covered |
| 3.1 Mean reversion (O-U) | Task 2 (compute_mean_reversion_ou) | Covered |
| 3.2 Kalman filter | Task 3 (compute_kalman_trend) | Covered |
| 3.2 HMM (3-state) | Task 4 (compute_hmm_regime) | Covered |
| 3.2 GARCH(1,1) | Task 3 (compute_garch_volatility) | Covered |
| 3.2 Cointegration | Task 4 (compute_cointegration) | Covered |
| 3.3 Kelly criterion | Task 5 (compute_kelly_fraction) | Covered |
| 3.3 Ledoit-Wolf | Task 5 (compute_ledoit_wolf_covariance) | Covered |
| 3.3 PCA factor extraction | Task 5 (compute_pca_factors) | Covered |
| 3.4 Information Coefficient | Task 6 (compute_information_coefficient) | Covered |
| 3.4 Hurst exponent | Task 6 (compute_hurst_exponent) | Covered |
| 3.4 Wavelet decomposition | Task 6 (compute_wavelet_decomposition) | Covered |
| 3.5 Regime-aware routing | Task 7 (RegimeRouter) | Covered |
| Integration with feature engine | Task 8 | Covered |

### Placeholder Scan
No TBD, TODO, or placeholder text found.

### Type Consistency
All function signatures verified consistent across test and implementation code.
