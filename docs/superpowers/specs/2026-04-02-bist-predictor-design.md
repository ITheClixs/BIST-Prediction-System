# BIST-100 Stock Market Prediction System — Design Spec

## Overview

A CLI-based daily trading signal system for BIST-100 stocks. Predicts next-day direction (UP/DOWN) with calibrated confidence scores and percentage price targets. Uses an ensemble of gradient boosting and deep learning models, powered by institutional-grade quantitative methods, free data sources, and a high-performance Rust feature engine.

## Goals

- Produce accurate daily trading signals for BIST-100 stocks
- Output direction + calibrated confidence + predicted % move per stock
- Use only free/open data sources (no paid subscriptions)
- Maximize predictive accuracy through ensemble ML + quantitative alpha methods
- Provide honest backtesting with no future leakage
- Track live prediction accuracy over time

## Non-Goals

- Web dashboard or API server (CLI only)
- Real-time intraday signals (daily close-to-close only)
- Automated trade execution or broker integration
- Portfolio management or multi-asset allocation

---

## Architecture

Modular pipeline with 6 layers:

```
CLI Interface
    │
    ├── Data Ingest (Python)
    ├── Feature Engine (Rust + PyO3)
    ├── Quantitative Alpha Layer (Python)
    ├── Model Layer (Python — ensemble)
    ├── Evaluation (Python)
    └── SQLite Storage
```

Each module has clear interfaces and can be tested/iterated independently.

---

## 1. Data Ingestion

### OHLCV Price Data

| Source | Role | Details |
|--------|------|---------|
| **Is Yatirim API** | Primary | Undocumented REST API. Most accurate BIST data. Daily OHLCV + volume for all BIST-100 tickers. |
| **Yahoo Finance** | Fallback | Via `yfinance`. BIST tickers use `.IS` suffix (e.g., `THYAO.IS`). Rate-limited but reliable backup. |

Fallback logic: try Is Yatirim first; on failure (timeout, rate limit, data gap), fall back to Yahoo Finance automatically.

### Macro Data (TCMB EVDS)

Requires free API key (register at evds2.tcmb.gov.tr).

- USD/TRY, EUR/TRY exchange rates (daily)
- TCMB policy interest rate (on change)
- CPI / inflation data (monthly)
- Gold price XAU/TRY (daily)
- Government bond yields (daily)

### Sentiment Data

| Source | Method | Details |
|--------|--------|---------|
| **Google News RSS** | Headline parsing | Search per ticker + "borsa" keywords. No API key needed. |
| **Turkish finance RSS** | Feed parsing | bloomberght, bigpara RSS feeds. Turkish NLP for sentiment. |
| **X/Twitter** | Cashtag search | Search `$THYAO` etc. + Turkish finance accounts. Free tier API or snscrape fallback. |
| **Investing.com** | Comment scraping | Stock-specific comment sections for BIST tickers. |

All sentiment sources feed into a Turkish NLP sentiment scorer (using `dbmdz/bert-base-turkish-cased` or `savasy/bert-base-turkish-sentiment-cased` from HuggingFace — both free, pre-trained on Turkish text) that outputs a per-stock aggregate score.

### Data Quality Rules

- Validate OHLCV: open/high/low/close must be logically consistent
- Detect and handle stock splits / dividends via adjusted close
- Flag gaps > 5 trading days (delistings, halts)
- Rate limiter: configurable delays per source, exponential backoff on failure
- Incremental fetch: only pull data newer than last stored date

---

## 2. Feature Engine (Rust + PyO3)

### Rust-Computed Features (performance-critical)

**Technical Indicators (~30+)**
- RSI (14), MACD, Bollinger Bands, Stochastic Oscillator
- ATR, OBV, VWAP, ADX, CCI, MFI, Williams %R
- 12 moving averages (SMA/EMA at 5, 10, 20, 50, 100, 200 periods)

**Pattern Detection**
- Candlestick patterns (doji, hammer, engulfing, etc.)
- Support/resistance levels
- Volume profiles
- Trend breakout detection

**Cross-Stock Analysis**
- Sector momentum scores
- Pair correlations
- Beta to BIST-100 index
- Lead-lag relationships
- Volume flow between sectors

### Python-Computed Features (I/O-bound)

**Macro Features**
- USD/TRY, EUR/TRY daily deltas and trends
- Interest rate change impact
- CPI trend direction
- Gold price delta
- Bond yield spread

**Sentiment Scores**
- Per-stock aggregate sentiment from all sources
- Sentiment momentum (change in sentiment over time)

**Temporal Features**
- Day of week effects
- Month-of-year seasonality
- Pre/post holiday patterns
- Earnings season indicators
- BIST session time patterns

### Feature Store

- SQLite table keyed by `(ticker, date)`
- ~80+ features per stock per date
- Incremental computation — only recalculate for new dates
- Feature versioning via schema migrations
- Missing value strategy: forward-fill for macro, NaN flag for sentiment gaps

---

## 3. Quantitative Alpha Layer

This layer sits between the Feature Engine and Model Layer. It serves three roles:

1. **Extra features** — factor scores, statistical measures feed into ML models
2. **Independent signals** — momentum, mean reversion, pairs trading generate direct signals
3. **Model routing control** — regime detection adjusts ensemble weights dynamically

### 3.1 Factor Models & Alpha Signals

**Fama-French (adapted for BIST)**
Compute SMB (size), HML (value), momentum, profitability factors for BIST-100. Rank stocks on factor exposures. Stocks with positive multi-factor alignment get alpha boost.

**Cross-sectional Momentum (Jegadeesh & Titman, 1993)**
Rank all BIST-100 stocks by trailing 3/6/12-month returns. Compute percentile rank as feature. Top decile historically outperforms.

**Time-series Momentum (Moskowitz, Ooi, Pedersen, 2012)**
Per-stock: if trailing 12-month excess return > 0, signal is long. Binary signal + magnitude as features.

**Mean Reversion (Ornstein-Uhlenbeck process)**
Fit O-U process per stock: dX = θ(μ - X)dt + σdW. Estimate mean-reversion speed (θ), long-term mean (μ), current deviation. Stocks far from mean with high θ → mean reversion signal.

### 3.2 Statistical Methods

**Kalman Filter**
Track hidden "true momentum" state, filtering out noise. Outputs: filtered trend estimate, prediction error variance. Adapts to changing volatility — superior to raw moving averages.

**Hidden Markov Model (3-state)**
States: bull, bear, sideways. Trained on BIST-100 index returns + volatility. Outputs: current regime probability, regime transition probability. Controls ensemble routing (see 3.5).

**GARCH(1,1)**
Per-stock volatility forecast: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1). Outputs: forecasted volatility, volatility surprise (actual vs predicted). Used for confidence calibration and position sizing.

**Cointegration (Engle-Granger)**
Test across BIST-100 pairs. Find cointegrated pairs → when spread deviates beyond 2σ, mean reversion signal. Outputs: spread z-score, half-life of mean reversion.

### 3.3 Risk & Position Sizing

**Kelly Criterion**
f* = (p·b - q) / b, where p = win probability, b = win/loss ratio, q = 1-p. Use fractional Kelly (0.25–0.5×) for safety. Output included in signal display.

**Ledoit-Wolf Shrinkage Covariance**
Robust covariance matrix estimation for BIST-100 returns. Prevents overfitting to noisy sample correlations. Used for portfolio-level risk estimation.

**PCA Factor Extraction**
Extract top 5–10 principal components from BIST-100 return matrix. These are latent market drivers (e.g., "banking factor", "export factor"). Used as both ML features and risk decomposition.

### 3.4 Signal Quality Measurement

**Information Coefficient (IC)**
IC = rank_correlation(predicted_return, actual_return). IC > 0.05 is meaningful. Measured per feature and per model — automatically prunes useless signals.

**Hurst Exponent**
H > 0.5 → trending (trust momentum). H < 0.5 → mean-reverting (trust O-U / pairs). H ≈ 0.5 → random walk (reduce confidence). Computed per-stock via R/S analysis.

**Wavelet Decomposition**
Discrete wavelet transform separates price into frequency bands: daily noise, weekly cycles, monthly trends. ML models receive decomposed signals instead of raw price.

### 3.5 Regime-Aware Routing

The HMM regime detection dynamically adjusts the ensemble:

- **Bull regime** → increase momentum model weight, decrease mean-reversion weight, Kelly at 0.5×
- **Bear regime** → increase mean-reversion weight, decrease momentum weight, Kelly at 0.25×
- **Sideways regime** → pairs trading signals dominate, reduce overall signal count, Kelly at 0.25×

---

## 4. Model Layer

### Individual Models

| Model | Input | Strength |
|-------|-------|----------|
| **XGBoost** | Tabular features (80+) | Best tabular performance, feature importance |
| **LightGBM** | Tabular features (80+) | Faster, handles categoricals natively, diversity |
| **LSTM** | 30-day feature sequences | Temporal dependencies, momentum shifts |
| **Transformer** | 60-day feature sequences | Long-range attention, event detection |

Each model has dual prediction heads:
- Classification head → direction (UP/DOWN)
- Regression head → predicted percentage move

### Ensemble Combiner

Meta-learner (logistic regression or small XGBoost) trained on validation set predictions from all 4 models. Learns optimal weighting. Supports per-stock weight overrides where one model dominates for certain tickers.

Ensemble weights are modulated by the HMM regime-aware routing from the Quantitative Alpha Layer.

### Confidence Calibration

Platt scaling fits a sigmoid to map raw ensemble scores to calibrated probabilities. When the model outputs "78% UP", historically ~78% of such predictions were correct. Configurable minimum confidence threshold (default: 60%).

### Training Strategy

- **Walk-forward validation** — train on past, validate on next N days, slide window forward. No future leakage.
- **Retrain cadence** — full retrain monthly, incremental update weekly
- **Lookback** — 3–5 years of history for tree models, 1–2 years for deep learning
- **Model registry** — version each trained model, track performance, auto-rollback if new model underperforms

---

## 5. Evaluation & Backtesting

### Backtest Rules

- Walk-forward only — train on past, test on future, slide window
- Realistic costs — commission (0.1%) and slippage (0.05%) per trade
- No survivorship bias — include stocks that left BIST-100 during test period
- Signal delay — prediction at market close, trade at next-day open price
- Confidence threshold — only count trades above configurable confidence

### Metrics

**Prediction Quality:** accuracy, precision, recall, F1, AUC-ROC, Brier score, MAE on % predictions

**Trading Quality:** Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, average win/loss ratio, Calmar ratio

**Benchmarks:** vs buy-and-hold, vs BIST-100 index, vs random baseline, vs each sub-model individually, per-sector breakdown

### Live Accuracy Tracking

- Every prediction logged with timestamp, confidence, predicted direction and % move
- Actual result recorded automatically on next data fetch
- Rolling 30/60/90-day accuracy windows
- Per-stock accuracy breakdown
- Confidence bucket analysis (is 80%+ confidence actually more accurate than 60–70%?)
- Automatic model degradation alerts if accuracy drops below threshold

---

## 6. CLI Interface

### Commands

| Command | Description |
|---------|-------------|
| `bist-predict fetch` | Pull latest market data from all sources |
| `bist-predict features` | Compute features for latest data |
| `bist-predict train` | Train or retrain prediction models |
| `bist-predict signals` | Get today's trading signals |
| `bist-predict signals THYAO --detail` | Detailed signal for a single stock |
| `bist-predict backtest` | Run walk-forward backtest |
| `bist-predict accuracy` | Show prediction accuracy history |
| `bist-predict stocks` | List tracked BIST-100 stocks |
| `bist-predict config` | View/edit configuration |

### Signal Output Format

Signals grouped by confidence tier:
- **STRONG BUY** (≥80% UP confidence)
- **BUY** (70–80% UP confidence)
- **SELL** (70–80% DOWN confidence)
- **STRONG SELL** (≥80% DOWN confidence)

Each signal shows: ticker, direction, confidence %, target %, sentiment indicator, Kelly-suggested position size.

Detailed view (`--detail`) adds: per-model breakdown, top 5 features driving prediction, historical per-stock accuracy.

---

## 7. Project Structure

```
bist-predictor/
├── CLAUDE.md
├── pyproject.toml
├── Cargo.toml
├── config.toml
│
├── src/
│   └── bist_predict/
│       ├── __init__.py
│       ├── cli.py                  # Click CLI entry point
│       ├── config.py
│       │
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── scheduler.py        # Orchestrates collectors
│       │   ├── isyatirim.py
│       │   ├── yahoo.py
│       │   ├── tcmb.py
│       │   ├── sentiment.py
│       │   └── quality.py
│       │
│       ├── quant/
│       │   ├── __init__.py
│       │   ├── factors.py          # Fama-French, momentum, mean reversion
│       │   ├── statistical.py      # Kalman, HMM, GARCH, cointegration
│       │   ├── risk.py             # Kelly, Ledoit-Wolf, PCA
│       │   ├── signal_quality.py   # IC, Hurst, wavelets
│       │   └── regime.py           # Regime-aware routing
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── xgboost_model.py
│       │   ├── lightgbm_model.py
│       │   ├── lstm_model.py
│       │   ├── transformer_model.py
│       │   ├── ensemble.py
│       │   ├── calibration.py
│       │   └── registry.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── backtest.py
│       │   ├── metrics.py
│       │   └── tracker.py
│       │
│       └── storage/
│           ├── __init__.py
│           ├── database.py
│           └── migrations.py
│
├── rust/
│   └── bist_features/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── indicators.rs
│           ├── patterns.rs
│           └── correlations.rs
│
├── tests/
│   ├── test_ingest/
│   ├── test_features/
│   ├── test_quant/
│   ├── test_models/
│   └── test_evaluation/
│
├── data/
│   └── bist.db
│
└── docs/
    └── superpowers/
        └── specs/
```

---

## 8. Configuration

```toml
[data]
tcmb_api_key = ""
fetch_retries = 3
rate_limit_delay = 1.0

[signals]
min_confidence = 0.70
lookback_days = 30

[models]
retrain_interval = "monthly"
ensemble_weights = "learned"

[quant]
hmm_states = 3
kelly_fraction = 0.25
hurst_window = 252

[backtest]
commission = 0.001
slippage = 0.0005
```

---

## 9. Tech Stack

| Component | Technology |
|-----------|-----------|
| CLI | Click (Python) |
| HTTP client | httpx |
| Market data | yfinance, custom Is Yatirim client |
| RSS parsing | feedparser |
| ML (trees) | XGBoost, LightGBM |
| ML (deep) | PyTorch (LSTM, Transformer) |
| NLP | transformers (Turkish sentiment model) |
| Quant | scipy, statsmodels, hmmlearn, arch (GARCH), pywt (wavelets) |
| Rust binding | PyO3 + maturin |
| Storage | SQLite |
| Build | uv (Python), maturin (Rust→Python) |

---

## 10. Testing Strategy

- **Unit tests** per module: ingest, features, quant, models, evaluation
- **Integration tests**: full pipeline from fetch → features → predict → evaluate
- **Backtest validation**: verify no future leakage by checking that training data timestamps always precede test data
- **Data quality tests**: synthetic OHLCV data with known anomalies to verify quality checks
- **Model sanity tests**: predictions should beat random baseline on historical data
- **Rust tests**: native Rust tests for indicator calculations against known reference values
