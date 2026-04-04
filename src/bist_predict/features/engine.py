"""Feature computation engine — orchestrates Rust and Python feature computation."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np

from bist_predict.features.macro_features import compute_macro_features
from bist_predict.features.sentiment_features import compute_sentiment_features
from bist_predict.features.store import FeatureStore
from bist_predict.features.temporal_features import compute_temporal_features
from bist_predict.quant.factors import compute_mean_reversion_ou, compute_time_series_momentum
from bist_predict.quant.signal_quality import compute_hurst_exponent
from bist_predict.quant.statistical import compute_garch_volatility, compute_kalman_trend
from bist_predict.storage.database import Database

logger = logging.getLogger(__name__)

# Import Rust module — may not be available if not compiled
try:
    import bist_features as rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    logger.warning("Rust bist_features module not available — using Python fallback")

# Moving average periods
SMA_PERIODS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [5, 10, 20, 50, 100, 200]


class FeatureEngine:
    """Orchestrates feature computation from raw price data."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._store = FeatureStore(db)

    def compute_for_ticker(self, ticker: str, target_date: str) -> dict[str, float]:
        """Compute all features for a ticker on a given date.

        Loads historical price data, computes technical indicators (via Rust),
        macro features, sentiment features, and temporal features.
        """
        # Load price history
        prices = self._load_price_history(ticker, target_date, lookback=252)
        if len(prices) == 0:
            return {}

        features: dict[str, float] = {}

        # Technical indicators (Rust)
        close = np.array([p[5] for p in prices], dtype=np.float64)  # close
        high = np.array([p[3] for p in prices], dtype=np.float64)
        low = np.array([p[4] for p in prices], dtype=np.float64)
        open_ = np.array([p[2] for p in prices], dtype=np.float64)
        volume = np.array([p[7] for p in prices], dtype=np.float64)

        if HAS_RUST and len(close) > 0:
            features.update(self._compute_rust_features(open_, high, low, close, volume))

        # Quant alpha features
        if len(close) >= 30:
            features.update(compute_kalman_trend(close))
            features.update(compute_mean_reversion_ou(close))

        if len(close) >= 100:
            daily_returns = np.diff(close) / close[:-1]
            features.update(compute_garch_volatility(daily_returns))
            features.update(compute_hurst_exponent(close))

        if len(close) >= 253:
            features.update(compute_time_series_momentum(close, period=252))

        # Temporal features
        try:
            dt = date.fromisoformat(target_date)
            features.update(compute_temporal_features(dt))
        except ValueError:
            pass

        # Macro features
        features.update(compute_macro_features(self._db, target_date))

        # Sentiment features
        features.update(compute_sentiment_features(self._db, ticker, target_date))

        return features

    def compute_and_store(self, ticker: str, target_date: str) -> dict[str, float]:
        """Compute features and persist them in the feature store."""
        features = self.compute_for_ticker(ticker, target_date)
        if features:
            self._store.save(ticker, target_date, features)
        return features

    def _load_price_history(
        self, ticker: str, end_date: str, lookback: int = 252
    ) -> list[tuple]:
        """Load up to `lookback` days of price history ending at end_date."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT id, ticker, open, high, low, close, adj_close, volume
                   FROM raw_prices
                   WHERE ticker = ? AND date <= ?
                   ORDER BY date ASC""",
                (ticker, end_date),
            ).fetchall()
        # Return last `lookback` rows
        return rows[-lookback:] if len(rows) > lookback else rows

    def _compute_rust_features(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> dict[str, float]:
        """Compute all Rust-based technical indicators. Returns feature dict with latest values."""
        features: dict[str, float] = {}
        last = len(close) - 1

        # RSI
        rsi = rust.compute_rsi(close, period=14)
        features["rsi_14"] = float(rsi[last])

        # SMAs
        for period in SMA_PERIODS:
            if len(close) >= period:
                sma = rust.compute_sma(close, period)
                features[f"sma_{period}"] = float(sma[last])

        # EMAs
        for period in EMA_PERIODS:
            if len(close) >= period:
                ema = rust.compute_ema(close, period)
                features[f"ema_{period}"] = float(ema[last])

        # MACD
        macd, signal, hist = rust.compute_macd(close)
        features["macd"] = float(macd[last])
        features["macd_signal"] = float(signal[last])
        features["macd_hist"] = float(hist[last])

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = rust.compute_bollinger_bands(close, period=20)
        features["bb_upper"] = float(bb_upper[last])
        features["bb_middle"] = float(bb_middle[last])
        features["bb_lower"] = float(bb_lower[last])
        if not np.isnan(bb_upper[last]) and not np.isnan(bb_lower[last]):
            bb_width = bb_upper[last] - bb_lower[last]
            features["bb_width"] = float(bb_width)
            if bb_width > 0:
                features["bb_position"] = float((close[last] - bb_lower[last]) / bb_width)

        # Stochastic
        k, d = rust.compute_stochastic(high, low, close)
        features["stoch_k"] = float(k[last])
        features["stoch_d"] = float(d[last])

        # ATR
        atr = rust.compute_atr(high, low, close, period=14)
        features["atr_14"] = float(atr[last])

        # OBV
        obv = rust.compute_obv(close, volume)
        features["obv"] = float(obv[last])

        # VWAP
        vwap = rust.compute_vwap(high, low, close, volume)
        features["vwap"] = float(vwap[last])

        # ADX
        adx = rust.compute_adx(high, low, close, period=14)
        features["adx_14"] = float(adx[last])

        # CCI
        cci = rust.compute_cci(high, low, close, period=20)
        features["cci_20"] = float(cci[last])

        # MFI
        mfi = rust.compute_mfi(high, low, close, volume, period=14)
        features["mfi_14"] = float(mfi[last])

        # Williams %R
        wr = rust.compute_williams_r(high, low, close, period=14)
        features["williams_r_14"] = float(wr[last])

        # Price-derived features
        features["close"] = float(close[last])
        features["volume"] = float(volume[last])
        if len(close) >= 2:
            features["return_1d"] = float((close[last] - close[last - 1]) / close[last - 1])
        if len(close) >= 6:
            features["return_5d"] = float((close[last] - close[last - 5]) / close[last - 5])
        if len(close) >= 21:
            features["return_20d"] = float((close[last] - close[last - 20]) / close[last - 20])

        # Volume features
        if len(volume) >= 20:
            vol_sma = np.mean(volume[-20:])
            features["volume_ratio_20d"] = float(volume[last] / vol_sma) if vol_sma > 0 else 1.0

        return features
