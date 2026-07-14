"""Accepted pooled-model features with point-in-time, scale-free formulas."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, time
from statistics import fmean, pstdev
from typing import Iterable
from zoneinfo import ZoneInfo

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.ingest.types import OHLCVBar
from bist_predict.research.panel import FeatureSnapshot, MissingReason

ISTANBUL = ZoneInfo("Europe/Istanbul")
FEATURE_AVAILABLE_TIME = time(18, 10)


class FeatureHistoryError(ValueError):
    """Raised when configured features cannot be reached from loaded history."""


def _spec(name: str, formula: str, lookback: int) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        formula=formula,
        formula_version="1",
        lookback=lookback,
        availability_rule="after_official_session_close",
        missing_value_policy="preserve_with_reason",
        normalization_policy="none",
    )


STATIONARY_FEATURE_MANIFEST = FeatureManifest(
    schema_version="1.0.0",
    features=(
        _spec("log_return_1d", "log(adj_close_t / adj_close_t_minus_1)", 2),
        _spec("log_return_5d", "log(adj_close_t / adj_close_t_minus_5)", 6),
        _spec("log_return_20d", "log(adj_close_t / adj_close_t_minus_20)", 21),
        _spec("close_over_sma20_minus_1", "adj_close / mean_20(adj_close) - 1", 20),
        _spec("sma20_over_sma100_minus_1", "mean_20(adj_close) / mean_100(adj_close) - 1", 100),
        _spec("atr14_over_close", "atr_14(adjusted_ohlc) / adj_close", 15),
        _spec("vwap20_over_close_minus_1", "vwap_20(adjusted_hlc, volume) / adj_close - 1", 20),
        _spec("log_volume", "log1p(raw_volume)", 1),
        _spec("volume_zscore_20", "zscore_20(raw_volume)", 20),
        _spec("realized_volatility_20", "std_20(log_return_1d) * sqrt(252)", 21),
        _spec("intraday_range_over_close", "(adjusted_high - adjusted_low) / adj_close", 1),
        _spec("overnight_gap", "adjusted_open / prior_adj_close - 1", 2),
        _spec("drawdown_20", "adj_close / max_20(adj_close) - 1", 20),
        _spec("cross_sectional_return_rank", "midrank_percentile_by_date(log_return_20d)", 21),
        _spec("market_relative_return_20d", "log_return_20d - date_mean(log_return_20d)", 21),
        _spec("day_of_week_sin", "sin(2*pi*weekday/7)", 1),
        _spec("day_of_week_cos", "cos(2*pi*weekday/7)", 1),
        _spec("month_sin", "sin(2*pi*(month-1)/12)", 1),
        _spec("month_cos", "cos(2*pi*(month-1)/12)", 1),
    ),
)


def _ordered_bars(prices: Iterable[OHLCVBar]) -> dict[str, list[OHLCVBar]]:
    by_ticker: dict[str, list[OHLCVBar]] = defaultdict(list)
    keys: set[tuple[str, date]] = set()
    for bar in prices:
        key = (bar.ticker, bar.date)
        if key in keys:
            raise FeatureHistoryError(f"duplicate price record: {bar.ticker} {bar.date}")
        keys.add(key)
        by_ticker[bar.ticker].append(bar)
    for bars in by_ticker.values():
        bars.sort(key=lambda bar: bar.date)
    return dict(by_ticker)


def _adjusted_ohlc(bar: OHLCVBar) -> tuple[float, float, float, float]:
    if bar.close <= 0.0 or bar.adj_close <= 0.0:
        raise FeatureHistoryError(
            f"non-positive price for {bar.ticker} on {bar.date.isoformat()}"
        )
    factor = bar.adj_close / bar.close
    values = (
        bar.open * factor,
        bar.high * factor,
        bar.low * factor,
        bar.adj_close,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise FeatureHistoryError(
            f"invalid adjusted OHLC for {bar.ticker} on {bar.date.isoformat()}"
        )
    return values


def _midrank_percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    result: dict[str, float] = {}
    index = 0
    while index < count:
        stop = index + 1
        while stop < count and ordered[stop][1] == ordered[index][1]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        percentile = (average_rank - 0.5) / count
        for ticker, _ in ordered[index:stop]:
            result[ticker] = percentile
        index = stop
    return result


def _ticker_values(bars: list[OHLCVBar], index: int) -> dict[str, float | None]:
    adjusted = [_adjusted_ohlc(bar) for bar in bars[: index + 1]]
    opens = [row[0] for row in adjusted]
    highs = [row[1] for row in adjusted]
    lows = [row[2] for row in adjusted]
    closes = [row[3] for row in adjusted]
    volumes = [float(bar.volume) for bar in bars[: index + 1]]

    sma20 = fmean(closes[-20:])
    sma100 = fmean(closes[-100:])
    true_ranges = [
        max(
            highs[position] - lows[position],
            abs(highs[position] - closes[position - 1]),
            abs(lows[position] - closes[position - 1]),
        )
        for position in range(index - 13, index + 1)
    ]
    atr14 = fmean(true_ranges)
    typical = [
        (high + low + close) / 3.0
        for high, low, close in zip(highs[-20:], lows[-20:], closes[-20:])
    ]
    volume20 = volumes[-20:]
    volume_sum = sum(volume20)
    vwap20 = (
        sum(price * volume for price, volume in zip(typical, volume20)) / volume_sum
        if volume_sum > 0.0
        else None
    )
    daily_log_returns = [
        math.log(closes[position] / closes[position - 1])
        for position in range(index - 19, index + 1)
    ]
    volume_std = pstdev(volume20)
    weekday_angle = 2.0 * math.pi * bars[index].date.weekday() / 7.0
    month_angle = 2.0 * math.pi * (bars[index].date.month - 1) / 12.0

    return {
        "log_return_1d": math.log(closes[index] / closes[index - 1]),
        "log_return_5d": math.log(closes[index] / closes[index - 5]),
        "log_return_20d": math.log(closes[index] / closes[index - 20]),
        "close_over_sma20_minus_1": closes[index] / sma20 - 1.0,
        "sma20_over_sma100_minus_1": sma20 / sma100 - 1.0,
        "atr14_over_close": atr14 / closes[index],
        "vwap20_over_close_minus_1": (
            vwap20 / closes[index] - 1.0 if vwap20 is not None else None
        ),
        "log_volume": math.log1p(volumes[index]),
        "volume_zscore_20": (
            (volumes[index] - fmean(volume20)) / volume_std
            if volume_std > 0.0
            else 0.0
        ),
        "realized_volatility_20": pstdev(daily_log_returns) * math.sqrt(252.0),
        "intraday_range_over_close": (highs[index] - lows[index]) / closes[index],
        "overnight_gap": opens[index] / closes[index - 1] - 1.0,
        "drawdown_20": closes[index] / max(closes[-20:]) - 1.0,
        "cross_sectional_return_rank": None,
        "market_relative_return_20d": None,
        "day_of_week_sin": math.sin(weekday_angle),
        "day_of_week_cos": math.cos(weekday_angle),
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
    }


def build_stationary_snapshots(
    prices: Iterable[OHLCVBar],
    *,
    manifest: FeatureManifest = STATIONARY_FEATURE_MANIFEST,
    target_horizon_sessions: int = 1,
) -> tuple[FeatureSnapshot, ...]:
    """Build eligible point-in-time feature snapshots for a pooled benchmark."""
    if target_horizon_sessions <= 0:
        raise ValueError("target_horizon_sessions must be positive")
    if manifest.manifest_hash != STATIONARY_FEATURE_MANIFEST.manifest_hash:
        raise FeatureHistoryError("stationary calculator requires its exact manifest")

    by_ticker = _ordered_bars(prices)
    maximum_lookback = max(spec.lookback for spec in manifest.features)
    required_rows = maximum_lookback + target_horizon_sessions
    candidates: dict[tuple[date, str], dict[str, float | None]] = {}
    for ticker, bars in by_ticker.items():
        if len(bars) < required_rows:
            raise FeatureHistoryError(
                f"{ticker} requires {required_rows} rows for configured lookbacks and target horizon; got {len(bars)}"
            )
        stop = len(bars) - target_horizon_sessions
        for index in range(maximum_lookback - 1, stop):
            key = (bars[index].date, ticker)
            candidates[key] = _ticker_values(bars, index)

    by_date: dict[date, dict[str, dict[str, float | None]]] = defaultdict(dict)
    for (session, ticker), values in candidates.items():
        by_date[session][ticker] = values

    snapshots: list[FeatureSnapshot] = []
    for session in sorted(by_date):
        date_values = by_date[session]
        returns = {
            ticker: float(values["log_return_20d"])
            for ticker, values in date_values.items()
            if values["log_return_20d"] is not None
        }
        ranks = _midrank_percentiles(returns)
        market_return = fmean(returns.values())
        for ticker in sorted(date_values):
            values = date_values[ticker]
            values["cross_sectional_return_rank"] = ranks[ticker]
            values["market_relative_return_20d"] = returns[ticker] - market_return
            reasons = {
                name: MissingReason.CALCULATION_FAILURE
                for name, value in values.items()
                if value is None
            }
            ordered_values = {
                name: values[name] for name in manifest.ordered_feature_names
            }
            snapshots.append(
                FeatureSnapshot(
                    date=session,
                    ticker=ticker,
                    feature_available_at=datetime.combine(
                        session, FEATURE_AVAILABLE_TIME, tzinfo=ISTANBUL
                    ),
                    values=ordered_values,
                    missing_reasons=reasons,
                    feature_manifest_hash=manifest.manifest_hash,
                )
            )

    return tuple(snapshots)
