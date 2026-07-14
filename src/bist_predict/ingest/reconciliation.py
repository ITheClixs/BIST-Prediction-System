"""Provider overlap diagnostics and partial-interval reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from bist_predict.ingest.types import OHLCVBar


@dataclass(frozen=True)
class ProviderComparison:
    """Field-level differences for one overlapping provider observation."""

    ticker: str
    date: str
    timestamp_aligned: bool
    open_absolute_difference: float
    high_absolute_difference: float
    low_absolute_difference: float
    close_absolute_difference: float
    return_difference: float | None
    volume_absolute_difference: int
    corporate_action_consistent: bool


@dataclass(frozen=True)
class ReconciliationReport:
    """Provenance and quality summary for one provider merge."""

    overlaps: tuple[ProviderComparison, ...] = ()
    missing_from_primary: tuple[str, ...] = ()
    missing_from_fallback: tuple[str, ...] = ()
    fallback_fill_count: int = 0


def reconcile_price_bars(
    primary: Sequence[OHLCVBar],
    fallback: Sequence[OHLCVBar],
) -> tuple[list[OHLCVBar], ReconciliationReport]:
    """Prefer primary bars while filling missing dates from the fallback."""
    primary_by_key = _unique_by_key(primary, "primary")
    fallback_by_key = _unique_by_key(fallback, "fallback")
    primary_keys = set(primary_by_key)
    fallback_keys = set(fallback_by_key)
    overlap_keys = sorted(primary_keys & fallback_keys)
    primary_returns = _close_returns(primary_by_key)
    fallback_returns = _close_returns(fallback_by_key)

    overlaps = tuple(
        _comparison(
            primary_by_key[key],
            fallback_by_key[key],
            primary_returns.get(key),
            fallback_returns.get(key),
        )
        for key in overlap_keys
    )
    missing_primary_keys = sorted(fallback_keys - primary_keys)
    missing_fallback_keys = sorted(primary_keys - fallback_keys)
    canonical = dict(fallback_by_key)
    canonical.update(primary_by_key)
    bars = [canonical[key] for key in sorted(canonical)]
    report = ReconciliationReport(
        overlaps=overlaps,
        missing_from_primary=tuple(key[1].isoformat() for key in missing_primary_keys),
        missing_from_fallback=tuple(key[1].isoformat() for key in missing_fallback_keys),
        fallback_fill_count=len(missing_primary_keys),
    )
    return bars, report


BarKey = tuple[str, date]


def _unique_by_key(bars: Sequence[OHLCVBar], label: str) -> dict[BarKey, OHLCVBar]:
    result: dict[BarKey, OHLCVBar] = {}
    for bar in bars:
        key = (bar.ticker, bar.date)
        if key in result:
            raise ValueError(f"duplicate {label} provider bar: {bar.ticker} {bar.date}")
        result[key] = bar
    return result


def _close_returns(bars: dict[BarKey, OHLCVBar]) -> dict[BarKey, float]:
    returns: dict[BarKey, float] = {}
    by_ticker: dict[str, list[OHLCVBar]] = {}
    for bar in bars.values():
        by_ticker.setdefault(bar.ticker, []).append(bar)
    for ticker_bars in by_ticker.values():
        ticker_bars.sort(key=lambda bar: bar.date)
        for previous, current in zip(ticker_bars, ticker_bars[1:]):
            if previous.close > 0.0:
                returns[(current.ticker, current.date)] = current.close / previous.close - 1.0
    return returns


def _adjustment_ratio(bar: OHLCVBar) -> float | None:
    if bar.close == 0.0:
        return None
    return bar.adj_close / bar.close


def _comparison(
    primary: OHLCVBar,
    fallback: OHLCVBar,
    primary_return: float | None,
    fallback_return: float | None,
) -> ProviderComparison:
    primary_ratio = _adjustment_ratio(primary)
    fallback_ratio = _adjustment_ratio(fallback)
    ratios_consistent = (
        primary_ratio is None
        and fallback_ratio is None
        or primary_ratio is not None
        and fallback_ratio is not None
        and abs(primary_ratio - fallback_ratio) <= 1e-9
    )
    return_difference = None
    if primary_return is not None and fallback_return is not None:
        return_difference = primary_return - fallback_return
    return ProviderComparison(
        ticker=primary.ticker,
        date=primary.date.isoformat(),
        timestamp_aligned=primary.date == fallback.date,
        open_absolute_difference=abs(primary.open - fallback.open),
        high_absolute_difference=abs(primary.high - fallback.high),
        low_absolute_difference=abs(primary.low - fallback.low),
        close_absolute_difference=abs(primary.close - fallback.close),
        return_difference=return_difference,
        volume_absolute_difference=abs(primary.volume - fallback.volume),
        corporate_action_consistent=ratios_consistent,
    )
