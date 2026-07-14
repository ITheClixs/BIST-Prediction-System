"""Small live provider-schema check for scheduled CI, not research evidence."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from typing import Sequence

from bist_predict.ingest.types import OHLCVBar, OpenQuality, VolumeQuality
from bist_predict.ingest.yahoo import YahooFinanceClient


def validate_provider_bars(bars: Sequence[OHLCVBar], *, ticker: str) -> dict[str, object]:
    """Fail on schema, quality, chronology, or provenance drift."""
    if not bars:
        raise ValueError(f"provider returned no rows for {ticker}")
    keys: set[tuple[str, date]] = set()
    for bar in bars:
        if bar.ticker != ticker:
            raise ValueError(f"provider returned unexpected ticker: {bar.ticker}")
        key = (bar.ticker, bar.date)
        if key in keys:
            raise ValueError(f"provider returned duplicate row: {bar.ticker} {bar.date}")
        keys.add(key)
        if bar.date.weekday() >= 5:
            raise ValueError(f"provider returned a weekend row: {bar.date}")
        if bar.open_quality is not OpenQuality.OBSERVED or bar.open <= 0.0:
            raise ValueError(f"provider open is not observed: {bar.date}")
        if bar.volume_quality is not VolumeQuality.OBSERVED or bar.volume <= 0:
            raise ValueError(f"provider volume is not observed: {bar.date}")
        if not bar.provider_symbol or not bar.provider_record_id or not bar.source_retrieved_at:
            raise ValueError(f"provider provenance is incomplete: {bar.date}")
    ordered = sorted(bars, key=lambda bar: bar.date)
    return {
        "ticker": ticker,
        "row_count": len(ordered),
        "start": ordered[0].date.isoformat(),
        "end": ordered[-1].date.isoformat(),
        "sources": sorted({bar.source for bar in ordered}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="THYAO")
    parser.add_argument("--days", type=int, default=21)
    args = parser.parse_args()
    if args.days < 7 or args.days > 60:
        parser.error("--days must be between 7 and 60")
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=args.days)
    bars = YahooFinanceClient().fetch_sync(args.ticker, start, end)
    print(json.dumps(validate_provider_bars(bars, ticker=args.ticker), sort_keys=True))


if __name__ == "__main__":
    main()
