"""Compute aggregated sentiment features from stored sentiment data."""

from __future__ import annotations

import math

from bist_predict.storage.database import Database


def compute_sentiment_features(
    db: Database, ticker: str, date: str
) -> dict[str, float]:
    """Compute aggregated sentiment features for a ticker on a date.

    Returns:
    - sentiment_mean: average sentiment score
    - sentiment_count: number of sentiment records
    - sentiment_positive_ratio: fraction of positive (>0) scores
    - sentiment_max: maximum sentiment score
    - sentiment_min: minimum sentiment score
    """
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT sentiment_score FROM sentiment_data
               WHERE ticker = ? AND date = ? AND sentiment_score IS NOT NULL""",
            (ticker, date),
        ).fetchall()

    if not rows:
        return {
            "sentiment_mean": math.nan,
            "sentiment_count": 0,
            "sentiment_positive_ratio": math.nan,
            "sentiment_max": math.nan,
            "sentiment_min": math.nan,
        }

    scores = [row[0] for row in rows]
    positive_count = sum(1 for s in scores if s > 0)

    return {
        "sentiment_mean": sum(scores) / len(scores),
        "sentiment_count": len(scores),
        "sentiment_positive_ratio": positive_count / len(scores),
        "sentiment_max": max(scores),
        "sentiment_min": min(scores),
    }
