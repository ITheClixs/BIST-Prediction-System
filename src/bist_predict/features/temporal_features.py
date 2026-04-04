"""Compute calendar/temporal features from a date."""

from __future__ import annotations

import calendar
from datetime import date


def compute_temporal_features(d: date) -> dict[str, float]:
    """Compute temporal features for a given date.

    Returns day-of-week effects, month seasonality, and calendar position features.
    """
    day_of_week = d.weekday()  # 0=Monday, 4=Friday
    _, days_in_month = calendar.monthrange(d.year, d.month)

    return {
        "day_of_week": float(day_of_week),
        "month": float(d.month),
        "quarter": float((d.month - 1) // 3 + 1),
        "day_of_month": float(d.day),
        "week_of_year": float(d.isocalendar()[1]),
        "is_monday": float(day_of_week == 0),
        "is_friday": float(day_of_week == 4),
        "is_month_start": float(d.day <= 3),
        "is_month_end": float(d.day >= days_in_month - 2),
        "is_quarter_start": float(d.month in (1, 4, 7, 10) and d.day <= 5),
        "is_quarter_end": float(d.month in (3, 6, 9, 12) and d.day >= days_in_month - 4),
        "is_january": float(d.month == 1),  # January effect
    }
