"""Compute macro-economic features from stored TCMB data."""

from __future__ import annotations

import math

from bist_predict.storage.database import Database

MACRO_INDICATORS = ["USD_TRY", "EUR_TRY", "GOLD_TRY", "POLICY_RATE", "CPI", "BOND_2Y"]


def compute_macro_features(db: Database, date: str) -> dict[str, float]:
    """Compute macro feature deltas and percentage changes for a given date.

    For each indicator, computes:
    - {indicator}_value: current value
    - {indicator}_delta: change from previous available value
    - {indicator}_pct: percentage change from previous
    """
    features: dict[str, float] = {}

    with db.connect() as conn:
        for indicator in MACRO_INDICATORS:
            key = indicator.lower()

            # Get current value
            row = conn.execute(
                "SELECT value FROM macro_data WHERE indicator = ? AND date = ?",
                (indicator, date),
            ).fetchone()

            if row is None:
                features[f"{key}_value"] = math.nan
                features[f"{key}_delta"] = math.nan
                features[f"{key}_pct"] = math.nan
                continue

            current = row[0]
            features[f"{key}_value"] = current

            # Get previous value
            prev_row = conn.execute(
                """SELECT value FROM macro_data
                   WHERE indicator = ? AND date < ?
                   ORDER BY date DESC LIMIT 1""",
                (indicator, date),
            ).fetchone()

            if prev_row is None:
                features[f"{key}_delta"] = math.nan
                features[f"{key}_pct"] = math.nan
            else:
                prev = prev_row[0]
                features[f"{key}_delta"] = current - prev
                features[f"{key}_pct"] = (current - prev) / prev if prev != 0 else math.nan

    return features
