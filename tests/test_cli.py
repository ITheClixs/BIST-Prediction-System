"""CLI helper tests."""

from __future__ import annotations

from bist_predict.cli import _get_feature_dates_to_compute, _resolve_tickers
from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


def test_get_feature_dates_to_compute_backfills_missing_history(tmp_db_path) -> None:
    """Missing feature dates should be backfilled from available price dates."""
    db = Database(tmp_db_path)
    db.initialize()

    with db.connect() as conn:
        for day, close in [
            ("2026-04-01", 10.0),
            ("2026-04-02", 10.5),
            ("2026-04-03", 10.8),
        ]:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("BIOEN", day, close, close, close, close, close, 1000, "test"),
            )
        conn.commit()

    FeatureStore(db).save("BIOEN", "2026-04-01", {"close": 10.0, "volume": 1000.0, "rsi_14": 50.0})

    assert _get_feature_dates_to_compute(db, "BIOEN", None) == [
        "2026-04-02",
        "2026-04-03",
    ]


def test_get_feature_dates_to_compute_explicit_date_overrides_backfill(tmp_db_path) -> None:
    """Explicit target dates should compute exactly one date."""
    db = Database(tmp_db_path)
    db.initialize()

    with db.connect() as conn:
        conn.execute(
            """INSERT INTO raw_prices
               (ticker, date, open, high, low, close, adj_close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("BIOEN", "2026-04-01", 10.0, 10.0, 10.0, 10.0, 10.0, 1000, "test"),
        )
        conn.commit()

    assert _get_feature_dates_to_compute(db, "BIOEN", "2026-04-05") == ["2026-04-05"]


def test_get_feature_dates_to_compute_ignores_future_snapshot(tmp_db_path) -> None:
    """A single future-dated feature row should not suppress backfill."""
    db = Database(tmp_db_path)
    db.initialize()

    with db.connect() as conn:
        for day, close in [
            ("2026-04-01", 10.0),
            ("2026-04-02", 10.5),
            ("2026-04-03", 10.8),
        ]:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("BIOEN", day, close, close, close, close, close, 1000, "test"),
            )
        conn.commit()

    FeatureStore(db).save("BIOEN", "2026-04-05", {"close": 10.8, "volume": 1000.0, "rsi_14": 50.0})

    assert _get_feature_dates_to_compute(db, "BIOEN", None) == [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03",
    ]


def test_resolve_tickers_uses_db_universe_and_persists_manual_add(tmp_db_path) -> None:
    """CLI ticker resolution should come from tracked_stocks state."""
    db = Database(tmp_db_path)
    db.initialize()

    tickers = _resolve_tickers(db, None)
    assert "THYAO" in tickers

    assert _resolve_tickers(db, "BIOEN") == ["BIOEN"]
    assert "BIOEN" in db.list_tracked_stocks()
