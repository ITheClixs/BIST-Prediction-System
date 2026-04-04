"""Live accuracy tracking -- logs predictions and measures rolling accuracy."""

from __future__ import annotations

from bist_predict.storage.database import Database


class AccuracyTracker:
    """Track prediction accuracy over time using the predictions table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def log_prediction(
        self,
        ticker: str,
        prediction_date: str,
        target_date: str,
        direction: str,
        confidence: float,
        predicted_pct_move: float,
        model_version: str,
    ) -> None:
        """Log a new prediction."""
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (ticker, prediction_date, target_date, direction, confidence,
                    predicted_pct_move, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, target_date, model_version) DO UPDATE SET
                       direction = excluded.direction,
                       confidence = excluded.confidence,
                       predicted_pct_move = excluded.predicted_pct_move""",
                (ticker, prediction_date, target_date, direction, confidence,
                 predicted_pct_move, model_version),
            )
            conn.commit()

    def record_actual(
        self, ticker: str, target_date: str, actual_pct_move: float, model_version: str,
    ) -> None:
        """Record the actual outcome for a previously logged prediction."""
        with self._db.connect() as conn:
            conn.execute(
                """UPDATE predictions SET actual_pct_move = ?
                   WHERE ticker = ? AND target_date = ? AND model_version = ?""",
                (actual_pct_move, ticker, target_date, model_version),
            )
            conn.commit()

    def get_predictions(
        self, ticker: str, limit: int = 100,
    ) -> list[dict]:
        """Get recent predictions for a ticker."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT ticker, prediction_date, target_date, direction, confidence,
                          predicted_pct_move, actual_pct_move, model_version
                   FROM predictions WHERE ticker = ?
                   ORDER BY target_date DESC LIMIT ?""",
                (ticker, limit),
            ).fetchall()

        return [
            {
                "ticker": r[0], "prediction_date": r[1], "target_date": r[2],
                "direction": r[3], "confidence": r[4], "predicted_pct_move": r[5],
                "actual_pct_move": r[6], "model_version": r[7],
            }
            for r in rows
        ]

    def rolling_accuracy(self, ticker: str, window: int = 30) -> float:
        """Compute rolling directional accuracy over last N predictions with known outcomes."""
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT direction, actual_pct_move FROM predictions
                   WHERE ticker = ? AND actual_pct_move IS NOT NULL
                   ORDER BY target_date DESC LIMIT ?""",
                (ticker, window),
            ).fetchall()

        if not rows:
            return 0.0

        correct = sum(
            1 for direction, actual in rows
            if (direction == "UP" and actual > 0) or (direction == "DOWN" and actual <= 0)
        )
        return correct / len(rows)

    def confidence_buckets(
        self, ticker: str,
    ) -> dict[str, dict[str, float]]:
        """Analyze accuracy by confidence bucket.

        Returns {bucket_label: {accuracy, count}} for buckets:
        60-70%, 70-80%, 80-90%, 90-100%.
        """
        with self._db.connect() as conn:
            rows = conn.execute(
                """SELECT confidence, direction, actual_pct_move FROM predictions
                   WHERE ticker = ? AND actual_pct_move IS NOT NULL""",
                (ticker,),
            ).fetchall()

        buckets: dict[str, dict[str, int]] = {
            "60-70": {"correct": 0, "total": 0},
            "70-80": {"correct": 0, "total": 0},
            "80-90": {"correct": 0, "total": 0},
            "90-100": {"correct": 0, "total": 0},
        }

        for conf, direction, actual in rows:
            if conf < 0.6:
                continue
            elif conf < 0.7:
                bucket = "60-70"
            elif conf < 0.8:
                bucket = "70-80"
            elif conf < 0.9:
                bucket = "80-90"
            else:
                bucket = "90-100"

            buckets[bucket]["total"] += 1
            is_correct = (direction == "UP" and actual > 0) or (direction == "DOWN" and actual <= 0)
            if is_correct:
                buckets[bucket]["correct"] += 1

        result = {}
        for label, data in buckets.items():
            if data["total"] > 0:
                result[label] = {
                    "accuracy": data["correct"] / data["total"],
                    "count": float(data["total"]),
                }

        return result
