"""Model version registry -- tracks trained models in SQLite."""

from __future__ import annotations

import json

from bist_predict.storage.database import Database


class ModelRegistry:
    """Register, activate, and query trained model versions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def register(
        self, model_name: str, version: str, model_path: str, metrics: dict,
    ) -> None:
        """Register a trained model version."""
        metrics_json = json.dumps(metrics)
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO model_registry (model_name, version, model_path, metrics_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(model_name, version) DO UPDATE SET
                       model_path = excluded.model_path,
                       metrics_json = excluded.metrics_json""",
                (model_name, version, model_path, metrics_json),
            )
            conn.commit()

    def activate(self, model_name: str, version: str) -> None:
        """Set a model version as the active one. Deactivates all others for that model."""
        with self._db.connect() as conn:
            conn.execute(
                "UPDATE model_registry SET is_active = 0 WHERE model_name = ?",
                (model_name,),
            )
            conn.execute(
                "UPDATE model_registry SET is_active = 1 WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
            conn.commit()

    def get_active(self, model_name: str) -> dict | None:
        """Get the active version for a model. Returns None if no active version."""
        with self._db.connect() as conn:
            row = conn.execute(
                """SELECT model_name, version, model_path, metrics_json, trained_at
                   FROM model_registry WHERE model_name = ? AND is_active = 1""",
                (model_name,),
            ).fetchone()

        if row is None:
            return None

        return {
            "model_name": row[0], "version": row[1], "model_path": row[2],
            "metrics_json": row[3], "trained_at": row[4],
        }

    def list_models(self, model_name: str | None = None) -> list[dict]:
        """List all registered models, optionally filtered by name."""
        with self._db.connect() as conn:
            if model_name:
                rows = conn.execute(
                    """SELECT model_name, version, model_path, metrics_json, trained_at, is_active
                       FROM model_registry WHERE model_name = ? ORDER BY trained_at DESC""",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT model_name, version, model_path, metrics_json, trained_at, is_active
                       FROM model_registry ORDER BY model_name, trained_at DESC""",
                ).fetchall()

        return [
            {
                "model_name": r[0], "version": r[1], "model_path": r[2],
                "metrics_json": r[3], "trained_at": r[4], "is_active": bool(r[5]),
            }
            for r in rows
        ]
