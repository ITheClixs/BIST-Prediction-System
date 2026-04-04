"""Tests for model version registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bist_predict.models.registry import ModelRegistry
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


class TestModelRegistry:
    def test_register_model(self, db: Database) -> None:
        registry = ModelRegistry(db)
        registry.register("xgboost", "v1", "/models/xgb_v1", {"accuracy": 0.72})
        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["model_name"] == "xgboost"
        assert models[0]["version"] == "v1"

    def test_activate_model(self, db: Database) -> None:
        registry = ModelRegistry(db)
        registry.register("xgboost", "v1", "/models/xgb_v1", {"accuracy": 0.70})
        registry.register("xgboost", "v2", "/models/xgb_v2", {"accuracy": 0.75})

        registry.activate("xgboost", "v2")
        active = registry.get_active("xgboost")
        assert active is not None
        assert active["version"] == "v2"

    def test_activate_deactivates_previous(self, db: Database) -> None:
        registry = ModelRegistry(db)
        registry.register("xgboost", "v1", "/models/xgb_v1", {})
        registry.register("xgboost", "v2", "/models/xgb_v2", {})
        registry.activate("xgboost", "v1")
        registry.activate("xgboost", "v2")

        active = registry.get_active("xgboost")
        assert active["version"] == "v2"

    def test_get_active_no_model(self, db: Database) -> None:
        registry = ModelRegistry(db)
        assert registry.get_active("nonexistent") is None

    def test_metrics_stored_as_json(self, db: Database) -> None:
        registry = ModelRegistry(db)
        metrics = {"accuracy": 0.72, "mae": 0.015, "sharpe": 1.2}
        registry.register("xgboost", "v1", "/models/xgb_v1", metrics)

        models = registry.list_models()
        stored_metrics = json.loads(models[0]["metrics_json"])
        assert stored_metrics["accuracy"] == 0.72
