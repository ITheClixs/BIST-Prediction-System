"""Tests for XGBoost model with dual prediction heads."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.models.xgboost_model import XGBoostModel


@pytest.fixture
def sample_data() -> tuple:
    rng = np.random.default_rng(42)
    n = 200
    n_features = 10
    X = rng.normal(0, 1, (n, n_features))
    y_dir = (X[:, 0] > 0).astype(np.int64)
    y_pct = X[:, 0] * 0.01 + rng.normal(0, 0.005, n)
    return X, y_dir, y_pct


class TestXGBoostModel:
    def test_train_returns_metrics(self, sample_data: tuple) -> None:
        X, y_dir, y_pct = sample_data
        model = XGBoostModel()
        metrics = model.train(X[:150], y_dir[:150], y_pct[:150], X[150:], y_dir[150:], y_pct[150:])
        assert "val_accuracy" in metrics
        assert "val_mae" in metrics

    def test_predict_returns_probs_and_pct(self, sample_data: tuple) -> None:
        X, y_dir, y_pct = sample_data
        model = XGBoostModel()
        model.train(X[:150], y_dir[:150], y_pct[:150])
        probs, pct_pred = model.predict(X[150:])
        assert probs.shape == (50,)
        assert pct_pred.shape == (50,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_better_than_random(self, sample_data: tuple) -> None:
        X, y_dir, y_pct = sample_data
        model = XGBoostModel()
        model.train(X[:150], y_dir[:150], y_pct[:150])
        probs, _ = model.predict(X[150:])
        predicted_dir = (probs > 0.5).astype(int)
        accuracy = np.mean(predicted_dir == y_dir[150:])
        assert accuracy > 0.55

    def test_save_and_load(self, sample_data: tuple, tmp_path) -> None:
        X, y_dir, y_pct = sample_data
        model = XGBoostModel()
        model.train(X[:150], y_dir[:150], y_pct[:150])

        path = str(tmp_path / "xgb_model")
        model.save(path)

        model2 = XGBoostModel()
        model2.load(path)
        probs1, pct1 = model.predict(X[150:])
        probs2, pct2 = model2.predict(X[150:])
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_name(self) -> None:
        model = XGBoostModel()
        assert model.name == "xgboost"
