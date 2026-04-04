"""Tests for Transformer model with dual prediction heads."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.models.transformer_model import TransformerModel


@pytest.fixture
def sample_seq_data() -> tuple:
    rng = np.random.default_rng(42)
    n = 100
    seq_len = 60
    n_features = 10
    X = rng.normal(0, 1, (n, seq_len, n_features))
    y_dir = rng.integers(0, 2, n).astype(np.int64)
    y_pct = rng.normal(0, 0.01, n)
    return X, y_dir, y_pct


class TestTransformerModel:
    def test_train_returns_metrics(self, sample_seq_data: tuple) -> None:
        X, y_dir, y_pct = sample_seq_data
        model = TransformerModel(input_size=10, d_model=32, nhead=4, epochs=2, batch_size=32)
        metrics = model.train(X[:80], y_dir[:80], y_pct[:80], X[80:], y_dir[80:], y_pct[80:])
        assert "val_accuracy" in metrics
        assert "val_mae" in metrics

    def test_predict_shape(self, sample_seq_data: tuple) -> None:
        X, y_dir, y_pct = sample_seq_data
        model = TransformerModel(input_size=10, d_model=32, nhead=4, epochs=2, batch_size=32)
        model.train(X[:80], y_dir[:80], y_pct[:80])
        probs, pct_pred = model.predict(X[80:])
        assert probs.shape == (20,)
        assert pct_pred.shape == (20,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_save_and_load(self, sample_seq_data: tuple, tmp_path) -> None:
        X, y_dir, y_pct = sample_seq_data
        model = TransformerModel(input_size=10, d_model=32, nhead=4, epochs=2, batch_size=32)
        model.train(X[:80], y_dir[:80], y_pct[:80])

        path = str(tmp_path / "transformer_model")
        model.save(path)

        model2 = TransformerModel(input_size=10, d_model=32, nhead=4)
        model2.load(path)
        probs1, _ = model.predict(X[80:])
        probs2, _ = model2.predict(X[80:])
        np.testing.assert_array_almost_equal(probs1, probs2, decimal=5)

    def test_name(self) -> None:
        assert TransformerModel(input_size=10).name == "transformer"
