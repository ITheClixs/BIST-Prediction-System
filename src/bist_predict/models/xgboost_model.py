"""XGBoost model with dual prediction heads (classification + regression)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from xgboost import XGBClassifier, XGBRegressor


class XGBoostModel:
    """XGBoost with separate classifier and regressor heads."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> None:
        self._classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self._regressor = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="mae",
            random_state=42,
            verbosity=0,
        )

    @property
    def name(self) -> str:
        return "xgboost"

    def train(
        self,
        X_train: NDArray[np.float64],
        y_dir_train: NDArray[np.int64],
        y_pct_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_dir_val: NDArray[np.int64] | None = None,
        y_pct_val: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        self._classifier.fit(X_train, y_dir_train)
        self._regressor.fit(X_train, y_pct_train)

        metrics: dict[str, float] = {}
        if X_val is not None and y_dir_val is not None and y_pct_val is not None:
            probs, pct_pred = self.predict(X_val)
            pred_dir = (probs > 0.5).astype(int)
            metrics["val_accuracy"] = float(np.mean(pred_dir == y_dir_val))
            metrics["val_mae"] = float(np.mean(np.abs(pct_pred - y_pct_val)))

        return metrics

    def predict(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        probs = self._classifier.predict_proba(X)[:, 1]
        pct = self._regressor.predict(X)
        return probs.astype(np.float64), pct.astype(np.float64)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._classifier.save_model(str(p / "classifier.json"))
        self._regressor.save_model(str(p / "regressor.json"))

    def load(self, path: str) -> None:
        p = Path(path)
        self._classifier.load_model(str(p / "classifier.json"))
        self._regressor.load_model(str(p / "regressor.json"))
