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
        early_stopping_rounds: int = 20,
        seed: int = 42,
    ) -> None:
        self._early_stopping_rounds = early_stopping_rounds
        self._best_iterations: dict[str, int] = {}
        self._classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        self._regressor = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="mae",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def n_features(self) -> int | None:
        return getattr(self._classifier, "n_features_in_", None)

    @property
    def best_iterations(self) -> dict[str, int]:
        """Best validation iterations for the two independently stopped heads."""
        return dict(self._best_iterations)

    def train(
        self,
        X_train: NDArray[np.float64],
        y_dir_train: NDArray[np.int64],
        y_pct_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_dir_val: NDArray[np.int64] | None = None,
        y_pct_val: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        validation_values = (X_val, y_dir_val, y_pct_val)
        has_validation = all(value is not None for value in validation_values)
        if any(value is not None for value in validation_values) and not has_validation:
            raise ValueError("validation features and both targets must be supplied together")

        if has_validation:
            self._classifier.set_params(
                early_stopping_rounds=self._early_stopping_rounds
            )
            self._regressor.set_params(early_stopping_rounds=self._early_stopping_rounds)
            self._classifier.fit(
                X_train,
                y_dir_train,
                eval_set=[(X_val, y_dir_val)],
                verbose=False,
            )
            self._regressor.fit(
                X_train,
                y_pct_train,
                eval_set=[(X_val, y_pct_val)],
                verbose=False,
            )
            self._best_iterations = {
                "classifier": int(self._classifier.best_iteration),
                "regressor": int(self._regressor.best_iteration),
            }
        else:
            self._classifier.set_params(early_stopping_rounds=None)
            self._regressor.set_params(early_stopping_rounds=None)
            self._classifier.fit(X_train, y_dir_train)
            self._regressor.fit(X_train, y_pct_train)
            self._best_iterations = {}

        metrics: dict[str, float] = {}
        if has_validation:
            probs, pct_pred = self.predict(X_val)
            pred_dir = (probs > 0.5).astype(int)
            metrics["val_accuracy"] = float(np.mean(pred_dir == y_dir_val))
            metrics["val_mae"] = float(np.mean(np.abs(pct_pred - y_pct_val)))
            metrics["classifier_best_iteration"] = float(
                self._best_iterations["classifier"]
            )
            metrics["regressor_best_iteration"] = float(
                self._best_iterations["regressor"]
            )

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
