"""LightGBM model with dual prediction heads (classification + regression)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray


class LightGBMModel:
    """LightGBM with separate classifier and regressor heads."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
        seed: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._early_stopping_rounds = early_stopping_rounds
        self._best_iterations: dict[str, int] = {}
        self._classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        self._regressor = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        self._clf_booster: lgb.Booster | None = None
        self._reg_booster: lgb.Booster | None = None

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def n_features(self) -> int | None:
        if self._clf_booster is not None:
            return self._clf_booster.num_feature()
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
            callbacks = [
                lgb.early_stopping(self._early_stopping_rounds, verbose=False)
            ]
            self._classifier.fit(
                X_train,
                y_dir_train,
                eval_set=[(X_val, y_dir_val)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
            self._regressor.fit(
                X_train,
                y_pct_train,
                eval_set=[(X_val, y_pct_val)],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(self._early_stopping_rounds, verbose=False)
                ],
            )
            self._best_iterations = {
                "classifier": int(self._classifier.best_iteration_),
                "regressor": int(self._regressor.best_iteration_),
            }
        else:
            self._classifier.fit(X_train, y_dir_train)
            self._regressor.fit(X_train, y_pct_train)
            self._best_iterations = {}
        # Cache boosters for predict
        self._clf_booster = self._classifier.booster_
        self._reg_booster = self._regressor.booster_

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
        clf_booster = self._clf_booster if self._clf_booster is not None else self._classifier.booster_
        reg_booster = self._reg_booster if self._reg_booster is not None else self._regressor.booster_

        # LightGBM booster.predict() for binary classification already returns
        # probabilities (sigmoid applied internally). For regression, raw scores.
        probs = clf_booster.predict(X)
        reg_raw = reg_booster.predict(X)

        return probs.astype(np.float64), reg_raw.astype(np.float64)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        booster = self._clf_booster if self._clf_booster is not None else self._classifier.booster_
        booster.save_model(str(p / "classifier.txt"))
        booster = self._reg_booster if self._reg_booster is not None else self._regressor.booster_
        booster.save_model(str(p / "regressor.txt"))

    def load(self, path: str) -> None:
        p = Path(path)
        self._clf_booster = lgb.Booster(model_file=str(p / "classifier.txt"))
        self._reg_booster = lgb.Booster(model_file=str(p / "regressor.txt"))
