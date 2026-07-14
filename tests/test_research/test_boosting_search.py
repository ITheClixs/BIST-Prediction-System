"""Bounded boosting trial and immutable manifest tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bist_predict.research.boosting_search import run_bounded_boosting_search


class _FakeBoostingModel:
    def __init__(self, parameters: dict[str, object], seed: int) -> None:
        self._parameters = parameters
        self._seed = seed
        self.best_iterations = {"classifier": 7, "regressor": 9}

    def train(self, *args: object) -> dict[str, float]:
        return {
            "val_accuracy": 0.6 + self._seed * 0.001,
            "val_mae": float(self._parameters["learning_rate"]) + self._seed * 1e-5,
            "classifier_best_iteration": 7.0,
            "regressor_best_iteration": 9.0,
        }


def _factory(model_name: str, parameters: dict[str, object], seed: int) -> _FakeBoostingModel:
    assert model_name in {"xgboost", "lightgbm"}
    return _FakeBoostingModel(parameters, seed)


def _arrays() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(30, 3))
    validation = rng.normal(size=(10, 3))
    train_return = train[:, 0] * 0.01
    validation_return = validation[:, 0] * 0.01
    return (
        train,
        (train_return > 0.0).astype(np.int64),
        train_return,
        validation,
        (validation_return > 0.0).astype(np.int64),
        validation_return,
    )


def test_search_records_every_bounded_trial_and_selection_evidence(tmp_path) -> None:
    output = tmp_path / "search"
    result = run_bounded_boosting_search(
        "xgboost",
        *_arrays(),
        training_dates=("2020-01-02", "2023-12-29"),
        validation_dates=("2024-01-02", "2024-03-29"),
        candidates=(
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03},
        ),
        seeds=(11, 29),
        output_dir=output,
        model_factory=_factory,
    )

    trials = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "search_manifest.json").read_text())
    assert len(trials) == 4
    assert {trial["seed"] for trial in trials} == {11, 29}
    assert all(trial["duration_seconds"] >= 0.0 for trial in trials)
    assert all(trial["training_dates"]["end"] == "2023-12-29" for trial in trials)
    assert all(trial["validation_dates"]["start"] == "2024-01-02" for trial in trials)
    assert all(trial["best_iterations"] == {"classifier": 7, "regressor": 9} for trial in trials)
    assert manifest["best_trial_id"] == result.best_trial_id
    assert len(manifest["trials_sha256"]) == 64
    with pytest.raises(FileExistsError):
        run_bounded_boosting_search(
            "xgboost",
            *_arrays(),
            training_dates=("2020-01-02", "2023-12-29"),
            validation_dates=("2024-01-02", "2024-03-29"),
            candidates=({"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},),
            seeds=(11,),
            output_dir=output,
            model_factory=_factory,
        )


def test_search_rejects_unbounded_candidates_and_seed_counts(tmp_path) -> None:
    candidate = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
    with pytest.raises(ValueError, match="at most 4 candidates"):
        run_bounded_boosting_search(
            "lightgbm",
            *_arrays(),
            training_dates=("2020-01-02", "2023-12-29"),
            validation_dates=("2024-01-02", "2024-03-29"),
            candidates=(candidate,) * 5,
            seeds=(11,),
            output_dir=tmp_path / "too-many-candidates",
            model_factory=_factory,
        )
    with pytest.raises(ValueError, match="at most 3 seeds"):
        run_bounded_boosting_search(
            "lightgbm",
            *_arrays(),
            training_dates=("2020-01-02", "2023-12-29"),
            validation_dates=("2024-01-02", "2024-03-29"),
            candidates=(candidate,),
            seeds=(1, 2, 3, 4),
            output_dir=tmp_path / "too-many-seeds",
            model_factory=_factory,
        )
