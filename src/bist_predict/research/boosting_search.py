"""Small, immutable, validation-only searches for existing boosting models."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bist_predict.models.lightgbm_model import LightGBMModel
from bist_predict.models.xgboost_model import XGBoostModel


class BoostingModel(Protocol):
    best_iterations: dict[str, int]

    def train(self, *args: object) -> dict[str, float]: ...


ModelFactory = Callable[[str, dict[str, object], int], BoostingModel]


@dataclass(frozen=True)
class BoostingSearchResult:
    """Identity and paths for one completed bounded search."""

    best_trial_id: str
    best_metric: float
    trials_path: str
    manifest_path: str


def _default_factory(
    model_name: str,
    parameters: dict[str, object],
    seed: int,
) -> BoostingModel:
    if model_name == "xgboost":
        return XGBoostModel(**parameters, seed=seed)  # type: ignore[arg-type]
    if model_name == "lightgbm":
        return LightGBMModel(**parameters, seed=seed)  # type: ignore[arg-type]
    raise ValueError(f"unsupported boosting model: {model_name}")


def _validate_candidates(
    candidates: Sequence[dict[str, object]],
    seeds: Sequence[int],
) -> None:
    if not candidates or len(candidates) > 4:
        raise ValueError("boosting search requires 1 to at most 4 candidates")
    if not seeds or len(seeds) > 3:
        raise ValueError("boosting search requires 1 to at most 3 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("boosting search seeds must be unique")
    allowed = {"n_estimators", "max_depth", "learning_rate", "early_stopping_rounds"}
    for parameters in candidates:
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            raise ValueError(f"unknown boosting parameters: {', '.join(unknown)}")
        estimators = int(parameters.get("n_estimators", 200))
        depth = int(parameters.get("max_depth", 6))
        learning_rate = float(parameters.get("learning_rate", 0.05))
        stopping = int(parameters.get("early_stopping_rounds", 20))
        if not 10 <= estimators <= 1_000:
            raise ValueError("n_estimators must remain between 10 and 1000")
        if not 1 <= depth <= 10:
            raise ValueError("max_depth must remain between 1 and 10")
        if not 0.001 <= learning_rate <= 0.3:
            raise ValueError("learning_rate must remain between 0.001 and 0.3")
        if not 1 <= stopping <= 100:
            raise ValueError("early_stopping_rounds must remain between 1 and 100")


def _validate_dates(
    training_dates: tuple[str, str],
    validation_dates: tuple[str, str],
) -> None:
    train_start, train_end = map(pd.Timestamp, training_dates)
    validation_start, validation_end = map(pd.Timestamp, validation_dates)
    if not train_start <= train_end < validation_start <= validation_end:
        raise ValueError("training and validation dates must be ordered and disjoint")


def _validate_arrays(arrays: tuple[NDArray[np.generic], ...]) -> None:
    X_train, y_dir_train, y_return_train, X_validation, y_dir_validation, y_return_validation = arrays
    if X_train.ndim != 2 or X_validation.ndim != 2:
        raise ValueError("boosting feature arrays must be two-dimensional")
    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("training and validation feature widths differ")
    if not (len(X_train) == len(y_dir_train) == len(y_return_train)):
        raise ValueError("training feature and target lengths differ")
    if not (
        len(X_validation) == len(y_dir_validation) == len(y_return_validation)
    ):
        raise ValueError("validation feature and target lengths differ")


def _trial_id(
    model_name: str,
    parameters: dict[str, object],
    seed: int,
    training_dates: tuple[str, str],
    validation_dates: tuple[str, str],
) -> str:
    payload = {
        "model_name": model_name,
        "parameters": parameters,
        "seed": seed,
        "training_dates": training_dates,
        "validation_dates": validation_dates,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_bounded_boosting_search(
    model_name: str,
    X_train: NDArray[np.float64],
    y_dir_train: NDArray[np.int64],
    y_return_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    y_dir_validation: NDArray[np.int64],
    y_return_validation: NDArray[np.float64],
    *,
    training_dates: tuple[str, str],
    validation_dates: tuple[str, str],
    candidates: Sequence[dict[str, object]],
    seeds: Sequence[int],
    output_dir: Path,
    model_factory: ModelFactory = _default_factory,
) -> BoostingSearchResult:
    """Run at most twelve declared trials and freeze their complete evidence."""
    if model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"unsupported boosting model: {model_name}")
    _validate_candidates(candidates, seeds)
    _validate_dates(training_dates, validation_dates)
    _validate_arrays(
        (
            X_train,
            y_dir_train,
            y_return_train,
            X_validation,
            y_dir_validation,
            y_return_validation,
        )
    )
    if output_dir.exists():
        raise FileExistsError(f"boosting search artifact already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    trials: list[dict[str, object]] = []
    for candidate in candidates:
        parameters = dict(candidate)
        parameters.setdefault("early_stopping_rounds", 20)
        for seed in seeds:
            identifier = _trial_id(
                model_name,
                parameters,
                seed,
                training_dates,
                validation_dates,
            )
            model = model_factory(model_name, parameters, seed)
            started = time.perf_counter()
            metrics = model.train(
                X_train,
                y_dir_train,
                y_return_train,
                X_validation,
                y_dir_validation,
                y_return_validation,
            )
            duration = time.perf_counter() - started
            if "val_mae" not in metrics:
                raise ValueError("boosting trial did not report validation MAE")
            trials.append(
                {
                    "trial_id": identifier,
                    "model_name": model_name,
                    "parameters": parameters,
                    "training_dates": {
                        "start": training_dates[0],
                        "end": training_dates[1],
                    },
                    "validation_dates": {
                        "start": validation_dates[0],
                        "end": validation_dates[1],
                    },
                    "metric": {"name": "val_mae", "value": float(metrics["val_mae"])},
                    "metrics": {name: float(value) for name, value in metrics.items()},
                    "seed": int(seed),
                    "duration_seconds": duration,
                    "best_iterations": model.best_iterations,
                }
            )

    trials.sort(key=lambda trial: str(trial["trial_id"]))
    trials_path = output_dir / "trials.jsonl"
    rendered_trials = "".join(
        json.dumps(trial, sort_keys=True, separators=(",", ":")) + "\n"
        for trial in trials
    )
    trials_path.write_text(rendered_trials)
    best = min(
        trials,
        key=lambda trial: (
            float(trial["metric"]["value"]),  # type: ignore[index]
            str(trial["trial_id"]),
        ),
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "candidate_count": len(candidates),
        "seeds": list(seeds),
        "trial_count": len(trials),
        "selection_metric": "val_mae",
        "selection_direction": "minimize",
        "best_trial_id": best["trial_id"],
        "best_metric": best["metric"]["value"],  # type: ignore[index]
        "trials_sha256": hashlib.sha256(rendered_trials.encode()).hexdigest(),
        "training_shape": list(X_train.shape),
        "validation_shape": list(X_validation.shape),
    }
    manifest_path = output_dir / "search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return BoostingSearchResult(
        best_trial_id=str(best["trial_id"]),
        best_metric=float(best["metric"]["value"]),  # type: ignore[index]
        trials_path=str(trials_path),
        manifest_path=str(manifest_path),
    )

