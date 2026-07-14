"""Create-only live predictions and one-time point-in-time maturation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bist_predict.ingest.types import OHLCVBar, OpenQuality
from bist_predict.research.portfolio_backtest import Signal, prediction_identifier
from bist_predict.research.predictions import validate_predictions

ISTANBUL = ZoneInfo("Europe/Istanbul")
TARGET_CLOSE = time(18, 0)


@dataclass(frozen=True)
class PredictionRecord:
    """The exact model output known when a signal was generated."""

    prediction_id: str
    signal_date: str
    execution_date: str
    target_date: str
    ticker: str
    model_run_id: str
    feature_manifest_hash: str
    predicted_return: float
    predicted_probability: float
    created_at: str

    def __post_init__(self) -> None:
        signal = date.fromisoformat(self.signal_date)
        execution = date.fromisoformat(self.execution_date)
        target = date.fromisoformat(self.target_date)
        created = datetime.fromisoformat(self.created_at)
        if not signal < execution <= target:
            raise ValueError("prediction dates must follow signal < execution <= target")
        if created.tzinfo is None:
            raise ValueError("prediction created_at must be timezone-aware")
        if not self.prediction_id or not self.model_run_id or not self.ticker:
            raise ValueError("prediction identity fields must not be empty")
        if len(self.feature_manifest_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.feature_manifest_hash
        ):
            raise ValueError("feature manifest hash must be a lowercase SHA-256 value")
        if not math.isfinite(self.predicted_return):
            raise ValueError("predicted return must be finite")
        if not 0.0 <= self.predicted_probability <= 1.0:
            raise ValueError("predicted probability must lie in [0, 1]")


@dataclass(frozen=True)
class PredictionOutcome:
    """Frozen realized target tied to the original prediction and price record."""

    prediction_id: str
    model_run_id: str
    feature_manifest_hash: str
    ticker: str
    target_date: str
    target_open: float
    target_close: float
    realized_return: float
    realized_direction: int
    source: str
    source_record_id: str
    matured_at: str


class ImmutablePredictionStore:
    """Store each prediction and outcome as a separate create-only JSON record."""

    def __init__(self, root: Path) -> None:
        self._records = root / "records"
        self._outcomes = root / "outcomes"
        self._records.mkdir(parents=True, exist_ok=True)
        self._outcomes.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_new(path: Path, payload: dict[str, object]) -> Path:
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("x") as handle:
            handle.write(rendered)
        return path

    def persist(self, prediction: PredictionRecord) -> Path:
        """Persist a signal-time record without upsert or model substitution."""
        return self._write_new(
            self._records / f"{prediction.prediction_id}.json",
            asdict(prediction),
        )

    @staticmethod
    def _read(path: Path, record_type: type[PredictionRecord] | type[PredictionOutcome]):
        return record_type(**json.loads(path.read_text()))

    def records(self) -> tuple[PredictionRecord, ...]:
        return tuple(
            self._read(path, PredictionRecord)
            for path in sorted(self._records.glob("*.json"))
        )

    def outcomes(self) -> tuple[PredictionOutcome, ...]:
        return tuple(
            self._read(path, PredictionOutcome)
            for path in sorted(self._outcomes.glob("*.json"))
        )

    def unresolved(self) -> tuple[PredictionRecord, ...]:
        resolved = {outcome.prediction_id for outcome in self.outcomes()}
        return tuple(
            prediction
            for prediction in self.records()
            if prediction.prediction_id not in resolved
        )

    def mature(
        self,
        *,
        as_of: datetime,
        prices: Iterable[OHLCVBar],
    ) -> tuple[PredictionOutcome, ...]:
        """Freeze outcomes whose declared open-to-close interval has completed."""
        if as_of.tzinfo is None:
            raise ValueError("maturation as_of must be timezone-aware")
        by_key: dict[tuple[str, date], OHLCVBar] = {}
        for bar in prices:
            key = (bar.ticker, bar.date)
            if key in by_key:
                raise ValueError(f"duplicate maturation price: {bar.ticker} {bar.date}")
            by_key[key] = bar

        matured: list[PredictionOutcome] = []
        for prediction in self.unresolved():
            target_date = date.fromisoformat(prediction.target_date)
            target_close = datetime.combine(target_date, TARGET_CLOSE, tzinfo=ISTANBUL)
            if as_of < target_close:
                continue
            bar = by_key.get((prediction.ticker, target_date))
            if bar is None:
                raise ValueError(
                    f"missing exact target price for {prediction.ticker} {target_date}"
                )
            if bar.open_quality is not OpenQuality.OBSERVED:
                raise ValueError("prediction maturation requires an observed open")
            if bar.open <= 0.0:
                raise ValueError("prediction maturation requires a positive open")
            if not bar.provider_record_id:
                raise ValueError("prediction maturation requires source record provenance")
            realized = bar.close / bar.open - 1.0
            outcome = PredictionOutcome(
                prediction_id=prediction.prediction_id,
                model_run_id=prediction.model_run_id,
                feature_manifest_hash=prediction.feature_manifest_hash,
                ticker=prediction.ticker,
                target_date=prediction.target_date,
                target_open=bar.open,
                target_close=bar.close,
                realized_return=realized,
                realized_direction=int(realized > 0.0),
                source=bar.source,
                source_record_id=bar.provider_record_id,
                matured_at=as_of.isoformat(),
            )
            self._write_new(
                self._outcomes / f"{prediction.prediction_id}.json",
                asdict(outcome),
            )
            matured.append(outcome)
        return tuple(matured)

    def accuracy_metrics(self) -> dict[str, int | float]:
        """Compute accuracy only from paired immutable predictions and outcomes."""
        predictions = {item.prediction_id: item for item in self.records()}
        outcomes = self.outcomes()
        if not outcomes:
            return {"resolved_predictions": 0, "directional_accuracy": 0.0, "mae": 0.0}
        predicted_returns = np.asarray(
            [predictions[item.prediction_id].predicted_return for item in outcomes],
            dtype=np.float64,
        )
        realized_returns = np.asarray(
            [item.realized_return for item in outcomes], dtype=np.float64
        )
        return {
            "resolved_predictions": len(outcomes),
            "directional_accuracy": float(
                np.mean((predicted_returns > 0.0) == (realized_returns > 0.0))
            ),
            "mae": float(np.mean(np.abs(predicted_returns - realized_returns))),
        }


def persist_signal_predictions(
    signals: Iterable[Signal],
    predictions: pd.DataFrame,
    store: ImmutablePredictionStore,
    *,
    model_run_id: str,
) -> tuple[Path, ...]:
    """Bind every actionable signal to the immutable model output that formed it."""
    validate_predictions(predictions)
    if not model_run_id:
        raise ValueError("model_run_id must not be empty")
    by_identifier = {
        prediction_identifier(
            str(row["fold_id"]),
            str(row["model_name"]),
            str(row["date"]),
            str(row["ticker"]),
        ): row
        for _, row in predictions.iterrows()
    }
    paths: list[Path] = []
    for signal in signals:
        if not signal.eligible:
            continue
        if signal.execution_date is None:
            raise ValueError("actionable signal requires an execution date")
        prediction = by_identifier.get(signal.prediction_id)
        if prediction is None:
            raise ValueError(f"signal has no immutable prediction: {signal.signal_id}")
        signal_date = date.fromisoformat(signal.signal_date)
        created_at = datetime.combine(
            signal_date, time(18, 10), tzinfo=ISTANBUL
        ).isoformat()
        paths.append(
            store.persist(
                PredictionRecord(
                    prediction_id=signal.prediction_id,
                    signal_date=signal.signal_date,
                    execution_date=signal.execution_date,
                    target_date=signal.execution_date,
                    ticker=signal.ticker,
                    model_run_id=model_run_id,
                    feature_manifest_hash=str(
                        prediction["feature_manifest_hash"]
                    ),
                    predicted_return=float(prediction["predicted_return"]),
                    predicted_probability=float(
                        prediction["predicted_probability"]
                    ),
                    created_at=created_at,
                )
            )
        )
    return tuple(paths)
