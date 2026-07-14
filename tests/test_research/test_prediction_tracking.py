"""Immutable live prediction and maturation tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from bist_predict.ingest.types import OHLCVBar, OpenQuality
from bist_predict.research.prediction_tracking import (
    ImmutablePredictionStore,
    PredictionRecord,
    persist_signal_predictions,
)
from bist_predict.research.portfolio_backtest import (
    CostModel,
    PortfolioBacktester,
    StrategyConfig,
)
from bist_predict.research.predictions import PREDICTION_COLUMNS


def _record() -> PredictionRecord:
    return PredictionRecord(
        prediction_id="prediction-001",
        signal_date="2024-01-02",
        execution_date="2024-01-03",
        target_date="2024-01-03",
        ticker="THYAO",
        model_run_id="20240102T180000Z-abc1234-deadbe",
        feature_manifest_hash="a" * 64,
        predicted_return=0.02,
        predicted_probability=0.72,
        created_at="2024-01-02T15:10:00+00:00",
    )


def _bar(*, close: float = 102.0, quality: OpenQuality = OpenQuality.OBSERVED) -> OHLCVBar:
    return OHLCVBar(
        ticker="THYAO",
        date=date(2024, 1, 3),
        open=100.0,
        high=103.0,
        low=99.0,
        close=close,
        adj_close=close,
        volume=1_000_000,
        source="yahoo",
        open_quality=quality,
        provider_record_id="yahoo:THYAO:2024-01-03",
    )


def test_prediction_record_is_create_only_and_binds_original_model(tmp_path) -> None:
    store = ImmutablePredictionStore(tmp_path)

    path = store.persist(_record())

    assert path.exists()
    assert store.unresolved()[0].model_run_id == _record().model_run_id
    with pytest.raises(FileExistsError):
        store.persist(_record())


def test_maturation_waits_for_target_close_and_requires_observed_open(tmp_path) -> None:
    store = ImmutablePredictionStore(tmp_path)
    store.persist(_record())

    assert store.mature(
        as_of=datetime(2024, 1, 3, 14, 59, tzinfo=UTC), prices=[_bar()]
    ) == ()
    with pytest.raises(ValueError, match="observed open"):
        store.mature(
            as_of=datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
            prices=[_bar(quality=OpenQuality.PROXY)],
        )


def test_maturation_freezes_exact_realized_target_and_accuracy_uses_it(tmp_path) -> None:
    store = ImmutablePredictionStore(tmp_path)
    store.persist(_record())

    outcomes = store.mature(
        as_of=datetime(2024, 1, 3, 15, 1, tzinfo=UTC), prices=[_bar()]
    )
    repeated = store.mature(
        as_of=datetime(2024, 1, 4, 15, 1, tzinfo=UTC), prices=[_bar(close=50.0)]
    )
    metrics = store.accuracy_metrics()

    assert outcomes[0].realized_return == pytest.approx(0.02)
    assert outcomes[0].model_run_id == _record().model_run_id
    assert outcomes[0].source_record_id == "yahoo:THYAO:2024-01-03"
    assert repeated == ()
    assert store.outcomes()[0] == outcomes[0]
    assert metrics == {
        "resolved_predictions": 1,
        "directional_accuracy": 1.0,
        "mae": pytest.approx(0.0),
    }


def test_actionable_signal_generation_persists_original_prediction(tmp_path) -> None:
    predictions = pd.DataFrame.from_records(
        [
            {
                "date": "2024-01-02",
                "ticker": "THYAO",
                "fold_id": "fold_0001",
                "model_name": "ridge",
                "model_version": "ridge-v1",
                "training_end": "2023-12-29",
                "feature_manifest_hash": "a" * 64,
                "target": 0.02,
                "prediction": 1,
                "predicted_probability": 0.72,
                "predicted_return": 0.02,
            }
        ],
        columns=PREDICTION_COLUMNS,
    )
    result = PortfolioBacktester(
        strategy=StrategyConfig(top_k=1, decision_cost_rate=0.001),
        costs=CostModel(0.0, 0.0, 0.0, 0.0, 0.0),
    ).run(predictions, [_bar()], model_name="ridge", starting_equity=100_000.0)
    store = ImmutablePredictionStore(tmp_path)

    paths = persist_signal_predictions(
        result.signals,
        predictions,
        store,
        model_run_id="20240102T180000Z-abc1234-deadbe",
    )

    assert len(paths) == 1
    record = store.records()[0]
    assert record.prediction_id == result.signals[0].prediction_id
    assert record.predicted_return == 0.02
    assert record.feature_manifest_hash == "a" * 64
