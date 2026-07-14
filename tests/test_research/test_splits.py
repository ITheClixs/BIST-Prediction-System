"""Research invariants for date-grouped walk-forward validation."""

from __future__ import annotations

import json
from importlib import import_module

import pandas as pd
import pytest

from bist_predict.research.splits import ExpandingWindowSplitter, WalkForwardFold


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=9, tz="UTC")
    rows: list[dict[str, object]] = []
    for date in dates:
        for ticker in ("AKBNK", "THYAO"):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "feature_available_at": date + pd.Timedelta(hours=18),
                    "target_end": date + pd.Timedelta(days=1, hours=15),
                }
            )
    return pd.DataFrame(rows)


def test_split_module_exposes_expanding_window_api() -> None:
    module = import_module("bist_predict.research.splits")

    assert hasattr(module, "ExpandingWindowSplitter")
    assert hasattr(module, "WalkForwardFold")


def test_folds_keep_every_trading_date_in_one_partition() -> None:
    panel = _panel()
    splitter = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    )

    folds = splitter.split(panel)

    assert len(folds) == 2
    for fold in folds:
        train_dates = set(panel.loc[list(fold.train_indices), "date"])
        validation_dates = set(panel.loc[list(fold.validation_indices), "date"])
        assert train_dates == set(fold.train_dates)
        assert validation_dates == set(fold.validation_dates)
        assert train_dates.isdisjoint(validation_dates)


def test_folds_enforce_feature_and_target_time_chronology() -> None:
    panel = _panel()
    folds = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    ).split(panel)

    for fold in folds:
        train = panel.loc[list(fold.train_indices)]
        validation = panel.loc[list(fold.validation_indices)]
        validation_feature_start = validation["feature_available_at"].min()
        assert train["feature_available_at"].max() < validation_feature_start
        assert train["target_end"].max() < validation_feature_start


def test_target_overlap_purges_the_entire_training_date() -> None:
    panel = _panel()
    overlapping_date = panel["date"].drop_duplicates().iloc[3]
    validation_start = panel["date"].drop_duplicates().iloc[4]
    overlap_row = panel.index[
        (panel["date"] == overlapping_date) & (panel["ticker"] == "AKBNK")
    ][0]
    panel.loc[overlap_row, "target_end"] = pd.Timestamp(validation_start, tz="UTC") + (
        pd.Timedelta(hours=19)
    )

    fold = ExpandingWindowSplitter(
        min_train_dates=4,
        validation_dates=2,
        step_dates=2,
    ).split(panel)[0]

    train = panel.loc[list(fold.train_indices)]
    assert overlapping_date not in set(train["date"])
    assert set(train["ticker"]) == {"AKBNK", "THYAO"}


def test_embargo_dates_are_excluded_between_train_and_validation() -> None:
    panel = _panel()
    fold = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    ).split(panel)[0]

    assert fold.train_dates == ("2024-01-02", "2024-01-03", "2024-01-04")
    assert fold.embargo_dates == ("2024-01-05",)
    assert fold.validation_dates == ("2024-01-08", "2024-01-09")


def test_fold_membership_is_invariant_to_ticker_and_row_order() -> None:
    panel = _panel()
    splitter = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    )
    original = splitter.split(panel)
    reordered = splitter.split(
        panel.sort_values(["ticker", "date"], ascending=[False, False])
    )
    shuffled = splitter.split(panel.sample(frac=1.0, random_state=17))

    def signature(folds: list[WalkForwardFold]) -> list[tuple[object, ...]]:
        return [
            (
                fold.fold_id,
                fold.train_indices,
                fold.validation_indices,
                fold.train_dates,
                fold.validation_dates,
            )
            for fold in folds
        ]

    assert signature(original) == signature(reordered) == signature(shuffled)


def test_all_tickers_share_the_fold_date_boundaries() -> None:
    panel = _panel()
    folds = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    ).split(panel)

    for fold in folds:
        for indices, expected_dates in (
            (fold.train_indices, fold.train_dates),
            (fold.validation_indices, fold.validation_dates),
        ):
            partition = panel.loc[list(indices)]
            dates_by_ticker = partition.groupby("ticker")["date"].agg(tuple).to_dict()
            assert dates_by_ticker == {
                "AKBNK": expected_dates,
                "THYAO": expected_dates,
            }


def test_fold_metadata_and_indices_are_json_serializable() -> None:
    panel = _panel()
    folds = ExpandingWindowSplitter(
        min_train_dates=3,
        validation_dates=2,
        step_dates=2,
        embargo_dates=1,
    ).split(panel)

    payload = [fold.to_dict() for fold in folds]
    encoded = json.dumps(payload, sort_keys=True)

    assert '"fold_id": "fold_0001"' in encoded
    assert payload[0]["train_window"]["feature_time_start"].endswith("+00:00")
    assert payload[0]["validation_window"]["feature_time_start"].endswith("+00:00")


def test_splitter_rejects_fractional_embargo_dates() -> None:
    with pytest.raises(ValueError, match="embargo_dates"):
        ExpandingWindowSplitter(
            min_train_dates=3,
            validation_dates=2,
            embargo_dates=0.5,  # type: ignore[arg-type]
        )
