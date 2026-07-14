"""Missingness report lineage tests."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_predict.features.manifest import FeatureManifest, FeatureSpec
from bist_predict.research.missingness import build_missingness_report


def _manifest() -> FeatureManifest:
    return FeatureManifest(
        schema_version="1",
        features=(
            FeatureSpec(
                name="return_1d",
                formula="log return",
                formula_version="1",
                lookback=2,
                availability_rule="after close",
                missing_value_policy="preserve_with_reason",
                normalization_policy="none",
            ),
        ),
    )


def test_report_records_feature_ticker_date_source_fold_and_reason() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06"],
            "ticker": ["THYAO", "THYAO"],
            "return_1d": [0.01, None],
            "return_1d__missing_reason": [None, "stale_source"],
        },
        index=[10, 11],
    )

    report = build_missingness_report(
        panel,
        _manifest(),
        source_by_sample={10: "yahoo", 11: "isyatirim"},
        fold_by_sample={10: "fold_0001_train", 11: "fold_0001_validation"},
    )

    assert list(report.columns) == [
        "feature",
        "ticker",
        "date",
        "source",
        "fold",
        "is_missing",
        "missing_reason",
    ]
    assert report.loc[1].to_dict() == {
        "feature": "return_1d",
        "ticker": "THYAO",
        "date": "2026-01-06",
        "source": "isyatirim",
        "fold": "fold_0001_validation",
        "is_missing": True,
        "missing_reason": "stale_source",
    }


def test_report_rejects_unexplained_missing_values() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "ticker": ["THYAO"],
            "return_1d": [None],
            "return_1d__missing_reason": [None],
        }
    )

    with pytest.raises(ValueError, match="missing reason required"):
        build_missingness_report(
            panel,
            _manifest(),
            source_by_sample={0: "yahoo"},
            fold_by_sample={0: "fold_0001_validation"},
        )
