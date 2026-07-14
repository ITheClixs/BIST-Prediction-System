"""Fold-aware missingness reports for accepted research panels."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from bist_predict.features.manifest import FeatureManifest


def build_missingness_report(
    panel: pd.DataFrame,
    manifest: FeatureManifest,
    *,
    source_by_sample: Mapping[object, str],
    fold_by_sample: Mapping[object, str],
) -> pd.DataFrame:
    """Return one auditable missingness record per sample and feature."""
    required = {"date", "ticker"}
    for feature in manifest.ordered_feature_names:
        required.add(feature)
        required.add(f"{feature}__missing_reason")
    missing_columns = sorted(required.difference(panel.columns))
    if missing_columns:
        raise ValueError(f"panel missing columns: {', '.join(missing_columns)}")

    records: list[dict[str, object]] = []
    for sample_index, sample in panel.iterrows():
        if sample_index not in source_by_sample:
            raise ValueError(f"source missing for sample: {sample_index}")
        if sample_index not in fold_by_sample:
            raise ValueError(f"fold missing for sample: {sample_index}")
        for feature in manifest.ordered_feature_names:
            value = sample[feature]
            reason_value = sample[f"{feature}__missing_reason"]
            is_missing = bool(pd.isna(value))
            has_reason = not bool(pd.isna(reason_value))
            if is_missing and not has_reason:
                raise ValueError(
                    f"missing reason required for {feature} at sample {sample_index}"
                )
            if not is_missing and has_reason:
                raise ValueError(
                    f"observed feature has missing reason for {feature} at sample {sample_index}"
                )
            records.append(
                {
                    "feature": feature,
                    "ticker": str(sample["ticker"]),
                    "date": str(sample["date"]),
                    "source": source_by_sample[sample_index],
                    "fold": fold_by_sample[sample_index],
                    "is_missing": is_missing,
                    "missing_reason": str(reason_value) if has_reason else None,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "feature",
            "ticker",
            "date",
            "source",
            "fold",
            "is_missing",
            "missing_reason",
        ],
    )

