"""Date-grouped expanding-window validation for research panels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True)
class SampleWindow:
    """JSON-safe date and timestamp boundaries for one fold partition."""

    date_start: str
    date_end: str
    feature_time_start: str
    feature_time_end: str
    target_end_start: str
    target_end_end: str


@dataclass(frozen=True)
class WalkForwardFold:
    """One immutable date-grouped research fold."""

    fold_id: str
    train_indices: tuple[JsonScalar, ...]
    validation_indices: tuple[JsonScalar, ...]
    train_dates: tuple[str, ...]
    embargo_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    train_window: SampleWindow
    validation_window: SampleWindow

    def to_dict(self) -> dict[str, Any]:
        """Return boundaries and sample indices ready for JSON persistence."""
        payload = asdict(self)
        payload["train_indices"] = list(self.train_indices)
        payload["validation_indices"] = list(self.validation_indices)
        payload["train_dates"] = list(self.train_dates)
        payload["embargo_dates"] = list(self.embargo_dates)
        payload["validation_dates"] = list(self.validation_dates)
        return payload


@dataclass(frozen=True)
class ExpandingWindowSplitter:
    """Create expanding-window folds without splitting trading dates.

    ``min_train_dates`` determines the first validation boundary before purge.
    The actual training set can contain fewer dates when a whole date group is
    removed because its feature or label timestamps overlap validation.
    """

    min_train_dates: int
    validation_dates: int
    step_dates: int | None = None
    embargo_dates: int = 0
    date_column: str = "date"
    ticker_column: str = "ticker"
    feature_time_column: str = "feature_available_at"
    target_end_column: str = "target_end"

    def __post_init__(self) -> None:
        self._require_positive("min_train_dates", self.min_train_dates)
        self._require_positive("validation_dates", self.validation_dates)
        if self.step_dates is not None:
            self._require_positive("step_dates", self.step_dates)
        if (
            isinstance(self.embargo_dates, bool)
            or not isinstance(self.embargo_dates, int)
            or self.embargo_dates < 0
        ):
            raise ValueError("embargo_dates must be a non-negative integer")

    @staticmethod
    def _require_positive(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def split(self, panel: pd.DataFrame) -> list[WalkForwardFold]:
        """Split a point-in-time panel using trading-date boundaries.

        Training date groups are purged when any sample's feature timestamp or
        target end reaches the first validation feature timestamp. This keeps
        every ticker on a date together while enforcing the temporal invariant.
        """
        working = self._validated_panel(panel)
        trading_dates = tuple(sorted(working["_split_date"].unique()))
        step = self.step_dates or self.validation_dates
        validation_start_position = self.min_train_dates + self.embargo_dates
        folds: list[WalkForwardFold] = []

        while validation_start_position + self.validation_dates <= len(trading_dates):
            validation_dates = trading_dates[
                validation_start_position : validation_start_position
                + self.validation_dates
            ]
            validation_rows = self._rows_for_dates(working, validation_dates)
            validation_feature_start = validation_rows["_feature_time"].min()

            train_stop = validation_start_position - self.embargo_dates
            candidate_train_dates = trading_dates[:train_stop]
            train_dates = tuple(
                split_date
                for split_date in candidate_train_dates
                if self._date_precedes_validation(
                    working,
                    split_date,
                    validation_feature_start,
                )
            )
            embargo = trading_dates[train_stop:validation_start_position]
            train_rows = self._rows_for_dates(working, train_dates)

            if not train_rows.empty:
                folds.append(
                    self._build_fold(
                        ordinal=len(folds) + 1,
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        train_dates=train_dates,
                        embargo_dates=embargo,
                        validation_dates=validation_dates,
                    )
                )

            validation_start_position += step

        return folds

    def _validated_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.date_column,
            self.ticker_column,
            self.feature_time_column,
            self.target_end_column,
        }
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(f"Panel is missing required columns: {missing}")
        if not panel.index.is_unique:
            raise ValueError("Panel index must uniquely identify every sample")

        working = panel[
            [
                self.date_column,
                self.ticker_column,
                self.feature_time_column,
                self.target_end_column,
            ]
        ].copy()
        working["_sample_index"] = [self._json_index(value) for value in panel.index]
        working["_index_sort"] = working["_sample_index"].map(
            lambda value: f"{type(value).__name__}:{value}"
        )
        working["_split_date"] = self._parse_timestamps(
            working[self.date_column], self.date_column
        ).dt.normalize()
        working["_feature_time"] = self._parse_timestamps(
            working[self.feature_time_column], self.feature_time_column
        )
        working["_target_end"] = self._parse_timestamps(
            working[self.target_end_column], self.target_end_column
        )
        working["_ticker_sort"] = working[self.ticker_column].astype(str)
        return working.sort_values(
            ["_split_date", "_ticker_sort", "_index_sort"],
            kind="stable",
        )

    @staticmethod
    def _parse_timestamps(values: pd.Series, name: str) -> pd.Series:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{name} must contain valid, non-missing timestamps")
        return parsed

    @staticmethod
    def _json_index(value: object) -> JsonScalar:
        if hasattr(value, "item"):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError("Panel indices must be JSON scalar values")

    @staticmethod
    def _rows_for_dates(
        working: pd.DataFrame,
        dates: tuple[pd.Timestamp, ...],
    ) -> pd.DataFrame:
        return working.loc[working["_split_date"].isin(dates)]

    @staticmethod
    def _date_precedes_validation(
        working: pd.DataFrame,
        split_date: pd.Timestamp,
        validation_feature_start: pd.Timestamp,
    ) -> bool:
        date_rows = working.loc[working["_split_date"] == split_date]
        return bool(
            date_rows["_feature_time"].max() < validation_feature_start
            and date_rows["_target_end"].max() < validation_feature_start
        )

    @staticmethod
    def _iso_dates(dates: tuple[pd.Timestamp, ...]) -> tuple[str, ...]:
        return tuple(date.date().isoformat() for date in dates)

    def _build_fold(
        self,
        *,
        ordinal: int,
        train_rows: pd.DataFrame,
        validation_rows: pd.DataFrame,
        train_dates: tuple[pd.Timestamp, ...],
        embargo_dates: tuple[pd.Timestamp, ...],
        validation_dates: tuple[pd.Timestamp, ...],
    ) -> WalkForwardFold:
        return WalkForwardFold(
            fold_id=f"fold_{ordinal:04d}",
            train_indices=tuple(train_rows["_sample_index"]),
            validation_indices=tuple(validation_rows["_sample_index"]),
            train_dates=self._iso_dates(train_dates),
            embargo_dates=self._iso_dates(embargo_dates),
            validation_dates=self._iso_dates(validation_dates),
            train_window=self._window(train_rows),
            validation_window=self._window(validation_rows),
        )

    @staticmethod
    def _window(rows: pd.DataFrame) -> SampleWindow:
        return SampleWindow(
            date_start=rows["_split_date"].min().date().isoformat(),
            date_end=rows["_split_date"].max().date().isoformat(),
            feature_time_start=rows["_feature_time"].min().isoformat(),
            feature_time_end=rows["_feature_time"].max().isoformat(),
            target_end_start=rows["_target_end"].min().isoformat(),
            target_end_end=rows["_target_end"].max().isoformat(),
        )
