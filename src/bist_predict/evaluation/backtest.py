"""Walk-forward backtesting engine with realistic costs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestFold:
    """A single walk-forward fold with train/validation boundaries."""

    train_start: int
    train_end: int
    val_start: int
    val_end: int


class WalkForwardBacktest:
    """Walk-forward backtesting -- train on past, validate on future, slide window.

    Rules:
    - No future leakage (train always before validation)
    - Realistic costs (commission + slippage per trade)
    - Signal delay (prediction at close, trade at next-day open)
    """

    def __init__(
        self,
        train_window: int = 252,
        val_window: int = 63,
        step_size: int = 21,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self._train_window = train_window
        self._val_window = val_window
        self._step_size = step_size
        self._commission = commission
        self._slippage = slippage

    def generate_folds(self, n_dates: int) -> list[tuple[int, int, int, int]]:
        """Generate walk-forward fold indices.

        Returns list of (train_start, train_end, val_start, val_end) tuples.
        """
        folds = []
        start = 0
        while start + self._train_window + self._val_window <= n_dates:
            train_start = start
            train_end = start + self._train_window
            val_start = train_end
            val_end = val_start + self._val_window

            folds.append((train_start, train_end, val_start, val_end))
            start += self._step_size

        return folds

    def apply_costs(self, gross_return: float) -> float:
        """Apply commission and slippage to a gross return.

        Costs are applied on both entry and exit.
        """
        total_cost = 2 * self._commission + 2 * self._slippage
        return gross_return - total_cost
