"""Prediction quality and trading quality metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_prediction_metrics(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    y_pct_true: NDArray[np.float64],
    y_pct_pred: NDArray[np.float64],
) -> dict[str, float]:
    """Compute prediction quality metrics.

    Returns dict with: accuracy, precision, recall, f1, auc_roc, brier_score, mae.
    """
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "mae": float(np.mean(np.abs(y_pct_true - y_pct_pred))),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = math.nan

    return metrics


def compute_trading_metrics(
    daily_returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute trading quality metrics from a series of daily returns."""
    if len(daily_returns) == 0:
        return {
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "avg_win_loss_ratio": 0.0,
            "calmar_ratio": 0.0, "total_return": 0.0,
        }

    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf

    # Sharpe ratio (annualized)
    std = np.std(excess, ddof=1) if len(excess) > 1 else 1e-10
    sharpe = float(np.mean(excess) / std * math.sqrt(252)) if std > 0 else 0.0

    # Sortino ratio (downside deviation only)
    downside = excess[excess < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = float(np.mean(excess) / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Win rate
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = float(len(wins) / len(daily_returns))

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Average win/loss ratio
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 1e-10
    avg_wl_ratio = avg_win / avg_loss

    # Total return
    total_return = float(np.prod(1 + daily_returns) - 1)

    # Calmar ratio
    calmar = float(total_return / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_loss_ratio": avg_wl_ratio,
        "calmar_ratio": calmar,
        "total_return": total_return,
    }
