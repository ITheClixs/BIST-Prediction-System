"""Data-snooping tests for the best of several models against one benchmark.

Selecting the best of ``m`` models and then testing that model alone is not a
valid procedure: the selection already used the data.  White (2000) and Hansen
(2005) test the whole family jointly, using a stationary bootstrap (Politis and
Romano, 1994) that resamples every model on the same index draw so the
cross-model dependence is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "DataSnoopingResult",
    "reality_check_and_spa",
    "stationary_bootstrap_indices",
    "stationary_block_length",
]


@dataclass(frozen=True)
class DataSnoopingResult:
    """Joint tests that no candidate beats the benchmark out of sample."""

    benchmark: str
    candidates: tuple[str, ...]
    observation_count: int
    replications: int
    block_length: float
    seed: int
    best_candidate: str
    best_mean_outperformance: float
    reality_check_statistic: float
    reality_check_p_value: float
    spa_statistic: float
    spa_p_value_lower: float
    spa_p_value_consistent: float
    spa_p_value_upper: float
    per_candidate_mean_outperformance: tuple[tuple[str, float], ...]

    @property
    def verdict(self) -> str:
        """Return whether any candidate survives the joint null at 5%."""
        return (
            "no_candidate_beats_benchmark"
            if self.spa_p_value_consistent > 0.05
            else "at_least_one_candidate_beats_benchmark"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of both joint tests."""
        return {
            "benchmark": self.benchmark,
            "candidates": list(self.candidates),
            "observation_count": self.observation_count,
            "replications": self.replications,
            "block_length": self.block_length,
            "seed": self.seed,
            "best_candidate": self.best_candidate,
            "best_mean_outperformance": self.best_mean_outperformance,
            "reality_check": {
                "statistic": self.reality_check_statistic,
                "p_value": self.reality_check_p_value,
            },
            "superior_predictive_ability": {
                "statistic": self.spa_statistic,
                "p_value_lower": self.spa_p_value_lower,
                "p_value_consistent": self.spa_p_value_consistent,
                "p_value_upper": self.spa_p_value_upper,
            },
            "mean_outperformance": {
                name: value for name, value in self.per_candidate_mean_outperformance
            },
            "convention": "outperformance = loss(benchmark) - loss(candidate); positive favours the candidate",
            "verdict": self.verdict,
        }


def stationary_block_length(sample: np.ndarray) -> float:
    """Return the Politis and White (2009) stationary-bootstrap block length.

    The selection is delegated to ``arch``.  The largest column-wise choice is
    used so the block is long enough for the most persistent series in the
    family.
    """
    import warnings

    from arch.bootstrap import optimal_block_length

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lengths = optimal_block_length(np.asarray(sample, dtype=np.float64))
    selected = np.asarray(lengths["stationary"].to_numpy(), dtype=np.float64)
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return 1.0
    return float(max(1.0, float(np.max(finite))))


def stationary_bootstrap_indices(
    count: int,
    *,
    block_length: float,
    replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return ``replications x count`` stationary-bootstrap index draws.

    Politis and Romano (1994) start each replication at a uniform position and
    then, at every step, either continue the current block with probability
    :math:`1 - q` or restart uniformly with probability :math:`q = 1/L`.  Block
    lengths are therefore geometric with mean ``L`` and the resampled series is
    stationary, unlike a fixed-length block bootstrap.
    """
    if count < 2:
        raise ValueError("stationary bootstrap requires at least two observations")
    if replications < 1:
        raise ValueError("replications must be positive")
    if not np.isfinite(block_length) or block_length < 1.0:
        raise ValueError("block_length must be finite and at least one")
    restart_probability = 1.0 / block_length
    indices = np.empty((replications, count), dtype=np.int64)
    indices[:, 0] = rng.integers(0, count, size=replications)
    restarts = rng.random((replications, count - 1)) < restart_probability
    fresh = rng.integers(0, count, size=(replications, count - 1))
    for step in range(1, count):
        continued = (indices[:, step - 1] + 1) % count
        indices[:, step] = np.where(restarts[:, step - 1], fresh[:, step - 1], continued)
    return indices


def reality_check_and_spa(
    losses: pd.DataFrame,
    *,
    benchmark: str,
    replications: int = 10_000,
    seed: int = 42,
    block_length: float | None = None,
    studentize: bool = True,
) -> DataSnoopingResult:
    r"""Run White's Reality Check and Hansen's SPA on a loss panel.

    ``losses`` holds one column per model and one row per evaluation session.
    Define the relative performance of candidate ``k`` as
    :math:`Z_{k,t} = L_{0,t} - L_{k,t}`, so a positive mean favours the
    candidate.  The Reality Check statistic is
    :math:`V_n = \max_k \sqrt n \bar Z_k`, and the SPA statistic studentizes it,
    :math:`T_n = \max[0, \max_k \sqrt n \bar Z_k / \hat\omega_k]`, where
    :math:`\hat\omega_k` is the stationary-bootstrap standard deviation of
    :math:`\sqrt n \bar Z_k`.

    Hansen's three recentrings bracket the p-value.  ``lower`` recentres at
    :math:`\max(\bar Z_k, 0)` and is liberal, ``upper`` recentres at
    :math:`\bar Z_k` and is conservative, and ``consistent`` keeps only models
    that are not clearly inferior, using the threshold
    :math:`\sqrt n \bar Z_k / \hat\omega_k \ge -\sqrt{2 \log\log n}`.
    """
    if benchmark not in losses.columns:
        raise ValueError(f"benchmark column is missing: {benchmark}")
    candidates = tuple(str(name) for name in losses.columns if str(name) != benchmark)
    if not candidates:
        raise ValueError("at least one candidate model is required")
    values = losses.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("loss panel must be finite")
    count = values.shape[0]
    if count < 8:
        raise ValueError("data-snooping tests require at least eight evaluation sessions")
    if replications < 100:
        raise ValueError("data-snooping tests require at least one hundred replications")

    relative = losses[[benchmark]].to_numpy(dtype=np.float64) - losses[list(candidates)].to_numpy(
        np.float64
    )
    observed_means = relative.mean(axis=0)
    chosen_block = (
        stationary_block_length(relative) if block_length is None else float(block_length)
    )

    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(
        count, block_length=chosen_block, replications=replications, rng=rng
    )
    resampled_means = relative[indices].mean(axis=1)
    root = np.sqrt(count)

    centred = root * (resampled_means - observed_means)
    omega = np.sqrt(np.mean(np.square(centred), axis=0))
    scale_floor = 1e-12 * max(float(np.max(np.abs(relative))), 1e-12)
    omega = np.where(omega <= scale_floor, np.nan, omega)
    if np.isnan(omega).all():
        raise ValueError("every candidate is numerically identical to the benchmark")

    reality_check_statistic = float(np.max(root * observed_means))
    reality_check_p_value = float(np.mean(np.max(centred, axis=1) >= reality_check_statistic))

    studentized = np.where(np.isnan(omega), -np.inf, root * observed_means / omega)
    threshold = -np.sqrt(2.0 * np.log(np.log(count)))
    recentrings = {
        "lower": np.maximum(observed_means, 0.0),
        "consistent": np.where(studentized >= threshold, observed_means, 0.0),
        "upper": observed_means,
    }
    # Studentizing is what separates Hansen's SPA from White's Reality Check:
    # without it, a candidate with a large loss variance dominates the maximum
    # whatever its mean. ``studentize=False`` reproduces the unstudentized
    # comparison used by ``arch.bootstrap.SPA`` and exists so the two can be
    # checked against each other.
    scale = np.where(np.isnan(omega), np.inf, omega) if studentize else np.ones_like(omega)
    observed_maximum = float(np.max(np.where(np.isnan(omega), -np.inf, observed_means / scale)))
    spa_statistic = float(max(0.0, float(np.max(studentized))))
    p_values: dict[str, float] = {}
    for name, recentring in recentrings.items():
        adjusted = (resampled_means - recentring) / scale
        # Compare untruncated maxima. Hansen defines the statistic with a floor
        # at zero, but applying that floor to both the observed value and every
        # bootstrap draw collapses the comparison onto the atom at zero as soon
        # as no candidate has a positive mean, and returns an arbitrary p-value
        # in exactly the case where the evidence is weakest.
        p_values[name] = float(np.mean(np.max(adjusted, axis=1) > observed_maximum))

    best_position = int(np.argmax(observed_means))
    return DataSnoopingResult(
        benchmark=str(benchmark),
        candidates=candidates,
        observation_count=count,
        replications=replications,
        block_length=chosen_block,
        seed=seed,
        best_candidate=candidates[best_position],
        best_mean_outperformance=float(observed_means[best_position]),
        reality_check_statistic=reality_check_statistic,
        reality_check_p_value=reality_check_p_value,
        spa_statistic=spa_statistic,
        spa_p_value_lower=p_values["lower"],
        spa_p_value_consistent=p_values["consistent"],
        spa_p_value_upper=p_values["upper"],
        per_candidate_mean_outperformance=tuple(
            (name, float(observed_means[position])) for position, name in enumerate(candidates)
        ),
    )
