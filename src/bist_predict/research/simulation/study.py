"""The complete calibration study, and the artifact it writes.

Six experiments are run, each against a null that holds by construction:

``dependence``
    How the size of the equal-predictive-accuracy test moves with the number of
    panel units and their correlation, against the closed-form prediction.
``robustness``
    Whether the session-aggregated test stays calibrated when the innovations
    are heavy-tailed, the volatility clusters, the level switches regime, or the
    record is short.
``power``
    The smallest population out-of-sample :math:`R^2` the design separates from
    zero at 80% power, for the record length used and for longer ones.
``nested``
    What the two tests do when the zero benchmark is the correct model and the
    fitted forecast is pure estimation noise: the size of the Clark--West test,
    and the rate at which a squared-error comparison calls the fitted model
    significantly worse for no reason but having been fitted.
``family``
    The family-wise error rate of Holm, the Reality Check and both variants of
    the SPA test when every member of the family is skill-free.
``search``
    What fraction of skill-free configuration grids clear the False Strategy
    Theorem's expected maximum, and what fraction clear a synchronised bootstrap
    quantile of the same grid.
``anchor``
    The single design point matched to the measured panel, reported separately
    because it is the one the empirical chapter is entitled to cite.

Every cell is seeded from one root seed by a deterministic offset, so the study
replays exactly and any single cell can be re-run in isolation.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from bist_predict.research.simulation.calibration import (
    SizePowerCell,
    asymptotic_pooled_size,
    minimum_detectable_effect,
    nested_null_cell,
    size_power_cell,
)
from bist_predict.research.simulation.panels import PanelDesign
from bist_predict.research.simulation.search_calibration import (
    family_wise_cell,
    search_threshold_cell,
)

__all__ = [
    "StudyConfiguration",
    "run_study",
    "write_study",
]

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class StudyConfiguration:
    """Sizes and seeds for the whole study, in one place.

    The defaults are the values the manuscript reports.  ``quick`` shrinks every
    replication count by a fixed factor so the test suite can exercise the same
    code path in seconds without a second, drifting definition of the study.
    """

    seed: int = 20260806
    size_replications: int = 10_000
    power_replications: int = 4000
    nested_replications: int = 10_000
    family_replications: int = 2000
    search_replications: int = 1000
    bootstrap_replications: int = 999
    alpha: float = 0.05
    anchor_correlation: float = 0.5697
    anchor_volatility: float = 0.019
    anchor_predictable_share: float = 0.136
    anchor_forecast_variance_ratio: float = 0.272
    anchor_forecast_correlation: float = 0.947
    power_predictable_share: float = 0.35
    family_size: int = 6
    grid_trial_count: int = 72

    def quick(self, factor: int = 40) -> StudyConfiguration:
        """Return the same study with every replication count divided down."""
        if factor < 1:
            raise ValueError("factor must be positive")
        return StudyConfiguration(
            seed=self.seed,
            size_replications=max(50, self.size_replications // factor),
            power_replications=max(50, self.power_replications // factor),
            nested_replications=max(50, self.nested_replications // factor),
            family_replications=max(25, self.family_replications // factor),
            search_replications=max(25, self.search_replications // factor),
            bootstrap_replications=max(199, self.bootstrap_replications // 2),
            alpha=self.alpha,
            anchor_correlation=self.anchor_correlation,
            anchor_volatility=self.anchor_volatility,
            anchor_predictable_share=self.anchor_predictable_share,
            anchor_forecast_variance_ratio=self.anchor_forecast_variance_ratio,
            anchor_forecast_correlation=self.anchor_forecast_correlation,
            power_predictable_share=self.power_predictable_share,
            family_size=self.family_size,
            grid_trial_count=self.grid_trial_count,
        )

    def anchor_design(self) -> PanelDesign:
        """Return the design point matched to the measured panel.

        Every parameter here is a measurement, not a preference: the four names,
        the 120 evaluated sessions, the target's cross-sectional correlation and
        volatility, and the variance and cross-sectional correlation of the
        fitted forecast all come from the committed run.  The predictable share
        is the one quantity that cannot be measured, so it is set to make the
        simulated forecast's variance ratio reproduce the measured one at a
        population accuracy of zero.
        """
        return PanelDesign(
            unit_count=4,
            session_count=120,
            target_correlation=self.anchor_correlation,
            target_volatility=self.anchor_volatility,
            predictable_share=self.anchor_predictable_share,
            forecast_variance_ratio=self.anchor_forecast_variance_ratio,
            forecast_correlation=self.anchor_forecast_correlation,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe record of the configuration."""
        return {
            "seed": self.seed,
            "size_replications": self.size_replications,
            "power_replications": self.power_replications,
            "nested_replications": self.nested_replications,
            "family_replications": self.family_replications,
            "search_replications": self.search_replications,
            "bootstrap_replications": self.bootstrap_replications,
            "alpha": self.alpha,
            "anchor_correlation": self.anchor_correlation,
            "anchor_volatility": self.anchor_volatility,
            "anchor_predictable_share": self.anchor_predictable_share,
            "anchor_forecast_variance_ratio": self.anchor_forecast_variance_ratio,
            "anchor_forecast_correlation": self.anchor_forecast_correlation,
            "power_predictable_share": self.power_predictable_share,
            "family_size": self.family_size,
            "grid_trial_count": self.grid_trial_count,
        }


UNIT_COUNTS = (2, 4, 10, 30, 100)
CORRELATIONS = (0.0, 0.2, 0.4, 0.5697, 0.8)
POWER_SESSION_COUNTS = (120, 250, 500, 1000, 2000)
# The effect axis is the population covariance ratio 2 cov(r, yhat) / var(r),
# because it is the one quantity both tests respond to. The population
# out-of-sample R-squared of the same forecast is this number minus the
# forecast's own variance ratio, so a covariance below that ratio is a forecast
# that carries information and still loses on squared error.
POWER_COVARIANCES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.272, 0.35, 0.42, 0.50)
TRIAL_CORRELATIONS = (0.0, 0.5, 0.9, 0.98)


def _seeds(root: int) -> Iterator[int]:
    """Yield an unbounded stream of distinct, reproducible cell seeds."""
    stream = np.random.default_rng(root)
    while True:
        yield int(stream.integers(0, 2**31 - 1))


def _dependence_experiment(
    configuration: StudyConfiguration, seeds: Iterator[int]
) -> list[dict[str, object]]:
    anchor = configuration.anchor_design()
    cells: list[dict[str, object]] = []
    for units in UNIT_COUNTS:
        cell = size_power_cell(
            anchor.with_(unit_count=units),
            population_r_squared=0.0,
            replications=configuration.size_replications,
            seed=next(seeds),
            alpha=configuration.alpha,
        )
        cells.append({"varied": "unit_count", **cell.to_dict()})
    for correlation in CORRELATIONS:
        for units in (4, 30):
            # Sweeping the target's correlation while holding the forecast's at
            # the measured 0.947 would ask a nearly common forecast to track a
            # nearly idiosyncratic panel, which is not a harder inference
            # problem but an unattainable one. The forecast inherits the panel's
            # factor structure instead, so the cells differ only in dependence.
            cell = size_power_cell(
                anchor.with_(
                    unit_count=units,
                    target_correlation=correlation,
                    forecast_correlation=correlation,
                ),
                population_r_squared=0.0,
                replications=configuration.size_replications,
                seed=next(seeds),
                alpha=configuration.alpha,
            )
            cells.append({"varied": "target_correlation", **cell.to_dict()})
    return cells


def _robustness_experiment(
    configuration: StudyConfiguration, seeds: Iterator[int]
) -> list[dict[str, object]]:
    anchor = configuration.anchor_design()
    variants = [
        ("gaussian_constant", anchor),
        ("student_t5", anchor.with_(innovation="student_t", degrees_of_freedom=5.0)),
        ("garch", anchor.with_(volatility="garch")),
        ("regime_switching", anchor.with_(volatility="regime")),
        (
            "heavy_tailed_garch",
            anchor.with_(innovation="student_t", degrees_of_freedom=5.0, volatility="garch"),
        ),
        ("sixty_sessions", anchor.with_(session_count=60)),
        ("five_hundred_sessions", anchor.with_(session_count=500)),
        ("idiosyncratic_forecast", anchor.with_(forecast_correlation=0.4)),
    ]
    cells: list[dict[str, object]] = []
    for label, design in variants:
        cell = size_power_cell(
            design,
            population_r_squared=0.0,
            replications=configuration.size_replications,
            seed=next(seeds),
            alpha=configuration.alpha,
        )
        cells.append({"variant": label, **cell.to_dict()})
    return cells


def _power_experiment(configuration: StudyConfiguration, seeds: Iterator[int]) -> dict[str, object]:
    anchor = configuration.anchor_design().with_(
        predictable_share=configuration.power_predictable_share
    )
    curves: list[dict[str, object]] = []
    detectable: list[dict[str, object]] = []
    ratio = anchor.forecast_variance_ratio
    for sessions in POWER_SESSION_COUNTS:
        design = anchor.with_(session_count=sessions)
        cells: list[SizePowerCell] = []
        for covariance in POWER_COVARIANCES:
            cell = size_power_cell(
                design,
                population_r_squared=covariance - ratio,
                replications=configuration.power_replications,
                seed=next(seeds),
                alpha=configuration.alpha,
            )
            cells.append(cell)
            curves.append({"session_count": sessions, **cell.to_dict()})
        detectable.append(
            {
                "session_count": sessions,
                "diebold_mariano_r_squared": minimum_detectable_effect(cells),
                "clark_west_covariance_ratio": minimum_detectable_effect(
                    cells, test="clark_west", scale="covariance_ratio"
                ),
                "power": 0.80,
            }
        )
    return {
        "predictable_share": configuration.power_predictable_share,
        "forecast_variance_ratio": ratio,
        "covariance_grid": list(POWER_COVARIANCES),
        "curves": curves,
        "minimum_detectable": detectable,
    }


def _nested_experiment(
    configuration: StudyConfiguration, seeds: Iterator[int]
) -> list[dict[str, object]]:
    anchor = configuration.anchor_design()
    variants = [
        ("anchor", anchor, configuration.anchor_forecast_variance_ratio),
        ("quiet_forecast", anchor, 0.05),
        ("loud_forecast", anchor, 0.60),
        (
            "independent_rows",
            anchor.with_(target_correlation=0.0, forecast_correlation=0.0),
            0.272,
        ),
        ("thirty_names", anchor.with_(unit_count=30), 0.272),
        ("five_hundred_sessions", anchor.with_(session_count=500), 0.272),
    ]
    cells: list[dict[str, object]] = []
    for label, design, ratio in variants:
        cell = nested_null_cell(
            design,
            variance_ratio=ratio,
            replications=configuration.nested_replications,
            seed=next(seeds),
            alpha=configuration.alpha,
        )
        cells.append({"variant": label, **cell.to_dict()})
    return cells


def _family_experiment(
    configuration: StudyConfiguration, seeds: Iterator[int]
) -> list[dict[str, object]]:
    anchor = configuration.anchor_design()
    designs = [
        ("anchor", anchor),
        ("independent_rows", anchor.with_(target_correlation=0.0, forecast_correlation=0.0)),
        ("thirty_names", anchor.with_(unit_count=30)),
        (
            "thirty_names_independent",
            anchor.with_(unit_count=30, target_correlation=0.0, forecast_correlation=0.0),
        ),
    ]
    cells: list[dict[str, object]] = []
    for label, design in designs:
        cell = family_wise_cell(
            design,
            family_size=configuration.family_size,
            replications=configuration.family_replications,
            seed=next(seeds),
            alpha=configuration.alpha,
            bootstrap_replications=configuration.bootstrap_replications,
        )
        cells.append({"variant": label, **cell.to_dict()})
    return cells


def _search_experiment(
    configuration: StudyConfiguration, seeds: Iterator[int]
) -> list[dict[str, object]]:
    return [
        search_threshold_cell(
            trial_count=configuration.grid_trial_count,
            session_count=120,
            trial_correlation=correlation,
            replications=configuration.search_replications,
            seed=next(seeds),
            alpha=configuration.alpha,
            bootstrap_replications=configuration.bootstrap_replications,
        ).to_dict()
        for correlation in TRIAL_CORRELATIONS
    ]


def run_study(configuration: StudyConfiguration | None = None) -> dict[str, object]:
    """Run every experiment and return the assembled study record."""
    settings = configuration or StudyConfiguration()
    seeds = _seeds(settings.seed)
    started = time.perf_counter()
    dependence = _dependence_experiment(settings, seeds)
    robustness = _robustness_experiment(settings, seeds)
    power = _power_experiment(settings, seeds)
    nested = _nested_experiment(settings, seeds)
    family = _family_experiment(settings, seeds)
    search = _search_experiment(settings, seeds)
    anchor = settings.anchor_design()
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": settings.to_dict(),
        "anchor_design": anchor.to_dict(),
        "closed_form_anchor_size": asymptotic_pooled_size(
            anchor.unit_count, anchor.target_correlation, alpha=settings.alpha
        ),
        "experiments": {
            "dependence": dependence,
            "robustness": robustness,
            "power": power,
            "nested": nested,
            "family": family,
            "search": search,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def write_study(destination: Path, study: dict[str, object]) -> Path:
    """Write the study to ``destination`` and return the path.

    ``elapsed_seconds`` is excluded from the file's content hash: the study is
    deterministic in its numbers but not in how long a machine takes to produce
    them, and a hash that changes with load would defeat its own purpose.
    """
    import hashlib

    payload = {key: value for key, value in study.items() if key != "elapsed_seconds"}
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    recorded = dict(study)
    recorded["content_hash"] = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
