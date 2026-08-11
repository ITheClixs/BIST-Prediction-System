"""Simulation-based calibration of the evaluation stack.

The rest of the research package measures a market.  This package measures the
instrument: it generates panels whose truth is fixed by construction, runs the
same estimators the empirical chapter runs, and reports how often they are
wrong.  Nothing here touches market data, so the results hold for any study
that uses the same tests on a panel with the same dependence structure.
"""

from bist_predict.research.simulation.calibration import (
    NestedNullCell,
    RejectionRate,
    SizePowerCell,
    asymptotic_pooled_size,
    loss_differentials,
    minimum_detectable_effect,
    nested_null_cell,
    rejection_rate,
    size_power_cell,
)
from bist_predict.research.simulation.panels import (
    ForecastMoments,
    PanelDesign,
    SimulatedPanel,
    attainable_r_squared,
    forecast_family,
    forecast_moments,
    noise_forecast_family,
    population_loss_differential,
    simulate_panel,
    standardised_innovations,
)
from bist_predict.research.simulation.search_calibration import (
    FamilyWiseCell,
    SearchThresholdCell,
    family_wise_cell,
    search_threshold_cell,
)
from bist_predict.research.simulation.study import (
    StudyConfiguration,
    run_study,
    write_study,
)

__all__ = [
    "FamilyWiseCell",
    "ForecastMoments",
    "NestedNullCell",
    "PanelDesign",
    "RejectionRate",
    "SearchThresholdCell",
    "SimulatedPanel",
    "SizePowerCell",
    "StudyConfiguration",
    "asymptotic_pooled_size",
    "attainable_r_squared",
    "family_wise_cell",
    "forecast_family",
    "forecast_moments",
    "loss_differentials",
    "minimum_detectable_effect",
    "nested_null_cell",
    "noise_forecast_family",
    "population_loss_differential",
    "rejection_rate",
    "run_study",
    "search_threshold_cell",
    "simulate_panel",
    "size_power_cell",
    "standardised_innovations",
    "write_study",
]
