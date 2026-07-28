"""Statistical inference for the accepted benchmark's out-of-sample evidence."""

from bist_predict.research.inference.dependence import (
    CrossSectionalDependence,
    cross_sectional_dependence,
    effective_sample_size,
    variance_inflation_factor,
)
from bist_predict.research.inference.forecast_tests import (
    LOSS_DIFFERENTIAL_CONVENTION,
    DieboldMarianoResult,
    diebold_mariano,
    squared_error_differential,
)
from bist_predict.research.inference.hac import (
    automatic_bartlett_bandwidth,
    bartlett_long_run_variance,
    mean_standard_error,
    sample_autocovariance,
)
from bist_predict.research.inference.multiplicity import HolmCorrection, holm_step_down
from bist_predict.research.inference.sharpe import (
    SharpeInference,
    annualisation_factor,
    deflated_sharpe_ratio,
    deflated_sharpe_threshold,
    per_period_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_inference,
    sharpe_standard_error,
)
from bist_predict.research.inference.snooping import (
    DataSnoopingResult,
    reality_check_and_spa,
    stationary_block_length,
    stationary_bootstrap_indices,
)

__all__ = [
    "CrossSectionalDependence",
    "DataSnoopingResult",
    "DieboldMarianoResult",
    "HolmCorrection",
    "LOSS_DIFFERENTIAL_CONVENTION",
    "SharpeInference",
    "annualisation_factor",
    "automatic_bartlett_bandwidth",
    "bartlett_long_run_variance",
    "cross_sectional_dependence",
    "deflated_sharpe_ratio",
    "deflated_sharpe_threshold",
    "diebold_mariano",
    "effective_sample_size",
    "holm_step_down",
    "mean_standard_error",
    "per_period_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "reality_check_and_spa",
    "sample_autocovariance",
    "sharpe_inference",
    "sharpe_standard_error",
    "squared_error_differential",
    "stationary_block_length",
    "stationary_bootstrap_indices",
    "variance_inflation_factor",
]
