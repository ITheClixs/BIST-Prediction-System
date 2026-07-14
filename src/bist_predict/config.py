"""Configuration management — loads and validates config.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "bist.db"


@dataclass(frozen=True)
class DataConfig:
    tcmb_api_key: str = ""
    fetch_retries: int = 3
    rate_limit_delay: float = 1.0


@dataclass(frozen=True)
class SignalsConfig:
    min_confidence: float = 0.70
    lookback_days: int = 30


@dataclass(frozen=True)
class ModelsConfig:
    retrain_interval: str = "monthly"
    ensemble_weights: str = "learned"
    active_models: str = "xgboost,lightgbm"
    include_neural: bool = False
    seq_len: int = 30
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class QuantConfig:
    hmm_states: int = 3
    kelly_fraction: float = 0.25
    hurst_window: int = 252


@dataclass(frozen=True)
class BacktestConfig:
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class ResearchConfig:
    """Methodological scope of the accepted benchmark.

    Advanced feature and model modules remain importable for experiments, but
    they are deliberately excluded from the accepted default until a saved
    chronological experiment demonstrates incremental value.
    """

    experiment_scope: str = "fixed_bist_large_cap_prototype"
    enabled_feature_families: tuple[str, ...] = (
        "stationary_price",
        "cross_sectional",
        "temporal",
    )
    experimental_feature_families: tuple[str, ...] = (
        "sentiment",
        "macro",
        "hmm_regime",
        "wavelet",
        "cointegration",
        "kelly_sizing",
    )
    accepted_models: tuple[str, ...] = (
        "zero_return",
        "majority_direction",
        "previous_return",
        "market_direction",
        "rolling_mean",
        "logistic",
        "ridge",
    )
    experimental_models: tuple[str, ...] = (
        "xgboost",
        "lightgbm",
        "lstm",
        "transformer",
        "stacking",
        "calibration",
    )


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    db_path: Path = DEFAULT_DB_PATH


def _research_config(raw: dict[str, Any]) -> ResearchConfig:
    """Build an immutable research config from TOML lists or defaults."""
    defaults = ResearchConfig()

    def values(name: str) -> tuple[str, ...]:
        configured = raw.get(name, getattr(defaults, name))
        if isinstance(configured, str):
            configured = configured.split(",")
        return tuple(str(value).strip() for value in configured if str(value).strip())

    return ResearchConfig(
        experiment_scope=str(raw.get("experiment_scope", defaults.experiment_scope)),
        enabled_feature_families=values("enabled_feature_families"),
        experimental_feature_families=values("experimental_feature_families"),
        accepted_models=values("accepted_models"),
        experimental_models=values("experimental_models"),
    )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from a TOML file. Returns defaults if file missing."""
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return Config(
        data=DataConfig(**raw.get("data", {})),
        signals=SignalsConfig(**raw.get("signals", {})),
        models=ModelsConfig(**raw.get("models", {})),
        quant=QuantConfig(**raw.get("quant", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        research=_research_config(raw.get("research", {})),
    )
