"""Configuration invariants for the accepted research benchmark."""

from pathlib import Path

from bist_predict.config import Config, load_config


EXPECTED_BASELINES = (
    "zero_return",
    "majority_direction",
    "previous_return",
    "market_direction",
    "rolling_mean",
    "logistic",
    "ridge",
)


def test_default_research_scope_is_truthful_and_baseline_only() -> None:
    config = Config()

    assert config.research.experiment_scope == "fixed_bist_large_cap_prototype"
    assert config.research.accepted_models == EXPECTED_BASELINES
    assert config.research.enabled_feature_families == (
        "stationary_price",
        "cross_sectional",
        "temporal",
    )
    assert set(config.research.experimental_feature_families).isdisjoint(
        config.research.enabled_feature_families
    )


def test_research_scope_and_enabled_families_load_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[research]
experiment_scope = "fixed_bist_large_cap_prototype"
enabled_feature_families = ["stationary_price", "temporal"]
accepted_models = ["zero_return", "ridge"]
""".strip()
    )

    config = load_config(config_path)

    assert config.research.experiment_scope == "fixed_bist_large_cap_prototype"
    assert config.research.enabled_feature_families == ("stationary_price", "temporal")
    assert config.research.accepted_models == ("zero_return", "ridge")
