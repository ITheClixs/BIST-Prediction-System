"""Render the inferential and sensitivity blocks from immutable run metrics.

Every number these functions emit is read out of ``metrics.json``. Nothing here
is typed by hand, so a rerun that moves a number moves the document with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "render_inference_block",
    "render_sensitivity_block",
    "inference_headlines",
]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics field must be an object: {field}")
    return value


def _number(mapping: Mapping[str, Any], key: str, field: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics field must be numeric: {field}.{key}")
    return float(value)


_VERDICT_LABELS = {
    "candidate_better": "beats the null",
    "benchmark_better": "loses to the null",
    "indistinguishable": "indistinguishable",
}


def render_inference_block(metrics: Mapping[str, Any]) -> str:
    """Render dependence, equal-accuracy, data-snooping and Sharpe evidence."""
    inference = _mapping(metrics.get("inference"), "metrics.inference")
    dependence = _mapping(inference.get("cross_sectional_dependence"), "dependence")
    accuracy = _mapping(inference.get("equal_predictive_accuracy"), "equal_predictive_accuracy")
    session = _mapping(accuracy.get("session_aggregated"), "session_aggregated")
    row_level = _mapping(accuracy.get("row_level_for_comparison"), "row_level_for_comparison")
    holm = _mapping(accuracy.get("family_wise_correction"), "family_wise_correction")
    adjusted = _mapping(holm.get("adjusted_p_values"), "adjusted_p_values")
    snooping = _mapping(inference.get("data_snooping"), "data_snooping")
    spa = _mapping(snooping.get("superior_predictive_ability"), "superior_predictive_ability")
    reality = _mapping(snooping.get("reality_check"), "reality_check")
    sharpe = _mapping(inference.get("portfolio_sharpe"), "portfolio_sharpe")

    lines = [
        "### Effective sample size",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Panel units | {int(_number(dependence, 'unit_count', 'dependence'))} |",
        f"| Evaluated sessions | {int(_number(dependence, 'session_count', 'dependence'))} |",
        f"| Out-of-sample rows | {int(_number(dependence, 'row_count', 'dependence'))} |",
        "| Mean within-session correlation | "
        f"{_number(dependence, 'mean_pairwise_correlation', 'dependence'):.4f} |",
        "| Variance inflation factor | "
        f"{_number(dependence, 'variance_inflation_factor', 'dependence'):.4f} |",
        "| Effective independent rows | "
        f"{_number(dependence, 'effective_row_count', 'dependence'):.1f} |",
        "",
        "### Equal predictive accuracy against the zero-return null",
        "",
        "Squared-error loss, Diebold-Mariano with the Harvey-Leybourne-Newbold",
        "correction. The differential is `loss(model) - loss(null)`, so a positive",
        "statistic means the model loses to the null.",
        "",
        "| Model | DM statistic | p | Holm-adjusted p | Verdict | Row-level p |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name in sorted(session):
        test = _mapping(session[name], f"session_aggregated.{name}")
        row_test = _mapping(row_level[name], f"row_level_for_comparison.{name}")
        verdict = str(test.get("verdict"))
        lines.append(
            f"| {name} | {_number(test, 'statistic', name):+.3f} | "
            f"{_number(test, 'p_value', name):.4f} | "
            f"{float(adjusted[name]):.4f} | "
            f"{_VERDICT_LABELS.get(verdict, verdict)} | "
            f"{_number(row_test, 'p_value', name):.4f} |"
        )
    rejected = list(holm.get("rejected") or [])
    survivors = [
        name
        for name in rejected
        if str(_mapping(session[name], name).get("verdict")) == "candidate_better"
    ]
    lines += [
        "",
        f"Holm rejects the null of equal accuracy for {len(rejected)} of "
        f"{int(_number(holm, 'family_size', 'holm'))} models; "
        f"{len(survivors)} of those rejections favour the model.",
        "",
        "### Data snooping across the model family",
        "",
        "| Test | Statistic | p |",
        "|---|---:|---:|",
        f"| White Reality Check | {_number(reality, 'statistic', 'reality_check'):.6f} | "
        f"{_number(reality, 'p_value', 'reality_check'):.4f} |",
        f"| Hansen SPA (lower) | {_number(spa, 'statistic', 'spa'):.4f} | "
        f"{_number(spa, 'p_value_lower', 'spa'):.4f} |",
        f"| Hansen SPA (consistent) | {_number(spa, 'statistic', 'spa'):.4f} | "
        f"{_number(spa, 'p_value_consistent', 'spa'):.4f} |",
        f"| Hansen SPA (upper) | {_number(spa, 'statistic', 'spa'):.4f} | "
        f"{_number(spa, 'p_value_upper', 'spa'):.4f} |",
        "",
        f"Best candidate by mean outperformance: `{snooping.get('best_candidate')}` at "
        f"{_number(snooping, 'best_mean_outperformance', 'snooping'):.3e} squared-error units, "
        f"over {int(_number(snooping, 'replications', 'snooping')):,} stationary-bootstrap "
        f"replications with mean block length "
        f"{_number(snooping, 'block_length', 'snooping'):.2f}.",
        "",
        "### Portfolio Sharpe ratio under search",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Sessions | {int(_number(sharpe, 'observation_count', 'sharpe'))} |",
        f"| Per-session Sharpe | {_number(sharpe, 'per_period_sharpe', 'sharpe'):.4f} |",
        "| Annualised Sharpe (square-root rule) | "
        f"{_number(sharpe, 'annualised_sharpe', 'sharpe'):.4f} |",
        "| Annualised Sharpe (Lo autocorrelation-adjusted) | "
        f"{_number(sharpe, 'autocorrelation_adjusted_annualised_sharpe', 'sharpe'):.4f} |",
        f"| Skewness | {_number(sharpe, 'skewness', 'sharpe'):.4f} |",
        f"| Kurtosis | {_number(sharpe, 'kurtosis', 'sharpe'):.4f} |",
        "| Probabilistic Sharpe ratio, threshold 0 | "
        f"{_number(sharpe, 'probabilistic_sharpe_ratio', 'sharpe'):.4f} |",
        f"| Configurations examined | {int(_number(sharpe, 'trial_count', 'sharpe'))} |",
        "| Search threshold (expected maximum under no skill) | "
        f"{_number(sharpe, 'deflated_sharpe_threshold', 'sharpe'):.4f} |",
        f"| Deflated Sharpe ratio | {_number(sharpe, 'deflated_sharpe_ratio', 'sharpe'):.4f} |",
    ]
    return "\n".join(lines)


def render_sensitivity_block(metrics: Mapping[str, Any]) -> str:
    """Render the configuration grid and the bootstrap block-length sweep."""
    sensitivity = _mapping(
        metrics.get("configuration_sensitivity"), "metrics.configuration_sensitivity"
    )
    sharpe = _mapping(sensitivity.get("per_period_sharpe"), "per_period_sharpe")
    returns = _mapping(sensitivity.get("net_return"), "net_return")
    best = _mapping(sensitivity.get("best_trial"), "best_trial")
    worst = _mapping(sensitivity.get("worst_trial"), "worst_trial")
    reported = _mapping(sensitivity.get("reported_trial"), "reported_trial")
    trials = sensitivity.get("trials")
    if not isinstance(trials, Sequence):
        raise ValueError("metrics field must be a list: configuration_sensitivity.trials")
    null_wins = sum(
        1 for trial in trials if _mapping(trial, "trial")["best_model"] == "zero_return"
    )

    lines = [
        "### Configuration grid",
        "",
        "| Configuration | Net return | Per-session Sharpe | Sessions | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, trial in (("Best", best), ("Reported", reported), ("Worst", worst)):
        lines.append(
            f"| {label}: `{trial['trial_id']}` | "
            f"{_number(trial, 'net_return', label) * 100:.4f}% | "
            f"{_number(trial, 'per_period_sharpe', label):.4f} | "
            f"{int(_number(trial, 'session_count', label))} | "
            f"{int(_number(trial, 'trade_count', label))} |"
        )
    lines += [
        "",
        "| Grid summary | Value |",
        "|---|---:|",
        f"| Configurations evaluated | {int(_number(sensitivity, 'trial_count', 'grid'))} |",
        f"| Reported rank by Sharpe | "
        f"{int(_number(sensitivity, 'reported_rank_by_sharpe', 'grid'))} |",
        f"| Median per-session Sharpe | {_number(sharpe, 'median', 'sharpe'):.4f} |",
        f"| Per-session Sharpe range | {_number(sharpe, 'minimum', 'sharpe'):.4f} to "
        f"{_number(sharpe, 'maximum', 'sharpe'):.4f} |",
        f"| Configurations with positive net return | "
        f"{_number(returns, 'share_positive', 'returns') * 100:.2f}% |",
        f"| Configurations where the zero-return null had the best out-of-sample "
        f"R-squared | {null_wins} of {len(trials)} |",
        "| Expected maximum Sharpe under no skill | "
        f"{_number(sensitivity, 'expected_maximum_sharpe_under_no_skill', 'grid'):.4f} |",
        "",
        "### Bootstrap block-length sensitivity",
        "",
        "| Mean block length | Annualised return 95% interval |",
        "|---:|---|",
    ]
    blocks = _mapping(
        metrics.get("bootstrap_block_sensitivity"), "metrics.bootstrap_block_sensitivity"
    )
    for key in sorted(blocks, key=int):
        interval = _mapping(
            _mapping(blocks[key], key).get("annualized_return"), f"{key}.annualized_return"
        )
        lines.append(
            f"| {int(key)} | "
            f"{_number(interval, 'lower', key) * 100:+.2f}% to "
            f"{_number(interval, 'upper', key) * 100:+.2f}% |"
        )
    return "\n".join(lines)


def inference_headlines(metrics: Mapping[str, Any]) -> list[str]:
    """Render the inferential conclusions as sentences a reader can check."""
    inference = _mapping(metrics.get("inference"), "metrics.inference")
    dependence = _mapping(inference.get("cross_sectional_dependence"), "dependence")
    accuracy = _mapping(inference.get("equal_predictive_accuracy"), "equal_predictive_accuracy")
    session = _mapping(accuracy.get("session_aggregated"), "session_aggregated")
    holm = _mapping(accuracy.get("family_wise_correction"), "family_wise_correction")
    snooping = _mapping(inference.get("data_snooping"), "data_snooping")
    spa = _mapping(snooping.get("superior_predictive_ability"), "superior_predictive_ability")
    sharpe = _mapping(inference.get("portfolio_sharpe"), "portfolio_sharpe")
    sensitivity = _mapping(
        metrics.get("configuration_sensitivity"), "metrics.configuration_sensitivity"
    )
    trials = sensitivity.get("trials")
    if not isinstance(trials, Sequence):
        raise ValueError("metrics field must be a list: configuration_sensitivity.trials")

    beaten = sorted(
        name
        for name, test in session.items()
        if str(_mapping(test, name).get("verdict")) == "benchmark_better"
        and name in set(holm.get("rejected") or [])
    )
    favouring = [
        name
        for name in (holm.get("rejected") or [])
        if str(_mapping(session[str(name)], str(name)).get("verdict")) == "candidate_better"
    ]
    null_wins = sum(
        1 for trial in trials if _mapping(trial, "trial")["best_model"] == "zero_return"
    )
    returns = _mapping(sensitivity.get("net_return"), "net_return")
    return [
        f"- Same-session rows correlate at "
        f"{_number(dependence, 'mean_pairwise_correlation', 'dependence'):.3f}, so the "
        f"{int(_number(dependence, 'row_count', 'dependence'))} out-of-sample rows carry about "
        f"{_number(dependence, 'effective_row_count', 'dependence'):.0f} independent "
        f"observations, not {int(_number(dependence, 'row_count', 'dependence'))}.",
        f"- No model beats the zero-return null on squared error after Holm correction "
        f"({len(favouring)} of {int(_number(holm, 'family_size', 'holm'))} favour the model); "
        f"{len(beaten)} models are significantly worse than it: "
        + ", ".join(f"`{name}`" for name in beaten)
        + ".",
        f"- Hansen's test of superior predictive ability does not reject the null that no "
        f"candidate beats the zero-return benchmark "
        f"(p = {_number(spa, 'p_value_consistent', 'spa'):.4f}, consistent recentring).",
        f"- The strategy's deflated Sharpe ratio is "
        f"{_number(sharpe, 'deflated_sharpe_ratio', 'sharpe'):.4f} against a search threshold "
        f"of {_number(sharpe, 'deflated_sharpe_threshold', 'sharpe'):.4f} per session across "
        f"{int(_number(sharpe, 'trial_count', 'sharpe'))} configurations.",
        f"- Across the {int(_number(sensitivity, 'trial_count', 'grid'))}-configuration grid, "
        f"{_number(returns, 'share_positive', 'returns') * 100:.2f}% of settings produced a "
        f"positive net return, and the zero-return null had the best out-of-sample R-squared "
        f"in {null_wins} of {len(trials)}.",
    ]
