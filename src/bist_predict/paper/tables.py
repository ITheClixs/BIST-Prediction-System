"""Generate the manuscript's result tables from the immutable run artifacts.

The manuscript never contains a hand-typed number. Each table below is emitted
from ``metrics.json`` and written into ``paper/generated``; ``make paper``
regenerates them before typesetting, so the document cannot state a figure the
pipeline does not produce.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "TABLE_BUILDERS",
    "escape_latex",
    "render_all_tables",
    "split_row",
]

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_UNESCAPED_AMPERSAND = re.compile(r"(?<!\\)&")


def escape_latex(value: object) -> str:
    """Escape a value for use inside a LaTeX table cell."""
    text = str(value)
    return "".join(_ESCAPES.get(character, character) for character in text)


def split_row(row: str) -> list[str]:
    r"""Split a LaTeX table row on its real column separators.

    ``\&`` is a literal ampersand inside a cell, not a separator. A splitter
    that ignores the escape reports an extra column for any row containing one,
    which is a defect in the checker rather than in the table.
    """
    return [cell.strip() for cell in _UNESCAPED_AMPERSAND.split(row.rstrip("\\ \n"))]


def _percent(value: float, digits: int = 2) -> str:
    return rf"{value * 100:.{digits}f}\%"


def _table(caption: str, label: str, spec: str, header: str, rows: Sequence[str]) -> str:
    body = "\n".join(f"    {row} \\\\" for row in rows)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{tab:{label}}}",
            rf"  \begin{{tabular}}{{{spec}}}",
            r"    \toprule",
            f"    {header} \\\\",
            r"    \midrule",
            body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def prediction_table(metrics: Mapping[str, Any]) -> str:
    """Out-of-sample accuracy for every evaluated forecaster."""
    prediction = metrics["prediction"]
    rows = []
    for name in sorted(prediction, key=lambda key: -float(prediction[key]["zero_mean_r_squared"])):
        model = prediction[name]
        spearman = model.get("spearman_ic")
        rows.append(
            " & ".join(
                (
                    rf"\texttt{{{escape_latex(name)}}}",
                    str(int(model["sample_count"])),
                    _percent(float(model["mae"]), 3),
                    _percent(float(model["rmse"]), 3),
                    f"{float(model['zero_mean_r_squared']):.4f}",
                    "n/a" if spearman is None else f"{float(spearman):.4f}",
                    _percent(float(model["directional_accuracy"])),
                )
            )
        )
    return _table(
        "Out-of-sample accuracy by model, recomputed from the saved prediction rows. "
        "The zero-mean $R^2$ compares against a zero forecast, so the null sits at "
        "exactly zero by construction.",
        "prediction",
        "lrrrrrr",
        "Model & $n$ & MAE & RMSE & $R^2_0$ & Spearman IC & Dir.\\ acc.",
        rows,
    )


def accuracy_test_table(metrics: Mapping[str, Any]) -> str:
    """Diebold-Mariano tests against the null, with and without correction."""
    accuracy = metrics["inference"]["equal_predictive_accuracy"]
    session = accuracy["session_aggregated"]
    row_level = accuracy["row_level_for_comparison"]
    adjusted = accuracy["family_wise_correction"]["adjusted_p_values"]
    labels = {
        "candidate_better": "beats null",
        "benchmark_better": "loses to null",
        "indistinguishable": "indistinguishable",
    }
    rows = []
    for name in sorted(session, key=lambda key: -float(session[key]["statistic"])):
        test = session[name]
        rows.append(
            " & ".join(
                (
                    rf"\texttt{{{escape_latex(name)}}}",
                    f"{float(test['statistic']):+.3f}",
                    f"{float(test['p_value']):.4f}",
                    f"{float(adjusted[name]):.4f}",
                    labels[str(test["verdict"])],
                    f"{float(row_level[name]['p_value']):.4f}",
                )
            )
        )
    return _table(
        "Equal predictive accuracy against the zero-return null under squared-error loss. "
        "The differential is $\\mathrm{loss}(\\mathrm{model}) - \\mathrm{loss}(\\mathrm{null})$, "
        "so a positive statistic means the model is worse. The final column repeats each test "
        "treating all panel rows as independent draws.",
        "accuracy",
        "lrrrlr",
        "Model & $DM^{\\ast}$ & $p$ & Holm $p$ & Verdict & Row-level $p$",
        rows,
    )


def nested_test_table(metrics: Mapping[str, Any]) -> str:
    """Clark-West tests of predictive content against the zero benchmark."""
    content = metrics["inference"]["predictive_content"]
    session = content["session_aggregated"]
    adjusted = content["family_wise_correction"]["adjusted_p_values"]
    labels = {
        "predictive_content": "content",
        "no_predictive_content": "none",
    }
    rows = []
    for name in sorted(session, key=lambda key: -float(session[key]["statistic"])):
        test = session[name]
        rows.append(
            " & ".join(
                (
                    rf"\texttt{{{escape_latex(name)}}}",
                    f"{float(test['mean_adjusted_differential']):+.3e}",
                    f"{float(test['statistic']):+.3f}",
                    f"{float(test['p_value']):.4f}",
                    f"{float(adjusted[name]):.4f}",
                    labels[str(test["verdict"])],
                )
            )
        )
    return _table(
        "Clark--West tests of predictive content against the zero-return benchmark. The adjusted "
        "differential is $f_t = 2 r_t \\hat r_t$, whose expectation is zero under the null that "
        "the population forecast is zero and positive whenever the forecast covaries with the "
        "target. The test is one-sided and, unlike the squared-error comparison in "
        "Table~\\ref{tab:accuracy}, is correctly sized for this nested pair.",
        "nested",
        "lrrrrl",
        "Model & $\\bar f$ & $CW$ & $p$ & Holm $p$ & Verdict",
        rows,
    )


def joint_search_table(metrics: Mapping[str, Any]) -> str:
    """The best configuration against a bootstrap of the whole grid."""
    joint = metrics["inference"]["joint_search"]
    rows = [
        f"Configurations resampled jointly & {int(joint['trial_count'])}",
        f"Sessions common to every configuration & {int(joint['session_count'])}",
        "Mean pairwise correlation between configurations & "
        f"{float(joint['mean_pairwise_correlation']):.4f}",
        f"Stationary bootstrap block length & {float(joint['block_length']):.2f}",
        rf"Best configuration & \texttt{{{escape_latex(joint['best_trial'])}}}",
        f"Its per-session Sharpe ratio & {float(joint['best_per_period_sharpe']):.4f}",
        f"Expected maximum under the joint null & {float(joint['null_expected_maximum']):.4f}",
        f"95th percentile under the joint null & {float(joint['null_quantile_95']):.4f}",
        f"Exact bootstrap p-value & {float(joint['p_value']):.4f}",
        f"Independent-equivalent trials & {float(joint['independent_equivalent_trials']):.2f}",
    ]
    return _table(
        "The best of the configuration grid against a stationary bootstrap that recentres every "
        "configuration to zero expected return and resamples all of them on a single index draw. "
        "Unlike the deflated Sharpe ratio this assumes nothing about how the trials disperse, and "
        "it returns a critical value and a p-value rather than an expectation.",
        "joint_search",
        "lr",
        "Quantity & Value",
        rows,
    )


def snooping_table(metrics: Mapping[str, Any]) -> str:
    """Joint tests over the whole model family."""
    snooping = metrics["inference"]["data_snooping"]
    spa = snooping["superior_predictive_ability"]
    rows = [
        " & ".join(
            (
                "White Reality Check",
                f"{float(snooping['reality_check']['statistic']):.6f}",
                f"{float(snooping['reality_check']['p_value']):.4f}",
            )
        ),
        " & ".join(
            (
                "Hansen SPA, lower",
                f"{float(spa['statistic']):.4f}",
                f"{float(spa['p_value_lower']):.4f}",
            )
        ),
        " & ".join(
            (
                "Hansen SPA, consistent",
                f"{float(spa['statistic']):.4f}",
                f"{float(spa['p_value_consistent']):.4f}",
            )
        ),
        " & ".join(
            (
                "Hansen SPA, upper",
                f"{float(spa['statistic']):.4f}",
                f"{float(spa['p_value_upper']):.4f}",
            )
        ),
    ]
    return _table(
        f"Joint tests that no candidate beats the zero-return benchmark, over "
        f"{int(snooping['replications']):,} stationary-bootstrap replications with mean block "
        f"length {float(snooping['block_length']):.2f}. Hansen's three recentrings bracket the "
        f"p-value.",
        "snooping",
        "lrr",
        "Test & Statistic & $p$",
        rows,
    )


def sharpe_table(metrics: Mapping[str, Any]) -> str:
    """Sharpe-ratio inference for the executed strategy."""
    sharpe = metrics["inference"]["portfolio_sharpe"]
    rows = [
        f"Evaluated sessions & {int(sharpe['observation_count'])}",
        f"Per-session Sharpe & {float(sharpe['per_period_sharpe']):.4f}",
        f"Annualised, square-root rule & {float(sharpe['annualised_sharpe']):.4f}",
        "Annualised, Lo autocorrelation-adjusted & "
        f"{float(sharpe['autocorrelation_adjusted_annualised_sharpe']):.4f}",
        f"Skewness & {float(sharpe['skewness']):.4f}",
        f"Kurtosis & {float(sharpe['kurtosis']):.4f}",
        f"Probabilistic Sharpe ratio, threshold 0 & {float(sharpe['probabilistic_sharpe_ratio']):.4f}",
        f"Configurations examined & {int(sharpe['trial_count'])}",
        f"Search threshold & {float(sharpe['deflated_sharpe_threshold']):.4f}",
        f"Deflated Sharpe ratio & {float(sharpe['deflated_sharpe_ratio']):.4f}",
    ]
    return _table(
        "Sharpe-ratio inference. The deflated ratio evaluates the probabilistic Sharpe ratio at "
        "the threshold the best of $N$ skill-free configurations would be expected to reach.",
        "sharpe",
        "lr",
        "Quantity & Value",
        rows,
    )


def portfolio_table(metrics: Mapping[str, Any]) -> str:
    """The executed portfolio result, straight from the ledger."""
    portfolio = metrics["portfolio"]
    rows = [
        f"Gross return & {_percent(float(portfolio['gross_return']), 4)}",
        f"Net return & {_percent(float(portfolio['net_return']), 4)}",
        f"Annualised return & {_percent(float(portfolio['annualized_return']), 4)}",
        f"Annualised volatility & {_percent(float(portfolio['annualized_volatility']), 4)}",
        f"Sharpe & {float(portfolio['sharpe']):.4f}",
        f"Maximum drawdown & {_percent(float(portfolio['maximum_drawdown']), 4)}",
        f"Turnover & {float(portfolio['turnover']):.2f}$\\times$",
        f"Round trips & {int(portfolio['trade_count'])}",
        f"Sessions holding risk & {int(portfolio['invested_sessions'])} of "
        f"{int(portfolio['session_count'])}",
        f"Equal-weight universe & {_percent(float(portfolio['benchmark_return']), 4)}",
        f"Modelled cost & TRY {float(portfolio['cost_decomposition']['total']):,.2f}",
    ]
    return _table(
        "The executed long-only top-$k$ portfolio, computed entirely from the persisted event "
        "ledgers. Sessions holding risk is reported because annualised figures assume "
        "continuous deployment.",
        "portfolio",
        "lr",
        "Measure & Value",
        rows,
    )


def cost_table(metrics: Mapping[str, Any]) -> str:
    """Cost sensitivity with the trading decisions held fixed."""
    cases = metrics["cost_sensitivity"]
    ordered = sorted(cases.items(), key=lambda item: float(item[1]["cost_multiplier"]))
    rows = []
    for label, case in ordered:
        case_metrics = case["metrics"]
        rows.append(
            " & ".join(
                (
                    escape_latex(label),
                    _percent(float(case_metrics["gross_return"]), 4),
                    _percent(float(case_metrics["net_return"]), 4),
                    f"TRY {float(case_metrics['cost_decomposition']['total']):,.2f}",
                    str(int(case_metrics["trade_count"])),
                )
            )
        )
    return _table(
        "Cost sensitivity. The same trading decisions are reused across the three cases, so only "
        "the bill changes; net return cannot improve as costs rise.",
        "costs",
        "lrrrr",
        "Case & Gross & Net & Total cost & Round trips",
        rows,
    )


def sensitivity_table(metrics: Mapping[str, Any]) -> str:
    """The configuration grid, summarised."""
    sensitivity = metrics["configuration_sensitivity"]
    sharpe = sensitivity["per_period_sharpe"]
    returns = sensitivity["net_return"]
    trials = sensitivity["trials"]
    null_wins = sum(1 for trial in trials if trial["best_model"] == "zero_return")
    rows = [
        f"Configurations evaluated & {int(sensitivity['trial_count'])}",
        f"Reported configuration rank by Sharpe & {int(sensitivity['reported_rank_by_sharpe'])}",
        f"Median per-session Sharpe & {float(sharpe['median']):.4f}",
        f"Per-session Sharpe range & {float(sharpe['minimum']):.4f} to {float(sharpe['maximum']):.4f}",
        f"Configurations with positive net return & {_percent(float(returns['share_positive']))}",
        f"Best configuration net return & {_percent(float(sensitivity['best_trial']['net_return']), 4)}",
        f"Null had the best $R^2_0$ & {null_wins} of {len(trials)}",
        "Expected maximum Sharpe under no skill & "
        f"{float(sensitivity['expected_maximum_sharpe_under_no_skill']):.4f}",
    ]
    return _table(
        "The configuration grid. Each trial is a complete re-run of the evaluation under one "
        "fold geometry and one portfolio breadth.",
        "sensitivity",
        "lr",
        "Quantity & Value",
        rows,
    )


def dependence_table(metrics: Mapping[str, Any]) -> str:
    """Cross-sectional dependence, the sample size it implies, and its ceiling."""
    dependence = metrics["inference"]["cross_sectional_dependence"]
    panel = metrics["inference"]["detectability"]["panel"]
    rows = [
        f"Panel units & {int(dependence['unit_count'])}",
        f"Evaluated sessions & {int(dependence['session_count'])}",
        f"Out-of-sample rows & {int(dependence['row_count'])}",
        f"Mean within-session correlation & {float(dependence['mean_pairwise_correlation']):.4f}",
        f"Variance inflation factor & {float(dependence['variance_inflation_factor']):.4f}",
        f"Effective independent rows & {float(dependence['effective_row_count']):.1f}",
        "Mean within-session correlation of the loss differential & "
        f"{float(panel['loss_differential_correlation']):.4f}",
        "Independent rows per session, from the loss differential & "
        f"{float(panel['independent_rows_per_session']):.4f}",
        "Ceiling on independent rows per session ($1/\\bar\\rho$) & "
        f"{float(panel['independent_rows_per_session_ceiling']):.4f}",
        "Standard-error gain from an unlimited universe & "
        f"{(float(panel['standard_error_headroom']) - 1.0) * 100:.1f}\\%",
    ]
    return _table(
        "Cross-sectional dependence in the evaluation panel, and the most an arbitrarily wide "
        "cross-section could add at the same correlation.",
        "dependence",
        "lr",
        "Quantity & Value",
        rows,
    )


def detectability_table(metrics: Mapping[str, Any]) -> str:
    """The effect sizes the design was powered to find."""
    report = metrics["inference"]["detectability"]
    rows = [
        f"Best-powered candidate & {escape_latex(str(report['reference_model']))}",
        "Standard error of its session-mean loss differential & "
        f"{float(report['reference_standard_error']):.3e}",
        f"Test size $\\alpha$ & {float(report['alpha']):.2f}",
        f"Target power & {float(report['power']):.2f}",
        "Smallest detectable loss differential & "
        f"{float(report['minimum_detectable_loss_differential']):.3e}",
        f"Mean squared error of the null & {float(report['benchmark_mean_squared_error']):.3e}",
        "Smallest detectable out-of-sample $R^2_0$ & "
        f"{float(report['minimum_detectable_r_squared']):.4f}",
        f"Reference effect assumed plausible & {float(report['reference_r_squared']):.4f}",
        "Sessions required for that effect & "
        f"{int(report['sessions_required_for_reference_r_squared']):,}",
        "Per-session Sharpe required for a deflated ratio of $0.95$ & "
        f"{float(report['per_period_sharpe_required']):.4f}",
        f"Its annualised equivalent & {float(report['annualised_sharpe_required']):.4f}",
    ]
    return _table(
        "Inverting the accepted tests: the smallest effects this design could have separated "
        "from the null at conventional size and power.",
        "detectability",
        "lr",
        "Quantity & Value",
        rows,
    )


def feasibility_table(metrics: Mapping[str, Any]) -> str:
    """The information coefficient the cost schedule demands of any forecaster."""
    report = metrics["inference"]["detectability"]
    feasibility = report["feasibility"]
    required = float(feasibility["required_information_coefficient"])
    session = float(report["session_information_coefficient"])
    pooled = float(report["pooled_information_coefficient"])
    error = float(report["session_information_coefficient_standard_error"])
    breadth = report["feasible_breadth_at_unit_holding"]
    rows = [
        f"Round-trip cost on notional & {float(feasibility['round_trip_cost_rate']) * 1e4:.2f} bp",
        "Standard deviation of the executable target & "
        f"{_percent(float(feasibility['target_volatility']))}",
        f"Names ranked & {int(feasibility['universe_size'])}",
        f"Names held & {int(feasibility['selected'])}",
        f"Selection score $\\lambda(N,k)$ & {float(feasibility['selection_score']):.4f}",
        f"Information coefficient required & {required:.4f}",
        f"Per-session cross-sectional IC achieved & {session:.4f} ($\\pm$ {error:.4f})",
        f"Shortfall factor & {required / session:.1f}$\\times$",
        f"Pooled IC over stock-sessions, for contrast & {pooled:.4f}",
        "Universe clearing the bound while holding one name & "
        + ("$>10^{6}$" if breadth is None else f"{int(breadth):,}"),
    ]
    return _table(
        "Proposition~\\ref{prop:feasibility} evaluated on the accepted design. The requirement "
        "depends on the cost schedule, the volatility of the target and the breadth of the rule, "
        "and on nothing the forecaster does. Selection happens within a session, so the "
        "per-session cross-sectional IC is what instantiates $\\rho$; the pooled correlation is "
        "larger because it also absorbs the common time-series component that ranking cannot "
        "use. The final row is the fixed-count breadth at which the bound would be met: "
        "$\\lambda$ is unbounded in $N$, so this is a statement about attainable universes, not "
        "an impossibility result.",
        "feasibility",
        "lr",
        "Quantity & Value",
        rows,
    )


def search_dependence_table(metrics: Mapping[str, Any]) -> str:
    """The search correction under both readings of trial independence."""
    report = metrics["inference"]["detectability"]
    rows = [
        f"Configurations searched & {int(report['trial_count'])}",
        "Realised variance of trial Sharpe ratios & "
        f"{float(report['realised_trial_variance']):.6f}",
        "Variance implied by independent trials & "
        f"{float(report['independent_trial_variance']):.6f}",
        f"Effective independent trials & {float(report['effective_trial_count']):.2f}",
        f"Threshold at the realised dispersion & {float(report['deflated_sharpe_threshold']):.4f}",
        "Threshold if the trials were independent & "
        f"{float(report['independent_trial_threshold']):.4f}",
        f"Best configuration in the grid & {float(report['grid_maximum_sharpe']):.4f}",
    ]
    return _table(
        "The False Strategy threshold under both readings of the grid. The best configuration "
        "falls below either, so the conclusion does not turn on how the trials are counted.",
        "search",
        "lr",
        "Quantity & Value",
        rows,
    )


def block_length_table(metrics: Mapping[str, Any]) -> str:
    """The bootstrap interval repeated across block lengths."""
    blocks = metrics["bootstrap_block_sensitivity"]
    rows = []
    for key in sorted(blocks, key=int):
        interval = blocks[key]["annualized_return"]
        rows.append(
            " & ".join(
                (
                    str(int(key)),
                    _percent(float(interval["lower"])),
                    _percent(float(interval["upper"])),
                    "yes" if float(interval["lower"]) <= 0.0 <= float(interval["upper"]) else "no",
                )
            )
        )
    return _table(
        "The 95\\% bootstrap interval for annualised return, repeated across every candidate "
        "block length.",
        "blocks",
        "rrrl",
        "Block length & Lower & Upper & Contains zero",
        rows,
    )


TABLE_BUILDERS = {
    "dependence": dependence_table,
    "prediction": prediction_table,
    "accuracy": accuracy_test_table,
    "nested": nested_test_table,
    "joint_search": joint_search_table,
    "snooping": snooping_table,
    "sharpe": sharpe_table,
    "portfolio": portfolio_table,
    "costs": cost_table,
    "sensitivity": sensitivity_table,
    "blocks": block_length_table,
    "detectability": detectability_table,
    "feasibility": feasibility_table,
    "search": search_dependence_table,
}


def render_all_tables(metrics: Mapping[str, Any]) -> dict[str, str]:
    """Render every manuscript table from one metrics document."""
    return {name: builder(metrics) for name, builder in TABLE_BUILDERS.items()}
