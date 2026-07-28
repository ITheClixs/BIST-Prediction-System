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
    """Cross-sectional dependence and the sample size it implies."""
    dependence = metrics["inference"]["cross_sectional_dependence"]
    rows = [
        f"Panel units & {int(dependence['unit_count'])}",
        f"Evaluated sessions & {int(dependence['session_count'])}",
        f"Out-of-sample rows & {int(dependence['row_count'])}",
        f"Mean within-session correlation & {float(dependence['mean_pairwise_correlation']):.4f}",
        f"Variance inflation factor & {float(dependence['variance_inflation_factor']):.4f}",
        f"Effective independent rows & {float(dependence['effective_row_count']):.1f}",
    ]
    return _table(
        "Cross-sectional dependence in the evaluation panel.",
        "dependence",
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
    "snooping": snooping_table,
    "sharpe": sharpe_table,
    "portfolio": portfolio_table,
    "costs": cost_table,
    "sensitivity": sensitivity_table,
    "blocks": block_length_table,
}


def render_all_tables(metrics: Mapping[str, Any]) -> dict[str, str]:
    """Render every manuscript table from one metrics document."""
    return {name: builder(metrics) for name, builder in TABLE_BUILDERS.items()}
