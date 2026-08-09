"""Render the simulation study's tables for the manuscript.

Every table here reports a measured error rate against the level that error
rate was supposed to have. The run-bundle tables in :mod:`tables` say what the
estimators concluded on the market; these say what those same estimators do
when the answer is known in advance, so the two must be read together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "CALIBRATION_TABLE_BUILDERS",
    "render_all_calibration_tables",
]


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


def _rate(record: Mapping[str, Any], digits: int = 4) -> str:
    """Format a measured rejection rate with its Monte Carlo interval."""
    return (
        f"{float(record['rate']):.{digits}f} "
        rf"\small{{[{float(record['lower']):.{digits}f}, {float(record['upper']):.{digits}f}]}}"
    )


def size_by_cross_section_table(study: Mapping[str, Any]) -> str:
    """Measured size against the closed form, as the panel widens."""
    cells = [cell for cell in study["experiments"]["dependence"] if cell["varied"] == "unit_count"]
    rows = []
    for cell in cells:
        design = cell["design"]
        rows.append(
            " & ".join(
                (
                    str(int(design["unit_count"])),
                    f"{float(cell['predicted_row_size']):.4f}",
                    _rate(cell["row_rejection"]),
                    _rate(cell["session_rejection"]),
                )
            )
        )
    return _table(
        "Measured size of the two-sided equal-accuracy test at a nominal $5\\%$, against the "
        "closed form of Proposition~\\ref{prop:size}, as the cross-section widens at the panel's "
        "measured correlation. Brackets are $95\\%$ Monte Carlo intervals over "
        f"{int(study['configuration']['size_replications']):,} replications. Treating rows as "
        "independent is not a small approximation; aggregating to the session removes the error "
        "entirely.",
        "calibration-size",
        "rrll",
        "Names $k$ & Predicted (row) & Measured (row) & Measured (session)",
        rows,
    )


def robustness_table(study: Mapping[str, Any]) -> str:
    """Session-level size across innovation, volatility and length regimes."""
    labels = {
        "gaussian_constant": "Gaussian, constant volatility",
        "student_t5": "Student $t_5$ innovations",
        "garch": "GARCH(1,1) volatility",
        "regime_switching": "Regime-switching level",
        "heavy_tailed_garch": "Student $t_5$ with GARCH",
        "sixty_sessions": "60 sessions",
        "five_hundred_sessions": "500 sessions",
        "idiosyncratic_forecast": "Idiosyncratic forecast",
    }
    rows = []
    for cell in study["experiments"]["robustness"]:
        rows.append(
            " & ".join(
                (
                    labels.get(str(cell["variant"]), str(cell["variant"])),
                    _rate(cell["row_rejection"]),
                    _rate(cell["session_rejection"]),
                )
            )
        )
    return _table(
        "Measured size of the session-aggregated test at a nominal $5\\%$ under departures from "
        "the baseline generator. The aggregation is what holds the level; none of the stresses "
        "that break the row-level test disturbs it.",
        "calibration-robustness",
        "lll",
        "Generator & Row level & Session level",
        rows,
    )


def nested_table(study: Mapping[str, Any]) -> str:
    """What each test does when the benchmark is right and the model is noise."""
    labels = {
        "anchor": "Anchor design",
        "quiet_forecast": "Low-variance forecast",
        "loud_forecast": "High-variance forecast",
        "independent_rows": "Uncorrelated names",
        "thirty_names": "30 names",
        "five_hundred_sessions": "500 sessions",
    }
    rows = []
    for cell in study["experiments"]["nested"]:
        rows.append(
            " & ".join(
                (
                    labels.get(str(cell["variant"]), str(cell["variant"])),
                    f"{float(cell['variance_ratio']):.3f}",
                    _rate(cell["diebold_mariano_against_candidate"]),
                    _rate(cell["clark_west_session"]),
                )
            )
        )
    return _table(
        "The nested null: the zero benchmark is the correct population forecast and the fitted "
        "model is estimation noise. The squared-error comparison is not merely oversized, it is "
        "almost certain to declare the fitted model significantly \\emph{worse}, at a rate set by "
        "the forecast's own variance. The Clark--West statistic, which removes that term, holds "
        "its level.",
        "calibration-nested",
        "lrll",
        "Design & $\\mathbb{E}[\\hat y^2]/\\mathbb{E}[r^2]$ & DM rejects towards null & "
        "Clark--West",
        rows,
    )


def family_wise_table(study: Mapping[str, Any]) -> str:
    """Family-wise error of each multiple-comparison correction."""
    labels = {
        "anchor": "4 names, correlated",
        "independent_rows": "4 names, independent",
        "thirty_names": "30 names, correlated",
        "thirty_names_independent": "30 names, independent",
    }
    rows = []
    for cell in study["experiments"]["family"]:
        rows.append(
            " & ".join(
                (
                    labels.get(str(cell["variant"]), str(cell["variant"])),
                    _rate(cell["uncorrected_any"], 3),
                    _rate(cell["holm_row"], 3),
                    _rate(cell["holm_session"], 3),
                    _rate(cell["reality_check"], 3),
                    _rate(cell["spa_hansen"], 3),
                )
            )
        )
    return _table(
        "Family-wise error rate at a nominal $5\\%$ with every member of a family of "
        f"{int(study['configuration']['family_size'])} forecasts skill-free by construction. "
        "Holm controls the family but not the dependence: applied to row-level p-values on a "
        "correlated panel it fails outright, while the same procedure on session-aggregated "
        "p-values is correct. The bootstrap corrections are mildly liberal throughout.",
        "calibration-family",
        "llllll",
        "Design & Uncorrected & Holm (row) & Holm (session) & Reality Check & SPA",
        rows,
    )


def search_calibration_table(study: Mapping[str, Any]) -> str:
    """Whether each search threshold is a threshold at all."""
    rows = []
    for cell in study["experiments"]["search"]:
        rows.append(
            " & ".join(
                (
                    f"{float(cell['trial_correlation']):.2f}",
                    _rate(cell["false_strategy_expectation"], 3),
                    _rate(cell["joint_bootstrap_quantile"], 3),
                    f"{float(cell['mean_independent_equivalent_trials']):.2f}",
                )
            )
        )
    return _table(
        "A grid of "
        f"{int(study['configuration']['grid_trial_count'])} skill-free configurations, at four "
        "levels of dependence between them. The False Strategy expectation is exceeded about "
        "half the time whatever the dependence, because an expectation is not a critical value; "
        "the synchronised bootstrap quantile of the same grid holds the nominal level. The last "
        "column is the independent-trial count the grid behaves like, recovered from the "
        "simulated maximum.",
        "calibration-search",
        "rllr",
        "Trial corr.\\ & Clears $\\mathbb{E}[\\max]$ & Clears bootstrap $q_{0.95}$ & "
        "$N_{\\mathrm{eff}}$",
        rows,
    )


def detectable_effect_table(study: Mapping[str, Any]) -> str:
    """Simulation-calibrated minimum detectable effects, by record length."""
    rows = []
    for record in study["experiments"]["power"]["minimum_detectable"]:
        rows.append(
            " & ".join(
                (
                    f"{int(record['session_count']):,}",
                    f"{float(record['diebold_mariano_r_squared']):.4f}",
                    f"{float(record['clark_west_covariance_ratio']):.4f}",
                )
            )
        )
    return _table(
        "The smallest effect each test separates from its null at $80\\%$ power and $5\\%$ size, "
        "obtained by simulating the whole comparison rather than by inverting a normal "
        "approximation. The Diebold--Mariano column is a population out-of-sample $R^2_0$; the "
        "Clark--West column is a population covariance ratio, which is the quantity that test "
        "responds to. A forecast can carry information the second detects while losing the "
        "comparison the first makes.",
        "calibration-detectable",
        "rrr",
        "Sessions & DM: $R^2_0$ at $80\\%$ power & Clark--West: covariance ratio",
        rows,
    )


CALIBRATION_TABLE_BUILDERS = {
    "calibration_size": size_by_cross_section_table,
    "calibration_robustness": robustness_table,
    "calibration_nested": nested_table,
    "calibration_family": family_wise_table,
    "calibration_search": search_calibration_table,
    "calibration_detectable": detectable_effect_table,
}


def render_all_calibration_tables(study: Mapping[str, Any]) -> dict[str, str]:
    """Render every calibration table from one study document."""
    return {name: builder(study) for name, builder in CALIBRATION_TABLE_BUILDERS.items()}
