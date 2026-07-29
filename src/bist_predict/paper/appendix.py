"""Appendix tables, generated from the run bundle and from the source itself.

The body of the manuscript quotes results; the appendix has to be able to
reconstruct them.  Everything here is therefore read out of the committed
artifacts or out of the declarations the code already carries -- the feature
manifest, the run configuration, the executed fold boundaries, the trial grid,
and the catalogue of defects the mutation harness reintroduces.  Nothing is
transcribed by hand, so an appendix cannot drift away from the experiment it
documents.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from bist_predict.paper.tables import escape_latex
from bist_predict.research.stationary_features import STATIONARY_FEATURE_MANIFEST

__all__ = [
    "APPENDIX_BUILDERS",
    "configuration_grid_appendix",
    "execution_appendix",
    "feature_manifest_appendix",
    "fold_geometry_appendix",
    "mutation_appendix",
    "render_all_appendices",
]

_MUTATION_PATTERN = re.compile(
    r'^\s{4}\(\s*\n\s{8}"(?P<label>(?:[^"\\]|\\.)*)",',
    re.MULTILINE,
)

# Configuration keys grouped by what they govern, so the appendix reads as a
# specification rather than as an alphabetised dump.
_CONFIG_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Scope",
        ("experiment_scope", "methodology_version", "seed"),
    ),
    (
        "Fold geometry",
        ("min_train_dates", "validation_dates", "step_dates", "embargo_dates"),
    ),
    (
        "Portfolio",
        (
            "portfolio_model",
            "top_k",
            "starting_equity",
            "min_trade_value",
            "max_participation",
            "liquidity_lookback_sessions",
            "decision_cost_rate",
        ),
    ),
    (
        "Cost schedule",
        (
            "commission_rate",
            "bid_ask_spread_rate",
            "slippage_rate",
            "market_impact_coefficient",
            "tax_rate",
        ),
    ),
    (
        "Resampling and search",
        ("bootstrap_iterations", "bootstrap_block_sizes", "sensitivity_grid"),
    ),
)


def _longtable(
    caption: str,
    label: str,
    spec: str,
    header: str,
    rows: Sequence[str],
    *,
    size: str = "footnotesize",
) -> str:
    """Render a table that is allowed to break across pages.

    A ``table`` float that outgrows its page does not warn: it silently overruns
    the bottom margin and prints over the folio.  Every appendix table here is
    long enough for that to be a real risk.
    """
    lines = [
        r"\begin{center}",
        f"\\{size}",
        rf"\begin{{longtable}}{{{spec}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label}}} \\",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
        r"  \endfirsthead",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
        r"  \endhead",
        r"    \bottomrule",
        r"  \endfoot",
    ]
    lines.extend(f"    {row} \\\\" for row in rows)
    lines.extend([r"\end{longtable}", r"\end{center}", ""])
    return "\n".join(lines)


def configuration_grid_appendix(metrics: Mapping[str, Any]) -> str:
    """Render every configuration trial, ordered by per-session Sharpe ratio."""
    trials = metrics["configuration_sensitivity"]["trials"]
    ordered = sorted(trials, key=lambda trial: -float(trial["per_period_sharpe"]))
    rows = [
        " & ".join(
            (
                rf"\texttt{{{escape_latex(trial['trial_id'])}}}",
                str(int(trial["fold_count"])),
                str(int(trial["session_count"])),
                rf"{float(trial['net_return']) * 100:.2f}\%",
                f"{float(trial['per_period_sharpe']):.4f}",
                str(int(trial["trade_count"])),
                escape_latex(str(trial["best_model"])),
            )
        )
        for trial in ordered
    ]
    return _longtable(
        "Every configuration in the search grid, ordered by per-session Sharpe ratio. Each row "
        "is a complete re-run of the evaluation under one fold geometry and one portfolio "
        "breadth, not a re-weighting of a cached result. The final column names the model with "
        "the best out-of-sample $R^2_0$ in that trial.",
        "grid",
        "lrrrrrl",
        "Configuration & Folds & Sessions & Net return & Sharpe & Round trips & Best model",
        rows,
        size="scriptsize",
    )


def feature_manifest_appendix() -> str:
    """Render the accepted feature manifest exactly as the code declares it."""
    rows = []
    for index, spec in enumerate(STATIONARY_FEATURE_MANIFEST.features, start=1):
        rows.append(
            " & ".join(
                (
                    str(index),
                    rf"\texttt{{{escape_latex(spec.name)}}}",
                    rf"\texttt{{{escape_latex(spec.formula)}}}",
                    str(int(spec.lookback)),
                )
            )
        )
    manifest_hash = STATIONARY_FEATURE_MANIFEST.manifest_hash
    return _longtable(
        "The accepted feature manifest, schema version "
        rf"\texttt{{{escape_latex(STATIONARY_FEATURE_MANIFEST.schema_version)}}}, digest "
        rf"\texttt{{{escape_latex(manifest_hash[:16])}}}. Every entry declares its formula, its "
        "lookback in sessions, and a normalisation policy of \\texttt{none}: the accepted models "
        "consume scale-free quantities, so nominal price level cannot act as accidental ticker "
        "identification. A panel built against a different manifest digest is refused.",
        "features",
        "rlp{0.46\\textwidth}r",
        "\\# & Feature & Formula & Lookback",
        rows,
        size="scriptsize",
    )


def execution_appendix(config: Mapping[str, Any]) -> str:
    """Render the declared configuration of the committed run, grouped by role."""
    remaining = {str(key): value for key, value in config.items()}
    rows: list[str] = []
    for group, keys in _CONFIG_GROUPS:
        present = [key for key in keys if key in remaining]
        if not present:
            continue
        rows.append(rf"\multicolumn{{2}}{{l}}{{\emph{{{group}}}}}")
        for key in present:
            value = remaining.pop(key)
            rendered = f"{value:g}" if isinstance(value, float) else str(value)
            rows.append(
                rf"\quad \texttt{{{escape_latex(key)}}} & \texttt{{{escape_latex(rendered)}}}"
            )
    if remaining:
        rows.append(r"\multicolumn{2}{l}{\emph{Other}}")
        for key in sorted(remaining):
            value = remaining[key]
            rendered = f"{value:g}" if isinstance(value, float) else str(value)
            rows.append(
                rf"\quad \texttt{{{escape_latex(key)}}} & \texttt{{{escape_latex(rendered)}}}"
            )
    return _longtable(
        "The committed run configuration, verbatim from the bundle. These values enter the "
        "configuration hash, so a change to any of them produces a different run identity rather "
        "than a silently different result.",
        "configuration",
        "lr",
        "Parameter & Value",
        rows,
    )


def fold_geometry_appendix(folds: Sequence[Mapping[str, Any]]) -> str:
    """Render the executed fold boundaries, read from the run's fold artifact."""
    rows = []
    for fold in folds:
        train = fold["train_window"]
        validation = fold["validation_window"]
        embargo = fold.get("embargo_dates") or []
        rows.append(
            " & ".join(
                (
                    rf"\texttt{{{escape_latex(fold['fold_id'])}}}",
                    escape_latex(str(train["date_start"])),
                    escape_latex(str(train["date_end"])),
                    str(len(fold["train_dates"])),
                    str(len(embargo)),
                    escape_latex(str(validation["date_start"])),
                    escape_latex(str(validation["date_end"])),
                    str(len(fold["validation_dates"])),
                )
            )
        )
    return _longtable(
        "The partition that was executed, read from the run's fold artifact rather than "
        "recomputed for the manuscript. Training windows expand from a common start; the embargo "
        "column counts trading dates removed between training and validation; validation blocks "
        "are disjoint, so no session is evaluated twice.",
        "folds",
        "lllrrllr",
        "Fold & Train from & Train to & Dates & Emb. & Validate from & Validate to & Dates",
        rows,
    )


def mutation_appendix(source: str) -> str:
    """Render the catalogue of defects the mutation harness reintroduces.

    The labels are parsed out of the harness itself, so the appendix cannot list
    a defect the harness does not actually apply, and cannot omit one it does.
    """
    labels = [match.group("label") for match in _MUTATION_PATTERN.finditer(source)]
    if not labels:
        raise ValueError("no mutation labels were found in the harness source")
    rows = []
    for index, label in enumerate(labels, start=1):
        subject, _, description = label.partition(":")
        rows.append(
            " & ".join(
                (
                    str(index),
                    escape_latex(subject.strip()),
                    escape_latex(description.strip() or subject.strip()),
                )
            )
        )
    return _longtable(
        f"The {len(labels)} defects the mutation harness reintroduces into the source. For each "
        "one it runs the guarding test clean, applies the edit, requires the test to fail, "
        "restores the file, and requires the test to pass again. A defect that survives means "
        "the guarding test is decorative; three were found that way and repaired.",
        "mutations",
        "rlp{0.52\\textwidth}",
        "\\# & Subject & Defect reintroduced",
        rows,
        size="scriptsize",
    )


APPENDIX_BUILDERS = (
    "grid_appendix",
    "features_appendix",
    "configuration_appendix",
    "folds_appendix",
    "mutations_appendix",
)


def render_all_appendices(
    *,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    mutation_source: str,
) -> dict[str, str]:
    """Render every appendix table from the run bundle and the harness source."""
    return {
        "grid_appendix": configuration_grid_appendix(metrics),
        "features_appendix": feature_manifest_appendix(),
        "configuration_appendix": execution_appendix(config),
        "folds_appendix": fold_geometry_appendix(folds),
        "mutations_appendix": mutation_appendix(mutation_source),
    }
