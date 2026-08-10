"""Check every quantitative claim in the documents against the run artifacts.

Two kinds of check run here.

Generated blocks are regenerated from ``metrics.json`` and compared to what the
document actually contains, so a rerun that moves a number fails until the
document is regenerated.

Prose claims are numbers a human wrote outside those blocks. Each one is
declared below with the artifact field it must equal and the tolerance it is
allowed. A claim whose anchor text is missing from the document fails, so
deleting the sentence does not silently pass the check.

Run with ``make verify-claims``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bist_predict.research.readme_results import (  # noqa: E402
    BLOCK_RENDERERS,
    GENERATED_BLOCKS,
)

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Claim:
    """One number a document states, and the artifact field that decides it."""

    name: str
    pattern: str
    resolve: Callable[[Mapping[str, Any]], float]
    scale: float = 1.0
    tolerance: float = 5e-5

    def expected(self, metrics: Mapping[str, Any]) -> float:
        return self.resolve(metrics) * self.scale


def _dig(metrics: Mapping[str, Any], path: str) -> Any:
    node: Any = metrics
    for key in path.split("."):
        if isinstance(node, Mapping):
            node = node[key]
        else:
            node = node[int(key)]
    return node


def _field(path: str) -> Callable[[Mapping[str, Any]], float]:
    return lambda metrics: float(_dig(metrics, path))


def _cost_breakeven(metrics: Mapping[str, Any]) -> float:
    """Return the cost multiple at which net return crosses zero.

    Linear interpolation between the bracketing cases, computed from the
    artifact rather than copied out of a figure.
    """
    cases = _dig(metrics, "cost_sensitivity")
    points = sorted(
        (float(case["cost_multiplier"]), float(case["metrics"]["net_return"]))
        for case in cases.values()
    )
    for (low_x, low_y), (high_x, high_y) in zip(points, points[1:], strict=False):
        if low_y >= 0.0 >= high_y and low_y != high_y:
            return low_x + (high_x - low_x) * low_y / (low_y - high_y)
    raise ValueError("net return never crosses zero across the cost cases")


def _null_wins(metrics: Mapping[str, Any]) -> float:
    trials = _dig(metrics, "configuration_sensitivity.trials")
    return float(sum(1 for trial in trials if trial["best_model"] == "zero_return"))


def _worse_than_null(metrics: Mapping[str, Any]) -> float:
    accuracy = _dig(metrics, "inference.equal_predictive_accuracy")
    rejected = set(accuracy["family_wise_correction"]["rejected"])
    return float(
        sum(
            1
            for name, test in accuracy["session_aggregated"].items()
            if name in rejected and test["verdict"] == "benchmark_better"
        )
    )


def _better_than_null(metrics: Mapping[str, Any]) -> float:
    accuracy = _dig(metrics, "inference.equal_predictive_accuracy")
    rejected = set(accuracy["family_wise_correction"]["rejected"])
    return float(
        sum(
            1
            for name, test in accuracy["session_aggregated"].items()
            if name in rejected and test["verdict"] == "candidate_better"
        )
    )


def _cell(
    experiment: str, selector: str, value: str, *, key: str = "variant"
) -> Callable[[Mapping[str, Any]], float]:
    """Resolve one field of one labelled cell of the calibration study.

    The study stores each experiment as a list of cells identified by a label
    rather than by position, so a claim that named a position would silently
    follow the wrong cell if the sweep were reordered.
    """

    def resolve(metrics: Mapping[str, Any]) -> float:
        cells = _dig(metrics, f"calibration.experiments.{experiment}")
        for cell in cells:
            if str(cell[key]) == selector:
                return float(_dig(cell, value))
        raise KeyError(f"no cell with {key}={selector!r} in experiment {experiment!r}")

    return resolve


def _unit_cell(units: int, value: str) -> Callable[[Mapping[str, Any]], float]:
    """Resolve a field of the dependence sweep cell with a given cross-section."""

    def resolve(metrics: Mapping[str, Any]) -> float:
        cells = _dig(metrics, "calibration.experiments.dependence")
        for cell in cells:
            if cell["varied"] == "unit_count" and int(cell["design"]["unit_count"]) == units:
                return float(_dig(cell, value))
        raise KeyError(f"no dependence cell with {units} units")

    return resolve


def _search_cell(correlation: float, value: str) -> Callable[[Mapping[str, Any]], float]:
    """Resolve a field of the search cell at a given trial correlation."""

    def resolve(metrics: Mapping[str, Any]) -> float:
        for cell in _dig(metrics, "calibration.experiments.search"):
            if abs(float(cell["trial_correlation"]) - correlation) < 1e-9:
                return float(_dig(cell, value))
        raise KeyError(f"no search cell at trial correlation {correlation}")

    return resolve


def _detectable(sessions: int, value: str) -> Callable[[Mapping[str, Any]], float]:
    """Resolve a minimum detectable effect at a given record length."""

    def resolve(metrics: Mapping[str, Any]) -> float:
        for record in _dig(metrics, "calibration.experiments.power.minimum_detectable"):
            if int(record["session_count"]) == sessions:
                return float(record[value])
        raise KeyError(f"no minimum detectable effect at {sessions} sessions")

    return resolve


def _largest_closed_form_error(metrics: Mapping[str, Any]) -> float:
    """Return the worst gap between the predicted and measured row-level size."""
    cells = [
        cell
        for cell in _dig(metrics, "calibration.experiments.dependence")
        if cell["varied"] == "unit_count"
    ]
    return max(
        abs(float(cell["row_rejection"]["rate"]) - float(cell["predicted_row_size"]))
        for cell in cells
    )


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "out-of-sample rows",
        r"(?P<value>\d[\d,]*) out-of-sample (?:panel )?rows",
        _field("inference.cross_sectional_dependence.row_count"),
        tolerance=0.5,
    ),
    Claim(
        "evaluated sessions",
        r"(?P<value>\d+) evaluation sessions",
        _field("inference.cross_sectional_dependence.session_count"),
        tolerance=0.5,
    ),
    Claim(
        "within-session correlation",
        r"correlate at (?P<value>\d+(?:\.\d+)?)",
        _field("inference.cross_sectional_dependence.mean_pairwise_correlation"),
        tolerance=5e-4,
    ),
    Claim(
        "effective independent rows",
        r"about (?P<value>\d+) independent observations",
        _field("inference.cross_sectional_dependence.effective_row_count"),
        tolerance=0.5,
    ),
    Claim(
        "models significantly worse than the null",
        r"(?P<value>\d+) of the six fitted models are significantly worse",
        _worse_than_null,
        tolerance=0.5,
    ),
    Claim(
        "models significantly better than the null",
        r"(?P<value>\d+) are significantly better",
        _better_than_null,
        tolerance=0.5,
    ),
    Claim(
        "SPA consistent p-value",
        r"SPA[^.]*?p = (?P<value>\d+(?:\.\d+)?)",
        _field("inference.data_snooping.superior_predictive_ability.p_value_consistent"),
        tolerance=5e-5,
    ),
    Claim(
        "deflated Sharpe ratio",
        r"deflated Sharpe ratio of (?P<value>\d+(?:\.\d+)?)",
        _field("inference.portfolio_sharpe.deflated_sharpe_ratio"),
        tolerance=5e-5,
    ),
    Claim(
        "search threshold",
        r"search threshold of (?P<value>\d+(?:\.\d+)?)",
        _field("inference.portfolio_sharpe.deflated_sharpe_threshold"),
        tolerance=5e-5,
    ),
    Claim(
        "configurations examined",
        r"(?P<value>\d+)-configuration grid",
        _field("configuration_sensitivity.trial_count"),
        tolerance=0.5,
    ),
    Claim(
        "configurations with positive net return",
        r"(?P<value>\d+(?:\.\d+)?)% of (?:those )?configurations produced a positive net return",
        _field("configuration_sensitivity.net_return.share_positive"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "grid trials where the null won",
        r"null had the best out-of-sample .{0,3}R.{0,20}? in (?P<value>\d+) of",
        _null_wins,
        tolerance=0.5,
    ),
    Claim(
        "reported net return",
        r"loses (?P<value>\d+(?:\.\d+)?)% of its capital after costs",
        lambda metrics: -float(_dig(metrics, "portfolio.net_return")),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "reported gross return",
        r"gross return of (?P<value>\d+(?:\.\d+)?)%",
        _field("portfolio.gross_return"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "annualised Sharpe",
        r"annualised Sharpe ratio of (?P<value>-?\d+(?:\.\d+)?)",
        _field("portfolio.sharpe"),
        tolerance=5e-5,
    ),
    Claim(
        "provider rows",
        r"(?P<value>\d[\d,]*) provider rows",
        _field("data_manifest.row_count"),
        tolerance=0.5,
    ),
    Claim(
        "kurtosis of session returns",
        r"kurtosis of (?P<value>\d+(?:\.\d+)?)",
        _field("inference.portfolio_sharpe.kurtosis"),
        tolerance=5e-5,
    ),
    Claim(
        "probabilistic Sharpe ratio",
        r"Sharpe ratio exceeds zero is (?P<value>\d+(?:\.\d+)?)",
        _field("inference.portfolio_sharpe.probabilistic_sharpe_ratio"),
        tolerance=5e-5,
    ),
    Claim(
        "Reality Check p-value",
        r"Reality Check gives .{0,3}p = (?P<value>\d+(?:\.\d+)?)",
        _field("inference.data_snooping.reality_check.p_value"),
        tolerance=5e-5,
    ),
    Claim(
        "autocorrelation-adjusted Sharpe",
        r"annualisation gives (?P<value>-?\d+(?:\.\d+)?)",
        _field("inference.portfolio_sharpe.autocorrelation_adjusted_annualised_sharpe"),
        tolerance=5e-5,
    ),
    Claim(
        "sessions holding risk",
        r"carries risk on only (?P<value>\d+) of",
        _field("portfolio.invested_sessions"),
        tolerance=0.5,
    ),
    Claim(
        "cost breakeven multiplier",
        r"Breakeven sits at (?P<value>\d+(?:\.\d+)?)x",
        _cost_breakeven,
        tolerance=5e-3,
    ),
    Claim(
        "best grid net return",
        r"best configuration in the grid returns (?P<value>\d+(?:\.\d+)?)%",
        _field("configuration_sensitivity.best_trial.net_return"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "best grid per-session Sharpe",
        r"per-session Sharpe ratio of (?P<value>\d+(?:\.\d+)?)",
        _field("configuration_sensitivity.best_trial.per_period_sharpe"),
        tolerance=5e-5,
    ),
    Claim(
        "smallest detectable out-of-sample R-squared",
        r"design could separate from zero at 5% size and 80% power is (?P<value>\d+(?:\.\d+)?)",
        _field("inference.detectability.minimum_detectable_r_squared"),
        tolerance=5e-5,
    ),
    Claim(
        "sessions required for a plausible effect",
        r"would need (?P<value>\d[\d,]*) sessions",
        _field("inference.detectability.sessions_required_for_reference_r_squared"),
        tolerance=0.5,
    ),
    Claim(
        "independent rows per session ceiling",
        r"carry at most (?P<value>\d+(?:\.\d+)?) independent rows",
        _field("inference.detectability.panel.independent_rows_per_session_ceiling"),
        tolerance=5e-5,
    ),
    Claim(
        "information coefficient the cost schedule requires",
        r"cross-sectional information coefficient of (?P<value>\d+(?:\.\d+)?)",
        _field("inference.detectability.feasibility.required_information_coefficient"),
        tolerance=5e-5,
    ),
    Claim(
        "per-session information coefficient achieved",
        r"portfolio model achieved (?P<value>\d+(?:\.\d+)?) per session",
        _field("inference.detectability.session_information_coefficient"),
        tolerance=5e-5,
    ),
    Claim(
        "pooled information coefficient",
        r"pooled correlation of (?P<value>\d+(?:\.\d+)?)",
        _field("inference.detectability.pooled_information_coefficient"),
        tolerance=5e-5,
    ),
    Claim(
        "Sharpe ratio that would have established skill",
        r"required a Sharpe ratio of (?P<value>\d+(?:\.\d+)?) per session",
        _field("inference.detectability.per_period_sharpe_required"),
        tolerance=5e-5,
    ),
    Claim(
        "effective independent trials",
        r"behave like (?P<value>\d+(?:\.\d+)?) independent searches",
        _field("inference.detectability.effective_trial_count"),
        tolerance=5e-3,
    ),
    Claim(
        "search threshold under independent trials",
        r"independent puts the bar at (?P<value>\d+(?:\.\d+)?)",
        _field("inference.detectability.independent_trial_threshold"),
        tolerance=5e-5,
    ),
    # --- Clark-West and the joint search, from the run bundle -----------------
    Claim(
        "joint search p-value",
        r"exact (?:joint )?bootstrap p-value is \$?(?P<value>\d+(?:\.\d+)?)",
        _field("inference.joint_search.p_value"),
        tolerance=5e-5,
    ),
    Claim(
        "correlation between configurations",
        r"mean pairwise correlation between configurations is \$?(?P<value>\d+(?:\.\d+)?)",
        _field("inference.joint_search.mean_pairwise_correlation"),
        tolerance=5e-5,
    ),
    Claim(
        "independent-equivalent trials from the bootstrap",
        r"behaves like \$?(?P<value>\d+(?:\.\d+)?)\$? independent trials",
        _field("inference.joint_search.independent_equivalent_trials"),
        tolerance=5e-3,
    ),
    Claim(
        "sessions common to every configuration",
        r"on the (?P<value>\d+) sessions every configuration evaluated",
        _field("inference.joint_search.session_count"),
        tolerance=0.5,
    ),
    Claim(
        "expected maximum under the joint null",
        r"against an expected\s+maximum of \$?(?P<value>\d+(?:\.\d+)?)",
        _field("inference.joint_search.null_expected_maximum"),
        tolerance=5e-5,
    ),
    # --- The calibration study ------------------------------------------------
    Claim(
        "measured row-level size at four names",
        r"row-level test rejects a true null \$?(?P<value>\d+(?:\.\d+)?)",
        _unit_cell(4, "row_rejection.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "predicted row-level size at four names",
        r"time on four names against a predicted \$?(?P<value>\d+(?:\.\d+)?)",
        _unit_cell(4, "predicted_row_size"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "measured row-level size at thirty names",
        r"\$(?P<value>\d+(?:\.\d+)?)%\$ on thirty against a predicted",
        _unit_cell(30, "row_rejection.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "measured row-level size at a hundred names",
        r"and \$(?P<value>\d+(?:\.\d+)?)%\$ on a hundred against a predicted",
        _unit_cell(100, "row_rejection.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "largest closed-form discrepancy",
        r"largest discrepancy anywhere in the sweep is \$?(?P<value>\d+(?:\.\d+)?)",
        _largest_closed_form_error,
        tolerance=5e-5,
    ),
    Claim(
        "measured session-level size at four names",
        r"the session-level test \$(?P<value>\d+(?:\.\d+)?)%",
        _unit_cell(4, "session_rejection.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "nested-null rejection rate against the candidate",
        r"significantly worse than the benchmark in \$(?P<value>\d+(?:\.\d+)?)%",
        _cell("nested", "anchor", "diebold_mariano_against_candidate.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "nested-null rate at a loud forecast",
        r"variance ratio of \$0\.60\$ it reaches \$(?P<value>\d+(?:\.\d+)?)%",
        _cell("nested", "loud_forecast", "diebold_mariano_against_candidate.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "nested-null rate at a quiet forecast",
        r"ratio of \$0\.05\$, falls to \$(?P<value>\d+(?:\.\d+)?)%",
        _cell("nested", "quiet_forecast", "diebold_mariano_against_candidate.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "Clark-West row-level size at thirty names",
        r"at thirty names it reaches \$?(?P<value>\d+(?:\.\d+)?)\$?",
        _cell("nested", "thirty_names", "clark_west_row.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "Holm family-wise error at row level, four names",
        r"family-wise error is \$(?P<value>\d+(?:\.\d+)?)\$ at four names",
        _cell("family", "anchor", "holm_row.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "Holm family-wise error at row level, thirty names",
        r"and \$(?P<value>\d+(?:\.\d+)?)\$ at thirty, while the identical",
        _cell("family", "thirty_names", "holm_row.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "Holm family-wise error at session level, four names",
        r"session-aggregated statistics gives \$(?P<value>\d+(?:\.\d+)?)\$",
        _cell("family", "anchor", "holm_session.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "Holm family-wise error at session level, thirty names",
        r"statistics gives \$\d+(?:\.\d+)?\$ and \$(?P<value>\d+(?:\.\d+)?)\$",
        _cell("family", "thirty_names", "holm_session.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "false strategy expectation cleared at the highest trial correlation",
        r"clears the expected maximum in \$\d+(?:\.\d+)?\\?%\$ to \$(?P<value>\d+(?:\.\d+)?)%\$",
        _search_cell(0.5, "false_strategy_expectation.rate"),
        scale=100.0,
        tolerance=5e-3,
    ),
    Claim(
        "SPA size, untruncated variant",
        r"puts the variant at \$?(?P<value>\d+(?:\.\d+)?)\$?",
        _cell("family", "anchor", "spa_untruncated.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "SPA size, Hansen definition",
        r"own\s+definition at \$?(?P<value>\d+(?:\.\d+)?)\$?",
        _cell("family", "anchor", "spa_hansen.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "independent-equivalent trials recovered when trials are independent",
        r"it recovers \$?(?P<value>\d+(?:\.\d+)?)\$? of a nominal",
        _search_cell(0.0, "mean_independent_equivalent_trials"),
        tolerance=5e-3,
    ),
    Claim(
        "joint bootstrap size at the highest trial correlation",
        r"and \$?(?P<value>\d+(?:\.\d+)?)\$? at the four correlation levels",
        _search_cell(0.98, "joint_bootstrap_quantile.rate"),
        tolerance=5e-5,
    ),
    Claim(
        "minimum detectable R-squared by simulation at 120 sessions",
        r"equal-accuracy test requires a population \$R\^2_0\$ of \$?(?P<value>\d+(?:\.\d+)?)",
        _detectable(120, "diebold_mariano_r_squared"),
        tolerance=5e-5,
    ),
    Claim(
        "minimum detectable covariance ratio at 120 sessions",
        r"Clark-{2}West requires a covariance ratio of \$?(?P<value>\d+(?:\.\d+)?)",
        _detectable(120, "clark_west_covariance_ratio"),
        tolerance=5e-5,
    ),
)


_LATEX_LITERALS = (
    ("{,}", ","),  # a LaTeX thin-space digit separator, as in 1{,}004
    ("\\%", "%"),
    ("\\,", ""),
    ("~", " "),
)


def _normalise(text: str) -> str:
    r"""Return document text a single claim pattern can match in either format.

    The same claim is stated in Markdown prose and in LaTeX, where a thousands
    separator is ``1{,}004`` rather than ``1,004`` and a sentence may wrap
    mid-number. Without this, a pattern tuned to one document silently matches
    the wrong digits in the other: ``1{,}004 provider rows`` reads as ``4``.
    Line structure carries no meaning for these claims, so whitespace is
    collapsed and every pattern can be written as though the document were one
    line.
    """
    for source, replacement in _LATEX_LITERALS:
        text = text.replace(source, replacement)
    return re.sub(r"\s+", " ", text)


def _document_text(paths: Sequence[Path]) -> str:
    return _normalise("\n".join(path.read_text(encoding="utf-8") for path in paths))


def _block_content(document: str, block: str) -> str | None:
    start = f"<!-- {block}:START -->"
    end = f"<!-- {block}:END -->"
    if start not in document or end not in document:
        return None
    return document[document.index(start) + len(start) : document.index(end)].strip("\n")


def _artifacts(run_path: Path, calibration_path: Path) -> dict[str, Any]:
    def load(name: str) -> Any:
        return json.loads((run_path / name).read_text(encoding="utf-8"))

    metrics = load("metrics.json")
    # The calibration study is a second immutable artifact with its own hash. It
    # is merged under one key rather than flattened, so a calibration claim can
    # never accidentally resolve against a run-bundle field of the same name.
    return {
        **metrics,
        "metrics": metrics,
        "data_manifest": load("data_manifest.json"),
        "run_manifest": load("run_manifest.json"),
        "universe_manifest": load("universe_manifest.json"),
        "calibration": json.loads(calibration_path.read_text(encoding="utf-8")),
    }


def check(documents: Sequence[Path], run_path: Path, calibration_path: Path) -> int:
    metrics = _artifacts(run_path, calibration_path)
    rows: list[tuple[str, str, str, str]] = []
    failures = 0

    for path in documents:
        text = path.read_text(encoding="utf-8")
        for block in GENERATED_BLOCKS:
            content = _block_content(text, block)
            if content is None:
                continue
            expected = BLOCK_RENDERERS[block](run_path).strip("\n")
            passed = content == expected
            failures += 0 if passed else 1
            rows.append(
                (
                    f"{path.name}:{block}",
                    "regenerated block",
                    "matches artifacts" if passed else "STALE, rerun make readme-results",
                    "PASS" if passed else "FAIL",
                )
            )

    document = _document_text(documents)
    for claim in CLAIMS:
        matches = list(re.finditer(claim.pattern, document))
        if not matches:
            failures += 1
            rows.append((claim.name, "not stated", "anchor text absent", "FAIL"))
            continue
        expected = claim.expected(metrics)
        for match in matches:
            stated = float(match.group("value").replace(",", ""))
            passed = abs(stated - expected) <= claim.tolerance
            failures += 0 if passed else 1
            rows.append(
                (
                    claim.name,
                    f"{stated:g}",
                    f"{expected:g}",
                    "PASS" if passed else "FAIL",
                )
            )

    name_width = max(len(row[0]) for row in rows)
    stated_width = max(max(len(row[1]) for row in rows), 6)
    print(f"{'claim'.ljust(name_width)}  {'stated'.ljust(stated_width)}  expected")
    print("-" * (name_width + stated_width + 40))
    for name, stated, expected, verdict in rows:
        print(f"{name.ljust(name_width)}  {stated.ljust(stated_width)}  {expected}   [{verdict}]")
    print(f"\n{len(rows) - failures}/{len(rows)} checks passed")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the claim checker over the requested documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="accepted run directory")
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=None,
        help="document to check; repeatable",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "calibration" / "study.json",
        help="simulation calibration study",
    )
    arguments = parser.parse_args(argv)
    documents = arguments.document or [ROOT / "README.md", ROOT / "paper" / "main.tex"]
    return check(
        [Path(item) for item in documents],
        Path(arguments.run),
        Path(arguments.calibration),
    )


if __name__ == "__main__":
    raise SystemExit(main())
