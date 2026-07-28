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
)


def _document_text(paths: Sequence[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _block_content(document: str, block: str) -> str | None:
    start = f"<!-- {block}:START -->"
    end = f"<!-- {block}:END -->"
    if start not in document or end not in document:
        return None
    return document[document.index(start) + len(start) : document.index(end)].strip("\n")


def _artifacts(run_path: Path) -> dict[str, Any]:
    def load(name: str) -> Any:
        return json.loads((run_path / name).read_text(encoding="utf-8"))

    metrics = load("metrics.json")
    return {
        **metrics,
        "metrics": metrics,
        "data_manifest": load("data_manifest.json"),
        "run_manifest": load("run_manifest.json"),
        "universe_manifest": load("universe_manifest.json"),
    }


def check(documents: Sequence[Path], run_path: Path) -> int:
    metrics = _artifacts(run_path)
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
    arguments = parser.parse_args(argv)
    documents = arguments.document or [ROOT / "README.md"]
    return check([Path(item) for item in documents], Path(arguments.run))


if __name__ == "__main__":
    raise SystemExit(main())
