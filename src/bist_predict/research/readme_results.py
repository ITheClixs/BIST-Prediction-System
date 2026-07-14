"""Render accepted benchmark evidence into a marker-delimited README block."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from bist_predict.research.prediction_metrics import recompute_prediction_metrics
from bist_predict.research.run_artifacts import verify_artifact_hashes


START_MARKER = "<!-- ACCEPTED_RESULTS:START -->"
END_MARKER = "<!-- ACCEPTED_RESULTS:END -->"
REQUIRED_ARTIFACTS = (
    "artifact_hashes.json",
    "metrics.json",
    "run_manifest.json",
    "data_manifest.json",
    "universe_manifest.json",
)


class ReadmeResultsError(ValueError):
    """Raised when run evidence cannot produce a truthful README block."""


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReadmeResultsError(f"required run artifact is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReadmeResultsError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReadmeResultsError(f"required run artifact must contain an object: {path.name}")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReadmeResultsError(f"required field must be an object: {field}")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, field: str) -> Any:
    if key not in mapping:
        raise ReadmeResultsError(f"missing required field: {field}.{key}")
    return mapping[key]


def _number(mapping: Mapping[str, Any], key: str, *, field: str) -> float:
    value = _required(mapping, key, field=field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadmeResultsError(f"required field must be numeric: {field}.{key}")
    return float(value)


def _optional_number(mapping: Mapping[str, Any], key: str, *, field: str) -> float | None:
    value = _required(mapping, key, field=field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadmeResultsError(f"required field must be numeric or null: {field}.{key}")
    return float(value)


def _text(mapping: Mapping[str, Any], key: str, *, field: str) -> str:
    value = _required(mapping, key, field=field)
    if not isinstance(value, str) or not value:
        raise ReadmeResultsError(f"required field must be a non-empty string: {field}.{key}")
    return value


def _percent(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


def _decimal(value: float | None, digits: int = 4) -> str:
    return "not available" if value is None else f"{value:.{digits}f}"


def _prediction_table(prediction: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, float]]]:
    if not prediction:
        raise ReadmeResultsError(
            "required field must contain at least one model: metrics.prediction"
        )
    lines = [
        "| Model | Samples | MAE | RMSE | Zero-mean R-squared | Spearman IC | Directional accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    r_squared: list[tuple[str, float]] = []
    for model_name in sorted(prediction):
        model = _mapping(prediction[model_name], field=f"metrics.prediction.{model_name}")
        sample_count = _number(model, "sample_count", field=f"metrics.prediction.{model_name}")
        zero_mean_r_squared = _number(
            model, "zero_mean_r_squared", field=f"metrics.prediction.{model_name}"
        )
        r_squared.append((model_name, zero_mean_r_squared))
        lines.append(
            "| "
            + " | ".join(
                (
                    model_name,
                    str(int(sample_count)),
                    _percent(_number(model, "mae", field=f"metrics.prediction.{model_name}")),
                    _percent(_number(model, "rmse", field=f"metrics.prediction.{model_name}")),
                    f"{zero_mean_r_squared:.4f}",
                    _decimal(
                        _optional_number(
                            model, "spearman_ic", field=f"metrics.prediction.{model_name}"
                        )
                    ),
                    _percent(
                        _number(
                            model,
                            "directional_accuracy",
                            field=f"metrics.prediction.{model_name}",
                        ),
                        2,
                    ),
                    _percent(
                        _number(
                            model,
                            "balanced_accuracy",
                            field=f"metrics.prediction.{model_name}",
                        ),
                        2,
                    ),
                )
            )
            + " |"
        )
    return lines, r_squared


def _portfolio_table(portfolio: Mapping[str, Any]) -> list[str]:
    costs = _mapping(
        _required(portfolio, "cost_decomposition", field="metrics.portfolio"),
        field="metrics.portfolio.cost_decomposition",
    )
    rows = (
        ("Gross return", _percent(_number(portfolio, "gross_return", field="metrics.portfolio"))),
        ("Net return", _percent(_number(portfolio, "net_return", field="metrics.portfolio"))),
        (
            "Annualized return",
            _percent(_number(portfolio, "annualized_return", field="metrics.portfolio")),
        ),
        (
            "Annualized volatility",
            _percent(_number(portfolio, "annualized_volatility", field="metrics.portfolio")),
        ),
        ("Sharpe", f"{_number(portfolio, 'sharpe', field='metrics.portfolio'):.4f}"),
        (
            "Maximum drawdown",
            _percent(_number(portfolio, "maximum_drawdown", field="metrics.portfolio")),
        ),
        ("Turnover", f"{_number(portfolio, 'turnover', field='metrics.portfolio'):.4f}x"),
        ("Trade count", str(int(_number(portfolio, "trade_count", field="metrics.portfolio")))),
        (
            "Equal-weight benchmark return",
            _percent(_number(portfolio, "benchmark_return", field="metrics.portfolio")),
        ),
        (
            "Benchmark-relative return",
            _percent(_number(portfolio, "benchmark_relative_return", field="metrics.portfolio")),
        ),
        (
            "Total modeled costs",
            f"TRY {_number(costs, 'total', field='metrics.portfolio.cost_decomposition'):,.2f}",
        ),
    )
    return ["| Portfolio measure | Accepted result |", "|---|---:|"] + [
        f"| {label} | {value} |" for label, value in rows
    ]


def _cost_table(cost_sensitivity: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    if not cost_sensitivity:
        raise ReadmeResultsError("required field must not be empty: metrics.cost_sensitivity")
    cases: list[tuple[float, str, Mapping[str, Any]]] = []
    for label, raw_case in cost_sensitivity.items():
        case = _mapping(raw_case, field=f"metrics.cost_sensitivity.{label}")
        multiplier = _number(case, "cost_multiplier", field=f"metrics.cost_sensitivity.{label}")
        case_metrics = _mapping(
            _required(case, "metrics", field=f"metrics.cost_sensitivity.{label}"),
            field=f"metrics.cost_sensitivity.{label}.metrics",
        )
        cases.append((multiplier, label, case_metrics))
    cases.sort(key=lambda item: (item[0], item[1]))

    lines = [
        "| Cost case | Gross return | Net return | Total costs | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    net_returns: list[float] = []
    for _, label, metrics in cases:
        costs = _mapping(
            _required(
                metrics, "cost_decomposition", field=f"metrics.cost_sensitivity.{label}.metrics"
            ),
            field=f"metrics.cost_sensitivity.{label}.metrics.cost_decomposition",
        )
        net_return = _number(
            metrics, "net_return", field=f"metrics.cost_sensitivity.{label}.metrics"
        )
        net_returns.append(net_return)
        lines.append(
            f"| {label} | "
            f"{_percent(_number(metrics, 'gross_return', field=f'metrics.cost_sensitivity.{label}.metrics'))} | "
            f"{_percent(net_return)} | "
            f"TRY {_number(costs, 'total', field=f'metrics.cost_sensitivity.{label}.metrics.cost_decomposition'):,.2f} | "
            f"{int(_number(metrics, 'trade_count', field=f'metrics.cost_sensitivity.{label}.metrics'))} |"
        )
    return lines, net_returns


def _negative_results(
    metrics: Mapping[str, Any],
    r_squared: Sequence[tuple[str, float]],
    cost_net_returns: Sequence[float],
) -> list[str]:
    lines: list[str] = []
    best_model, best_r_squared = max(r_squared, key=lambda item: (item[1], item[0]))
    if best_r_squared <= 0.0:
        lines.append(
            "- No evaluated model achieved positive zero-mean R-squared; "
            f"the best observed value was {best_r_squared:.4f} for `{best_model}`."
        )

    bootstrap = _mapping(
        _required(metrics, "bootstrap", field="metrics"), field="metrics.bootstrap"
    )
    annualized = _mapping(
        _required(bootstrap, "annualized_return", field="metrics.bootstrap"),
        field="metrics.bootstrap.annualized_return",
    )
    confidence = _number(
        annualized, "confidence_level", field="metrics.bootstrap.annualized_return"
    )
    lower = _number(annualized, "lower", field="metrics.bootstrap.annualized_return")
    upper = _number(annualized, "upper", field="metrics.bootstrap.annualized_return")
    if lower <= 0.0 <= upper:
        lines.append(
            f"- The {confidence:.0%} block-bootstrap interval for annualized return spans zero "
            f"({_percent(lower, 2)} to {_percent(upper, 2)})."
        )

    benchmarks = _mapping(
        _required(metrics, "benchmarks", field="metrics"), field="metrics.benchmarks"
    )
    index_benchmark = _mapping(
        _required(benchmarks, "relevant_bist_index", field="metrics.benchmarks"),
        field="metrics.benchmarks.relevant_bist_index",
    )
    if index_benchmark.get("status") == "not_available_in_input_dataset":
        lines.append(
            "- No relevant BIST index benchmark was available in the accepted input dataset; "
            "the report therefore does not claim index-relative performance."
        )

    if len(cost_net_returns) > 1 and all(
        later <= earlier
        for earlier, later in zip(cost_net_returns, cost_net_returns[1:], strict=False)
    ):
        lines.append(
            "- Net return did not improve as modeled transaction costs increased "
            f"({_percent(cost_net_returns[0])} to {_percent(cost_net_returns[-1])})."
        )
    return lines


def render_accepted_results(run_path: Path | str) -> str:
    """Render deterministic Markdown from one immutable accepted run directory."""
    run_directory = Path(run_path)
    if not (run_directory / "artifact_hashes.json").is_file():
        raise ReadmeResultsError("required run artifact is missing: artifact_hashes.json")
    integrity_failures = verify_artifact_hashes(run_directory)
    if integrity_failures:
        detail = ", ".join(
            f"{name}={reason}" for name, reason in sorted(integrity_failures.items())
        )
        raise ReadmeResultsError(f"run artifact integrity check failed: {detail}")
    artifacts = {name: _load_mapping(run_directory / name) for name in REQUIRED_ARTIFACTS}
    metrics = artifacts["metrics.json"]
    run_manifest = artifacts["run_manifest.json"]
    data_manifest = artifacts["data_manifest.json"]
    universe_manifest = artifacts["universe_manifest.json"]
    prediction_path = run_directory / "predictions.parquet"
    if not prediction_path.is_file():
        raise ReadmeResultsError("required run artifact is missing: predictions.parquet")
    try:
        recomputed_prediction = recompute_prediction_metrics(pd.read_parquet(prediction_path))
    except (OSError, ValueError) as error:
        raise ReadmeResultsError(f"could not recompute predictions.parquet: {error}") from error

    run_id = _text(run_manifest, "run_id", field="run_manifest")
    git_sha = _text(run_manifest, "git_sha", field="run_manifest")
    dataset_id = _text(data_manifest, "dataset_id", field="data_manifest")
    universe_version = _text(universe_manifest, "universe_version", field="universe_manifest")
    tickers = _required(universe_manifest, "tickers", field="universe_manifest")
    if (
        not isinstance(tickers, list)
        or not tickers
        or not all(isinstance(item, str) for item in tickers)
    ):
        raise ReadmeResultsError(
            "required field must be a non-empty string list: universe_manifest.tickers"
        )

    prediction = _mapping(
        _required(metrics, "prediction", field="metrics"), field="metrics.prediction"
    )
    if prediction != recomputed_prediction:
        raise ReadmeResultsError(
            "metrics.prediction does not match recomputation from predictions.parquet"
        )
    portfolio = _mapping(
        _required(metrics, "portfolio", field="metrics"), field="metrics.portfolio"
    )
    cost_sensitivity = _mapping(
        _required(metrics, "cost_sensitivity", field="metrics"),
        field="metrics.cost_sensitivity",
    )
    prediction_lines, r_squared = _prediction_table(prediction)
    cost_lines, cost_net_returns = _cost_table(cost_sensitivity)

    clean = _required(run_manifest, "dirty_working_tree", field="run_manifest") is False
    lines = [
        "### Accepted run provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run | `{run_id}` |",
        f"| Git commit | `{git_sha[:12]}` ({'clean' if clean else 'dirty'} working tree recorded) |",
        f"| Dataset | `{dataset_id}` |",
        f"| Scope | `{universe_version}` |",
        f"| Tickers | {', '.join(tickers)} |",
        f"| Period | {_text(data_manifest, 'start', field='data_manifest')} to {_text(data_manifest, 'end', field='data_manifest')} |",
        f"| Provider rows | {int(_number(data_manifest, 'row_count', field='data_manifest')):,} |",
        "",
        "### Out-of-sample prediction metrics",
        "",
        *prediction_lines,
        "",
        "### Accepted portfolio result",
        "",
        *_portfolio_table(portfolio),
        "",
        "### Transaction-cost sensitivity",
        "",
        *cost_lines,
        "",
        "### Negative results and evidence limits",
        "",
        *_negative_results(metrics, r_squared, cost_net_returns),
    ]
    return "\n".join(lines)


def update_readme_results(readme_path: Path | str, run_path: Path | str) -> None:
    """Replace only the content inside the accepted-results marker pair."""
    path = Path(readme_path)
    try:
        current = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReadmeResultsError(f"could not read README: {error}") from error
    if current.count(START_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise ReadmeResultsError("README must contain exactly one accepted-results marker pair")
    content_start = current.index(START_MARKER) + len(START_MARKER)
    content_end = current.index(END_MARKER)
    if content_end < content_start:
        raise ReadmeResultsError("README must contain exactly one accepted-results marker pair")
    generated = render_accepted_results(run_path)
    updated = f"{current[:content_start]}\n{generated}\n{current[content_end:]}"
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as error:
        raise ReadmeResultsError(f"could not update README: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update the marker-delimited accepted-results block in a README."
    )
    parser.add_argument("--readme", type=Path, required=True, help="README file to update")
    parser.add_argument("--run", type=Path, required=True, help="accepted run directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the README results updater CLI."""
    arguments = _parser().parse_args(argv)
    update_readme_results(arguments.readme, arguments.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
