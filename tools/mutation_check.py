"""Reintroduce each guarded defect and confirm the suite actually catches it.

A test that passes on already-correct input proves nothing. Each entry below
names one real defect, the exact source edit that reintroduces it, and the test
that is supposed to fail as a result. The harness runs that test clean, applies
the edit, runs it again, restores the file, and runs it once more. A defect that
survives means the guarding test is decorative.

Run with ``make mutation-check``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    (
        "variance guard: compare dispersion against zero instead of the series scale",
        "src/bist_predict/research/reporting.py",
        "    return dispersion > _RELATIVE_VARIANCE_FLOOR * max(scale, _RELATIVE_VARIANCE_FLOOR)",
        "    return dispersion > 0.0",
        "tests/test_research/test_reporting.py::test_a_flat_equity_curve_does_not_produce_an_astronomical_sharpe",
    ),
    (
        "variance guard (sharpe module): same defect on the per-period estimator",
        "src/bist_predict/research/inference/sharpe.py",
        "    if deviation <= _RELATIVE_VARIANCE_FLOOR * max(scale, _RELATIVE_VARIANCE_FLOOR):",
        "    if deviation <= 0.0:",
        "tests/test_research/test_inference_sharpe.py::test_scale_relative_guard_returns_zero_for_a_constant_series",
    ),
    (
        "holding period: hardcode one session instead of reading the ledger",
        "src/bist_predict/research/reporting.py",
        "    return float(np.mean(holding_periods)) if holding_periods else 0.0",
        "    return 1.0 if holding_periods else 0.0",
        "tests/test_research/test_reporting.py::test_holding_period_counts_sessions_rather_than_assuming_one",
    ),
    (
        "Holm: drop the running maximum that enforces monotonicity",
        "src/bist_predict/research/inference/multiplicity.py",
        "        running_maximum = max(running_maximum, scaled)",
        "        running_maximum = scaled",
        "tests/test_research/test_inference_multiplicity.py::test_monotonicity_is_enforced_across_the_ordered_family",
    ),
    (
        "Diebold-Mariano: drop the Harvey-Leybourne-Newbold small-sample correction",
        "src/bist_predict/research/inference/forecast_tests.py",
        "    correction = np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count)",
        "    correction = 1.0",
        "tests/test_research/test_inference_forecast_tests.py::test_small_sample_correction_is_actually_applied",
    ),
    (
        "Diebold-Mariano: aggregate rows instead of sessions, ignoring cross-sectional dependence",
        "src/bist_predict/research/inference/forecast_tests.py",
        '    by_session = merged.groupby("date", sort=True)["differential"].mean()',
        '    by_session = merged.set_index("date")["differential"]',
        "tests/test_research/test_inference_forecast_tests.py::test_session_aggregation_collapses_same_date_rows",
    ),
    (
        "Diebold-Mariano: guard the differential variance against zero, not against scale",
        "src/bist_predict/research/inference/forecast_tests.py",
        "    if scale <= 1e-12 * max(reference, 1e-12):",
        "    if scale <= 0.0:",
        "tests/test_research/test_inference_forecast_tests.py::test_an_economically_constant_differential_is_rejected",
    ),
    (
        "HAC: normalise the autocovariance by n-k instead of n",
        "src/bist_predict/research/inference/hac.py",
        "    return float(np.dot(centred[lag:], centred[:-lag]) / count)",
        "    return float(np.dot(centred[lag:], centred[:-lag]) / (count - lag))",
        "tests/test_research/test_inference_hac.py::test_lag_one_autocovariance_matches_hand_computation",
    ),
    (
        "SPA: compare floored maxima, collapsing the comparison onto the atom at zero",
        "src/bist_predict/research/inference/snooping.py",
        "        p_values[name] = float(np.mean(draws > observed_maximum))",
        "        p_values[name] = float(np.mean(np.maximum(draws, 0.0) > max(observed_maximum, 0.0)))",
        "tests/test_research/test_inference_snooping.py::test_a_family_of_inferior_candidates_reports_no_evidence",
    ),
    (
        "SPA: skip the bootstrap recentring entirely",
        "src/bist_predict/research/inference/snooping.py",
        "        adjusted = (resampled_means - recentring) / scale",
        "        adjusted = resampled_means / scale",
        "tests/test_research/test_inference_snooping.py::test_a_family_of_inferior_candidates_reports_no_evidence",
    ),
    (
        "Lo annualisation: sum q-1 autocorrelations from a 120-session sample",
        "src/bist_predict/research/inference/sharpe.py",
        "    usable_lags = min(horizon - 1, series.size - 1, bandwidth)",
        "    usable_lags = min(horizon - 1, series.size - 1)",
        "tests/test_research/test_inference_sharpe.py::test_annualisation_lags_are_truncated_to_an_estimable_bandwidth",
    ),
    (
        "Deflated Sharpe: ignore the trial count and never deflate",
        "src/bist_predict/research/inference/sharpe.py",
        "    if trial_count == 1 or trial_sharpe_variance == 0.0:",
        "    if trial_count >= 1 or trial_sharpe_variance == 0.0:",
        "tests/test_research/test_inference_sharpe.py::test_deflated_threshold_grows_with_the_number_of_trials",
    ),
    (
        "Cross-sectional dependence: assume independence by ignoring the correlation",
        "src/bist_predict/research/inference/dependence.py",
        "    return float(max(1.0, 1.0 + (unit_count - 1) * mean_pairwise_correlation))",
        "    return 1.0",
        "tests/test_research/test_inference_dependence.py::test_perfectly_correlated_units_collapse_to_one_effective_unit",
    ),
    (
        "Stationary bootstrap: use fixed-length blocks instead of geometric ones",
        "src/bist_predict/research/inference/snooping.py",
        "        indices[:, step] = np.where(restarts[:, step - 1], fresh[:, step - 1], continued)",
        "        indices[:, step] = continued",
        "tests/test_research/test_inference_snooping.py::test_bootstrap_block_lengths_are_geometric_with_the_requested_mean",
    ),
    (
        "Sensitivity grid: decouple the step from the fold width",
        "src/bist_predict/research/sensitivity.py",
        '            "step_dates": int(validation),',
        '            "step_dates": 10,',
        "tests/test_research/test_sensitivity.py::test_step_is_tied_to_the_validation_width",
    ),
    (
        "Accepted config: allow a grid that omits the reported configuration",
        "src/bist_predict/research/accepted_benchmark.py",
        '    sensitivity_grid: str = "train=24,36,48|val=5,10,20|embargo=1,2|topk=1,2,3,4"',
        '    sensitivity_grid: str = "train=36,48|val=5,20|embargo=2|topk=1,2,4"',
        "tests/test_research/test_accepted_benchmark.py::test_accepted_configuration_appears_in_its_own_sensitivity_grid",
    ),
    (
        "Clark-West: drop the factor of two from the encompassing adjustment",
        "src/bist_predict/research/inference/nested.py",
        "    adjusted = (\n        2.0\n"
        '        * rows["target"].to_numpy(dtype=np.float64)\n'
        '        * rows["predicted_return"].to_numpy(dtype=np.float64)\n'
        "    )",
        "    adjusted = (\n"
        '        rows["target"].to_numpy(dtype=np.float64)\n'
        '        * rows["predicted_return"].to_numpy(dtype=np.float64)\n'
        "    )",
        "tests/test_research/test_inference_nested.py::test_the_adjustment_is_twice_the_product_of_target_and_forecast",
    ),
    (
        "Clark-West: make the one-sided test two-sided and halve its power",
        "src/bist_predict/research/inference/nested.py",
        "        p_value=float(stats.norm.sf(statistic)),",
        "        p_value=float(2.0 * stats.norm.sf(abs(statistic))),",
        "tests/test_research/test_inference_nested.py::test_the_test_is_one_sided",
    ),
    (
        # A skill-free grid cannot guard this: its means are already near zero,
        # so removing the recentring barely moves the null. The defect only
        # shows when a configuration has a mean worth removing, at which point
        # leaving it in the bootstrap raises the null to meet the observation.
        "Joint search: skip the recentring, so the joint null is never imposed",
        "src/bist_predict/research/inference/joint_search.py",
        "    centred = values - observed_mean",
        "    centred = values",
        "tests/test_research/test_inference_joint_search.py::test_a_genuinely_strong_configuration_is_rejected",
    ),
    (
        "Joint search: resample each configuration on its own index draw",
        "src/bist_predict/research/inference/joint_search.py",
        "    draws = _sharpe_along_axis(centred[indices])",
        "    draws = _sharpe_along_axis(\n"
        "        np.stack(\n"
        "            [\n"
        "                centred[\n"
        "                    stationary_bootstrap_indices(\n"
        "                        count,\n"
        "                        block_length=chosen_block,\n"
        "                        replications=replications,\n"
        "                        rng=np.random.default_rng(seed + column),\n"
        "                    )\n"
        "                ][:, :, column]\n"
        "                for column in range(values.shape[1])\n"
        "            ],\n"
        "            axis=2,\n"
        "        )\n"
        "    )",
        "tests/test_research/test_inference_joint_search.py::test_the_effective_trial_count_falls_as_the_grid_becomes_redundant",
    ),
]


def run(test_id: str) -> tuple[int, str]:
    result = subprocess.run(
        ["uv", "run", "pytest", test_id, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return result.returncode, (result.stdout + result.stderr).strip().splitlines()[-1]


def main() -> int:
    rows: list[tuple[str, str, str, str]] = []
    failures = 0
    for label, relative, original, mutated, test_id in MUTATIONS:
        path = ROOT / relative
        text = path.read_text()
        if text.count(original) != 1:
            rows.append((label, "SETUP-ERROR", f"anchor found {text.count(original)}x", test_id))
            failures += 1
            continue
        baseline_code, baseline_line = run(test_id)
        path.write_text(text.replace(original, mutated, 1))
        try:
            mutant_code, mutant_line = run(test_id)
        finally:
            path.write_text(text)
        restored_code, _ = run(test_id)
        detected = baseline_code == 0 and mutant_code != 0 and restored_code == 0
        failures += 0 if detected else 1
        rows.append(
            (
                label,
                "DETECTED" if detected else "NOT DETECTED",
                f"clean={baseline_line} | mutated={mutant_line}",
                test_id.split("::")[-1],
            )
        )

    width = max(len(row[0]) for row in rows)
    print(f"{'defect reintroduced'.ljust(width)}  verdict       guarding test")
    print("-" * (width + 60))
    for label, verdict, detail, test_name in rows:
        print(f"{label.ljust(width)}  {verdict:<13} {test_name}")
        print(f"{' ' * width}  {detail}")
    print(f"\n{len(rows) - failures}/{len(rows)} defects detected by the suite")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
