"""Power, breadth and search bounds pinned to values computable by hand."""

from __future__ import annotations

import math

import pytest
from scipy import stats

from bist_predict.research.inference.detectability import (
    detectability_report,
    effective_trial_count,
    expected_top_selection_score,
    false_strategy_quantile,
    minimum_detectable_mean,
    panel_information_ceiling,
    required_information_coefficient,
    sessions_required_for_effect,
    sharpe_required_for_confidence,
)

# Expected values of the largest standard normal order statistic, from the
# classical tables: E[Z_(2:2)] = 1/sqrt(pi), and E[Z_(4:4)] = 1.0293753730.
LARGEST_OF_TWO = 1.0 / math.sqrt(math.pi)
LARGEST_OF_FOUR = 1.0293753730123246


def test_the_largest_order_statistic_matches_the_published_table() -> None:
    assert expected_top_selection_score(2, 1) == pytest.approx(LARGEST_OF_TWO, abs=1e-9)
    assert expected_top_selection_score(4, 1) == pytest.approx(LARGEST_OF_FOUR, abs=1e-9)


def test_selecting_the_whole_universe_removes_the_selection_advantage() -> None:
    """Averaging every order statistic recovers the mean of the parent normal."""
    for size in (2, 4, 9, 25):
        assert expected_top_selection_score(size, size) == pytest.approx(0.0, abs=1e-9)


def test_the_selection_score_rises_as_the_rule_gets_more_selective() -> None:
    scores = [expected_top_selection_score(20, k) for k in (20, 10, 5, 2, 1)]
    assert scores == sorted(scores)


def test_the_top_three_of_four_average_the_three_upper_order_statistics() -> None:
    """The lower two of four are symmetric about zero and cancel."""
    assert expected_top_selection_score(4, 3) == pytest.approx(LARGEST_OF_FOUR / 3.0, abs=1e-9)


def test_the_detectable_mean_is_the_two_sided_power_calculation() -> None:
    expected = float(stats.t.ppf(0.975, 119)) + float(stats.t.ppf(0.80, 119))
    assert minimum_detectable_mean(1.0, observations=120) == pytest.approx(expected, rel=1e-12)


def test_a_larger_sample_detects_a_smaller_effect() -> None:
    wide = minimum_detectable_mean(1.0, observations=1000)
    narrow = minimum_detectable_mean(1.0, observations=30)
    assert wide < narrow


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"standard_error": 0.0, "observations": 120}, "standard error"),
        ({"standard_error": 1.0, "observations": 2}, "three observations"),
        ({"standard_error": 1.0, "observations": 120, "alpha": 1.0}, "alpha"),
        ({"standard_error": 1.0, "observations": 120, "power": 0.4}, "power"),
    ],
)
def test_the_power_calculation_refuses_meaningless_inputs(
    kwargs: dict[str, float], message: str
) -> None:
    standard_error = kwargs.pop("standard_error")
    with pytest.raises(ValueError, match=message):
        minimum_detectable_mean(standard_error, **kwargs)  # type: ignore[arg-type]


def test_the_required_sample_reproduces_the_effect_it_was_solved_for() -> None:
    """At the returned count the effect is detectable; one session fewer it is not."""
    standard_error = 1.4441533820655354e-05
    effect = 0.25 * minimum_detectable_mean(standard_error, observations=120)
    required = sessions_required_for_effect(effect, standard_error=standard_error, observations=120)

    def detectable_at(count: int) -> float:
        scaled = standard_error * math.sqrt(120 / count)
        return minimum_detectable_mean(scaled, observations=count)

    assert detectable_at(required) <= effect
    assert detectable_at(required - 1) > effect


def test_an_effect_already_detectable_needs_no_further_sessions() -> None:
    standard_error = 1e-5
    effect = 10.0 * minimum_detectable_mean(standard_error, observations=120)
    assert (
        sessions_required_for_effect(effect, standard_error=standard_error, observations=120) == 120
    )


def test_the_panel_ceiling_is_the_reciprocal_of_the_correlation() -> None:
    ceiling = panel_information_ceiling(0.5696754057159131, 4)
    assert ceiling["independent_rows_per_session"] == pytest.approx(4 / 2.7090262171477395)
    assert ceiling["independent_rows_per_session_ceiling"] == pytest.approx(
        1.0 / 0.5696754057159131
    )
    assert ceiling["headroom"] == pytest.approx(1.1888463983025985, rel=1e-9)


def test_widening_the_universe_never_reaches_the_ceiling() -> None:
    """The gain from more names is bounded, and the bound is approached slowly."""
    correlation = 0.57
    achieved = [
        panel_information_ceiling(correlation, size)["independent_rows_per_session"]
        for size in (2, 4, 10, 100, 10_000)
    ]
    assert achieved == sorted(achieved)
    assert max(achieved) < 1.0 / correlation


def test_the_feasibility_bound_scales_with_cost_and_against_volatility() -> None:
    base = required_information_coefficient(
        round_trip_cost_rate=0.00202,
        target_volatility=0.019,
        universe_size=4,
        selected=3,
    )
    doubled_cost = required_information_coefficient(
        round_trip_cost_rate=0.00404,
        target_volatility=0.019,
        universe_size=4,
        selected=3,
    )
    doubled_volatility = required_information_coefficient(
        round_trip_cost_rate=0.00202,
        target_volatility=0.038,
        universe_size=4,
        selected=3,
    )
    required = base["required_information_coefficient"]
    assert doubled_cost["required_information_coefficient"] == pytest.approx(2.0 * required)
    assert doubled_volatility["required_information_coefficient"] == pytest.approx(0.5 * required)


def test_a_wider_universe_lowers_the_information_coefficient_a_strategy_needs() -> None:
    narrow = required_information_coefficient(
        round_trip_cost_rate=0.00202, target_volatility=0.019, universe_size=4, selected=1
    )["required_information_coefficient"]
    wide = required_information_coefficient(
        round_trip_cost_rate=0.00202, target_volatility=0.019, universe_size=100, selected=25
    )["required_information_coefficient"]
    assert wide < narrow


def test_the_false_strategy_quantile_grows_with_the_number_of_trials() -> None:
    values = [false_strategy_quantile(count) for count in (2, 5, 20, 72, 1000)]
    assert values == sorted(values)


def test_independent_trials_are_their_own_effective_count() -> None:
    """With no dependence to absorb, the correction is the identity."""
    assert effective_trial_count(
        trial_count=72,
        realised_trial_variance=0.004,
        independent_trial_variance=0.004,
    ) == pytest.approx(72.0, rel=1e-6)


def test_trials_that_disperse_less_than_independent_ones_count_for_less() -> None:
    reduced = effective_trial_count(
        trial_count=72,
        realised_trial_variance=0.002757385313670684,
        independent_trial_variance=0.00833670141424204,
    )
    assert 2.0 < reduced < 72.0
    assert reduced == pytest.approx(7.010726452101544, rel=1e-6)


def test_the_effective_count_preserves_the_search_threshold() -> None:
    """Its defining property: both readings of the theorem give the same bar."""
    realised, independent, trials = 0.002757385313670684, 0.00833670141424204, 72
    reduced = effective_trial_count(
        trial_count=trials,
        realised_trial_variance=realised,
        independent_trial_variance=independent,
    )
    assert math.sqrt(realised) * false_strategy_quantile(trials) == pytest.approx(
        math.sqrt(independent) * false_strategy_quantile(reduced), rel=1e-8
    )


def test_the_required_sharpe_is_the_point_where_the_deflated_ratio_clears_the_bar() -> None:
    threshold, observations, skew, kurtosis = 0.12669711763142574, 120, -0.0722, 7.229
    required = sharpe_required_for_confidence(
        threshold=threshold, observations=observations, skewness=skew, kurtosis=kurtosis
    )
    variance = 1.0 - skew * required + (kurtosis - 1.0) / 4.0 * required * required
    deflated = float(
        stats.norm.cdf((required - threshold) * math.sqrt(observations - 1) / math.sqrt(variance))
    )
    assert deflated == pytest.approx(0.95, abs=1e-9)


def test_a_longer_record_lowers_the_sharpe_that_would_have_convinced() -> None:
    short = sharpe_required_for_confidence(
        threshold=0.1267, observations=120, skewness=0.0, kurtosis=3.0
    )
    long = sharpe_required_for_confidence(
        threshold=0.1267, observations=2520, skewness=0.0, kurtosis=3.0
    )
    assert 0.1267 < long < short


def test_the_search_threshold_survives_however_long_the_record_gets() -> None:
    """Sampling noise vanishes with n; the bar the search sets does not."""
    limit = sharpe_required_for_confidence(
        threshold=0.1267, observations=250_000, skewness=0.0, kurtosis=3.0
    )
    assert limit == pytest.approx(0.1267, abs=5e-3)


def _report_inputs() -> dict[str, object]:
    return {
        "session_standard_errors": {"logistic": 1.4441533820655354e-05, "ridge": 2.6e-05},
        "benchmark_mean_squared_error": 0.00036039396789081337,
        "session_count": 120,
        "dependence": {"mean_pairwise_correlation": 0.5696754057159131, "unit_count": 4},
        "sharpe": {
            "per_period_sharpe": -0.028431310523605955,
            "trial_count": 72,
            "trial_sharpe_variance": 0.002757385313670684,
            "deflated_sharpe_threshold": 0.12669711763142574,
            "skewness": -0.07217950871540205,
            "kurtosis": 7.228953086719694,
        },
        "grid_maximum_sharpe": 0.035570809845700116,
        "periods_per_year": 252,
        "round_trip_cost_rate": 0.00202,
        "target_volatility": 0.01900006876105416,
        "realised_information_coefficient": 0.03871243681199547,
        "universe_size": 4,
        "selected": 3,
    }


def test_the_report_quotes_the_candidate_the_design_could_best_have_separated() -> None:
    report = detectability_report(**_report_inputs())  # type: ignore[arg-type]
    assert report.reference_model == "logistic"
    assert report.reference_standard_error == pytest.approx(1.4441533820655354e-05)


def test_the_report_reproduces_the_accepted_run_bounds() -> None:
    report = detectability_report(**_report_inputs())  # type: ignore[arg-type]
    assert report.minimum_detectable_r_squared == pytest.approx(0.11319210647592144, rel=1e-9)
    assert report.sessions_required_for_reference_r_squared == 15126
    assert report.feasibility["required_information_coefficient"] == pytest.approx(
        0.3098444187805461, rel=1e-9
    )
    assert report.effective_trial_count == pytest.approx(7.010726452101544, rel=1e-6)
    assert report.annualised_sharpe_required == pytest.approx(4.5785187214125065, rel=1e-9)


def test_the_design_could_not_have_detected_a_plausible_effect() -> None:
    """The paper's central claim, asserted rather than merely reported."""
    report = detectability_report(**_report_inputs())  # type: ignore[arg-type]
    assert report.minimum_detectable_r_squared > 10.0 * report.reference_r_squared
    assert report.feasibility["required_information_coefficient"] > (
        5.0 * report.realised_information_coefficient
    )
    assert report.grid_maximum_sharpe < report.deflated_sharpe_threshold
    assert report.grid_maximum_sharpe < report.independent_trial_threshold


def test_the_report_is_json_safe() -> None:
    record = detectability_report(**_report_inputs()).to_dict()  # type: ignore[arg-type]
    import json

    assert json.loads(json.dumps(record))["effective_trial_count"] == pytest.approx(
        7.010726452101544, rel=1e-6
    )
