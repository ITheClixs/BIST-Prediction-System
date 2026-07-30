r"""What this experiment could have found, had it been there to find.

A negative result is only as informative as the design that produced it.  The
tests in the sibling modules answer "is there evidence of skill?"; every one of
them answers *no* here.  That leaves the question those tests cannot answer:
would they have said *yes* for an effect of the size anyone actually expects?

Three quantities settle it, and each is computed from the run's own artifacts
rather than assumed.

**Detectable accuracy.**  The Diebold-Mariano statistic is a mean divided by its
standard error.  Inverting the test at conventional size and power gives the
smallest mean loss differential the design could have separated from zero, and
dividing that by the null's mean squared error puts it on the scale of
out-of-sample :math:`R^2` --- the scale forecasting papers report.

**The design effect of a correlated cross-section.**  Widening the cross-section
is the usual answer to a short sample.  It buys less than it appears to when the
units move together.  For a mean over :math:`k` units that are equicorrelated at
:math:`\bar\rho`, the variance is inflated by :math:`1 + (k-1)\bar\rho`, so one
session is worth

.. math::
    \frac{k}{1 + (k - 1)\bar\rho}

independent observations: increasing in :math:`k`, but bounded above by
:math:`1/\bar\rho` however many names are added.

The quantity that governs the precision of a *forecast comparison* is the
cross-sectional correlation of the model-versus-null loss differential
:math:`d_{i,t}`, not of the raw return.  They are different objects and only the
first one enters the standard error of the test that is actually run, so both are
computed and the loss differential is the one used.  Applying an equicorrelated
approximation to either is a simplification of a correlation matrix whose
realised entries vary, and the resulting ceiling is an average-case statement.

**The cost floor on breadth.**  Breadth buys something the design effect does
not: selectivity.  Ranking :math:`N` names and holding the best :math:`k`
concentrates the forecast in its right tail, and the strength of that
concentration is the mean of the top-:math:`k` standard normal order statistics.
Proposition 1 turns this into a lower bound on the cross-sectional information
coefficient a strategy needs before costs --- a hurdle no amount of statistical
significance can substitute for.

    **Proposition 1.**  Let the forecast :math:`\hat r` and the realised return
    :math:`r \sim \mathcal N(0, \sigma^2)` be jointly normal with correlation
    :math:`\rho`.  A long-only rule that ranks :math:`N` names by :math:`\hat r`,
    holds the top :math:`k` in equal weight for one period and pays a round-trip
    cost :math:`c` on notional has positive expected net return only if

    .. math::
        \rho > \frac{c}{\sigma \, \lambda(N, k)},
        \qquad
        \lambda(N, k) = \frac{1}{k} \sum_{i = N-k+1}^{N} \mathbb{E}[Z_{(i:N)}].

    *Proof.*  Conditioning a bivariate normal on its ranked component gives
    :math:`\mathbb E[r \mid \hat r] = \rho \sigma \, (\hat r - \mu_{\hat r}) /
    \sigma_{\hat r}`, so the expected realised return of a name selected at
    standardised forecast score :math:`z` is :math:`\rho \sigma z`.  Selection by
    rank is a function of the scores alone; averaging over the top :math:`k` of
    :math:`N` gives :math:`\rho \sigma \lambda(N, k)`.  Net of cost the
    expectation is :math:`\rho \sigma \lambda(N,k) - c`, and requiring it to
    exceed zero rearranges to the claim. :math:`\square`

**The bound is not an impossibility result, and the two breadth regimes differ.**
This matters, because it is easy to read the proposition as saying that no
universe is wide enough.  It does not.

*Fixed holding fraction* :math:`q = k/N`.  Here
:math:`\lambda \to \varphi(\Phi^{-1}(1-q))/q`, the mean of the upper-:math:`q`
tail of the standard normal, which is finite.  The requirement converges, so
past a few dozen names extra breadth at a fixed fraction buys almost nothing and
only a more selective rule helps.

*Fixed holding count* :math:`k`.  Here :math:`\lambda(N,k)` grows without bound,
like :math:`\sqrt{2 \log N}` for :math:`k = 1`, so the requirement falls to zero
and **sufficient breadth at fixed concentration always restores feasibility in
this idealised model.**  Any claim that breadth cannot help is false as a
statement about the model; what is true is a statement about *attainable*
breadth, which is why :func:`breadth_for_feasibility` reports the universe size
required rather than asserting impossibility.

The bound is deliberately generous to the strategy: it ignores estimation error
in the ranking, charges cost only once per round trip, assumes an unbiased
forecast, and --- most importantly in the fixed-:math:`k` regime --- ignores
capacity, participation limits and market impact, all of which bind hard on a
rule that puts the whole book into one name.  A design that fails the bound fails
for reasons no model choice can fix; a design that clears it has cleared a
necessary condition, not a sufficient one.

**Which correlation instantiates** :math:`\rho`.  Proposition 1 selects within a
session, so :math:`\rho` is the *per-session cross-sectional* correlation between
forecast and realised return.  A correlation pooled over stock-sessions is a
different quantity: it also absorbs the common time-series component, so a model
that only tracks the market direction earns pooled correlation while ranking
nothing.  Both are reported, and the per-session mean is the one compared against
the bound.

References: Lo (2002) and Bailey and Lopez de Prado (2014) for the Sharpe-ratio
inversion, Grinold (1989) for the breadth argument, David and Nagaraja (2003) for
the order-statistic machinery, and Cohen (1988) for the conventional 80% power
target used when inverting a two-sided test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from scipy import integrate, special, stats

from bist_predict.research.inference.sharpe import EULER_MASCHERONI

__all__ = [
    "DetectabilityReport",
    "breadth_for_feasibility",
    "detectability_report",
    "effective_trial_count",
    "expected_top_selection_score",
    "false_strategy_quantile",
    "minimum_detectable_mean",
    "panel_information_ceiling",
    "required_information_coefficient",
    "sampling_search_threshold",
    "sessions_required_for_effect",
    "sharpe_required_for_confidence",
    "tail_mean_selection_score",
]

_MAX_SESSIONS = 10_000_000
_SOLVER_TOLERANCE = 1e-10


def minimum_detectable_mean(
    standard_error: float,
    *,
    observations: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    r"""Return the smallest mean a two-sided t-test would reject the null for.

    The Diebold-Mariano statistic refers ``mean / standard_error`` to a Student
    distribution on ``observations - 1`` degrees of freedom.  A true mean of
    :math:`\delta` is therefore rejected with probability ``power`` when

    .. math::
        \delta = \left(t_{1-\alpha/2,\,n-1} + t_{\text{power},\,n-1}\right)
                 \cdot \mathrm{SE},

    which is the usual power calculation carried out on the test that was
    actually run rather than on a normal approximation to it.
    """
    if standard_error <= 0.0:
        raise ValueError("the standard error must be positive")
    if observations < 3:
        raise ValueError("a power calculation needs at least three observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0.5 <= power < 1.0:
        raise ValueError("power must lie in [0.5, 1)")
    degrees = observations - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees))
    shift = float(stats.t.ppf(power, degrees))
    return (critical + shift) * standard_error


def sessions_required_for_effect(
    effect: float,
    *,
    standard_error: float,
    observations: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    r"""Return the session count at which ``effect`` becomes detectable.

    The standard error of a mean falls as :math:`n^{-1/2}`, so rescaling the
    observed standard error to a hypothetical sample size is a division by
    :math:`\sqrt{n / n_0}`.  The critical values also move with the sample size,
    so the answer is found by search rather than by the closed-form
    :math:`n_0 (\delta_0/\delta)^2`, which understates the requirement slightly
    at small :math:`n`.
    """
    if effect <= 0.0:
        raise ValueError("the target effect must be positive")
    if standard_error <= 0.0:
        raise ValueError("the standard error must be positive")

    def detectable_at(count: int) -> float:
        scaled = standard_error * math.sqrt(observations / count)
        return minimum_detectable_mean(scaled, observations=count, alpha=alpha, power=power)

    if detectable_at(observations) <= effect:
        return observations
    low, high = observations, observations
    while detectable_at(high) > effect:
        high *= 2
        if high > _MAX_SESSIONS:
            return _MAX_SESSIONS
    while high - low > 1:
        middle = (low + high) // 2
        if detectable_at(middle) > effect:
            low = middle
        else:
            high = middle
    return high


def panel_information_ceiling(
    mean_pairwise_correlation: float, unit_count: int
) -> dict[str, float]:
    r"""Return how much independent information one session of the panel carries.

    Under the equicorrelated approximation the variance inflation factor of a
    :math:`k`-unit cross-section is :math:`1 + (k-1)\bar\rho`, so a session
    contributes :math:`k / (1 + (k-1)\bar\rho)` independent observations.  That
    expression is increasing in :math:`k` and converges to :math:`1/\bar\rho`,
    which is the most a session can ever be worth at that correlation.  The
    ``headroom`` field is the factor by which an unlimited universe would
    improve on the one actually used.
    """
    if not 0.0 < mean_pairwise_correlation < 1.0:
        raise ValueError("the mean pairwise correlation must lie strictly in (0, 1)")
    if unit_count < 2:
        raise ValueError("a cross-section needs at least two units")
    achieved = unit_count / (1.0 + (unit_count - 1) * mean_pairwise_correlation)
    ceiling = 1.0 / mean_pairwise_correlation
    return {
        "mean_pairwise_correlation": float(mean_pairwise_correlation),
        "unit_count": int(unit_count),
        "independent_rows_per_session": float(achieved),
        "independent_rows_per_session_ceiling": float(ceiling),
        "headroom": float(ceiling / achieved),
        "standard_error_headroom": float(math.sqrt(ceiling / achieved)),
    }


def expected_top_selection_score(universe_size: int, selected: int) -> float:
    r"""Return the mean of the top-``selected`` standard normal order statistics.

    Summing :math:`k` separate order-statistic expectations costs :math:`k`
    quadratures and loses accuracy as they accumulate.  Exchangeability collapses
    them into one.  A given draw lies in the top :math:`k` of :math:`N` exactly
    when at most :math:`k-1` of the other :math:`N-1` draws exceed it, so

    .. math::
        \lambda(N, k) = \frac{N}{k} \int_0^1 \Phi^{-1}(u)\,
            \Pr\!\left[\mathrm{Bin}(N-1,\,1-u) \le k-1\right] \mathrm{d}u
        = \frac{N}{k} \int_0^1 \Phi^{-1}(u)\, I_u(N-k,\,k)\, \mathrm{d}u,

    with :math:`I_u` the regularised incomplete beta function.  Selecting the
    whole universe leaves nothing to select on and returns exactly zero, which
    is the mean of the parent distribution.
    """
    if universe_size < 1:
        raise ValueError("the universe must contain at least one name")
    if not 1 <= selected <= universe_size:
        raise ValueError("the selected count must lie between one and the universe size")
    if selected == universe_size:
        return 0.0

    def integrand(u: float) -> float:
        if u <= 0.0 or u >= 1.0:
            return 0.0
        inclusion = float(special.betainc(universe_size - selected, selected, u))
        return float(stats.norm.ppf(u)) * inclusion

    value, _error = integrate.quad(integrand, 0.0, 1.0, limit=400)
    return universe_size * value / selected


def tail_mean_selection_score(holding_fraction: float) -> float:
    r"""Return the limit of :math:`\lambda(N, qN)` as the universe grows.

    Holding a fixed fraction :math:`q` of an expanding universe converges on
    selecting the upper-:math:`q` tail of the score distribution, whose mean is

    .. math::
        \lim_{N \to \infty} \lambda(N, qN)
        = \frac{\varphi\!\left(\Phi^{-1}(1-q)\right)}{q}.

    The limit is finite, which is why breadth at a fixed holding fraction stops
    helping.  It is the fixed-*count* regime, where no such limit exists, that
    keeps the bound from being an impossibility result.
    """
    if not 0.0 < holding_fraction < 1.0:
        raise ValueError("the holding fraction must lie strictly in (0, 1)")
    quantile = float(stats.norm.ppf(1.0 - holding_fraction))
    return float(stats.norm.pdf(quantile) / holding_fraction)


def breadth_for_feasibility(
    *,
    round_trip_cost_rate: float,
    target_volatility: float,
    information_coefficient: float,
    selected: int = 1,
    maximum_universe: int = 1_000_000,
) -> int | None:
    """Return the smallest universe that satisfies Proposition 1 at a fixed hold count.

    Because :math:`\\lambda(N, k)` is unbounded in :math:`N` for fixed :math:`k`,
    such a universe always exists in the model; the useful question is whether it
    is remotely attainable.  ``None`` means the answer exceeds
    ``maximum_universe``, which is the honest way to say "not in any real market"
    without claiming the bound can never be met.
    """
    if information_coefficient <= 0.0:
        return None
    required = round_trip_cost_rate / (target_volatility * information_coefficient)
    universe = max(selected + 1, 2)
    while universe <= maximum_universe:
        if expected_top_selection_score(universe, selected) > required:
            low, high = max(selected + 1, universe // 2), universe
            while high - low > 1:
                middle = (low + high) // 2
                if expected_top_selection_score(middle, selected) > required:
                    high = middle
                else:
                    low = middle
            return high
        universe *= 2
    return None


def required_information_coefficient(
    *,
    round_trip_cost_rate: float,
    target_volatility: float,
    universe_size: int,
    selected: int,
) -> dict[str, float]:
    """Return the information coefficient Proposition 1 demands, and its inputs.

    The result is a floor, not a target: a strategy whose realised correlation
    with next-period returns sits below it loses money in expectation however
    the forecast is produced, and however significant it is.
    """
    if round_trip_cost_rate <= 0.0:
        raise ValueError("the round-trip cost rate must be positive")
    if target_volatility <= 0.0:
        raise ValueError("the target volatility must be positive")
    score = expected_top_selection_score(universe_size, selected)
    if score <= 0.0:
        raise ValueError("the selection rule must concentrate on the upper tail")
    return {
        "round_trip_cost_rate": float(round_trip_cost_rate),
        "target_volatility": float(target_volatility),
        "universe_size": int(universe_size),
        "selected": int(selected),
        "selection_score": float(score),
        "required_information_coefficient": float(
            round_trip_cost_rate / (target_volatility * score)
        ),
    }


def false_strategy_quantile(trial_count: float) -> float:
    r"""Return the standardised expected maximum of ``trial_count`` null trials.

    This is the bracketed term of the False Strategy Theorem,

    .. math::
        (1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
        + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right),

    factored out of the threshold so that the trial count and the dispersion of
    the trials can be varied independently.
    """
    if trial_count <= 1.0:
        raise ValueError("the expected maximum is only defined beyond a single trial")
    upper = float(stats.norm.ppf(1.0 - 1.0 / trial_count))
    tail = float(stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e)))
    return (1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * tail


def effective_trial_count(
    *,
    trial_count: int,
    realised_trial_variance: float,
    independent_trial_variance: float,
) -> float:
    r"""Return how many independent searches the observed grid behaves like.

    The False Strategy Theorem assumes the trials are independent draws.  A
    configuration grid is not: neighbouring configurations reuse most of the same
    sessions, so their Sharpe ratios move together and disperse less than
    independent trials would.  Plugging the *realised* dispersion into the
    theorem already absorbs that, but it hides how strong the dependence is.

    Solving

    .. math::
        \sqrt{V_{\text{realised}}}\; q(N)
        = \sqrt{V_{\text{independent}}}\; q(N_{\text{eff}})

    for :math:`N_{\text{eff}}` recovers it: the number of genuinely independent
    trials that would have produced the same search threshold.  Because
    :math:`q` is strictly increasing, the solution is unique and is found by
    bisection.
    """
    if trial_count < 2:
        raise ValueError("an effective trial count needs at least two trials")
    if realised_trial_variance <= 0.0 or independent_trial_variance <= 0.0:
        raise ValueError("both trial variances must be positive")
    target = false_strategy_quantile(trial_count) * math.sqrt(
        realised_trial_variance / independent_trial_variance
    )
    low, high = 1.0 + 1e-9, float(trial_count)
    if false_strategy_quantile(high) <= target:
        return float(trial_count)
    while false_strategy_quantile(low) > target:
        low = 1.0 + (low - 1.0) / 10.0
        if low - 1.0 < 1e-15:
            return 1.0
    while high - low > _SOLVER_TOLERANCE * high:
        middle = 0.5 * (low + high)
        if false_strategy_quantile(middle) > target:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def sharpe_required_for_confidence(
    *,
    threshold: float,
    observations: int,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    r"""Return the per-period Sharpe ratio at which the deflated ratio clears ``confidence``.

    The deflated Sharpe ratio is the probabilistic Sharpe ratio evaluated at the
    search threshold, so asking what would have counted as evidence means
    inverting

    .. math::
        \Phi\!\left[\frac{(\widehat{SR} - SR^*_0)\sqrt{n-1}}
        {\sqrt{1 - \hat\gamma_3 \widehat{SR}
        + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right] = c

    in :math:`\widehat{SR}`.  The left side is continuous and strictly
    increasing in :math:`\widehat{SR}` wherever the radicand stays positive, so
    bisection above the threshold is sufficient.  ``kurtosis`` is the
    non-excess fourth moment, matching the convention in the sibling module.
    """
    if observations < 3:
        raise ValueError("the inversion needs at least three observations")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between one half and one")
    scale = math.sqrt(observations - 1)
    target = float(stats.norm.ppf(confidence))

    def shortfall(sharpe: float) -> float:
        variance = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
        if variance <= 0.0:
            return -math.inf
        return (sharpe - threshold) * scale / math.sqrt(variance) - target

    low = threshold
    high = threshold + 1.0
    while shortfall(high) < 0.0:
        high += 1.0
        if high > threshold + 100.0:
            raise ValueError("no finite Sharpe ratio reaches the requested confidence")
    while high - low > _SOLVER_TOLERANCE:
        middle = 0.5 * (low + high)
        if shortfall(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def sampling_search_threshold(
    *,
    per_period_sharpe: float,
    observations: int,
    trial_count: int,
) -> float:
    r"""Return the search threshold implied by *sampling* variance alone.

    The False Strategy threshold is :math:`\sqrt{V}\,q(N)`, and what :math:`V`
    stands for decides how the threshold behaves as the record lengthens.  Under
    Lo's (2002) iid expression :math:`V = (1 + \widehat{SR}^2/2)/n`, so the
    threshold falls like :math:`n^{-1/2}`: a longer record does lower the bar the
    search sets.  The threshold is only a floor that a longer record cannot erode
    if :math:`V` measures persistent heterogeneity in the *true* Sharpe ratios of
    the configurations, which is a different quantity.

    The realised cross-trial variance the run plugs in is neither: it mixes true
    heterogeneity, estimation noise, and the fact that the trials cover different
    evaluation windows.  Both readings are therefore reported, and neither is
    extrapolated to a sample size the run did not observe.
    """
    if observations < 2:
        raise ValueError("a sampling variance needs at least two observations")
    variance = (1.0 + per_period_sharpe * per_period_sharpe / 2.0) / observations
    return float(math.sqrt(variance) * false_strategy_quantile(float(trial_count)))


@dataclass(frozen=True)
class DetectabilityReport:
    """The bounds on what the accepted design was capable of establishing."""

    alpha: float
    power: float
    session_count: int
    reference_model: str
    reference_standard_error: float
    benchmark_mean_squared_error: float
    minimum_detectable_loss_differential: float
    minimum_detectable_r_squared: float
    reference_r_squared: float
    sessions_required_for_reference_r_squared: int
    panel: dict[str, float]
    feasibility: dict[str, float]
    pooled_information_coefficient: float
    session_information_coefficient: float
    session_information_coefficient_standard_error: float
    feasible_breadth_at_unit_holding: int | None
    per_period_sharpe: float
    grid_maximum_sharpe: float
    deflated_sharpe_threshold: float
    per_period_sharpe_required: float
    annualised_sharpe_required: float
    trial_count: int
    realised_trial_variance: float
    independent_trial_variance: float
    effective_trial_count: float
    independent_trial_threshold: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record of the detectability bounds."""
        return {
            "alpha": self.alpha,
            "power": self.power,
            "session_count": self.session_count,
            "reference_model": self.reference_model,
            "reference_standard_error": self.reference_standard_error,
            "benchmark_mean_squared_error": self.benchmark_mean_squared_error,
            "minimum_detectable_loss_differential": self.minimum_detectable_loss_differential,
            "minimum_detectable_r_squared": self.minimum_detectable_r_squared,
            "reference_r_squared": self.reference_r_squared,
            "sessions_required_for_reference_r_squared": (
                self.sessions_required_for_reference_r_squared
            ),
            "panel": dict(self.panel),
            "feasibility": dict(self.feasibility),
            "pooled_information_coefficient": self.pooled_information_coefficient,
            "session_information_coefficient": self.session_information_coefficient,
            "session_information_coefficient_standard_error": (
                self.session_information_coefficient_standard_error
            ),
            "feasible_breadth_at_unit_holding": self.feasible_breadth_at_unit_holding,
            "per_period_sharpe": self.per_period_sharpe,
            "grid_maximum_sharpe": self.grid_maximum_sharpe,
            "deflated_sharpe_threshold": self.deflated_sharpe_threshold,
            "per_period_sharpe_required": self.per_period_sharpe_required,
            "annualised_sharpe_required": self.annualised_sharpe_required,
            "trial_count": self.trial_count,
            "realised_trial_variance": self.realised_trial_variance,
            "independent_trial_variance": self.independent_trial_variance,
            "effective_trial_count": self.effective_trial_count,
            "independent_trial_threshold": self.independent_trial_threshold,
        }


def detectability_report(
    *,
    session_standard_errors: Mapping[str, float],
    benchmark_mean_squared_error: float,
    session_count: int,
    dependence: Mapping[str, object],
    sharpe: Mapping[str, object],
    grid_maximum_sharpe: float,
    periods_per_year: int,
    round_trip_cost_rate: float,
    target_volatility: float,
    pooled_information_coefficient: float,
    session_information_coefficient: float,
    session_information_coefficient_standard_error: float,
    loss_differential_correlation: float,
    universe_size: int,
    selected: int,
    reference_r_squared: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.80,
) -> DetectabilityReport:
    """Assemble the detectability bounds from quantities the run already produced.

    The reference model is the candidate with the smallest standard error --- the
    one the design had the best chance of separating from the null.  Quoting the
    bound for any other candidate would flatter the design.
    """
    if not session_standard_errors:
        raise ValueError("at least one candidate standard error is required")
    if benchmark_mean_squared_error <= 0.0:
        raise ValueError("the benchmark mean squared error must be positive")
    if not 0.0 < reference_r_squared < 1.0:
        raise ValueError("the reference R-squared must lie strictly in (0, 1)")

    reference_model = min(session_standard_errors, key=lambda name: session_standard_errors[name])
    reference_error = float(session_standard_errors[reference_model])
    detectable = minimum_detectable_mean(
        reference_error, observations=session_count, alpha=alpha, power=power
    )
    detectable_r_squared = detectable / benchmark_mean_squared_error
    required_sessions = sessions_required_for_effect(
        reference_r_squared * benchmark_mean_squared_error,
        standard_error=reference_error,
        observations=session_count,
        alpha=alpha,
        power=power,
    )

    unit_count = int(cast(int, dependence["unit_count"]))
    per_period_sharpe = float(cast(float, sharpe["per_period_sharpe"]))
    trial_count = int(cast(int, sharpe["trial_count"]))
    realised_variance = float(cast(float, sharpe["trial_sharpe_variance"]))
    independent_variance = (1.0 + per_period_sharpe * per_period_sharpe / 2.0) / session_count
    threshold = float(cast(float, sharpe["deflated_sharpe_threshold"]))
    required_sharpe = sharpe_required_for_confidence(
        threshold=threshold,
        observations=session_count,
        skewness=float(cast(float, sharpe["skewness"])),
        kurtosis=float(cast(float, sharpe["kurtosis"])),
    )

    return DetectabilityReport(
        alpha=alpha,
        power=power,
        session_count=session_count,
        reference_model=reference_model,
        reference_standard_error=reference_error,
        benchmark_mean_squared_error=float(benchmark_mean_squared_error),
        minimum_detectable_loss_differential=float(detectable),
        minimum_detectable_r_squared=float(detectable_r_squared),
        reference_r_squared=float(reference_r_squared),
        sessions_required_for_reference_r_squared=int(required_sessions),
        # The precision of the forecast comparison is governed by the dependence
        # of the loss differential, not of the raw target, so the ceiling is
        # instantiated on the former; the target correlation stays alongside it
        # because it is what the panel diagnostic reports.
        panel={
            **panel_information_ceiling(loss_differential_correlation, int(unit_count)),
            "target_correlation": float(cast(float, dependence["mean_pairwise_correlation"])),
            "loss_differential_correlation": float(loss_differential_correlation),
        },
        feasibility=required_information_coefficient(
            round_trip_cost_rate=round_trip_cost_rate,
            target_volatility=target_volatility,
            universe_size=universe_size,
            selected=selected,
        ),
        pooled_information_coefficient=float(pooled_information_coefficient),
        session_information_coefficient=float(session_information_coefficient),
        session_information_coefficient_standard_error=float(
            session_information_coefficient_standard_error
        ),
        # Fixed-count breadth makes the requirement vanish in the limit, so the
        # honest answer to "could a wider universe fix this?" is a number, not a
        # denial. None means it exceeds a million names.
        feasible_breadth_at_unit_holding=breadth_for_feasibility(
            round_trip_cost_rate=round_trip_cost_rate,
            target_volatility=target_volatility,
            information_coefficient=session_information_coefficient,
        ),
        per_period_sharpe=per_period_sharpe,
        grid_maximum_sharpe=float(grid_maximum_sharpe),
        deflated_sharpe_threshold=threshold,
        per_period_sharpe_required=float(required_sharpe),
        annualised_sharpe_required=float(required_sharpe * math.sqrt(periods_per_year)),
        trial_count=trial_count,
        realised_trial_variance=realised_variance,
        independent_trial_variance=float(independent_variance),
        effective_trial_count=effective_trial_count(
            trial_count=trial_count,
            realised_trial_variance=realised_variance,
            independent_trial_variance=independent_variance,
        ),
        independent_trial_threshold=float(
            math.sqrt(independent_variance) * false_strategy_quantile(trial_count)
        ),
    )
