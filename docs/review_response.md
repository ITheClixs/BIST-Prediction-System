# Response to an external review

An external review of the manuscript raised eighteen objections. Each was checked
against the artifacts rather than against the prose. This document records what
was found, what changed, and what is still open, including the objections that
turned out to be wrong.

Verification code for every number below is in
`src/bist_predict/research/inference/detectability.py` and
`tests/test_research/test_inference_detectability.py`.

## Confirmed errors, now corrected

### 1. The breadth conclusion was false

**Claimed:** "no long-only top-$k$ rule fed by a forecast of this quality is
profitable at any breadth."

**Found:** false. Proposition 1 requires
$\rho > c / (\sigma \lambda(N,k))$, and $\lambda$ behaves differently in two
regimes. At a fixed holding *fraction* $q = k/N$ it converges to
$\varphi(\Phi^{-1}(1-q))/q$, which is what the figure drew and what supported the
prose. At a fixed holding *count* it diverges: $\lambda(N,1) \sim \sqrt{2\log N}$.
At the pooled correlation the run reported, $\lambda(500,1) = 3.037$ gives a
requirement of $0.0350$, below the $0.0387$ achieved — so sufficient breadth does
restore feasibility in the model.

**Fixed:** both regimes are now derived, stated as a second proposition with a
proof, drawn in the figure, and pinned by
`test_breadth_at_a_fixed_holding_count_does_not_converge`, which reproduces the
counterexample directly. `breadth_for_feasibility` reports the universe a design
would need — here in excess of $10^6$ names — instead of asserting impossibility.

### 2. The feasibility bound used the wrong correlation

**Claimed:** the design needed an information coefficient of $0.3098$ and
achieved $0.0387$, a shortfall of eight.

**Found:** $0.0387$ is a Pearson correlation pooled over 480 stock-sessions.
Proposition 1 selects *within* a session, so the quantity it needs is the
per-session cross-sectional correlation. That is $0.0092$ with a standard error
of $0.0549$ across the 120 sessions — indistinguishable from zero. Pooling
inflates it roughly fourfold because it also absorbs the common time-series
component, which a ranking rule cannot exploit.

**Fixed:** both are computed and reported; the bound is instantiated on the
per-session value. The shortfall is 34x, not 8x, and the conclusion is stronger
than the one originally published.

### 3. The panel ceiling was instantiated on the wrong quantity

**Claimed:** the design effect follows from the correlation of the target.

**Found:** the standard error of a Diebold–Mariano test is a function of the
cross-sectional dependence of the loss differential $d_{i,t}$, not of the raw
return. Measured directly, the differential correlates at $0.5662$ against the
target's $0.5697$ — close enough that the headline barely moves (independent rows
per session $1.4823$ rather than $1.4765$), but they are different objects and
only one enters the test.

**Fixed:** the differential correlation is computed per candidate and used for the
ceiling; the target correlation remains reported as the panel diagnostic it is.

### 4. The deflated-Sharpe "permanent floor" claim was unsound

**Claimed:** the search threshold does not shrink with the record length, so a
strategy below it can never establish skill.

**Found:** $SR^{\ast}_0 = \sqrt{V}\,q(N)$, and the behaviour depends entirely on
what $V$ is. Under Lo's sampling variance $(1 + SR^2/2)/n$ the threshold falls
like $n^{-1/2}$: at $n = 1200$ it is $0.0697$, at $n = 12000$ it is $0.0220$. The
floor reading needs $V$ to be persistent heterogeneity in *true* Sharpe ratios,
which the realised cross-trial variance is not — it mixes heterogeneity,
estimation noise and unequal evaluation windows.

**Fixed:** the remark is replaced by the distinction, the threshold is reported at
the observed sample size only, and the test that asserted the old behaviour —
which tested an assumption rather than a property — is replaced by one pinning the
actual $n^{-1/2}$ scaling.

### 5. The cost-sensitivity cases are not a frozen ledger

**Claimed:** the three cost cases hold decisions fixed so that "only the bill
moves."

**Found:** gross return moves by $5\times10^{-4}$ across the cases ($7.7671\%$,
$7.7694\%$, $7.7188\%$) while the trade count stays at 143. The decisions are
fixed; the capital path is not, so position sizing responds to the cash previous
costs consumed. The breakeven multiplier is also a linear interpolation through a
nonlinear simulation.

**Fixed:** the manuscript now says exactly this, describes the strictly frozen
ledger as the cleaner counterfactual that is *not* what is computed, and labels
the breakeven figure an approximation. Implementing the frozen counterfactual is
listed below as open work.

### 6. Overstated economic and testing language

`"the gross edge is real"` is now stated as a realised-sample outcome behind a
forecast that is not distinguishable from the null. The modified SPA statistic is
labelled a variant of Hansen's test whose finite-sample properties are not
established here, rather than being reported under Hansen's name alone. The
power calculation is labelled an order-of-magnitude approximation, with its three
weaknesses named: central-$t$ inversion, a post-hoc selected standard error, and
an $n^{-1/2}$ extrapolation across sixty years.

## Objections that do not hold

**The trading threshold is not 1 basis point.** The review argued that a strategy
trading whenever predicted return exceeds `decision_cost_rate = 0.0001` and then
evaluating at 20.2 bp is incoherent. The selection rule applies
`max(decision_cost_rate, round_trip_estimate)`, so the binding hurdle is 20.20 bp
(`portfolio_backtest.py`, `_estimated_round_trip_cost_rate`). The 1 bp figure is a
floor that never binds; 306 of 480 signals are rejected against the 20.20 bp
hurdle. The configuration key is misleadingly named and that is worth fixing, but
the economics are not incoherent.

**The effective trial count was already labelled a diagnostic.** The manuscript
carries a remark stating that it matches expected maxima and not the distribution
of the maximum, and that it should not be read as a distributional equivalence.
The criticism is fair as a warning; it does not describe an unqualified claim.

## Open, and not addressed here

These are real and are not repaired by any edit to the text.

- **Scale.** Four stocks and 120 evaluated sessions do not support a benchmark
  claim. A point-in-time investable universe over multiple years is required.
- **Simulation-calibrated inference.** No end-to-end null and alternative
  simulation exists. Type-I error, power and family-wise error for the whole
  stack — DM, Holm, Reality Check, SPA, PSR/DSR — are unmeasured under this
  design's dependence and tail behaviour.
- **Joint resampling of the search.** Configurations are compared across
  different evaluation windows. The correct object is the empirical distribution
  of the maximum statistic under synchronized resampling on common dates.
- **Nested-model tests.** Squared-error comparison of a fitted model against a
  zero forecast conflates estimation noise with absence of information; a
  Clark–West-style test answers the predictive-content question the current
  design does not.
- **Cost calibration.** The schedule is a declared scenario. It is not calibrated
  to BIST quotes, depth, auction mechanics or broker fees, and no claim about
  what a desk would pay is defensible without them.
- **Frozen-ledger counterfactual.** Not implemented; see item 5.
- **Development search.** The 72-configuration grid measures the declared
  sensitivity sweep, not the full researcher search over targets, features,
  universes and cost assumptions.

Until those are addressed this is a prototype and a methods note, not a
benchmark, and it is described that way.
