# Catalog Plan and Reader Guide

A single document combining three earlier files:

- **Reader guide** — curated learning paths through the catalog.
- **Quality audit** — which tutorials are thin and could be deepened.
- **Consolidation plan** — the more aggressive option of merging or
  cutting tutorials.

The catalog has **87 active tutorials** plus a small `_legacy/`
archive. They are organised by economic topic, but many readers come in
looking for a method or a research area. The first part of this
document gives them an entry point. The second and third parts are for
maintainers planning incremental improvements.

---

# Part 1: Reader Guide — Learning Paths

The catalog is organised by economic topic, but many readers come in
looking for a *method* and want a sequence. The paths below are
curated reading orders for the most common entry points. Each path is
4-7 tutorials with one sentence on what the next step adds.

## Continuous-time methods (HJB, KFE, optimal control)

For readers who want to learn the continuous-time toolkit.

1. **`optimal-control/ramsey-growth/`** — Ramsey planner solved by
   forward saddle-path shooting. Most economically transparent
   continuous-time entry point; introduces the planner problem.
2. **`optimal-control/phase-diagrams/`** — Same model in 2D state
   space, eigenanalysis at steady state, backward integration along
   the stable arm. Builds geometric intuition for what the HJB
   computes.
3. **`optimal-control/hjb-growth/`** — Same model again, now solved
   by upwind finite differences on the HJB. Derives the HJB from the
   discrete-time Bellman, justifies upwinding from information flow,
   and explains the implicit pseudo-time step.
4. **`heterogeneous-agents/huggett-incomplete-markets/`** — HJB
   coupled to a Kolmogorov-Forward equation for the household
   distribution. First example of HJB + KFE, with the operator
   duality made explicit.

## Discrete-time dynamic programming

For readers learning Bellman equations from scratch.

1. **`dynamic-programming/cake-eating/`** — Smallest possible Bellman
   problem. VFI on a fixed resource with a closed-form policy
   $c^{\ast}(W) = (1-\beta) W$ for sanity checking.
2. **`dynamic-programming/optimal-growth/`** — Same VFI machinery on
   Brock-Mirman growth with production. Shows the saving rate
   $s = \alpha \beta$ emerge from the Bellman fixed point.
3. **`dynamic-programming/shock-discretization/`** — Tauchen and
   Rouwenhorst as the discretisation step that lets stochastic
   shocks enter Bellman equations as finite Markov chains.
4. **`dynamic-programming/consumption-savings/`** — Buffer-stock
   saving with persistent income risk and a borrowing limit. The
   first Bellman with a non-trivial state space.
5. **`dynamic-programming/job-search-mccall/`** — Reservation-wage
   fixed point. Bellman as a single scalar equation.
6. **`dynamic-programming/aiyagari/`** — Aiyagari incomplete-markets
   equilibrium: Bellman plus stationary distribution plus a price
   that clears the bond market by bisection.
7. **`computational-methods/projection-methods/`** — Same Brock-Mirman
   growth model as step 2, but solved by Chebyshev polynomial
   projection of the policy rule. A different way to approximate.

## Heterogeneous-agent macro

For readers focused on incomplete-markets models with non-trivial
distributions.

1. **`heterogeneous-agents/endogenous-grid-points/`** — Buffer-stock
   saving by EGP. Inverts the Euler equation to skip the asset-choice
   search.
2. **`heterogeneous-agents/envelope-equation-iteration/`** — EEI
   variant that iterates on the marginal continuation value
   directly.
3. **`heterogeneous-agents/huggett-incomplete-markets/`** — Huggett
   equilibrium by HJB + KFE in continuous time. Bisection on the
   risk-free rate clears the bond market.
4. **`dynamic-programming/aiyagari/`** — Aiyagari capital-market
   clearing in discrete time, the Huggett analogue with capital and
   production.

## Linearised DSGE

For readers building first-order rational-expectations solutions.

1. **`computational-methods/perturbation-linearization/`** — The
   underlying Taylor expansion that DSGE solvers extend. Macro
   adjustment around a steady state.
2. **`dsge/rbc/`** — Linearised RBC by undetermined coefficients
   plus a Klein QZ cross-check.
3. **`dsge/nkdsge/`** — Sticky-price New Keynesian DSGE. Same
   linearisation framework, monetary shock IRFs.
4. **`dsge/behavioral-nk/`** — Cognitive-discounting variant; shows
   how a small change in the linearised system reshapes
   forward-guidance effects.
5. **`dsge/assetNews/`** — Lucas-tree dividend-news pricing as a
   first-order asset-pricing application of the same machinery.

## Global / nonlinear DSGE

For readers solving macro models with binding constraints or large
shocks.

1. **`global-dsge/rbc-capital-tax/`** — Global VFI on RBC with a
   capital-income tax wedge.
2. **`global-dsge/rbc-irreversible-investment/`** — Global VFI with
   an irreversibility constraint on investment.
3. **`global-dsge/heaton-lucas/`** — Risk sharing under portfolio
   constraints, by simultaneous policy/transition iteration.
4. **`global-dsge/deep-learning-optimal-growth/`** — Brock-Mirman
   solved by a JAX neural policy trained on Euler residuals. Shows
   how the same growth model can be approximated by a small network.

## Demand estimation (logit family)

For readers learning differentiated-products demand from clean to
hard.

1. **`choice/logit-discrete-choice/`** — Plain logit MLE on clean
   simulated choice data. Introduces softmax, IIA, and the
   likelihood contour.
2. **`industrial-organization/logit-supply-side/`** — Same logit
   demand, now with endogenous prices and unobserved quality. Berry
   inversion plus IV/2SLS, then markup recovery via Bertrand-Nash
   FOC.
3. **`choice/nested-logit/`** — Two-nest logit on cereal data with
   IV. The simplest break with IIA.
4. **`choice/mixed-logit-simulation/`** — Random coefficients via
   simulated MLE.
5. **`industrial-organization/blp-random-coefficients/`** —
   Differentiated-products BLP with the contraction mapping plus IV/GMM.
6. **`industrial-organization/merger-simulation/`** — Apply the
   demand estimates to a horizontal merger; HHI, GUPPI, and the
   Bertrand-Nash post-merger price increase.

## Structural estimation of dynamic discrete choice

For readers learning the Rust toolkit and its successors.

1. **`industrial-organization/dynamic-discrete-choice/`** — Rust bus
   replacement model. NFXP, CCP, and MPEC implemented side by side,
   plus a maximum-causal-entropy IRL benchmark.
2. **`structural-econometrics/q-learning-bus-engine/`** — Same Rust
   model recovered by soft Q-learning and a DQN. Shows that a
   sample-based RL learner reproduces NFXP's hazard without the
   transition matrix.
3. **`structural-econometrics/keane-wolpin-career-choice/`** —
   Multi-period dynamic discrete choice over education and
   occupation; finite-horizon Emax with regression interpolation.
4. **`structural-econometrics/dcegm-retirement-saving/`** — Discrete-
   continuous EGM for life-cycle retirement, where the kink at
   retirement becomes an upper-envelope step.

## Reinforcement learning in economics

For readers wanting to see RL applied to economic models.

1. **`dynamic-programming/q-learning-growth/`** — Tabular Q-learning
   on Brock-Mirman growth. Recovers the saving rule from sampled
   transitions.
2. **`structural-econometrics/q-learning-bus-engine/`** — Soft
   Q-learning recovers Rust's hazard. Shows that an entropy-
   regularised RL learner reproduces a structural-econometric
   estimator.
3. **`agent-based-models/algorithmic-collusion-q-learning/`** —
   Independent Q-learners in repeated Bertrand competition. Shows
   that decentralised learners can sustain supra-competitive prices.
4. **`global-dsge/deep-learning-optimal-growth/`** — Neural policy
   trained on Euler residuals. Bridge from RL to deep-learning
   global solvers.
5. **`game-theory/deep-optimal-auctions/`** — Deep mechanism-design
   network trained with a regret penalty. Closest the catalog gets
   to learning-based mechanism design.

## Search and matching

For readers building search-theoretic models.

1. **`dynamic-programming/job-search-mccall/`** — One-state
   reservation-wage fixed point.
2. **`dynamic-programming/diamond-mortensen-pissarides/`** — Two-sided
   matching equilibrium with free entry and Nash-bargained wages.
3. **`choice/sequential-search-ursu/`** — Sequential inspection model
   in a Weitzman-style stopping rule, applied to consumer choice.
4. **`computational-methods/simulation-based-estimation/`** — MSM and
   indirect inference applied to estimating the McCall reservation
   wage. Bridges search theory and structural estimation.

## Game theory and auctions

For readers building game-theoretic equilibria.

1. **`game-theory/normal-form-games/`** — Pure and mixed Nash in
   small payoff matrices.
2. **`game-theory/static-games/`** — Cournot quantities and the
   damped best-response iteration.
3. **`game-theory/quantal-response-equilibrium/`** — Logit QRE on
   market entry, the noisy-best-response fixed point.
4. **`game-theory/first-price-auctions/`** — Bayesian-Nash bidding
   under independent private values; deviation checks.
5. **`structural-econometrics/auction-valuation-recovery/`** —
   Inverse problem: recover the value distribution from observed
   first-price bids.
6. **`industrial-organization/dynamic-games/`** — Markov-perfect
   equilibrium in a two-firm quality ladder.
7. **`industrial-organization/dynamic-games-estimation/`** —
   Estimate the same game using CCPs to avoid resolving the MPE.
8. **`game-theory/deep-optimal-auctions/`** — Neural mechanism design
   with regret penalties, audited against Myerson.

## Filtering and time series

For readers doing forecasting, state-space estimation, or VARs.

1. **`time-series/fred-macro-data/`** — Business-cycle moments and
   HP filtering on a FRED-style panel.
2. **`time-series/ar-processes/`** — AR(1) impulse responses and the
   multiplier-accelerator recursion.
3. **`time-series/stock-watson/`** — Diffusion indexes via PCA for
   forecasting industrial production.
4. **`time-series/ridge-lasso-sparsity/`** — Shrinkage and selection
   for monetary-policy shock identification.
5. **`time-series/minnesota-svar/`** — Recursive SVAR with Minnesota
   priors.
6. **`computational-methods/kalman-filter/`** — Kalman filtering on a
   linear state-space model.
7. **`computational-methods/particle-filter/`** — Particle filter for
   nonlinear state-space; shows ESS-based proposal failure.

## Numerical methods foundations

For readers brushing up on the mechanical tools used everywhere
else.

1. **`numerical-methods/root-finding/`** — Scalar `f(x) = 0` solvers.
2. **`numerical-methods/interpolation/`** — Linear, cubic spline,
   and PCHIP fits.
3. **`numerical-methods/scalar-optimization-monopoly-pricing/`** —
   Grid search, golden section, Newton on a 1D profit function.
4. **`numerical-methods/constrained-optimization-kkt/`** — Projected
   gradient, log barrier, and SLSQP on a budget allocation.
5. **`numerical-methods/fixed-point-acceleration/`** — Picard,
   damped Picard, and Anderson on the BLP-style share inversion.
6. **`numerical-methods/global-search-multistart/`** — Multi-start,
   random search, simulated annealing on a non-convex profit.
7. **`computational-methods/numerical-optimization/`** — Custom
   Newton and BFGS restart grid on a latent-regime mixture
   likelihood.
8. **`computational-methods/metropolis-hastings/`** — Random-walk
   Metropolis on a two-regime structural posterior.

## Behavioural economics

For readers building behavioural variants of standard models.

1. **`choice/risk-aversion-monotone-choice/`** — CRRA risk aversion
   with monotone constraints on lottery shares.
2. **`choice/convex-time-budget-present-bias/`** — Quasi-hyperbolic
   beta-delta from continuous CTB allocations.
3. **`choice/probability-distortion-mixture/`** — Bruhin-Fehr-Duda-
   Epper finite-mixture EM on certainty equivalents.
4. **`choice/urn-behavioral-mixtures/`** — EM on decision-rule
   mixtures (Bayesian vs cutoff classifiers).
5. **`dsge/behavioral-nk/`** — Cognitive discounting in linearised
   New Keynesian DSGE.

## Same-model cross-references at a glance

Several models appear in multiple tutorials, each with a different
solver. Quick map:

| Model | Tutorials | What differs |
|---|---|---|
| Ramsey planner (continuous time) | `optimal-control/hjb-growth`, `optimal-control/phase-diagrams`, `optimal-control/ramsey-growth` | HJB upwind FD vs eigenanalysis + backward ODE vs forward shooting |
| Brock-Mirman growth | `dynamic-programming/optimal-growth`, `dynamic-programming/q-learning-growth`, `computational-methods/projection-methods`, `global-dsge/deep-learning-optimal-growth` | VFI vs tabular Q-learning vs Chebyshev projection vs neural policy |
| Rust bus replacement | `industrial-organization/dynamic-discrete-choice`, `structural-econometrics/q-learning-bus-engine` | NFXP/CCP/MPEC/IRL vs soft Q-learning + DQN |
| Plain logit demand | `choice/logit-discrete-choice`, `industrial-organization/logit-supply-side` | MLE on clean data vs Berry+IV+markup recovery |
| RBC | `dynamic-programming/rbc`, `dsge/rbc`, `global-dsge/rbc-capital-tax`, `global-dsge/rbc-irreversible-investment` | Discrete-time VFI vs linearisation vs global with tax wedge vs global with irreversibility |
| McCall search | `dynamic-programming/job-search-mccall`, `computational-methods/simulation-based-estimation` | Solve the model vs estimate it from data |

---

# Part 2: Quality Audit — Thin Tutorials That Deserve Deepening

The catalog has 86 active tutorials (excluding `_legacy/`). This audit
ranks them by combined size (run.py + README) and flags the ones whose
content is unusually thin relative to the rest of the catalog.

The premise: instead of merging adjacent tutorials, deepen the thin
ones. The recent `hjb-growth` and `huggett-incomplete-markets` updates
are the template — both moved from "states the equation" to "derives
the equation, justifies the discretisation, lists failure modes" and
roughly doubled in length. That made each piece more valuable as a
standalone reference.

## Catalog size distribution

Median total size (run.py + README): roughly **624 lines**.
75th percentile: roughly **800 lines**.
Recent reference deepenings:
- `optimal-control/hjb-growth`: 1007 lines (deepened May 2026)
- `heterogeneous-agents/huggett-incomplete-markets`: 1274 lines (deepened May 2026)

Both were under 700 lines before the deepening.

## Tutorials below the median (deepening candidates)

Sorted by total size ascending. The **Why thin** column is a one-line
diagnosis based on the file structure.

| Total | run.py | README | Tutorial | Why thin |
|---:|---:|---:|---|---|
| 344 | 229 | 115 | `game-theory/first-price-auctions` | Short README, 2 figures only. Could derive the optimal bid function from FOCs and add a deviation-test figure. |
| 369 | 269 | 100 | `game-theory/static-games` | Best-response derivation likely terse; could add a damped-iteration convergence-diagnostic figure. |
| 380 | 269 | 111 | `game-theory/quantal-response-equilibrium` | The QRE fixed point and its derivation from logit could get a proper $\lambda \to \infty$ Nash limit treatment. |
| 385 | 278 | 107 | `industrial-organization/vertical-relationships` | Double marginalisation result is canonical; the prose probably skips the derivation of the integrated-firm benchmark. |
| 399 | 271 | 128 | `industrial-organization/vertical-contracts` | Assortment exact enumeration is small; could add a parameter-sensitivity figure. |
| 414 | 309 | 105 | `industrial-organization/dynamic-games` | MPE fixed point on quality ladder; the convergence story and the value of investment-incentive comparison could be expanded. |
| 418 | 292 | 126 | `industrial-organization/theory-of-the-firm` | Hold-up model derivation; could add a hierarchy-cost-vs-incentives trade-off figure. |
| 423 | 299 | 124 | `industrial-organization/production-functions-markups` | Proxy-control regression is the headline; the markup-formula derivation and the bias correction could be more explicit. |
| 444 | 330 | 114 | `computational-methods/perturbation-linearization` | Taylor expansion exposition; could add the systematic comparison of first vs second order. |
| 467 | 343 | 124 | `dynamic-programming/cake-eating` | The simplest Bellman example. Short by design, but the audit-against-closed-form story could be a fuller "how to test a solver" subsection. |
| 469 | 355 | 114 | `choice/maximum-score-binary-choice` | Score function and smoothing; could derive the rate of convergence (cube-root) and the smoothed-score asymptotic distribution. |
| 475 | 363 | 113 | `game-theory/normal-form-games` | Pure/mixed Nash on small payoff matrices; could add a section on dominance and rationalisability. |
| 486 | 363 | 123 | `computational-methods/kalman-filter` | The Kalman filter derivation could include the projection interpretation and a small numerical example showing the gain. |
| 504 | 383 | 121 | `optimal-control/phase-diagrams` | Two figures, no tables. The eigenanalysis at steady state could be derived more explicitly with the Jacobian written out. |
| 508 | 382 | 126 | `computational-methods/metropolis-hastings` | MH derivation is canonical; could add the acceptance-ratio derivation and an effective-sample-size diagnostic figure. |
| 515 | 396 | 119 | `choice/revealed-price-preference` | Price-vector GAPP graph; could expand on when GARP and GAPP diverge and why. |
| 529 | 439 | 90 | `choice/money-pump-index` | README is unusually short (90 lines); the maximum-mean-cycle derivation deserves a fuller exposition. |
| 539 | 429 | 110 | `choice/bayesian-learning` | No tables; the belief-update derivation could be made more explicit and a Bayes-factor figure added. |
| 541 | 418 | 123 | `computational-methods/projection-methods` | Chebyshev projection; could derive the spectral convergence rate and add an error-vs-degree convergence figure. |
| 546 | 402 | 144 | `choice/risk-aversion-monotone-choice` | Could derive the monotone-constraint enforcement more carefully. |
| 550 | 412 | 138 | `choice/urn-behavioral-mixtures` | The EM derivation could be more explicit; a recovery-quality figure would help. |
| 554 | 416 | 138 | `optimal-control/ramsey-growth` | Shooting fully implemented but the Brent's-method convergence rate vs other root-finders could be added. |
| 555 | 397 | 158 | `dynamic-programming/shock-discretization` | Tauchen vs Rouwenhorst; could derive the moment-matching properties and add a tail-discretisation comparison. |
| 563 | 440 | 123 | `dsge/nkdsge` | Coefficient matching; the Klein QZ cross-check could be added explicitly. |
| 567 | 415 | 152 | `structural-econometrics/auction-valuation-recovery` | Bid-density inversion; could add the GPV bandwidth-choice discussion and a finite-sample bias plot. |
| 569 | 422 | 147 | `dynamic-programming/optimal-growth` | Standard VFI on Brock-Mirman; could explain the saving-rate $s = \alpha\beta$ derivation explicitly. |
| 575 | 459 | 116 | `choice/logit-discrete-choice` | The likelihood-contour figure is great; could add a closed-form check on the FOC-implied estimator and an information-matrix Cramer-Rao bound. |
| 575 | 482 | 93 | `choice/revealed-preference-afriat` | README is short; the Afriat-inequalities-to-utility step could be derived rather than asserted. |
| 586 | 450 | 136 | `dynamic-programming/solow-growth` | Could derive the convergence-rate formula and add a comparative-statics figure across savings rates. |
| 588 | 444 | 144 | `computational-methods/simulation-based-estimation` | MSM and indirect inference; could add a comparison of moment-choice impact and an asymptotic-variance discussion. |
| 591 | 476 | 115 | `choice/houtman-maks-rational-subsets` | Subset search; the algorithm derivation and the worst-case complexity discussion could be added. |
| 594 | 465 | 129 | `dsge/behavioral-nk` | Cognitive discounting; the deviation from the standard NK model could be derived line by line. |
| 596 | 430 | 166 | `numerical-methods/interpolation` | Linear/cubic spline/PCHIP; could add a clear discussion of when each shines and a pointwise error figure. |
| 605 | 498 | 107 | `computational-methods/numerical-optimization` | README is 107 lines — short for the depth of content (custom Newton implementation). The Hessian regularisation and line-search choices deserve more exposition. |
| 606 | 470 | 136 | `time-series/ar-processes` | AR(1) IRFs; could add the moving-average representation derivation and a roots-vs-stationarity discussion. |
| 610 | 449 | 161 | `agent-based-models/brock-hommes-asset-pricing` | Strategy-switching dynamics; the SMM moment-choice and the bifurcation parameter could be discussed. |

## Tutorials with unusually short README (under 110 lines)

These are the ones that look like "the run.py is fine, the README is
the bottleneck":

| README | Tutorial |
|---:|---|
| 90 | `choice/money-pump-index` |
| 93 | `choice/revealed-preference-afriat` |
| 100 | `game-theory/static-games` |
| 105 | `industrial-organization/dynamic-games` |
| 107 | `game-theory/first-price-auctions`, `industrial-organization/vertical-relationships`, `industrial-organization/logit-supply-side`, `computational-methods/numerical-optimization` |
| 110 | `choice/bayesian-learning` |

These are first-pass deepening targets: edit the `add_equations`,
`add_solution_method`, and `add_takeaway` strings in run.py to flesh
out the prose around the equations, regenerate, validate. The
`hjb-growth` deepening commit (`5c04910`) is the template.

## Recommended deepening priorities (top ten)

If only ten tutorials get deepened, my picks (highest pedagogical
upside relative to current state):

1. `dynamic-programming/cake-eating` — the canonical first Bellman
   example; an "audit your solver" subsection would be widely used.
2. `dynamic-programming/optimal-growth` — second Bellman example;
   the saving-rate derivation is too valuable to stay implicit.
3. `optimal-control/phase-diagrams` — the Jacobian-eigenanalysis
   derivation is the centrepiece and currently terse.
4. `optimal-control/ramsey-growth` — the shooting-vs-other-root-finders
   convergence comparison is informative and low effort.
5. `dsge/nkdsge` — adding the Klein QZ cross-check would mirror
   `dsge/rbc` and complete the linearised DSGE story.
6. `computational-methods/projection-methods` — derive spectral
   convergence and show the polynomial-degree error curve.
7. `dynamic-programming/shock-discretization` — Tauchen vs Rouwenhorst
   tail-fit comparison would be a one-figure addition with a clear
   teaching point.
8. `choice/logit-discrete-choice` — Cramer-Rao information bound and
   FOC-implied estimator check.
9. `game-theory/first-price-auctions` — derive the bid function from
   the equilibrium ODE and add the deviation-test figure.
10. `computational-methods/numerical-optimization` — the README is
    much shorter than the run.py, and the custom Newton
    implementation deserves more exposition (Hessian regularisation,
    line-search rationale).

## How a deepening commit looks

Use the `optimal-control/hjb-growth` deepening as the template
(commits `5c04910` and `c49e7ef`):

1. Read the existing `add_equations` and `add_solution_method` strings
   in `run.py`. Identify what's stated vs derived.
2. Expand them with: a derivation step ("from X by doing Y"), a
   justification step ("the alternative would fail because Z"), and
   a failure-mode line ("a naive implementation gets W wrong").
3. Re-run `python run.py` to regenerate the README and figures.
4. Run `python scripts/validate_catalog.py` from repo root.
5. Commit with a one-line title plus a paragraph explaining what the
   deepening clarified.

Each tutorial deepens independently. No catalog-wide refactor is
required.

---

# Part 3: Consolidation Options — More Aggressive Trimming

This section lists merger options for cases where the same economic
model is solved by three or four different tutorials. Each merger
collapses a cluster of tutorials into one comparative reference. None
of these have been executed yet (except the cut of
`continuous-cake-eating` in commit `300ebc0`); they are catalogued here
as a menu the maintainer can pick from.

The conservative alternative — leave folders alone, add cross-references
where two or three tutorials solve the same model — has already been
done (commits `b352b26` and `494fd7c`).

## Repo size benchmark

Reference tutorial sizes (run.py + README, lines):

| Tutorial | Total |
|---|---:|
| `dynamic-programming/cake-eating` | 467 |
| `industrial-organization/blp-random-coefficients` | 898 |
| `structural-econometrics/keane-wolpin-career-choice` | 1227 |

A consolidated tutorial in the **1000-1500 line range** is on the long
side but acceptable when it covers a genuine multi-method comparison.
Above 2000 lines is too long; that needs splitting.

## Merger option 1: Continuous-time Ramsey trio

**Sources** (all in `optimal-control/`):
- `hjb-growth/` — 730 + 274 = 1004 lines (recently deepened)
- `phase-diagrams/` — 381 + 121 = 502 lines
- `ramsey-growth/` — 413 + 138 = 551 lines

**Merged tutorial**: `optimal-control/ramsey-growth/`. Drop the other
two folders.

**Title**: "Ramsey Growth: HJB Upwinding, Saddle-Path Shooting, and
Phase Diagrams"

**Post-merge size estimate**: ~750 lines run.py + ~400 lines README =
**~1150 lines**.

**What survives**: one unified model setup; the deepened HJB exposition;
eigenanalysis + backward integration; forward shooting; three
`### Method N` subsections; one results section; one comparison table
(grid points, runtime, accuracy).

**What gets cut**: two duplicate model-setup sections (~200 lines);
five of the nine current figures; two duplicate steady-state
derivations.

**Why this is meatier**: the same model solved three ways back to back.
The deepened HJB exposition becomes one section of a comparative
reference.

**Risk**: the cross-references already added in commit `b352b26` solve
most of the discoverability problem at near-zero cost. The merge would
lose the citability of `optimal-control/hjb-growth/` as a standalone
HJB tutorial.

## Merger option 2: Bus-engine pair → six-estimator Rust tutorial

**Sources**:
- `industrial-organization/dynamic-discrete-choice/` — NFXP, CCP, MPEC, IRL
- `structural-econometrics/q-learning-bus-engine/` — soft Q-learning, DQN

**Merged tutorial**: `industrial-organization/dynamic-discrete-choice/`.
Drop the q-learning folder.

**Title**: "Bus Engine Replacement: Six Estimators on Rust's Model"

**Post-merge size estimate**: **~1300 lines**.

**What survives**: one Rust model setup + simulated data; six
estimators as subsections (NFXP, CCP, MPEC, IRL, soft Q-learning, DQN);
one headline figure showing the recovered hazard from all six; one
comparison table.

**What gets cut**: duplicate Rust mileage state and dataset
(~150 lines); duplicate "intuition for replacement" prose; two of the
duplicate hazard-curve figures.

**Why this is meatier**: becomes the canonical multi-estimator
showcase for a structural model. No comparable resource exists in any
public econ-ML repo I am aware of.

**Risk**: requires moving content across topic folders
(`structural-econometrics/` → `industrial-organization/`), which the
plan otherwise avoids.

## Merger option 3: Logit demand pair

**Sources**:
- `choice/logit-discrete-choice/` — 575 lines
- `industrial-organization/logit-supply-side/` — 716 lines

**Merged tutorial**: `industrial-organization/logit-supply-side/`.

**Title**: "Logit Demand: From MLE to Berry Inversion and Markup
Recovery"

**Post-merge size estimate**: **~1100 lines**.

**Pedagogical risk**: the deep code reads showed the two have distinct
scopes (single-market MLE vs multi-market IV+markup recovery). Merging
forces a demand reader through supply machinery and a supply reader
past basics they already know.

**Recommendation**: probably skip. Cross-reference suffices.

## Merger option 4: Cake-eating + optimal-growth

**Sources**:
- `dynamic-programming/cake-eating/` — 467 lines
- `dynamic-programming/optimal-growth/` — 569 lines

**Merged tutorial**: `dynamic-programming/optimal-growth/`.

**Title**: "Bellman Fundamentals: Cake-Eating and Optimal Growth by VFI"

**Post-merge size estimate**: **~900 lines**.

**Why this is meatier**: cake-eating becomes the closed-form sanity
check inside a more substantive Bellman tutorial. The reader sees the
algorithm work twice on increasingly realistic problems.

**Risk**: cake-eating is often cited as the canonical "first Bellman"
example. Folding it into a longer tutorial reduces its citability.

## Merger option 5: Multi-start global optimisation pair

**Sources**:
- `numerical-methods/global-search-multistart/` — 1005 lines
- `computational-methods/numerical-optimization/` — 605 lines

**Merged tutorial**: `numerical-methods/global-search-multistart/`.

**Title**: "Global Optimisation: Multi-Start, Random Search, and
Annealing on Two Non-Convex Objectives"

**Post-merge size estimate**: **~1650 lines** — exceeds the recommended
1500-line ceiling.

**Recommendation**: probably skip on size grounds alone. Cross-reference
instead.

## Merger option 6: Linearised DSGE pair (RBC + NK)

**Sources**:
- `dsge/rbc/` — linearised RBC by Klein QZ
- `dsge/nkdsge/` — sticky-price New Keynesian by coefficient matching

**Merged tutorial**: `dsge/linearised-dsge/` (new slug).

**Title**: "Linearised DSGE: RBC and New Keynesian by Klein QZ"

**Post-merge size estimate**: **~1200 lines**.

**Why this is meatier**: a reader interested in linearised DSGE
currently sees two parallel tutorials that re-derive the Klein QZ
machinery. The merged version teaches the machinery once and uses it
on two economically distinct models.

**Risk**: requires a new folder slug (`linearised-dsge`) and breaks two
existing URLs.

## Outright cut (already done)

### `optimal-control/continuous-cake-eating/` → `_legacy/`

**Status**: done in commit `300ebc0`.

The Pontryagin maximum principle appears nowhere else in the catalog;
the costate intuition is fully covered by HJB ($V'(k)$ is the shadow
price); the economics duplicate discrete-time cake-eating.

## No-touch list

Consolidations considered and rejected.

| Cluster | Why keep separate |
|---|---|
| Q-learning trio (growth + bus + collusion) | Three distinct applications: toy MDP, structural recovery, repeated oligopoly. |
| Search tutorials (McCall + DMP + Ursu + simulation-based-estimation) | McCall is acceptance, DMP is two-sided equilibrium, Ursu is sequential inspection, simulation-based-estimation is the *econometric* problem. |
| Mixed-logit + nested-logit + BLP | Three genuinely different methods for handling IIA violations. |
| RBC variants (RBC, capital-tax, irreversible, projection, deep-learning) | Each adds a real economic friction or a different solver paradigm. |
| Heterogeneous-agents trio (EGP + EEI + Huggett) | Three substantially different solvers with their own mathematical machinery. |
| Auction tutorials (first-price + auction-valuation-recovery + deep-optimal-auctions) | Different problems: bidding, identification, mechanism design. |

## Two paths forward

**Conservative path (already taken)**: cut `continuous-cake-eating`,
add cross-references, write learning paths and quality audit. Catalog
goes from 88 to 87 tutorials. No URL breakage. Most discoverability
benefit captured at near-zero risk.

**Aggressive path (optional)**: execute the six mergers above. Catalog
goes from 87 to 80 tutorials. Bigger trim, real risk of URL breakage
and prose loss.

The deepening priorities in Part 2 work under either path. They are
the form of "make pieces meatier" that does not require any folder
rearrangement.

## How to act on this document

For maintainers picking this up cold:

1. Read Part 1 to understand the catalog as it is today.
2. Read Part 2 to see which tutorials are thin; pick one or two to
   deepen.
3. Read Part 3 to see merger options; pick zero or one to execute.
4. Each deepening or merger is a separate commit with the format
   shown in the deepening template (Part 2 closing).
5. After any change, run `python scripts/validate_catalog.py` from
   the repo root.
