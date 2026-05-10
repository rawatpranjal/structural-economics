# Learning Paths

The catalog has 87 tutorials. They are organised by economic topic, but
many readers come in looking for a *method* and want a sequence. The
paths below are curated reading orders for the most common entry
points. Each path is 4-7 tutorials with one sentence on what the next
step adds.

These paths complement the topic-level catalog in the root README;
neither replaces the other.

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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
