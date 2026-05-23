# Missing Preliminary Readings: Gap-Analysis Proposals

**Status: DONE 2026-05-22.** All 14 ranked P0+P1+P2 proposals from this doc shipped as Wave 1 (see `spec.md > Wave 1`). This file is now historical; kept for the gap-analysis methodology and the deferred / rejected proposals at the bottom.

Generated 2026-05-21. Companion to commit `20b0242`, which added
`## Preliminary readings` sections to 21 dense tutorials by linking
*existing* simpler siblings. This report lists 15 proposed NEW
tutorials that would fill the remaining stepping-stone gaps.

Pipeline: 4 Sonnet explorers per subject cluster surveyed assumed-but-
unserved concepts in each dense tutorial. An Opus judge de-duplicated,
merged overlapping proposals, dropped weak ones, and ranked the
survivors by reach × severity.

This is a triage list, not an authoring queue. The user reviews and
decides which to greenlight as tutorial folders.

## Ranked proposals

### P0 — high reach and load-bearing

#### 1. `optimal-control/upwind-finite-differences/`
Upwind finite-difference schemes and state-constraint Kuhn-Tucker.
- **Description.** Numerical scheme for first-order PDEs / HJB on a bounded state space, with the borrowing-constraint multiplier from the KT clip.
- **Reach.** `optimal-control/hjb-growth/`, `heterogeneous-agents/huggett-incomplete-markets/`, `heterogeneous-agents/aiyagari-hact/`.
- **Severity.** Three dense tutorials currently re-derive the upwind sign convention and the boundary KT condition inline. Single most reused continuous-time HA primitive.
- Foundational.

#### 2. `heterogeneous-agents/kolmogorov-forward-equation/`
KFE / Fokker-Planck and the A vs A-transpose duality.
- **Description.** Stationary density solve as the null space of the transposed HJB generator; mass conservation derivation; sparse linear-system solve with normalisation. Includes the continuous-time Markov generator Q for the income process as a worked example (absorbs Sonnet's standalone CT-Markov proposal).
- **Reach.** `huggett-incomplete-markets/`, `aiyagari-hact/`, `sequence-space-jacobian-hank/`.
- **Severity.** The transpose-operator trick is the load-bearing identity behind every continuous-time HA stationary distribution; currently re-explained every time. Pairs with #1.
- Intermediate.

#### 3. `structural-econometrics/gmm-identification-and-weighting/`
GMM moments, identification, and optimal weighting (merged).
- **Description.** Unified GMM prelim: moment conditions, Hansen 2-step, efficient W, just- versus over-identification.
- **Reach.** `adversarial-estimation/`, `mixed-logit-simulation/`, `industrial-organization/logit-supply-side/`, `agent-based-models/brock-hommes-asset-pricing/`, `computational-methods/simulation-based-estimation/`.
- **Severity.** Five dense tutorials assume GMM machinery without a catalog source. Highest reach in the set.
- Foundational.

#### 4. `dsge/blanchard-kahn-determinacy/`
BK conditions and QZ / Schur partitioning.
- **Description.** Counting stable eigenvalues; generalized Schur for non-invertible systems; what happens when BK fails (indeterminacy, explosiveness).
- **Reach.** `dsge/rbc/`, `dsge/nkdsge/`, `dsge/behavioral-nk/`, `computational-methods/perturbation-linearization/`.
- **Severity.** Every linearized DSGE in the repo invokes BK; QZ is non-obvious and never derived. Bridges perturbation to DSGE.
- Bridge.

#### 5. `bayesian-methods/bayesian-foundations/`
Bayes rule, priors, likelihood, conjugacy.
- **Description.** Scalar posterior update derivation; conjugacy and the Gaussian-Gaussian conjugate linear regression as a worked example (absorbs Sonnet's standalone conjugate-regression proposal); prior sensitivity and posterior predictive.
- **Reach.** `computational-methods/metropolis-hastings/`, `computational-methods/hamiltonian-monte-carlo/`, `numerical-methods/bayesian-optimization/`, `structural-econometrics/bayesian-dsge-hmc/`, `time-series/minnesota-svar/`.
- **Severity.** `bayesian-methods/` is the entry-point subject and currently has only one tutorial. This is its missing foundation.
- Foundational.

### P1 — high local severity, narrower reach

#### 6. `computational-methods/mcmc-diagnostics/`
ESS, R-hat, integrated autocorrelation time, trace plots.
- **Reach.** `metropolis-hastings/`, `hamiltonian-monte-carlo/`, `bayesian-dsge-hmc/`, `minnesota-svar/`.
- **Severity.** Four MCMC tutorials cite ESS / R-hat without definition. Small tutorial, large compression.
- Intermediate.

#### 7. `time-series/reduced-form-var/`
VAR(p) estimation and Cholesky identification, without priors.
- **Reach.** `time-series/minnesota-svar/`, `time-series/stock-watson/`.
- **Severity.** `minnesota-svar/` jumps directly to BVAR shrinkage; the OLS-VAR + Cholesky baseline has no home.
- Foundational.

#### 8. `industrial-organization/bertrand-ownership-matrix/`
Multi-product Bertrand FOC in matrix form with ownership matrix Ω.
- **Description.** `p = c - (Ω ⊙ ∂s/∂p)^{-1} s` derivation; fixed-point price solver under alternative ownership structures.
- **Reach.** `merger-simulation/`, `logit-supply-side/`, `nash-in-nash/`.
- **Severity.** The ownership-Hadamard trick is the operational core of all three; currently inlined.
- Intermediate.

#### 9. `choice/simulated-likelihood-and-halton-draws/`
MSL, common random numbers, Halton sequences.
- **Reach.** `mixed-logit-simulation/`, `rum-choice-networks/`, `adversarial-estimation/`, `blp-random-coefficients/`.
- **Severity.** Fixed-seed CRN and Halton-versus-pseudorandom are silently assumed; a prelim removes paragraphs from each.
- Intermediate.

#### 10. `structural-econometrics/neural-networks-for-economists/`
Feedforward NN basics, training, regularization.
- **Reach.** `rum-choice-networks/`, `adversarial-estimation/`, `game-theory/deep-optimal-auctions/`.
- **Severity.** Three dense tutorials each give a half-page recap of cross-entropy + Adam + L2; consolidating is a clear win.
- Bridge.

#### 11. `heterogeneous-agents/young-distribution-iteration/`
Young (2010) lottery method for discrete-time HA distributions.
- **Reach.** `sequence-space-jacobian-hank/`, `endogenous-grid-points/`, `dynamic-programming/aiyagari/`.
- **Severity.** Lottery weights are the discrete-time analog of the KFE; SSJ-HANK invokes them without a source.
- Intermediate.

### P2 — single-target, cheap, optional

#### 12. `structural-econometrics/weitzman-search-rule/`
Reservation-value index and Pandora's box optimal-search ordering.
- **Reach.** `choice/sequential-search-ursu/`.
- **Severity.** Single-tutorial reach, but the Weitzman index is load-bearing and otherwise undefined anywhere in the catalog. Cheap.
- Foundational.

#### 13. `computational-methods/quadrature-and-numerical-differentiation/`
Gauss-Hermite, Simpson, finite-difference gradients (merged).
- **Reach.** `smolyak-sparse-grids/`, `hamiltonian-monte-carlo/`, `bayesian-optimization/`, `shock-discretization/`.
- **Severity.** Gauss-Hermite nodes are the standard `E[·]` under AR(1) discretization and are nowhere defined as a method tutorial.
- Intermediate.

#### 14. `numerical-methods/gaussian-processes/`
GP regression, kernels, marginal likelihood.
- **Reach.** `numerical-methods/bayesian-optimization/`.
- **Severity.** Single-target. BO's surrogate is opaque without it. Borderline; keep if the GP write-up stays short.
- Intermediate.

#### 15. `game-theory/regret-matching-and-no-regret/`
Regret matching and no-regret dynamics as the algorithmic prelim to CFR.
- **Reach.** `game-theory/cfr-asymmetric-auction/`.
- **Severity.** Single dense target, but CFR's convergence claim is hollow without it.
- Foundational.

## Rejected or absorbed proposals

| Proposal | Disposition | Reason |
|---|---|---|
| structural-econometrics/emax-interpolation/ | Reject | Too narrow; one section inside keane-wolpin suffices. |
| structural-econometrics/dc-egm-upper-envelope/ | Reject | Single-tutorial sub-step; belongs inside `dcegm-retirement-saving/`. |
| structural-econometrics/identification-simulation-models/ | Reject | Vague; identification is model-specific. Subsumed by GMM tutorial plus per-tutorial prose. |
| bayesian-methods/normalizing-flows/ | Reject (defer) | Only two consumers; both already explain MAF inline. Revisit when a third appears. |
| computational-methods/abc-rejection/ | Reject | Toy stepping stone; `simulation-based-estimation/` can host a short rejection-ABC subsection. |
| computational-methods/gauss-hermite-quadrature/ | Merged into #13. | |
| numerical-methods/numerical-differentiation-integration/ | Merged into #13. | |
| heterogeneous-agents/continuous-time-markov-income/ | Absorbed into #2 as worked example. | |
| structural-econometrics/gmm-optimal-weighting/ | Merged into #3. | |
| structural-econometrics/gmm-moment-conditions/ | Merged into #3. | |
| industrial-organization/upward-pricing-pressure/ | Reject | Policy screens, not a method gap; `merger-simulation/` can host a short section. |
| bayesian-methods/conjugate-linear-regression/ | Absorbed into #5 as worked example. | |

Net: 6 merges/absorptions, 5 outright rejections, 15 surviving
proposals.

## Suggested next moves

- Greenlight P0 items (5 tutorials). They have the highest reach and
  most compressive effect on the dense pages already in the catalog.
- After each new tutorial lands, re-run the prereq sweep on the
  affected dense tutorials so the new folder appears in their
  `## Preliminary readings`. The mechanical insertion script is at
  `/tmp/insert_prelim.py` from commit `20b0242`'s session and can be
  adapted.
- P2 items are cheap singletons; batch them once a P0 or P1 author is
  context-loaded on the relevant subject.
