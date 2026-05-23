# spec.md - Computational Economics Authoring Spec

This spec covers all authoring waves for the tutorial catalog.

- **Wave 1 - Preliminary-Readings Curriculum.** Closed. 14 prelims shipped 2026-05-22.
- **Wave 2 - Behavioral and Dynamic Learning Tutorials.** Open. 3 primary + 1 addendum tutorials, design ported from `plan.md` on 2026-05-22. `plan.md` is now a stub.

See each wave section below.

## Wave 1: Preliminary-Readings Curriculum (P0 + P1 + P2)

**Date:** 2026-05-21 (spec); 2026-05-22 (all 14 prelims shipped). **Scope:** all 14 prelims (P0 5 + P1 5 + P2 4). **Status:** complete on `main` (commits `51e25ae`, `b36454f`, `f602b3d`, `4bac3ed`).

This wave defines fourteen new tutorials that fill the gaps in the
catalog's "on-ramp" to dense tutorials. Companion documents:

- Gap analysis (input): `docs/missing-preliminary-readings.md`.
- Commit that wired up `## Preliminary readings` sections to the
  21 dense tutorials: `20b0242`.
- Style + folder contracts the new tutorials must follow:
  `STYLE_GUIDE.md` and `CLAUDE.md`.

P2 entries are cheap singletons (each serves 1-4 dense tutorials)
and were added in this same v1 of the spec because their research
overhead is small. The curriculum is now fully spec'd; tutorial
authoring is a separate pass.

## Notational alignment (HARD RULE for every prelim)

Each prelim must use the *same notation* as the dense tutorials it
serves so that a reader walking from prelim to dense tutorial sees
no symbol drift. Before authoring a prelim, the author reads the
Equations and Model Setup sections of every dense tutorial in its
"serves" list and pins the following symbols verbatim:

- State and policy symbols (e.g. asset grid `a`, capital `k`,
  consumption `c`, value `v`, density `g`).
- Operator names (e.g. upwind generator `A`, its transpose `Aᵀ`,
  Markov generator `Q`).
- Parameter letters (`ρ`, `β`, `σ`, `γ` - read the dense tutorial's
  symbol table, don't invent).
- Indexing conventions (asset index `i`, income index `j`,
  time-step `n`).
- Borrowing limit or boundary symbol (`a_min`, `k_min`, `\underline a`).

The author lists the symbols imported from each dense tutorial in
the prelim's Model Setup table with a `[from <dense tutorial>]`
annotation. New symbols introduced by the prelim get defined and
flagged "[prelim introduces this; dense tutorials adopt it on
cut]". When a dense tutorial currently uses a different symbol for
the same object, the cut PR for that tutorial renames the symbol
to match the prelim (this is part of the cut, not a separate
refactor).

Concretely, by prelim:

- **Upwind FD (#1).** Symbols `a` (asset), `k` (capital), `v`,
  `D⁺`, `D⁻`, `A`, `\underline a`, `\underline k` are pinned from
  `hjb-growth` and `huggett-incomplete-markets`. Drift symbol `s`
  on the asset grid follows Huggett's convention.
- **KFE (#2).** Density `g`, generator `A` (shared with #1), `Aᵀ`,
  income generator `Q`, normalisation row index `i*` - pinned from
  Huggett and Aiyagari-HACT. Young-lottery weights `Λ` follow
  the SSJ tutorial's symbol.
- **GMM foundations (#3).** Parameter `θ`, moment vector `g(θ,x)`,
  Jacobian `G`, weighting `W`, variance `Ω` follow the standard
  Hayashi notation that the dense tutorials use. The SMM moment
  vector reuses `m̃(θ) - m_data` from
  `simulation-based-estimation` and `brock-hommes-asset-pricing`.
- **Blanchard-Kahn (#4).** RE system letters `A`, `B`, `C` and
  state vector `y_t` follow Klein-style notation already in
  `dsge/rbc`. QZ matrices `S`, `T` and stable-root partition
  notation follow the same.
- **Bayesian foundations (#5).** Posterior `p(θ|y)`, likelihood
  `p(y|θ)`, prior `p(θ)`, prior parameters
  `α, β` (Beta-Binomial), `V₀⁻¹, b₀` (Gaussian) follow
  `metropolis-hastings` and `minnesota-svar` exactly so the
  refactor lands without renaming.
- **MCMC diagnostics (#6).** Chain length `T`, lag-`t`
  autocorrelation `ρ_t`, integrated autocorrelation time
  `τ = 1 + 2 Σ ρ_t`, effective sample size `ESS = T/τ` - pinned
  from the existing `metropolis-hastings` definition block.
  `R̂` symbol and within/between variance notation `W, B` follow
  Gelman-Rubin convention.
- **Reduced-form VAR (#7).** VAR lag matrices `A_1, ..., A_p`,
  reduced-form residual `u_t`, residual covariance `Σ_u`,
  Cholesky factor `P` (so `Σ_u = P Pᵀ`), structural shock
  `ε_t = P⁻¹ u_t`, impulse response `Φ_j`. Notation matches
  `minnesota-svar` so the cut PR is rename-free.
- **Bertrand ownership matrix (#8).** Price vector `p`, marginal
  cost `c`, market shares `s(p)`, demand Jacobian `Δ(p) =
  ∂s/∂p`, ownership matrix `Ω` (single-product `Ω_{jj} = 1`,
  multi-product `Ω_{jk} = 1` for co-owned products). FOC
  `p = c - (Ω ⊙ Δ)⁻¹ s` follows Nevo and BLP supply-side
  notation, matching `merger-simulation` and `logit-supply-side`.
- **Simulated likelihood (#9).** Number of draws `R`, draw
  index `r`, simulated probability `P̂(θ) = (1/R) Σ_r f(θ; ξ_r)`,
  fixed draws `{ξ_r}`. Halton sequence `{h_n}` and pseudorandom
  `{u_n}` follow Train (2009) Chapter 9 notation as used in
  `mixed-logit-simulation` and `blp-random-coefficients`.
- **Neural networks regression (#10).** Inputs `x`, hidden
  activations `h`, weights `W` (input-to-hidden) and `W̃`
  (hidden-to-output), biases `b, b̃`, regularisation strength
  `λ`. Activation `tanh` matches `rum-choice-networks` and
  `adversarial-estimation`. Loss `L(W, W̃)` and gradient steps
  follow Goodfellow et al. Chapter 6 notation.
- **Weitzman search (#11).** Reservation value `z_j`, inspection
  cost `c_j`, value distribution `F_j(v)`, current best `b`
  follow Weitzman (1979) and the existing
  `sequential-search-ursu` symbol table.
- **Quadrature (#12).** Nodes `{ξ_n}`, weights `{w_n}`, node
  count `N`. AR(1) conditional expectation `E[f(z')|z]` with
  innovation `σ_ε`. Hermite-polynomial notation `H_n(x)` from
  Stroud-Secrest. Matches `smolyak-sparse-grids` pseudocode.
- **Gaussian processes (#13).** Inputs `X`, observations `y`,
  mean function `μ(x)`, kernel `k(x, x')`, signal variance `σ_f²`,
  length scale `ℓ`, noise `σ_n²`, posterior mean `μ_*(x_*)`,
  posterior variance `σ_*²(x_*)`. Matches Rasmussen-Williams
  and the existing `bayesian-optimization` notation.
- **Regret matching (#14).** Instantaneous regret `r_i^t`,
  cumulative regret `R_i^T`, regret-matching probability
  `π_i^t ∝ max(R_i^t, 0)`, time-average strategy `π̄_i`.
  Matches Hart-Mas-Colell (2000) and the existing
  `cfr-asymmetric-auction` notation.

Cross-tutorial collisions to watch:

- `A` is the upwind generator in continuous-time HA (prelims #1,
  #2 and Huggett/HACT) AND the RE-system lead matrix in DSGE
  (prelim #4 and dsge/rbc). The collision is unavoidable; each
  prelim uses `A` in its own scope. The Model Setup table makes
  the scope explicit.
- `θ` appears as a Bayesian parameter (#5) and a GMM parameter
  (#3). Same letter, same meaning - fine.
- `σ` is income volatility in HA tutorials, prior scale in
  Bayesian-foundations, and posterior std in MH. Each prelim
  uses the dense-tutorial convention and labels it on first use.
- `W` is the GMM weighting matrix (#3), the input-layer NN
  weights (#10), and the within-chain variance in R̂ (#6).
  Three different scopes; each prelim labels its `W` on first
  use.
- `Ω` is the ownership matrix in IO (#8), the GMM moment-
  covariance in (#3), and Greek-lowercase `ω` is a separate
  weight in some demand models. Scope-local; labelled on first
  use.
- `R` is the number of simulation draws (#9) and the rejection
  count in Beta-Binomial (#5). Disjoint scopes.

## Style template (applies to every prelim)

Every prelim tutorial is a full catalog entry: it satisfies the
Tutorial Contract (`run.py`, hand-maintained `README.md`,
`figures/thumb.png`, optional `tables/`), follows the standard
section order (Overview → Equations → Model Setup → Solution
Method → Results → Takeaway → References), uses code-fence math
syntax exclusively, has no em-dashes, and obeys the Learned Rules
in `CLAUDE.md`. Each prelim is a real tutorial with executable
output, not a primer-only Markdown file. Reader is RA/pre-PhD
through early-PhD econ, per `STYLE_GUIDE.md > Reader`.

Prelims follow the same prose discipline as the rest of the
catalog: economic object in the subject position, examples before
notation, one core method per tutorial, equations map to results.
Overview is prose-only; all math lives in Equations onward.

Catalog placement (root `README.md` insertions) is settled in the
gap report and reproduced in each entry's `Catalog row` field.

## Curriculum map (P0 + P1 + P2)

| # | Prelim | Subject section | Serves | Priority |
|---|---|---|---|---|
| 1 | `optimal-control/upwind-finite-differences/` | Continuous-Time Macro / Optimal Control | hjb-growth, huggett-incomplete-markets, aiyagari-hact | P0 |
| 2 | `heterogeneous-agents/kolmogorov-forward-equation/` | Heterogeneous Agents | huggett-incomplete-markets, aiyagari-hact, sequence-space-jacobian-hank | P0 |
| 3 | `structural-econometrics/gmm-foundations/` | Structural Econometrics | adversarial-estimation, simulation-based-estimation, brock-hommes-asset-pricing (+ light pointer in mixed-logit, logit-supply-side) | P0 |
| 4 | `dsge/blanchard-kahn-determinacy/` | Linearized DSGE | dsge/rbc, dsge/nkdsge, perturbation-linearization | P0 |
| 5 | `bayesian-methods/bayesian-foundations/` | Bayesian Methods (entry point) | metropolis-hastings (major refactor), minnesota-svar, plus pointers from HMC, BO, bayesian-dsge-hmc | P0 (refactor-flagged) |
| 6 | `computational-methods/mcmc-diagnostics/` | Bayesian Methods view | metropolis-hastings, hamiltonian-monte-carlo, bayesian-dsge-hmc | P1 |
| 7 | `time-series/reduced-form-var/` | Time Series | minnesota-svar, stock-watson | P1 |
| 8 | `industrial-organization/bertrand-ownership-matrix/` | Industrial Organization | merger-simulation, logit-supply-side | P1 |
| 9 | `numerical-methods/simulated-likelihood/` | Numerical Methods | mixed-logit-simulation, rum-choice-networks, adversarial-estimation, blp-random-coefficients | P1 |
| 10 | `numerical-methods/neural-networks-regression/` | Numerical Methods | rum-choice-networks, adversarial-estimation, deep-optimal-auctions | P1 |
| 11 | `choice/weitzman-search-rule/` | Choice and Demand | sequential-search-ursu | P2 |
| 12 | `numerical-methods/quadrature/` | Numerical Methods | smolyak-sparse-grids (+ light pointer in shock-discretization, bayesian-optimization, bayesian-dsge-hmc) | P2 |
| 13 | `numerical-methods/gaussian-processes/` | Numerical Methods | bayesian-optimization | P2 |
| 14 | `game-theory/regret-matching/` | Computational Game Theory | cfr-asymmetric-auction | P2 |

---

## 1. `optimal-control/upwind-finite-differences/`

**Title.** Upwind Finite Differences for First-Order PDEs and the
HJB State Constraint.

**Description (catalog row).** A drift-driven 1D HJB on a bounded
interval is discretised with one-sided finite differences. Sign of
the policy-implied drift selects forward or backward, and a
Kuhn-Tucker clip enforces the state constraint at the boundary.

**Scope - in.** Forward and backward finite-difference operators;
the upwind selection rule; sparse upwind generator matrix `A`;
borrowing-limit Kuhn-Tucker clip; viscosity-solution motivation
(one paragraph); CFL-like stability note (one sentence).
**Scope - out.** Higher-order schemes (WENO, ENO); semi-Lagrangian
methods; PDE theory beyond a one-paragraph motivation.

**Dense tutorials served + concept supplied.**
- `optimal-control/hjb-growth/`: the forward/backward operators
  and the upwind rule.
- `heterogeneous-agents/huggett-incomplete-markets/`: the state-
  constraint Kuhn-Tucker clip and the upwind discretisation on
  the asset grid.
- `heterogeneous-agents/aiyagari-hact/`: indirectly served via
  the existing pointer to Huggett (no extra cut needed there).

**Cut list.**
- `optimal-control/hjb-growth/README.md`, `## Equations > ###
  Upwind finite-difference discretisation`, ~lines 104-138:
  cut the generic FD operator + upwind rule + zero-drift
  fallback exposition. Replace with one sentence pointing at the
  prelim.
- `optimal-control/hjb-growth/README.md`, `## Equations > ###
  Boundary conditions`, ~lines 142-148: cut the generic boundary
  forcing rule. Replace with one sentence.
- `heterogeneous-agents/huggett-incomplete-markets/README.md`,
  `## Equations > ### The borrowing limit as a state constraint`,
  ~lines 99-114: cut the KT clip derivation. Replace with one
  sentence.
- `heterogeneous-agents/huggett-incomplete-markets/README.md`,
  `## Solution Method > ### Upwind discretisation of the HJB`,
  opening paragraphs ~lines 199-225: cut the generic FD/upwind
  exposition. Keep the asset-grid-specific generator construction
  in the lines that follow.
- `heterogeneous-agents/aiyagari-hact/README.md`: no cut; the
  existing pointer to Huggett covers it.

**References (canonical, from Haiku research).**
1. Achdou, Han, Lasry, Lions, Moll (2022), "Income and Wealth
   Distribution in Macroeconomics: A Continuous-Time Approach,"
   *Review of Economic Studies* 89(1), 45-86. Online appendix on
   HJB discretisation + state-constraint enforcement. Primary
   anchor for upwinding in HA macro.
2. Achdou & Capuzzo-Dolcetta (2010), "Mean Field Games: Numerical
   Methods," *SIAM Journal on Numerical Analysis* 48(4), 1136-1162.
   §3 establishes stability and consistency of upwind schemes for
   first-order HJB.
3. Crandall, Evans & Lions (1984), "Some Properties of Viscosity
   Solutions of Hamilton-Jacobi Equations," *Trans. AMS* 282(2),
   487-502. §2-3 justify monotone schemes converging to viscosity
   solutions.
4. LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*,
   Cambridge. Chapter 4 §4.2 on upwind methods for advection
   (pedagogical anchor).

**Equations to include.**
- Forward / backward / central differences on a 1D grid.
- The upwind rule: pick `D⁺` if drift > 0, `D⁻` if drift < 0.
- The sparse generator matrix `A` (tridiagonal under one-state
  drift, with diagonals from the upwind branches).
- The Kuhn-Tucker state-constraint condition at the lower
  boundary, with the resulting clip on the policy.

**`run.py` sketch.** Toy 1D HJB `ρv = u(c) + v'(k)(f(k) − c)` on
`[k_min, k_max]` with `c = (u')⁻¹(v'(k))`, log utility, no shocks.
Implicit upwind solve to fixed point. Figures: (a) value function
+ drift sign coloured by which difference was used; (b) error vs
analytic when the analytic policy is available; (c) failure mode
under naive central differences. Thumbnail from (a).

**Catalog row.** Insert at the top of "Continuous-Time Macro and
Optimal Control" in root `README.md`, immediately above
`optimal-control/hjb-growth/`.

---

## 2. `heterogeneous-agents/kolmogorov-forward-equation/`

**Title.** Kolmogorov Forward Equation and the Stationary
Wealth Distribution.

**Description (catalog row).** Mass conservation gives the KFE
for the cross-sectional density. The discretised forward operator
is the transpose of the upwind HJB generator. A sparse linear
solve plus a normalisation row delivers the stationary
distribution.

**Scope - in.** KFE / Fokker-Planck derivation from mass
conservation; the operator-duality `A` vs `Aᵀ`; sparse stationary
solve `Aᵀ g = 0` with a normalisation row; continuous-time Markov
generator `Q` for a Poisson income process as a worked subexample;
brief comparison to Young (2010) lottery iteration as the
discrete-time analog.
**Scope - out.** Time-dependent KFE solves; reflected diffusions
beyond a state constraint; ergodicity proofs.

**Dense tutorials served + concept supplied.**
- `huggett-incomplete-markets/`: the KFE derivation and the `Aᵀ`
  duality.
- `aiyagari-hact/`: the same stationary solve, generalised to a
  multi-state income chain.
- `sequence-space-jacobian-hank/`: the Young (2010) lottery
  iteration is named in the prelim as the discrete-time analog,
  so SSJ's lottery step becomes a one-sentence cross-reference.

**Cut list.**
- `heterogeneous-agents/huggett-incomplete-markets/README.md`,
  `## Equations > ### The Kolmogorov forward equation`,
  ~lines 116-149: cut the full KFE derivation.
- `heterogeneous-agents/huggett-incomplete-markets/README.md`,
  `## Solution Method > ### KFE by transposing the same
  generator`, ~lines 270-286: cut the `Aᵀ g = 0` mechanics.
- `heterogeneous-agents/aiyagari-hact/README.md`,
  `## Equations > ### Stationary KFE and closure`,
  ~lines 68-80: replace with a one-sentence pointer.
- `heterogeneous-agents/aiyagari-hact/README.md`,
  `### KFE by transposing the same generator`, ~lines 162-168:
  collapse to a parenthetical inside Solution Method.
- `heterogeneous-agents/sequence-space-jacobian-hank/README.md`,
  steady-state distribution sentence ~lines 134-135: append a
  pointer to the prelim describing the lottery as the
  discrete-time analog.

**References.**
1. Achdou, Han, Lasry, Lions, Moll (2022), *RES* 89(1), 45-86.
   The canonical economic reference for the `A` vs `Aᵀ` duality
   in computational HA macro.
2. Pavliotis (2014), *Stochastic Processes and Applications:
   Diffusion Processes, Fokker-Planck and Langevin Equations*,
   Springer. Chapters 2-3 for the KFE derivation from
   Chapman-Kolmogorov.
3. Gardiner (2004), *Handbook of Stochastic Methods*, 3rd ed.,
   Springer. §3.2 for the Fokker-Planck derivation from mass
   balance with physical intuition.
4. Young (2010), "Solving the Incomplete Markets Model with
   Aggregate Uncertainty Using the Krusell-Smith Algorithm and
   Non-Stochastic Simulations," *JEDC* 34, 36-41. Cited as the
   discrete-time lottery analog used by SSJ-HANK.

**Equations.**
- Continuity equation `∂g/∂t + ∂(μ g)/∂x = 0` (deterministic
  drift) generalised to include diffusion and Poisson jumps.
- Stationary condition `Aᵀ g = 0`, where `A` is the upwind
  generator from prelim #1.
- Normalisation row that replaces one redundant equation.
- Continuous-time Markov generator `Q` for two-state Poisson
  income (off-diagonal jump rates, zero row sums).

**`run.py` sketch.** Ornstein-Uhlenbeck process on a bounded
interval. Build `A` by upwinding the drift; solve `Aᵀ g = 0`
with the normalisation; compare to the analytic Gaussian
stationary density. Also reuse the toy `A` from prelim #1 (the
HJB solve) and recompute its `Aᵀ` stationary distribution.
Figures: (a) stationary density vs analytic; (b) sparse `A`
pattern; (c) operator duality diagram (`A` for HJB, `Aᵀ` for
KFE, sharing structure). Thumbnail from (a).

**Catalog row.** Insert at the top of "Heterogeneous Agents" in
root `README.md`, immediately above
`heterogeneous-agents/endogenous-grid-points/`, paired with
prelim #1 as the continuous-time HA infrastructure.

---

## 3. `structural-econometrics/gmm-foundations/`

**Title.** GMM Foundations: Moment Conditions, Identification,
and Optimal Weighting.

**Description (catalog row).** A location-model GMM estimator
runs from a moment condition through Hansen's two-step optimal
weighting. Adding moments raises efficiency only when each one
adds independent identifying variation; redundant moments shrink
the gain.

**Scope - in.** Moment conditions; just- vs over-identification;
the GMM estimator and its asymptotic distribution; Hansen's
two-step optimal weighting matrix; identification through
exclusion restrictions sketched briefly; the SMM bridge
(simulated moments stand in for population moments) in one
short subsection.
**Scope - out.** Nonparametric IV; weak-identification theory
beyond a one-paragraph caveat; specification-test details
(J-test mentioned, derivation deferred).

**Dense tutorials served + concept supplied.**
- `computational-methods/simulation-based-estimation/`:
  optimal-weighting allusion gets a pointer.
- `agent-based-models/brock-hommes-asset-pricing/`: the silent
  `W` in the SMM pseudocode step 4 gets a pointer.
- `structural-econometrics/adversarial-estimation/`: no inline
  cut; the prelim is added as `## Preliminary readings` so the
  GMM equivalence discussion has a referent.
- `industrial-organization/logit-supply-side/` and
  `choice/mixed-logit-simulation/`: no inline cuts (neither
  tutorial re-derives GMM background). Prelim is added as a
  Preliminary reading.

**Cut list.**
- `computational-methods/simulation-based-estimation/README.md`,
  `## Equations > ### Method 1: MSM`, ~line 57: replace the
  "in production code one would use the inverse of the moment
  covariance" sentence with a pointer to the prelim.
- `agent-based-models/brock-hommes-asset-pricing/README.md`,
  `## Solution Method`, step 4 of the SMM pseudocode (~line 144):
  append a one-sentence pointer to the prelim explaining `W`.
- `structural-econometrics/adversarial-estimation/README.md`: no
  inline cut; add to `## Preliminary readings` list.

**References.**
1. Hansen (1982), "Large Sample Properties of Generalized Method
   of Moments Estimators," *Econometrica* 50(4), 1029-1054.
   Equations 3.1-3.7 for the two-step weighting matrix and
   efficiency result.
2. Hayashi (2000), *Econometrics*, Princeton, Chapter 3. The
   pedagogical anchor; develops GMM as a generalisation of OLS.
3. Hall (2005), *Generalized Method of Moments*, Oxford,
   advanced texts in econometrics. Book-length treatment of
   identification, testing, and finite-sample behaviour.
4. McFadden (1989) and Pakes-Pollard (1989), paired Econometrica
   papers on the simulated method of moments. Bridge to SMM as
   used in catalog tutorials.

**Equations.**
- The moment condition `E[g(θ₀, x)] = 0`.
- The GMM objective `Q(θ) = ḡ(θ)ᵀ W ḡ(θ)`.
- Asymptotic distribution `√n (θ̂ − θ₀) → N(0, (GᵀWG)⁻¹ Gᵀ W Ω W G (GᵀWG)⁻¹)`.
- Optimal `W = Ω⁻¹` and efficient variance `(Gᵀ Ω⁻¹ G)⁻¹`.
- Hansen's two-step procedure.
- A one-line analogue for SMM with `m̂(θ) = m̃(θ) − m_data`.

**`run.py` sketch.** A scalar location model `x_i = θ + ε_i`
with `ε_i` from a skewed mixture. Estimate `θ` via GMM with
1, 2, then 5 moments (mean, variance, third moment, fifth
quantile, ninetieth quantile). Two-step weighting on the over-
identified specifications. Figures: (a) sampling distribution
of `θ̂` under each moment set + each weighting; (b) efficiency
gain plot (variance vs moment count, identity vs optimal `W`);
(c) J-statistic histogram under correct + misspecified models.
Thumbnail from (b).

**Catalog row.** Insert at the top of "Structural Econometrics"
in root `README.md`, immediately above
`industrial-organization/dynamic-discrete-choice/` (the current
first row of that section).

---

## 4. `dsge/blanchard-kahn-determinacy/`

**Title.** Blanchard-Kahn Conditions and Saddle-Path Selection
in Linear Rational Expectations.

**Description (catalog row).** A 2x2 forward-looking RE model
is solved by counting stable generalised eigenvalues against
predetermined states. The same QZ partition runs on a small
DSGE and selects the unique non-explosive path.

**Scope - in.** The Blanchard-Kahn eigenvalue-counting rule;
generalised Schur (QZ) decomposition; reordering stable roots
into the leading block; reading off the state transition `F`
and the jump rule `P`; what happens when the count fails
(indeterminacy, explosiveness, sunspot equilibria stated but
not built).
**Scope - out.** Full DSGE construction; estimation; impulse
response analysis beyond a sanity-check plot.

**Dense tutorials served + concept supplied.**
- `dsge/rbc/`: the QZ + BK mechanics in Method 2 are replaced
  with a pointer; the RBC-specific matrix construction stays.
- `dsge/nkdsge/`: the BK-determinacy sentences become a pointer.
- `computational-methods/perturbation-linearization/`: no cut;
  add the prelim as a `## Preliminary readings` link.
- `dsge/behavioral-nk/`: no cut found.

**Cut list.**
- `dsge/rbc/README.md`, `### Method 2: Klein QZ on the
  augmented 4x4 system (endogenous labor)`, ~lines 168-188:
  cut steps 2-6 of the generic QZ/BK mechanics. Keep step 1
  (matrix construction) and steps 7-8 (IRF initialisation).
  Replace cut block with one sentence.
- `dsge/nkdsge/README.md`, ~line 96 and Takeaway ~line 122:
  collapse the two BK sentences into one pointer.
- `computational-methods/perturbation-linearization/README.md`:
  no cut; add prelim to `## Preliminary readings`.

**References.**
1. Blanchard & Kahn (1980), "The Solution of Linear Difference
   Models under Rational Expectations," *Econometrica* 48(5),
   1305-1311. Theorem 1 is the canonical eigenvalue-counting
   rule.
2. Klein (2000), "Using the Generalized Schur Form to Solve a
   Multivariate Linear Rational Expectations Model," *JEDC*
   24(10), 1405-1423. The QZ algorithm in the form modern
   solvers use.
3. Sims (2002), "Solving Linear Rational Expectations Models,"
   *Computational Economics* 20(1-2), 1-20. The gensys
   algorithm bridging ordinary Schur and QZ.
4. DeJong & Dave (2010), *Structural Macroeconometrics*, 2nd
   ed., Princeton, Chapter 2. Pedagogical map from BK counting
   to concrete indeterminacy.

**Equations.**
- The linear RE system `A E_t y_{t+1} = B y_t + C ε_t`.
- Generalised Schur decomposition: `Q A Z = S`, `Q B Z = T`
  with stable roots in the leading block.
- BK counting rule: `#{|s_ii / t_ii| < 1}` = number of
  predetermined states for unique stable solution.
- Recovery of `F` (state transition) and `P` (jump rule) from
  the partition.

**`run.py` sketch.** A two-equation toy RE model
(consumption-Euler + capital accumulation linearised around
steady state). Sweep one structural parameter through the BK
boundary: solve QZ at each value, plot |generalised eigenvalues|
and flag which side of unity each one lands on. Also run a
small RBC linearisation as a sanity check. Figures:
(a) eigenvalue trajectories as the parameter crosses the BK
boundary; (b) phase-plane sample paths under determinate vs
indeterminate calibrations; (c) BK-classification heatmap on a
two-parameter grid. Thumbnail from (c).

**Catalog row.** Insert at the top of "Linearized DSGE" in root
`README.md`, immediately above `dsge/rbc/`.

---

## 5. `bayesian-methods/bayesian-foundations/`

**Title.** Bayesian Foundations: Priors, Likelihoods, and
Conjugate Posteriors.

**Description (catalog row).** Bayes rule turns a prior into a
posterior. Beta-Binomial conjugacy gives a closed-form scalar
update; Gaussian-Gaussian conjugacy gives a closed-form linear
regression update with shrinkage as a precision-weighted average.

**Scope - in.** Bayes rule for a scalar parameter; conjugacy via
Beta-Binomial; Gaussian-Gaussian conjugate linear regression
(precision = prior precision + data precision; mean = precision-
weighted average); prior sensitivity; posterior predictive
distribution; positions itself as the bayesian-methods/ entry
point.
**Scope - out.** Hierarchical models; nonconjugate sampling
(that is metropolis-hastings/'s job); model selection; Bayesian
nonparametrics.

**Dense tutorials served + concept supplied.**
- `computational-methods/metropolis-hastings/`: the prelim
  absorbs ~30% of MH's current README (the Beta-Binomial
  primer). MH becomes a method tutorial about the algorithm
  itself, not a Bayes-primer + algorithm hybrid.
- `time-series/minnesota-svar/`: the Gaussian-Gaussian
  conjugate update derivation moves to the prelim; Minnesota
  keeps the working formulas + plug-in `σ̂_i²`.
- `computational-methods/hamiltonian-monte-carlo/`,
  `numerical-methods/bayesian-optimization/`,
  `structural-econometrics/bayesian-dsge-hmc/`: no inline cuts;
  add prelim as a Preliminary reading.

**Cut list.**
- `computational-methods/metropolis-hastings/README.md`,
  Overview paragraphs 1-2 (~lines 5-8) + Equations preamble +
  Bayes rule block (~lines 13-24) + the entire
  `### Method 1: Beta-Binomial conjugate posterior`
  subsection (~lines 26-92): cut ~70 lines. Replace with a
  one-paragraph recap pointing to the prelim. The conjugate
  Beta-Binomial sanity-check experiment in Solution Method
  (lines 182-196) and Results (lines 224-237) STAYS - the
  derivation moves out, the verification stays.
- `numerical-methods/bayesian-optimization/README.md`,
  ~lines 39-41: repoint the existing cross-reference from
  metropolis-hastings/ to bayesian-foundations/.
- `time-series/minnesota-svar/README.md`, `## Equations`
  ~lines 34-50: cut the Gaussian-Gaussian conjugate derivation
  prose. Keep the working formulas for `V_i⁻¹` and `b_i` at
  lines 80-90.
- `computational-methods/hamiltonian-monte-carlo/README.md`:
  no cut; add to `## Preliminary readings`.
- `structural-econometrics/bayesian-dsge-hmc/README.md`: no
  cut; add to `## Preliminary readings`.

**References.**
1. Gelman, Carlin, Stern, Dunson, Vehtari, Rubin (2013),
   *Bayesian Data Analysis*, 3rd ed., Chapman & Hall/CRC.
   Chapters 1-2 (foundations); Chapter 14 (normal linear
   regression conjugate update).
2. Koop (2003), *Bayesian Econometrics*, Wiley, Chapter 2.
   Field-specific exposition linking conjugacy directly to
   econometric practice.
3. Robert (2007), *The Bayesian Choice*, 2nd ed., Springer,
   Chapter 3. Decision-theoretic framing for conjugacy and
   prior selection.
4. Hamilton (1994), *Time Series Analysis*, Princeton,
   Chapter 12. Bridges scalar Bayes to BVAR.

**Equations.**
- Bayes rule `p(θ | y) ∝ p(y | θ) p(θ)`.
- Beta-Binomial conjugate update `Beta(α + s, β + n − s)`.
- Gaussian-Gaussian conjugate regression: prior precision
  `V₀⁻¹`, posterior precision `V⁻¹ = V₀⁻¹ + XᵀX/σ²`, posterior
  mean as precision-weighted average.
- Posterior predictive density (one display block, with
  Beta-Binomial as worked example).

**`run.py` sketch.** Three minimal worked examples in one
script: (a) Beta-Binomial coin flip with three priors, showing
posterior contraction; (b) Gaussian-Gaussian conjugate linear
regression on a synthetic dataset, showing how the posterior
mean is the precision-weighted average of OLS and prior;
(c) prior-sensitivity sweep. Figures: (a) Beta posteriors after
0, 10, 100, 1000 observations; (b) regression posterior bands
shrinking with sample size; (c) prior-sensitivity fan chart.
Thumbnail from (a).

**Catalog row.** Insert at the top of "Bayesian Methods" view in
root `README.md`, immediately above
`computational-methods/metropolis-hastings/`.

---

---

## 6. `computational-methods/mcmc-diagnostics/`

**Title.** MCMC Chain Diagnostics: ESS, R-hat, and Integrated
Autocorrelation Time.

**Description (catalog row).** A correlated-Gaussian target is
sampled with random-walk Metropolis. ESS, IAT, R-hat across
multiple chains, and trace plots each catch a different
pathology before any structural model is touched.

**Scope - in.** Lag autocorrelation `ρ_t`; integrated
autocorrelation time `τ = 1 + 2 Σ ρ_t`; effective sample size
`ESS = T/τ`; multi-chain potential scale reduction factor `R̂`
(classical Gelman-Rubin and rank-normalised split-R̂); trace
plots; burn-in heuristics; brief note on Roberts-Gelman-Gilks
optimal acceptance rate.
**Scope - out.** Stationarity tests beyond R̂; geometric-
ergodicity proofs; sample-size planning rules.

**Dense tutorials served + concept supplied.**
- `metropolis-hastings/`: ESS and IAT definitions live in this
  prelim; MH keeps reported ESS values as Results.
- `bayesian-dsge-hmc/`: R̂ column in the posterior summary
  table gains a footnote pointer.
- `hamiltonian-monte-carlo/`: no inline cut; add prelim to
  Preliminary readings.
- `minnesota-svar/`: no diagnostic content to cut (empirical
  Bayes, no MCMC).

**Cut list.**
- `computational-methods/metropolis-hastings/README.md`,
  `## Solution Method`, ~lines 218-220: cut the ESS and IAT
  definitions. Keep the Roberts-Gelman-Gilks acceptance-rate
  sentence. Replace cut with one pointer paragraph.
- `structural-econometrics/bayesian-dsge-hmc/README.md`,
  Results posterior-summary table, ~lines 131-141: add a
  footnote pointing R̂ at the prelim.
- `computational-methods/hamiltonian-monte-carlo/README.md`:
  no cut; add prelim to `## Preliminary readings`.

**References.**
1. Gelman & Rubin (1992), "Inference from Iterative Simulation
   Using Multiple Sequences," *Statistical Science* 7(4),
   457-472. §2-3 define R̂ via within-chain and between-chain
   variance.
2. Geyer (1992), "Practical Markov Chain Monte Carlo,"
   *Statistical Science* 7(4), 473-483. §2-3 derive IAT from
   the Kipnis-Varadhan CLT and connect to ESS.
3. Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021),
   "Rank-normalization, folding, and localization: An improved
   R̂ for assessing convergence of MCMC," *Bayesian Analysis*
   16(2), 667-718. §2-3 for modern rank-normalised split-R̂.
4. Robert & Casella (2004), *Monte Carlo Statistical Methods*,
   2nd ed., Springer. Chapter 12 (convergence diagnostics).

**Equations.**
- Lag-`t` autocorrelation `ρ_t = Corr(θ_s, θ_{s+t})`.
- IAT `τ = 1 + 2 Σ_{t≥1} ρ_t` (with truncation rule).
- ESS `= T / τ`.
- Within / between variance `W, B` and `R̂ = √((T−1)/T + B/(TW))`.
- Rank-normalised split-R̂ as the modern default.

**`run.py` sketch.** Run three random-walk MH chains on a
correlated 2D Gaussian with two step-size choices (too small,
optimal). Compute IAT via Geyer's monotone-positive estimator,
ESS, classical R̂, and rank-normalised split-R̂. Figures:
(a) trace plots side-by-side under good and pathological
tuning; (b) autocorrelation decay with truncation cutoff;
(c) R̂ trajectory as chain length grows. Thumbnail from (a).

**Catalog row.** Insert in the "Bayesian Methods" view of root
`README.md`, between `computational-methods/metropolis-hastings/`
and `computational-methods/hamiltonian-monte-carlo/`.

---

## 7. `time-series/reduced-form-var/`

**Title.** Reduced-Form VARs: Estimation, Impulse Responses,
and Cholesky Identification.

**Description (catalog row).** A bivariate VAR(p) is estimated
by OLS, stacked into companion form, and identified by a
recursive Cholesky ordering of the residual covariance. The
resulting impulse responses are the building block for
Bayesian and structural extensions in the rest of the section.

**Scope - in.** VAR(p) lag stacking; companion-form matrix;
OLS estimation; reduced-form residual covariance `Σ_u`;
recursive (lower-triangular) Cholesky identification; impulse
response propagation `Φ_j`; brief comparison with non-recursive
identification (sign restrictions named, not derived).
**Scope - out.** Bayesian priors (handled in `minnesota-svar/`);
SVAR with proxy / external instruments; cointegration / VECM.

**Dense tutorials served + concept supplied.**
- `minnesota-svar/`: the reduced-form VAR setup, companion form,
  and Cholesky identification all move to the prelim; the
  Minnesota prior structure stays.
- `stock-watson/`: a one-pointer Takeaway cross-reference for
  the AR-with-factor forecast comparison.

**Cut list.**
- `time-series/minnesota-svar/README.md`, `## Equations`,
  ~lines 19-52: cut the generic VAR lag-stacking + OLS setup.
- `time-series/minnesota-svar/README.md`, `## Equations`,
  ~lines 101-136: cut the recursive Cholesky identification
  derivation.
- `time-series/minnesota-svar/README.md`, `## Solution Method`,
  pseudocode steps 1-6, ~lines 173-190: cut the generic VAR
  estimation steps; keep step 4 (Minnesota prior) and any
  Minnesota-specific shock-scaling logic at ~lines 138-152.
- `time-series/stock-watson/README.md`, Takeaway ~line 132:
  append a one-sentence pointer to the prelim for the VAR
  forecast comparison.

**References.**
1. Sims (1980), "Macroeconomics and Reality," *Econometrica*
   48(1), 1-48. Founding identification paper introducing
   reduced-form VARs and Cholesky orthogonalisation.
2. Hamilton (1994), *Time Series Analysis*, Princeton,
   Chapter 11 (§11.3-11.4 for OLS VAR + IRF + recursive
   identification).
3. Stock & Watson (2001), "Vector Autoregressions," *Journal of
   Economic Perspectives* 15(4), 101-115. Practitioner's
   overview distinguishing reduced-form, recursive, and
   structural VARs.
4. Lütkepohl (2005), *New Introduction to Multiple Time Series
   Analysis*, Springer, Chapters 9-10 (SVAR identification).

**Equations.**
- VAR(p): `y_t = c + Σ A_i y_{t−i} + u_t`, `u_t ∼ (0, Σ_u)`.
- Companion form `Y_t = F Y_{t−1} + e_t`.
- OLS estimator for `(c, A_1, ..., A_p)`.
- Cholesky factor `P` with `Σ_u = P Pᵀ`; structural shock
  `ε_t = P⁻¹ u_t`.
- Impulse response `Φ_j = J Fʲ Jᵀ P` (top block of companion
  power).

**`run.py` sketch.** A simulated bivariate AR(2) with known
true coefficients. Estimate VAR(2) by OLS; recover coefficients
within sampling noise. Cholesky-identify two orderings and
plot the resulting impulse responses against the true ones.
Figures: (a) IRFs under ordering (output, inflation) vs
(inflation, output); (b) reduced-form residual scatter +
Cholesky factor visualisation; (c) RMSE of estimated vs true
IRFs across sample sizes. Thumbnail from (a).

**Catalog row.** Insert in "Time Series" of root `README.md`,
between `time-series/ar-processes/` and
`time-series/minnesota-svar/`.

---

## 8. `industrial-organization/bertrand-ownership-matrix/`

**Title.** Multi-Product Bertrand-Nash Pricing and the
Ownership Matrix.

**Description (catalog row).** A multi-product firm's pricing
FOC stacks into a linear system whose ownership matrix `Ω`
encodes the Hadamard product with the demand Jacobian. The
same system reads pre-merger markups, recovers marginal costs,
and produces post-merger counterfactual prices.

**Scope - in.** Ownership matrix construction (single-product,
multi-product, merged); demand Jacobian `Δ(p) = ∂s/∂p`; Hadamard
product `Ω ⊙ Δ`; the FOC `p = c − (Ω ⊙ Δ)⁻¹ s`; fixed-point
iteration on the price vector; cost recovery as one inversion;
post-merger price counterfactual as a second solve under a new
`Ω`.
**Scope - out.** Random-coefficient demand (lives in BLP);
merger-screen formulas UPP/GUPPI/CMCR (stay in
`merger-simulation/`); bargaining (Nash-in-Nash is a different
method).

**Dense tutorials served + concept supplied.**
- `merger-simulation/`: the entire Bertrand-Nash-with-ownership
  derivation moves to the prelim; merger-specific screens
  (UPP/GUPPI/CMCR) and welfare frontier stay.
- `logit-supply-side/`: the FOC + ownership-matrix machinery
  moves; logit-specific Jacobian and IV estimation stay.
- `nash-in-nash/`: no cut (bilateral bargaining is a distinct
  method).

**Cut list.**
- `industrial-organization/merger-simulation/README.md`,
  `## Equations`, section "Bertrand-Nash pricing with
  multi-product firms" ~lines 54-98: cut the generic ownership
  matrix + Hadamard FOC derivation; replace with one pointer.
- `industrial-organization/logit-supply-side/README.md`,
  `## Equations`, FOC + ownership-matrix block ~lines 42-60:
  cut the FOC derivation and `Ω m = s` recovery; replace with
  one pointer. Keep the logit demand Jacobian construction
  immediately above.
- `industrial-organization/nash-in-nash/README.md`: no cut.

**References.**
1. Berry, Levinsohn & Pakes (1995), "Automobile Prices in
   Market Equilibrium," *Econometrica* 63(4), 841-890.
   Supply-side derivation of the multi-product Bertrand FOC
   under logit demand.
2. Nevo (2000), "A Practitioner's Guide to Estimation of
   Random-Coefficients Logit Models of Demand," *Journal of
   Economics and Management Strategy* 9(4), 513-548. The
   supply appendix is the cleanest exposition of the FOC and
   ownership matrix.
3. Werden & Froeb (1994), "The Effects of Mergers in
   Differentiated Products Industries: Logit Demand and Merger
   Policy," *Journal of Law, Economics, and Organization*
   10(2), 407-426. Direct merger application of the same FOC.
4. Conlon & Gortmaker (2020), "Best Practices for
   Differentiated Products Demand Estimation with PyBLP,"
   *RAND Journal of Economics* 51(4), 1108-1161. Modern
   computational reference for ownership-matrix supply-side
   solves.

**Equations.**
- Single-product Bertrand FOC `p_j − c_j = s_j / (−∂s_j/∂p_j)`.
- Multi-product FOC in matrix form
  `s(p) + (Ω ⊙ Δ(p))ᵀ (p − c) = 0`.
- Solution `p = c − (Ω ⊙ Δ)⁻¹ s`.
- Cost recovery (inversion at observed `p, s`) and
  counterfactual price under merged `Ω`.

**`run.py` sketch.** A 5-product market with logit demand,
two single-product firms and one two-product firm. Compute
pre-merger equilibrium prices via fixed-point on the FOC.
Recover marginal costs by inversion. Simulate a horizontal
merger of two single-product firms; recompute prices under
the new `Ω`. Figures: (a) pre vs post merger prices per
product; (b) ownership-matrix heatmap (pre and post);
(c) convergence of the fixed-point iterator. Thumbnail from
(a).

**Catalog row.** Insert in "Industrial Organization" of root
`README.md`, before `industrial-organization/logit-supply-side/`.

---

## 9. `numerical-methods/simulated-likelihood/`

**Title.** Simulated Maximum Likelihood, Common Random
Numbers, and Halton Sequences.

**Description (catalog row).** An intractable integrated
likelihood is replaced by fixed simulation draws. Common
random numbers make the objective smooth in `θ`; Halton
sequences beat pseudorandom draws on bias and variance for the
same draw count.

**Scope - in.** Simulated likelihood `P̂(θ) = (1/R) Σ f(θ; ξ_r)`;
common random numbers (CRN) for smoothness; pseudo-random vs
Halton vs scrambled Sobol; draw-count bias-variance tradeoff;
brief note on antithetic variates.
**Scope - out.** Maximum simulated moments (handled in GMM
prelim); importance sampling for rare events; sequential
Monte Carlo.

**Dense tutorials served + concept supplied.**
- `mixed-logit-simulation/`: the integral approximation and CRN
  paragraphs move to the prelim.
- `rum-choice-networks/`: the fixed-latent-draws paragraph
  becomes a pointer.
- `adversarial-estimation/`: the CRN-for-smooth-outer-objective
  paragraph becomes a pointer.
- `blp-random-coefficients/`: the simulated-shares paragraph
  becomes a pointer.

**Cut list.**
- `choice/mixed-logit-simulation/README.md`, `## Equations`
  ~lines 49-61: cut the generic integral-approximation +
  fixed-draws setup.
- `choice/mixed-logit-simulation/README.md`, `## Solution
  Method` ~lines 104-106: cut the CRN-explanation sentence.
- `structural-econometrics/rum-choice-networks/README.md`,
  `## Solution Method` ~lines 155-157: cut the fixed-latent-
  draws paragraph; replace with pointer.
- `structural-econometrics/adversarial-estimation/README.md`,
  `## Equations` ~lines 25-31: cut the CRN paragraph; replace
  with pointer.
- `industrial-organization/blp-random-coefficients/README.md`,
  `## Equations` ~lines 38-41: cut the simulated-shares
  draw-grid paragraph; replace with pointer.

**References.**
1. Train (2009), *Discrete Choice Methods with Simulation*,
   2nd ed., Cambridge, Chapter 9 ("Drawing from Densities").
   Pedagogical anchor.
2. McFadden (1989), "A Method of Simulated Moments for
   Estimation of Discrete Response Models without Numerical
   Integration," *Econometrica* 57(5), 995-1026. Foundational
   asymptotic theory; shared with the GMM prelim's references.
3. Bhat (2001), "Quasi-random Maximum Simulated Likelihood
   Estimation of the Mixed Multinomial Logit Model,"
   *Transportation Research Part B* 35(7), 677-693. Halton
   draws beat pseudorandom on mixed logit.
4. Hess, Train & Polak (2006), "On the use of a Modified
   Latin Hypercube Sampling Method...," *Transportation
   Research Part B* 40(2), 147-163. Practical comparison of
   Halton, MLHS, scrambled.

**Equations.**
- True integrated likelihood `P(θ) = ∫ f(θ; ξ) dF(ξ)`.
- Simulated estimator `P̂(θ) = (1/R) Σ_r f(θ; ξ_r)` with fixed
  draws `{ξ_r}`.
- CRN property: same `{ξ_r}` across `θ` makes `P̂(θ)` smooth in
  `θ`.
- Halton sequence construction in base `b`.

**`run.py` sketch.** Estimate a one-parameter mixed-logit
model with three draw schemes (pseudorandom, Halton, scrambled
Sobol) and three draw counts (R=50, 200, 1000). Compute bias
and standard deviation of `θ̂` across 200 Monte Carlo
replications. Figures: (a) `θ̂` sampling distribution by draw
scheme; (b) bias-variance vs `R` curves; (c) Halton vs
pseudorandom point cloud in 2D. Thumbnail from (b).

**Catalog row.** Insert in "Numerical Methods" of root
`README.md`, before `numerical-methods/fixed-point-acceleration/`.

---

## 10. `numerical-methods/neural-networks-regression/`

**Title.** Feedforward Neural Networks for Regression and
Density Approximation.

**Description (catalog row).** A one-hidden-layer tanh network
fits a Cobb-Douglas production surface and a demand function.
Forward pass, backpropagation, L2 weight decay, and Adam are
the four moving parts that every neural utility, neural
discriminator, and neural mechanism in this catalog reuses.

**Scope - in.** One-hidden-layer feedforward architecture with
tanh activation; forward pass; squared-error loss for
regression and cross-entropy for classification; backpropagation
via JAX autodiff; L2 weight decay; Adam optimiser; basic
hyperparameter sensitivity.
**Scope - out.** Deep architectures (multiple hidden layers);
convolutional / recurrent / attention layers; normalising
flows; meta-learning. Architecture choices specific to consumer
tutorials (RUMnet structure, GAN discriminator design,
mechanism heads) stay in those tutorials.

**Dense tutorials served + concept supplied.**
- `rum-choice-networks/`: the generic "one-hidden-layer tanh
  with small init" recap becomes a pointer.
- `adversarial-estimation/`: the "shallow tanh discriminator
  with weight decay" intro becomes a pointer.
- `game-theory/deep-optimal-auctions/`: the generic
  backprop-via-autodiff sentence becomes a pointer.

**Cut list.**
- `structural-econometrics/rum-choice-networks/README.md`,
  `## Solution Method` ~lines 164-181: cut the generic neural
  initialisation paragraph; keep the RUM-specific reasoning
  for why RUMnet probabilities start close to logit.
- `structural-econometrics/adversarial-estimation/README.md`,
  `## Solution Method`, "Method 3: Shallow neural-network
  discriminator" ~lines 169-191: cut the generic feedforward
  + L-BFGS + weight-decay recap; keep the discriminator-as-SMM
  payoff sentence.
- `game-theory/deep-optimal-auctions/README.md`,
  `## Solution Method` ~lines 131-136: cut the generic
  "differentiable network trained by autodiff" sentence; keep
  the IC-via-grid-best-lie content.

**References.**
1. Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT
   Press, Chapters 6-7. Feedforward networks + regularisation.
2. Bishop (2006), *Pattern Recognition and Machine Learning*,
   Springer, Chapter 5. Classical neural networks with
   Bayesian regularisation framing.
3. Kingma & Ba (2015), "Adam: A Method for Stochastic
   Optimization," *ICLR*. The optimiser used everywhere in
   the consumer tutorials.
4. Athey & Imbens (2019), "Machine Learning Methods That
   Economists Should Know About," *Annual Review of
   Economics* 11, 685-725. Positioning for economist readers.

**Equations.**
- Forward pass `h = tanh(Wx + b)`, `ŷ = W̃h + b̃`.
- Loss `L = (1/n) Σ (y_i − ŷ_i)² + λ (‖W‖² + ‖W̃‖²)`.
- Backprop via the chain rule (one display block).
- Adam update rule.

**`run.py` sketch.** Fit a one-hidden-layer (H = 16) tanh
network to a Cobb-Douglas surface `y = A k^α ℓ^(1−α)` on a
2D grid, with Gaussian noise. Train via Adam + JAX autodiff
with three weight-decay strengths. Compare in-sample and
out-of-sample MSE against a linear baseline and a degree-3
polynomial. Figures: (a) fitted surface vs truth; (b) bias-
variance tradeoff across `λ`; (c) training loss curves with
and without Adam. Thumbnail from (a).

**Catalog row.** Insert in "Numerical Methods" of root
`README.md`, between `numerical-methods/interpolation/` and
`numerical-methods/scalar-optimization-monopoly-pricing/`.

---

---

## 11. `choice/weitzman-search-rule/`

**Title.** Pandora's Box: Optimal Sequential Search and the
Weitzman Reservation-Value Rule.

**Description (catalog row).** A buyer inspects boxes with
random values at a cost per inspection. Weitzman's reservation
value `z_j` summarises each box; opening boxes in decreasing
`z` is optimal under perfect recall.

**Scope - in.** Reservation value derivation
`z_j = max_v (v − ∫ G_j(w) dw / something)`; index-ordering
optimal rule (open in decreasing `z`); perfect-recall stopping
condition; brief comparison with myopic search and with the
McCall job-search reservation wage.
**Scope - out.** Search with bargaining; rational-inattention
search models; learning across boxes.

**Dense tutorials served + concept supplied.**
- `choice/sequential-search-ursu/`: the entire Weitzman
  reservation-value derivation and index-ordering rule move to
  the prelim.

**Cut list.**
- `choice/sequential-search-ursu/README.md`, `## Equations`,
  reservation-value block ~lines 61-75: cut the Weitzman
  derivation.
- `choice/sequential-search-ursu/README.md`, `## Equations`,
  index-ordering rule ~lines 91-95: cut the index-ordering
  exposition.
Both replaced with single-pointer sentences.

**References.**
1. Weitzman (1979), "Optimal Search for the Best Alternative,"
   *Econometrica* 47(3), 641-654. Introduces `z_j` and the
   index rule; shows perfect recall yields no advantage over
   the reservation rule.
2. Kohn & Shavell (1974), "The Theory of Search," *JET* 9(2),
   93-123. Earlier reservation-wage stopping rules.
3. Ljungqvist & Sargent (2018), *Recursive Macroeconomic
   Theory*, 4th ed., MIT Press, Chapters 6-7. Textbook
   treatment of sequential search.
4. Choi, Dai & Kim (2018), "Consumer Search and Price
   Competition," *Econometrica* 86(4), 1257-1281. Modern
   application of the Weitzman index in market design.

**Equations.**
- Reservation value `z_j` solving
  `c_j = ∫_{z_j}^∞ (v − z_j) dF_j(v)`.
- Index-ordering optimal policy.
- Stopping condition: stop when current best `b ≥ max_j z_j`
  over uninspected boxes.

**`run.py` sketch.** A 6-box Pandora's-box problem with
Gaussian value distributions and heterogeneous inspection
costs. Solve for `z_j` per box; simulate the optimal search
path over 1000 draws; compare expected payoff against the
myopic rule (open in decreasing `E[v_j]`). Figures: (a)
reservation values vs costs; (b) realised payoff distribution
under optimal vs myopic; (c) inspection-count distribution.
Thumbnail from (a).

**Catalog row.** Insert in "Choice and Demand" of root
`README.md`, before `choice/sequential-search-ursu/`.

---

## 12. `numerical-methods/quadrature/`

**Title.** Numerical Quadrature: Gauss-Hermite Nodes for
Conditional Expectations.

**Description (catalog row).** Gauss-Hermite nodes integrate
against a Gaussian weight in closed form. A change of variables
turns `E[f(z')|z]` under an AR(1) Gaussian innovation into a
small weighted sum that converges faster than Monte Carlo on
smooth integrands.

**Scope - in.** Gauss-Hermite nodes from Hermite polynomial
roots; weights from Christoffel-Darboux; change of variables
for an AR(1) conditional expectation under Gaussian
innovations; node-count vs error scaling; brief comparison
with Simpson's rule and Monte Carlo.
**Scope - out.** Adaptive quadrature; sparse-grid quadrature
(handled in `smolyak-sparse-grids/`); high-dimensional
quasi-Monte Carlo.

**Dense tutorials served + concept supplied.**
- `computational-methods/smolyak-sparse-grids/`: the Gauss-
  Hermite step inside the inner expectation loop moves to the
  prelim.
- `dynamic-programming/shock-discretization/`: no direct cut;
  add prelim to `## Preliminary readings` (Tauchen-Hussey uses
  GH-style nodes but the existing tutorial focuses on the
  Markov-chain mapping).
- `numerical-methods/bayesian-optimization/`: no cut; add to
  Preliminary readings if the acquisition integral cites GH.
- `structural-econometrics/bayesian-dsge-hmc/`: no cut; add to
  Preliminary readings.

**Cut list.**
- `computational-methods/smolyak-sparse-grids/README.md`,
  `## Solution Method`, GH quadrature step ~lines 199-205:
  cut the inline definition of nodes/weights; keep the
  Smolyak-specific algorithm.
- All other consumer tutorials: add prelim to `## Preliminary
  readings`, no inline cuts.

**References.**
1. Stroud & Secrest (1966), *Gaussian Quadrature Formulas*,
   Prentice-Hall. Tabulated nodes/weights for Hermite weight.
2. Judd (1998), *Numerical Methods in Economics*, MIT Press,
   Chapter 7. Gaussian quadrature for economic applications.
3. Heer & Maußner (2009), *Dynamic General Equilibrium
   Modeling*, 2nd ed., Springer, Chapter 6. GH inside
   parametric-expectations and projection methods.
4. Tauchen & Hussey (1991), "Quadrature-Based Methods for
   Obtaining Approximate Solutions to Nonlinear Asset Pricing
   Models," *Econometrica* 59(2), 371-396. GH nodes for
   AR(1) Markov-chain approximation.

**Equations.**
- Gauss-Hermite identity
  `∫ f(x) e^{−x²} dx ≈ Σ w_n f(ξ_n)`.
- Change of variables `z' = ρ z + σ_ε √2 ξ` for AR(1) with
  innovation `σ_ε`.
- Resulting estimator
  `E[f(z')|z] ≈ (1/√π) Σ w_n f(ρ z + σ_ε √2 ξ_n)`.
- Error rate `O(N⁻²ᵏ)` for `Cᵏ` integrands (one line
  citing Stroud-Secrest).

**`run.py` sketch.** Integrate three test functions against a
standard Gaussian: a polynomial (exact at finite `N`), a smooth
non-polynomial (`exp(-x²/4)`), and a non-smooth one (`|x|`).
Compute GH error vs node count `N = 3, 5, 10, 30`; compare
against Simpson's rule and Monte Carlo with the same effort.
Apply to an AR(1) conditional expectation computed by GH and
by Monte Carlo. Figures: (a) error vs `N` (log-log) by method
and integrand; (b) GH nodes/weights visualisation; (c) AR(1)
`E[f(z')|z]` accuracy comparison. Thumbnail from (a).

**Catalog row.** Insert in "Numerical Methods" of root
`README.md`, between `numerical-methods/interpolation/` and
`numerical-methods/scalar-optimization-monopoly-pricing/`
(close to where #10 NN-regression sits; pick adjacent).

---

## 13. `numerical-methods/gaussian-processes/`

**Title.** Gaussian Process Regression and Uncertainty
Quantification.

**Description (catalog row).** A GP places a prior over
functions; conditioning on observed `(X, y)` returns a
posterior with closed-form mean and variance. Marginal
likelihood tunes the kernel hyperparameters, giving a
data-adaptive smoothness scale.

**Scope - in.** GP prior with constant mean; squared-
exponential (RBF) kernel; closed-form posterior conditioning;
marginal-likelihood-II hyperparameter tuning (length scale,
signal variance, noise); brief note on Matérn kernel choices.
**Scope - out.** Deep kernels; sparse / inducing-point GPs;
scalable variational inference; GP classification.

**Dense tutorials served + concept supplied.**
- `numerical-methods/bayesian-optimization/`: the entire GP
  surrogate derivation moves to the prelim; the BO tutorial
  keeps Expected-Improvement and the acquisition loop.

**Cut list.**
- `numerical-methods/bayesian-optimization/README.md`,
  `## Equations > ### Method 1` ~lines 43-72: cut the GP
  prior + posterior conditioning derivation.
- `numerical-methods/bayesian-optimization/README.md`,
  `## Solution Method > ### Method 1` ~lines 130-147: cut
  the length-scale tuning paragraph; replace with a pointer.

**References.**
1. Rasmussen & Williams (2006), *Gaussian Processes for
   Machine Learning*, MIT Press (free online). Chapter 2
   (regression) and Chapter 5 (model selection / marginal
   likelihood).
2. Bishop (2006), *Pattern Recognition and Machine Learning*,
   Springer, §6.4. Kernel-methods framing.
3. Kennedy & O'Hagan (2001), "Bayesian Calibration of Computer
   Models," *JRSS B* 63(3), 425-464. GP as a surrogate /
   emulator for expensive evaluations.
4. Snoek, Larochelle & Adams (2012), "Practical Bayesian
   Optimization of Machine Learning Algorithms," *NIPS* 25.
   Bridge to the BO consumer tutorial.

**Equations.**
- GP prior `f ∼ GP(μ, k)`.
- Squared-exponential kernel
  `k(x, x') = σ_f² exp(−‖x − x'‖² / (2 ℓ²))`.
- Posterior mean
  `μ_*(x_*) = k_*ᵀ (K + σ_n² I)⁻¹ y`.
- Posterior variance
  `σ_*²(x_*) = k(x_*, x_*) − k_*ᵀ (K + σ_n² I)⁻¹ k_*`.
- Log marginal likelihood (one display block).

**`run.py` sketch.** Fit a GP to noisy samples of a 1D test
function (e.g. a Forrester function or
`x sin x` on `[0, 10]`). Compare the squared-exponential
kernel against a Matérn-5/2 kernel; tune the length scale by
marginal-likelihood maximisation. Figures: (a) posterior mean
+ 95% credible band over the input domain; (b) marginal-
likelihood curve as a function of `ℓ`; (c) prior vs posterior
samples from the GP. Thumbnail from (a).

**Catalog row.** Insert in "Numerical Methods" of root
`README.md`, immediately before
`numerical-methods/bayesian-optimization/`.

---

## 14. `game-theory/regret-matching/`

**Title.** Regret Matching and No-Regret Dynamics.

**Description (catalog row).** Each player plays in proportion
to positive cumulative regret. The time-average strategy
converges to a correlated equilibrium without any opponent
model, and the same primitive scales to CFR for extensive-form
games.

**Scope - in.** Instantaneous regret `r_i^t`; cumulative
regret `R_i^T`; regret-matching update rule (probability
proportional to positive cumulative regret); convergence of
the time-average strategy to correlated equilibrium;
counterfactual regret as the extensive-form generalisation
(named, briefly motivated).
**Scope - out.** Specific CFR variants (vanilla CFR is in
`cfr-asymmetric-auction/`); deep CFR; multi-agent learning
theory beyond Hart-Mas-Colell.

**Dense tutorials served + concept supplied.**
- `game-theory/cfr-asymmetric-auction/`: the regret-matching
  primitive derivation moves to the prelim; CFR application
  to the asymmetric auction stays.

**Cut list.**
- `game-theory/cfr-asymmetric-auction/README.md`, `## Equations`,
  cumulative-regret + regret-matching block ~lines 45-65: cut
  the primitive derivation; replace with one pointer sentence.

**References.**
1. Hart & Mas-Colell (2000), "A Simple Adaptive Procedure
   Leading to Correlated Equilibrium," *Econometrica* 68(5),
   1127-1150. The founding regret-matching paper.
2. Zinkevich, Johanson, Bowling & Piccione (2008), "Regret
   Minimization in Games with Incomplete Information," *NIPS*
   20. CFR foundation.
3. Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and
   Games*, Cambridge, Chapter 4. Learning-theoretic
   foundations of no-regret dynamics.
4. Brown & Sandholm (2019), "Superhuman AI for Multiplayer
   Poker," *Science* 365(6456), 885-890. Modern CFR-variant
   benchmark.

**Equations.**
- Instantaneous regret
  `r_i^t(a) = u(a, a_{-i}^t) − u(a_i^t, a_{-i}^t)`.
- Cumulative regret `R_i^T(a) = Σ_{t=1}^T r_i^t(a)`.
- Regret-matching probability
  `π_i^{T+1}(a) ∝ max(R_i^T(a), 0)`.
- Convergence statement: time-average empirical play
  `π̄_i → ` correlated equilibrium (Theorem A, Hart-Mas-Colell).

**`run.py` sketch.** A 2x2 coordination game and a 3x3
rock-paper-scissors game. Run regret matching for 10,000
iterations; plot cumulative regret per action, the running
strategy mix, and the convergence of the time-average to
the correlated equilibrium. Compare against fictitious play
and best-response iteration on the same games. Figures:
(a) cumulative regret trajectories; (b) time-average strategy
convergence; (c) regret matching vs fictitious play on a
non-zero-sum game. Thumbnail from (b).

**Catalog row.** Insert in "Computational Game Theory" of root
`README.md`, between `game-theory/first-price-auctions/` and
`game-theory/cfr-asymmetric-auction/`.

---

## Refactor ledger (consolidated)

All cuts across the five prelims, by dense-tutorial target:

- `optimal-control/hjb-growth/README.md`: cut FD/upwind
  derivation (~lines 104-138) and boundary forcing rule
  (~lines 142-148). Two pointer sentences in.
- `heterogeneous-agents/huggett-incomplete-markets/README.md`:
  cut KT-clip derivation (~lines 99-114), upwind discretisation
  opening (~lines 199-225), KFE derivation (~lines 116-149),
  `Aᵀ` mechanics (~lines 270-286). Four pointer sentences in.
- `heterogeneous-agents/aiyagari-hact/README.md`: cut
  stationary-KFE prose (~lines 68-80) and `### KFE by
  transposing the same generator` (~lines 162-168). Two pointer
  sentences in.
- `heterogeneous-agents/sequence-space-jacobian-hank/README.md`:
  append pointer to Young-as-discrete-analog on the steady-
  state distribution sentence (~lines 134-135).
- `computational-methods/simulation-based-estimation/README.md`:
  cut one sentence on optimal weighting (~line 57) → pointer.
- `agent-based-models/brock-hommes-asset-pricing/README.md`:
  append one pointer sentence to SMM pseudocode step 4
  (~line 144).
- `structural-econometrics/adversarial-estimation/README.md`,
  `industrial-organization/logit-supply-side/README.md`,
  `choice/mixed-logit-simulation/README.md`: no inline cut;
  add GMM prelim to `## Preliminary readings`.
- `dsge/rbc/README.md`: cut QZ/BK pseudocode steps 2-6
  (~lines 168-188). One pointer sentence in.
- `dsge/nkdsge/README.md`: collapse BK sentences at ~line 96
  and Takeaway ~line 122 into one pointer.
- `computational-methods/perturbation-linearization/README.md`:
  no inline cut; add BK prelim to Preliminary readings.
- `computational-methods/metropolis-hastings/README.md`: major
  refactor. Cut Overview Bayes paragraphs (~lines 5-8), Bayes
  rule block (~lines 13-24), full Beta-Binomial conjugate
  subsection (~lines 26-92). Keep the sanity-check experiment
  in Solution Method + Results. Net cut: ~70 lines.
- `numerical-methods/bayesian-optimization/README.md`: repoint
  cross-reference at ~lines 39-41.
- `time-series/minnesota-svar/README.md`: cut conjugate prose
  (~lines 34-50); keep working formulas at lines 80-90.
- `computational-methods/hamiltonian-monte-carlo/README.md`,
  `structural-econometrics/bayesian-dsge-hmc/README.md`: no
  inline cut; add Bayesian-foundations prelim to Preliminary
  readings.

P1 cuts:

- `computational-methods/metropolis-hastings/README.md`,
  `## Solution Method` ~lines 218-220: cut ESS + IAT
  definitions (P1 #6 - MCMC diagnostics). One pointer
  paragraph in. Note: this is in addition to the larger MH
  refactor in P0 #5.
- `structural-econometrics/bayesian-dsge-hmc/README.md`,
  Results table ~lines 131-141: add a footnote pointing R̂ at
  P1 #6.
- `computational-methods/hamiltonian-monte-carlo/README.md`:
  no inline cut; add P1 #6 to Preliminary readings.
- `time-series/minnesota-svar/README.md`, `## Equations`
  ~lines 19-52 and ~lines 101-136, `## Solution Method`
  pseudocode steps 1-6 ~lines 173-190: cut generic VAR
  estimation + Cholesky identification (P1 #7). Keep
  Minnesota-specific prior + shock-scaling logic.
- `time-series/stock-watson/README.md`, Takeaway ~line 132:
  append a pointer to P1 #7.
- `industrial-organization/merger-simulation/README.md`,
  `## Equations` ~lines 54-98: cut the Bertrand FOC +
  ownership-matrix derivation (P1 #8). One pointer in.
- `industrial-organization/logit-supply-side/README.md`,
  `## Equations` ~lines 42-60: cut the FOC + Ω m = s
  recovery (P1 #8). One pointer in.
- `choice/mixed-logit-simulation/README.md`, `## Equations`
  ~lines 49-61 and `## Solution Method` ~lines 104-106: cut
  the integral-approximation + CRN paragraphs (P1 #9).
- `structural-econometrics/rum-choice-networks/README.md`,
  `## Solution Method` ~lines 155-157 (P1 #9) and ~lines
  164-181 (P1 #10): two pointer sentences.
- `structural-econometrics/adversarial-estimation/README.md`,
  `## Equations` ~lines 25-31 (P1 #9) and `## Solution
  Method` ~lines 169-191 (P1 #10): two pointer sentences.
- `industrial-organization/blp-random-coefficients/README.md`,
  `## Equations` ~lines 38-41: cut simulated-shares paragraph
  (P1 #9).
- `game-theory/deep-optimal-auctions/README.md`, `## Solution
  Method` ~lines 131-136: cut generic NN-via-autodiff sentence
  (P1 #10).

P2 cuts:

- `choice/sequential-search-ursu/README.md`, `## Equations`
  reservation-value block ~lines 61-75 and index-ordering rule
  ~lines 91-95: cut Weitzman derivation (P2 #11). Two pointer
  sentences in.
- `computational-methods/smolyak-sparse-grids/README.md`,
  `## Solution Method` GH step ~lines 199-205: cut inline GH
  node/weight definition (P2 #12). One pointer in.
- `numerical-methods/bayesian-optimization/README.md`,
  `## Equations` ~lines 43-72 (P2 #13) and `## Solution Method`
  ~lines 130-147 (P2 #13): cut GP posterior derivation and
  length-scale tuning. Two pointers in. (These cuts are
  separate from any P2 #12 quadrature reference.)
- `game-theory/cfr-asymmetric-auction/README.md`,
  `## Equations` ~lines 45-65: cut regret-matching primitive
  derivation (P2 #14). One pointer in.
- `dynamic-programming/shock-discretization/`,
  `structural-econometrics/bayesian-dsge-hmc/`: no inline cuts;
  add P2 #12 (quadrature) to Preliminary readings.

Total (P0 + P1 + P2): 22 dense tutorials touched, ~210 lines
cut, ~34 pointer sentences in. MH absorbs the largest single
cut (P0 #5 ~70 lines + P1 #6 ~3 lines); minnesota-svar absorbs
the second largest (P0 #5 + P1 #7 ~60 lines combined);
bayesian-optimization absorbs the third largest (P2 #13 ~50
lines, since the GP derivation is currently inline). Per-
tutorial cut totals are listed in each prelim entry above.

## Authoring order (suggested)

P0 first, then P1 interleaved so each consumer tutorial gets
both its P0 and P1 prelim before any final QC pass:

1. [DONE 2026-05-22] #1 upwind-finite-differences - foundation for #2.
2. [DONE 2026-05-22] #2 KFE - depends on #1's notation.
3. [DONE 2026-05-22] #4 Blanchard-Kahn - independent.
4. [DONE 2026-05-22] #3 GMM foundations - independent.
5. [DONE 2026-05-22] #7 reduced-form-VAR - commit `b36454f`.
6. [DONE 2026-05-22] #8 Bertrand ownership matrix - commit `b36454f`.
7. [DONE 2026-05-22] #9 simulated-likelihood - commit `b36454f`.
8. [DONE 2026-05-22] #10 NN regression - commit `f602b3d`.
9. [DONE 2026-05-22] #5 Bayesian foundations - paired with #6 in one
    PR per the alternative below.
10. [DONE 2026-05-22] #6 MCMC diagnostics - paired with #5 (the MH
    refactor: Bayes primer out, diagnostics primer out, MH stays as
    the algorithm tutorial).

P2 prelims (shipped together as one omnibus PR, commit `4bac3ed`):

11. [DONE 2026-05-22] #13 gaussian-processes.
12. [DONE 2026-05-22] #12 quadrature.
13. [DONE 2026-05-22] #11 weitzman-search-rule.
14. [DONE 2026-05-22] #14 regret-matching.

Each prelim is one PR: tutorial folder + cuts to the targeted
dense READMEs + catalog row insertion + validator pass.

## Verification

After each prelim PR:
- `python scripts/validate_catalog.py` passes.
- `python run.py` inside the prelim folder regenerates
  `figures/` and `figures/thumb.png`.
- Manual github.com render of the prelim README plus every
  dense README cut by the PR.
- `git diff --stat` matches the cut list above for that prelim.

## Out of scope (Wave 1)

- P1 and P2 prelims were originally out of scope at spec time; they were folded in on 2026-05-22 (see "Authoring order" above). This bullet is preserved for historical context.
- No catalog-row insertions in this pass; rows land with each
  prelim PR.
- No edits to dense READMEs in this pass; cuts are documented
  here and applied per prelim PR.

---

## Wave 2: Behavioral and Dynamic Learning Tutorials

**Date opened:** 2026-05-22 (merge of plan.md into spec.md). **Source:** plan.md (now retired and stubbed). **Status:** W2.1, W2.2, W2.3 already shipped May 20-21, 2026 - predating this merge. plan.md was retroactively a record of work that had already shipped, not a forward design. W2.4 (online-pricing-partial-identification) remains the only undone item; treat the rest of Wave 2 as a closed historical record.

Shipped tutorials (root README catalog rows):

- W2.1 `choice/convex-time-budget-present-bias/` - root README line 195.
- W2.2 `choice/consideration-set-estimation/` - root README line 196 (shipped framing: Manzini-Mariotti stochastic-choice + random consideration, slightly different from plan.md's "exact enumeration over consideration sets" framing).
- W2.3 `choice/probability-distortion-mixture/` - root README line 197 (shipped under a different folder name than plan.md's `probability-weighting-lottery-choice`, and a different framing: Bruhin-Fehr-Duda-Epper finite-mixture EM over latent CPT/EUT types instead of Prelec-weighting NLS. The economic object is closely related; the computational object is finite-mixture EM, not constrained nonlinear estimation. The user pivoted at authoring time. This is documented here rather than rewriting the Wave 2 spec; the shipped tutorial is the canonical artifact).

The PyBehavior repo and the `rawatpranjal/interactive-pricing-theory` repo were surveyed for tutorials that teach a computational tool not already covered cleanly by the catalog. The original output: three primary behavioral tutorials in `choice/` plus one IO addendum. The IO addendum outranks the three on raw computational novelty (a real online-learning algorithm with revealed-preference dominance elimination). It is the only Wave 2 item still to author.

### Notational policy (Wave 2)

Unlike Wave 1, Wave 2 tutorials do not serve existing dense tutorials. No cross-folder symbol pinning is required. Each tutorial pins its own canonical letters once in `Model Setup` and uses them consistently:

- **W2.1 CTB.** `β` (present bias), `δ` (long-run discount), `ρ` (CRRA curvature), `c_t`, `c_{t+k}`, `q` (gross price), `m` (budget), `k` (delay length).
- **W2.2 Consideration.** `J` (universe size), `C` (consideration set), `π_j` (consideration probability), `u_j` (utility), `p_j` (price), `q_j` (quality).
- **W2.3 Probability weighting.** `p` (objective probability), `w(p)` (decision weight), `α`, `η` (Prelec parameters), `v(x)` (value function), `EU` vs `PT`.
- **W2.4 Online pricing.** `K` (price grid), `S` (segments), `v_s` (segment valuation), `D_L(p)`, `D_U(p)` (demand bounds), `r_t` (regret at round `t`).

Cross-wave collisions to label on first use: `β` is present bias in W2.1 and the discount factor in many Wave-1 tutorials (disjoint scopes). `π` is consideration probability in W2.2 and regret-matching probability in Wave-1 prelim #14 (disjoint scopes).

### Cuts list (Wave 2)

None. Wave-2 tutorials introduce new economic objects, not new exposition of objects already in dense tutorials. No surgical cuts to existing READMEs are required. Each Wave-2 tutorial is a single-folder addition plus a root-catalog row.

### Curriculum map (Wave 2)

| # | Tutorial | Subject section | Core method | Status |
|---|---|---|---|---|
| W2.1 | `choice/convex-time-budget-present-bias/` | Choice and Demand | NLS + Tobit MLE on continuous allocations | shipped (root README line 195) |
| W2.2 | `choice/consideration-set-estimation/` | Choice and Demand | Manzini-Mariotti stochastic choice + random consideration (shipped framing) | shipped (root README line 196) |
| W2.3 | `choice/probability-distortion-mixture/` | Choice and Demand | Bruhin-Fehr-Duda-Epper finite-mixture EM (shipped framing) | shipped (root README line 197) |
| W2.4 | `industrial-organization/online-pricing-partial-identification/` | Industrial Organization | UCB with WARP-bound elimination | open |

---

### W2.1. `choice/convex-time-budget-present-bias/`

**Title.** Convex Time Budgets and the Identification of Present Bias.

**Description (catalog row).** Continuous experimental allocations between sooner and later payments identify present bias, long-run discounting, and utility curvature jointly. Nonlinear estimation on β-δ utility recovers preferences from designed intertemporal choices.

**Scope - in.** β-δ quasi-hyperbolic utility; CRRA curvature; CTB budget design (front-end delay, delay length, gross interest rate, budget variation); FOC-based and likelihood-based estimation via `scipy.optimize`; corner handling at allocation extremes; weak-design vs full-design identification comparison; brief framing of why front-end delay separates present bias from patience.
**Scope - out.** Sophisticated vs naive quasi-hyperbolic types; habit formation; non-CRRA utility families; field choice data (synthetic only).

**Economic object served.** Time preferences in experimental economics. Not currently in the catalog. Distinct from binary discrete-choice intertemporal work because the observed object is a continuous interior allocation, which is what makes the estimation problem nonlinear-structural rather than discrete.

**References (anchors; verify and expand at authoring time).**
1. Andreoni & Sprenger (2012), "Estimating Time Preferences from Convex Budgets," *AER* 102(7), 3333-3356. Founding CTB paper.
2. Laibson (1997), "Golden Eggs and Hyperbolic Discounting," *QJE* 112(2), 443-477. β-δ foundational.
3. Augenblick, Niederle & Sprenger (2015), "Working Over Time: Dynamic Inconsistency in Real Effort Tasks," *QJE* 130(3), 1067-1115. CTB with real-effort allocations.
4. Cohen, Ericson, Laibson & White (2020), "Measuring Time Preferences," *JEL* 58(2), 299-347. Comprehensive identification review.

**Equations.**
- Budget: `c_t + q c_{t+k} = m`.
- β-δ utility: `U = u(c_t) + β δ^k u(c_{t+k})`.
- CRRA utility: `u(c) = c^{1-ρ} / (1-ρ)`.
- Interior FOC: `u'(c_t) = β δ^k q u'(c_{t+k})`.
- NLS objective: `min_{β,δ,ρ} Σ (c_t^{obs} - c_t^{pred}(β, δ, ρ; design))²`.

**`run.py` sketch.** Simulate a CTB design with three front-end delays, three gross interest rates, and Gaussian mean-zero allocation noise. Generate continuous sooner/later allocations from known `(β, δ, ρ)`. Estimate by NLS and likelihood; bootstrap CIs. Compare a weak design (today-vs-future only) against a strong design (today-vs-future + future-vs-future). Figures: (a) CTB budget lines with simulated allocations; (b) sooner allocation by interest rate and delay; (c) `(β, δ)` profile-likelihood surface. Tables: `parameter-recovery.csv`, `design-comparison.csv`. Thumbnail from (a).

**Catalog row.** Insert in "Choice and Demand" of root `README.md`, near other static demand / preference-estimation rows. Suggested position: above `choice/sequential-search-ursu/`.

---

### W2.2. `choice/consideration-set-estimation/`

**Title.** Latent Consideration Sets in Product Choice.

**Description (catalog row).** Consumers may not evaluate every product. Enumerating hidden choice sets and summing the observed-choice likelihood over sets containing the chosen product separates attention from preference in observed demand.

**Scope - in.** Two-stage choice (consideration + final choice); consideration probability `π_j`; exact enumeration over consideration sets for small `J ≤ 6`; MLE of attention and preference parameters jointly; comparison with full-choice-set multinomial logit; display / prominence counterfactual.
**Scope - out.** Bayesian latent-set inference with MCMC; consideration-set sampling for large `J`; structural models of search effort (distinct from one-shot inattention).

**Economic object served.** Attention vs preference in demand. Not currently in the catalog. Teaches a genuinely new computational operation - likelihood evaluation over a latent combinatorial object - different from logit, BLP, or static demand.

**References (anchors).**
1. Hauser & Wernerfelt (1990), "An Evaluation Cost Model of Consideration Sets," *JCR* 16(4), 393-408. Founding two-stage choice model.
2. Goeree (2008), "Limited Information and Advertising in the U.S. Personal Computer Industry," *Econometrica* 76(5), 1017-1074. Latent consideration in IO.
3. Abaluck & Adams-Prassl (2021), "What Do Consumers Consider Before They Choose? Identification from Asymmetric Demand Responses," *QJE* 136(3), 1611-1663. Identification of inattention.
4. Crawford, Griffith & Iaria (2021), "A Survey of Preference Estimation with Unobserved Choice Set Heterogeneity," *Journal of Econometrics*. Pedagogical survey.

**Equations.**
- Stage 1 (consideration): `P(j ∈ C) = π_j`, products enter independently.
- Stage 2 (choice given set): `P(choose j | C) = exp(u_j) / Σ_{k ∈ C} exp(u_k)`.
- Observed-choice likelihood: `P(choose j) = Σ_{C : j ∈ C} P(choose j | C) P(C)`.
- Full enumeration size: `2^{J-1}` sets containing `j` for universe of size `J`.

**`run.py` sketch.** Universe of `J = 5` products with price, quality, and display status. Display affects consideration; price and quality affect utility. Simulate 5000 choices from known parameters. Estimate by MLE with exact enumeration. Compare against a full-choice-set logit fit to the same data. Counterfactual: raise display on one product, recompute predicted demand. Figures: (a) true vs estimated consideration probabilities; (b) observed shares, full-info predictions, latent-consideration predictions side-by-side; (c) demand shift from display counterfactual. Tables: `parameter-recovery.csv`, `model-comparison.csv`. Thumbnail from (a).

**Catalog row.** Insert in "Choice and Demand" of root `README.md`. Suggested position: near `choice/mixed-logit-simulation/`, before random-coefficient demand entries.

---

### W2.3. `choice/probability-weighting-lottery-choice/`

**Title.** Probability Weighting and Prospect-Theory Estimation.

**Description (catalog row).** Risky choices reveal distorted probability weights. Estimating a Prelec weighting function from lottery choices or certainty equivalents requires nonlinear transformation estimation under shape restrictions, and exposes the classic identification problem between utility curvature and probability weighting.

**Scope - in.** Expected utility, probability weighting only, and full prospect-theory value weighting; Prelec one-parameter and two-parameter weighting functions; certainty-equivalent inversion; shape restrictions via parameter bounds; identification comparison between strong-design (varied `p`, varied prizes) and weak-design (limited `p` variation).
**Scope - out.** Rank-dependent expected utility with general weighting families; loss aversion estimation; field-data prospect-theory applications.

**Economic object served.** Risk preferences with distorted probabilities. The catalog currently has no risk-preference-estimation tutorial. Teaches nonlinear function estimation with shape constraints - a reusable tool.

**References (anchors).**
1. Kahneman & Tversky (1979), "Prospect Theory: An Analysis of Decision under Risk," *Econometrica* 47(2), 263-291. Founding paper.
2. Prelec (1998), "The Probability Weighting Function," *Econometrica* 66(3), 497-527. Prelec one- and two-parameter forms.
3. Wu & Gonzalez (1996), "Curvature of the Probability Weighting Function," *Management Science* 42(12), 1676-1690. Curvature estimation.
4. Bruhin, Fehr-Duda & Epper (2010), "Risk and Rationality: Uncovering Heterogeneity in Probability Distortion," *Econometrica* 78(4), 1375-1412. Modern estimation.

**Equations.**
- Expected utility: `EU = p v(x_1) + (1-p) v(x_0)`.
- Prospect-theory value: `PT = w(p) v(x_1) + [1 - w(p)] v(x_0)`.
- Prelec weighting: `w(p) = exp(-η (-log p)^α)`.
- CRRA value: `v(x) = x^{1-γ} / (1-γ)`.
- NLS / MLE objective with bounds: `α > 0`, `η > 0`, `γ ≥ 0`.

**`run.py` sketch.** Simulate lottery choices and certainty equivalents across a grid of `(p, x_1, x_0)`. Compare three nested models: EU (`α = 1, η = 1`), probability weighting only (`v` linear), full PT. Solve certainty equivalents by Brent root-finding. Run identification comparison: strong design vs weak design. Figures: (a) objective probability vs Prelec decision weight at recovered `(α, η)`; (b) certainty equivalents by probability and prize; (c) fit-quality comparison across the three models; (d) strong vs weak parameter-recovery distributions. Tables: `parameter-recovery.csv`, `model-comparison.csv`. Thumbnail from (a).

**Catalog row.** Insert in "Choice and Demand" of root `README.md`. Suggested position: near the consideration-set entry; both are static demand with latent or distorted choice processes.

---

### W2.4. `industrial-organization/online-pricing-partial-identification/` (addendum)

**Status.** Optional in Wave 2. Outranks W2.1-W2.3 on raw computational novelty (real online-learning algorithm). Promote to first authoring slot if user prioritises a hardcore method over behavioral coverage. Otherwise defer to Wave 3.

**Title.** Online Pricing with Revealed-Preference Bounds: UCB on Partial Identification.

**Description (catalog row).** A seller posts prices and observes only buy / no-buy. WARP-style monotonicity converts each observation into a valuation bound, then bounds become demand bounds, then dominated prices are eliminated. The active price set shrinks as observations accumulate, faster than any bandit without economic structure.

**Scope - in.** Multi-segment demand on a discrete price grid; bandit algorithms (ε-greedy, learn-then-earn, UCB1, Thompson sampling); revealed-preference (WARP-style) valuation bound updates; profit-bound dominance elimination; UCB-PI as the hybrid; regret comparison against an oracle best fixed price.
**Scope - out.** Continuous price space; competing sellers; dynamic demand with inventory (that is `dynamic-pricing-sawtooth`, deferred). Formal regret-rate theorems (cite Auer et al. but do not re-prove).

**Economic object served.** Online learning under economic structure. Currently absent from the catalog. Distinct from existing bandit work (none) and from BLP because the learning signal is binary purchase, not allocation, and identification is partial via bounds.

**References (anchors).**
1. Auer, Cesa-Bianchi & Fischer (2002), "Finite-time Analysis of the Multiarmed Bandit Problem," *Machine Learning* 47, 235-256. UCB1 foundation.
2. Cohen, Lobel & Paes Leme (2020), "Feature-Based Dynamic Pricing," *Management Science* 66(11), 4921-4943. Dynamic pricing under structure.
3. Lattimore & Szepesvári (2020), *Bandit Algorithms*, Cambridge University Press. Textbook.
4. Manski (2003), *Partial Identification of Probability Distributions*, Springer. Partial-identification foundation.

**Equations.**
- Per-round observation: at price `p_t`, segment `s` either buys (`v_s ≥ p_t`) or does not (`v_s < p_t`).
- Bound update: `v_s^L ← max(v_s^L, p_t × 1[buy])`, `v_s^U ← min(v_s^U, p_t × 1[no buy] + ∞ × 1[buy])`.
- Demand bounds: `D_L(p) = Σ_s 1[v_s^L ≥ p]`, `D_U(p) = Σ_s 1[v_s^U ≥ p]`.
- Profit bounds: `π_L(p) = p D_L(p)`, `π_U(p) = p D_U(p)`.
- Elimination rule: drop `p` if `π_U(p) ≤ max_q π_L(q)`.

**`run.py` sketch.** `S = 4` segments with disjoint valuation intervals on a `K = 20`-price grid. Run `T = 5000` rounds for each of five algorithms (ε-greedy, learn-then-earn, UCB1, Thompson sampling, UCB-PI). Track cumulative regret against the oracle best fixed price. Show how UCB-PI's active price set shrinks while UCB1's does not. Figures: (a) cumulative regret on log-log axes; (b) active-price count over time under UCB-PI; (c) profit bounds across the grid at selected rounds; (d) segment valuation intervals tightening with observations. Tables: `final-regret.csv`, `elimination-diagnostics.csv`. Thumbnail from (a).

**Catalog row.** Insert in "Industrial Organization" of root `README.md`. Suggested position: near `industrial-organization/merger-simulation/`, or open a "Pricing and Demand" subsection if one fits.

---

### Authoring order (Wave 2)

1. [DONE 2026-05-21] **W2.1 `convex-time-budget-present-bias`** - shipped before this spec was written.
2. [DONE 2026-05-21] **W2.2 `consideration-set-estimation`** - shipped before this spec was written.
3. [DONE 2026-05-21] **W2.3 `probability-distortion-mixture`** - shipped before this spec was written, under a different folder name and framing than the plan.md original; see status note above.
4. **W2.4 `online-pricing-partial-identification`** - only undone item. Author next.

Each tutorial is one PR: folder + `run.py` + hand-maintained `README.md` + `figures/` + `tables/` + catalog row + validator pass.

### Verification (Wave 2)

Per tutorial:
- `python scripts/validate_catalog.py` passes from repo root.
- `python run.py` inside the tutorial folder regenerates `figures/`, `tables/`, and `figures/thumb.png`.
- Manual github.com render of the tutorial README (KaTeX visual check, no em-dashes).
- Root README catalog row inserted at the position named in each entry.
- Adversarial pass (per local memory rule: separate Sonnet agent runs `/bullshit` audit against this spec).

### Out of scope (Wave 2)

- PyBehavior and PyTorch are NOT runtime dependencies. Stack stays `numpy`, `scipy`, `pandas`, `matplotlib`. JAX may be used if it adds clarity.
- Runner-up methods from `rawatpranjal/interactive-pricing-theory` deferred to Wave 3:
  - `dynamic-pricing-sawtooth` (Gallego-van Ryzin finite-inventory by backward induction).
  - `network-rm` (LP-based network revenue management, bid-price controls).
  - `markdown-management` (discrete price-ladder DP with markdown-only constraint).
  - `dynamic-durable-games` (Markov-perfect dynamic oligopoly - already covered elsewhere).
- Behavioral tutorials beyond W2.1-W2.3 (e.g. habit formation, reference dependence) deferred to Wave 3.
