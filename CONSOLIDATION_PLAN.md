# Catalog Consolidation Plan

Goal: trim the fat across the 88-tutorial catalog by collecting shared
economic models under one roof, so the surviving tutorials become real
multi-method comparative references rather than thin standalone pieces.

The repo's redundancy lives at the **model level**, not the method
level: the same Ramsey planner appears in three folders, the same Rust
bus problem in two, the same plain-logit-MLE in two. Each pairing forces
a reader to splice content across folders to see one model solved by
different solvers. Merging fixes that.

Catalog today: **88 tutorials**. Catalog after this plan: **80
tutorials**. Eight removals via six mergers and one outright cut. About
9% of the surface area trimmed; the survivors get noticeably more
substantive.

## Repo size benchmark

Reference tutorial sizes (run.py + README, lines):

| Tutorial | Total |
|---|---:|
| `dynamic-programming/cake-eating` | 468 |
| `industrial-organization/blp-random-coefficients` | 898 |
| `structural-econometrics/keane-wolpin-career-choice` | 1227 |

A consolidated tutorial in the **1000-1500 line range** is on the long
side but acceptable when it covers a genuine multi-method comparison.
Above 2000 lines is too long; that needs splitting.

---

## Recommended mergers

Six mergers ranked by how much real value the consolidation creates.
Each listing shows: source tutorials, proposed merged title and folder,
post-merge size estimate, what survives, what gets cut, and why this is
meatier.

### Merger 1: Continuous-time Ramsey trio → one comparative HJB tutorial

**Sources** (all in `optimal-control/`):

- `hjb-growth/` — 730 + 274 = 1004 lines (recently deepened)
- `phase-diagrams/` — 381 + 121 = 502 lines
- `ramsey-growth/` — 413 + 138 = 551 lines

**Merged tutorial**: `optimal-control/ramsey-growth/` (keep this slug;
drop `hjb-growth` and `phase-diagrams` as folders).

**Title**: "Ramsey Growth: HJB Upwinding, Saddle-Path Shooting, and
Phase Diagrams"

**Post-merge size estimate**: ~750 lines run.py + ~400 lines README =
**~1150 lines**. Below the original sum (2057) because the model
exposition is written once instead of three times.

**What survives**:

- One unified model setup (Ramsey planner, calibration, steady state).
- The HJB derivation from `hjb-growth` (the recent deepening — the
  Δt → 0 limit, upwind discretisation justification, implicit step
  diagonal-dominance argument). This is the longest section.
- Eigenanalysis + backward integration code from `phase-diagrams`.
- Forward shooting with Brent's method from `ramsey-growth`.
- Three separate `### Method N` subsections in Solution Method:
  Method 1 = HJB upwind, Method 2 = phase-plane, Method 3 = shooting.
- One results section with: value function (HJB), stable arm (phase),
  convergent paths from multiple $k_0$ (shooting). Three figures
  instead of nine.
- One comparison table at the end: number of grid points, runtime,
  recovered $k_{ss}$ accuracy, code complexity.

**What gets cut**:

- The two duplicate model-setup sections (~200 lines).
- Five of the nine current figures (each method had its own value-
  function and dynamics plots; the merged version keeps one each).
- Two duplicate "Ramsey steady state" derivations.
- Slightly inconsistent calibrations (current files use
  $\alpha = 0.30, 0.33, 0.36$ across the three; merged version picks
  one).

**Why this is meatier**: a reader gets to see the same model solved
three ways back to back. The current state forces them to open three
folders and re-read the model setup three times. The deepened HJB
exposition becomes one section of a comparative reference, not a
standalone piece.

**Catalog row**: replace three rows under **Continuous-Time Macro and
Optimal Control** with one row whose description is "Ramsey planner
solved by HJB upwind FD, by phase-plane eigenanalysis with backward
integration, and by saddle-path shooting. Side-by-side comparison of
three boundary-value-problem solvers on one model."

---

### Merger 2: Bus-engine pair → one six-estimator Rust tutorial

**Sources**:

- `industrial-organization/dynamic-discrete-choice/` — NFXP, CCP, MPEC, IRL
- `structural-econometrics/q-learning-bus-engine/` — soft Q-learning, DQN

**Merged tutorial**: `industrial-organization/dynamic-discrete-choice/`
(drop the q-learning folder).

**Title**: "Bus Engine Replacement: Six Estimators on Rust's Model"

**Post-merge size estimate**: **~1300 lines** total. Both tutorials use
the same Rust data structure and the same hazard parameter as their
target.

**What survives**:

- One Rust model setup + simulated data (currently written twice).
- Six estimator implementations as `### Method N` subsections:
  NFXP, CCP, MPEC, IRL, soft Q-learning, DQN.
- One headline figure: replacement hazard recovered by all six methods
  (currently the comparison is split across two tutorials).
- One results table comparing computational cost and parameter
  accuracy across the six estimators.

**What gets cut**:

- Duplicate Rust mileage state, transition matrix, simulated dataset
  (saves ~150 lines).
- Duplicate "intuition for the replacement decision" exposition.
- Two of the four figures in each tutorial that show the same hazard
  curve.

**Why this is meatier**: this becomes the canonical multi-estimator
showcase for a structural model. Right now if you want to compare NFXP
to soft Q-learning on the same data you have to mentally splice two
tutorials. The merged version is a one-stop reference for "all the ways
to estimate Rust".

**Catalog**: drop the `q-learning-bus-engine` row from Structural
Econometrics; expand the `dynamic-discrete-choice` row's description to
mention the six estimators.

---

### Merger 3: Logit demand pair → one demand-and-markup tutorial

**Sources**:

- `choice/logit-discrete-choice/` — 575 lines
- `industrial-organization/logit-supply-side/` — 716 lines

**Merged tutorial**: `industrial-organization/logit-supply-side/` keeps
the slug since the supply-side material is the more advanced half.

**Title**: "Logit Demand: From MLE to Berry Inversion and Markup
Recovery"

**Post-merge size estimate**: **~1100 lines**. Smaller than the sum
because shared softmax/elasticity machinery written once.

**What survives**:

- Plain logit MLE on clean simulated data (the entry point from
  `logit-discrete-choice`).
- IIA exposition with the choice-set reduction example (best teaching
  asset in `logit-discrete-choice`).
- Berry log-share-ratio inversion under endogenous prices (from
  `logit-supply-side`).
- OLS vs IV/2SLS comparison.
- Bertrand-Nash FOC and ownership-matrix markup recovery.
- One unified results table: parameter recovery under MLE / OLS / IV.

**What gets cut**:

- One copy of the softmax + log-likelihood code (~30 lines).
- One copy of the elasticity-matrix derivation.
- Two of seven figures: the duplicate "share fit" figures and one of
  the two elasticity heatmaps.

**Why this is meatier**: the merged tutorial follows the natural
pedagogical arc — start with clean MLE, motivate IV by introducing
endogeneity, end with markup recovery as the payoff. A reader currently
has to figure out on their own that `logit-discrete-choice` is the
prequel to `logit-supply-side`.

**Catalog**: keep the row in IO; drop the `choice/logit-discrete-choice`
row from Choice and Demand. Add a forward-reference link from the
choice-section overview prose.

---

### Merger 4: Cake-eating + optimal-growth → one Bellman-fundamentals tutorial

**Sources**:

- `dynamic-programming/cake-eating/` — 468 lines
- `dynamic-programming/optimal-growth/` — 571 lines

**Merged tutorial**: `dynamic-programming/optimal-growth/`.

**Title**: "Bellman Fundamentals: Cake-Eating and Optimal Growth by VFI"

**Post-merge size estimate**: **~900 lines**.

**What survives**:

- VFI algorithm explained once with the cake-eating closed form
  $c^{\ast}(W) = (1-\beta) W$ as the auditable warm-up.
- Optimal-growth as the immediate next step: same algorithm, one
  production function added.
- Two closed forms for sanity checks ($V$ for cake-eating and
  $V(k) = E + B \log k$ for optimal-growth).
- Side-by-side convergence rates: cake-eating converges in 68
  iterations, optimal-growth in 143; the comparison is itself a
  teaching point.

**What gets cut**:

- Duplicate VFI scaffolding (~100 lines).
- One pair of value-function and policy-function plots.

**Why this is meatier**: the current cake-eating tutorial is so
minimal it reads as half a tutorial. Pairing it with optimal-growth
turns "the simplest Bellman" into a launchpad for "Bellman with
production". The reader sees the algorithm work twice on increasingly
realistic problems.

**Catalog**: drop the `cake-eating` row; expand the `optimal-growth`
row's description.

---

### Merger 5: Multi-start global optimisation pair

**Sources**:

- `numerical-methods/global-search-multistart/` — 1005 lines
- `computational-methods/numerical-optimization/` — 605 lines

**Merged tutorial**: `numerical-methods/global-search-multistart/`.

**Title**: "Global Optimisation: Multi-Start, Random Search, and
Annealing on Two Non-Convex Objectives"

**Post-merge size estimate**: **~1300 lines**. On the high side for
`numerical-methods/`, but justified by the side-by-side application
contrast.

**What survives**:

- Multi-start convergence curves (probability of finding global vs
  N starts) — the unique strength of `global-search-multistart`.
- Custom Newton with finite-difference Hessian and ridge regularisation
  — the unique strength of `numerical-optimization`.
- Both applications as separate worked examples: 1D piecewise-quadratic
  monopoly profit (economic kink) and 2D Gaussian-mixture latent-regime
  likelihood (statistical multi-modality).
- One unified comparison table: each algorithm × each application.

**What gets cut**:

- Duplicate scipy boilerplate.
- One of the two basin-map figures.
- The duplicate exposition of "convergence to a local minimum is not
  global discovery".

**Why this is meatier**: the merged tutorial becomes the canonical
"how do I know my optimiser found the right answer" reference. The
current pair forces a reader to sample two folders to see both an
economic and a statistical instance of the same problem.

**Catalog**: drop the `computational-methods/numerical-optimization`
row.

---

### Merger 6: Linearised DSGE pair (RBC + NK)

**Sources**:

- `dsge/rbc/` — linearised RBC by Klein QZ
- `dsge/nkdsge/` — sticky-price New Keynesian by coefficient matching

**Merged tutorial**: `dsge/linearised-dsge/` (new slug, since the merged
content is genuinely "linearised DSGE methods" rather than one model).

**Title**: "Linearised DSGE: RBC and New Keynesian by Klein QZ"

**Post-merge size estimate**: **~1200 lines**.

**What survives**:

- One unified linearisation + Klein QZ exposition (currently written
  partially in both).
- RBC application: TFP shock IRFs, both fixed-labor and endogenous-labor
  versions.
- NK application: monetary-policy shock IRFs under sticky prices.
- One comparison table of policy responses across the two models.

**What gets cut**:

- Duplicate Klein QZ derivations.
- Duplicate Blanchard-Kahn condition discussion.
- One of the two "IRF on a single shock" exposition paragraphs.

**Why this is meatier**: a reader interested in linearised DSGE
currently sees two parallel tutorials that re-derive the same matrix
machinery. The merged version teaches the machinery once and uses it
on two economically distinct models.

**Leave alone**:

- `dsge/behavioral-nk/` — adds cognitive discounting; distinct enough
  to stay separate.
- `dsge/assetNews/` — Lucas-tree dividend-news pricing; not actually
  about DSGE solution methods, so it does not belong here.

**Catalog**: replace two rows under Linearised DSGE with one.

---

## Recommended outright cut

### Cut 1: `optimal-control/continuous-cake-eating/`

This tutorial teaches Pontryagin's maximum principle on a fixed-stock
depletion problem. The Pontryagin framework appears nowhere else in the
catalog and is not the method of choice for any of the macro/IO
tutorials downstream. The economic content (cake-eating in continuous
time) duplicates what cake-eating already does in discrete time. The
costate / shadow-price intuition is already covered by the HJB tutorial
($V'(k)$ is the shadow price; same object as the costate).

**Action**: delete the folder; add one paragraph to the merged Ramsey
trio tutorial mentioning that the same problem can be solved by
Pontryagin and pointing the reader to a textbook (e.g. Acemoglu Chapter
7) if they want the costate framework.

If Pontryagin deserves first-class treatment it should get its own
non-trivial application (an investment-with-adjustment-cost model, say)
rather than a continuous-time cake-eating exercise.

---

## No-touch list

Consolidations considered and rejected, with the reason:

| Cluster | Why keep separate |
|---|---|
| Q-learning trio (growth + bus + collusion) | Three distinct applications: toy MDP, structural recovery, repeated oligopoly. Merging would dilute each. |
| Search tutorials (McCall + DMP + Ursu + simulation-based-estimation) | McCall is acceptance, DMP is two-sided equilibrium, Ursu is sequential inspection, simulation-based-estimation is the *econometric* problem. Different objects. |
| Mixed-logit + nested-logit + BLP | Three genuinely different methods for handling IIA violations. |
| RBC variants (RBC, capital-tax, irreversible, projection, deep-learning) | Each adds a real economic friction or a different solver paradigm; all have load-bearing distinct content. |
| Heterogeneous-agents trio (EGP + EEI + Huggett) | EGP, EEI, and HJB+KFE are three substantially different solvers with their own mathematical machinery. |
| Auction tutorials (first-price + auction-valuation-recovery + deep-optimal-auctions) | Different problems: bidding, identification, mechanism design. |

---

## Phasing

The mergers are independent. Do them one at a time, lowest risk first:

1. **Cut `continuous-cake-eating`** — single delete, no merge logic.
2. **Merge cake-eating + optimal-growth** — small files, identical
   algorithm, lowest risk.
3. **Merge logit-discrete-choice + logit-supply-side** — clean
   pedagogical arc, well-bounded.
4. **Merge multi-start global-opt pair** — medium-sized, two
   applications.
5. **Merge bus-engine pair** — six estimators on one model, the
   "showcase" merger.
6. **Merge linearised DSGE pair** — slightly more delicate because the
   models live in different files today.
7. **Merge Ramsey trio** — biggest job; the recently-deepened
   `hjb-growth` material has to land cleanly inside the merged
   tutorial.

After all seven actions:

- Cut continuous-cake-eating: -1
- Five two-into-one mergers: -5
- Ramsey three-into-one merger: -2
- **Total reduction: 8 tutorials**
- **Catalog after: 80 tutorials**

---

## What this plan deliberately does NOT do

- It does not propose deleting any tutorial that has a unique solver
  family, even if the economics overlap with another tutorial. The
  repo's value comes from showing the same problem solved different
  ways; the cuts target tutorials where the **solver itself** is
  shared, not just the topic.
- It does not propose moving tutorials across topic folders. Folder
  moves cause catalog churn and broken links.
- It does not propose changing `lib/` utilities. The shared library
  has earned its place; the duplication is in the tutorial text and
  the calibration scripts, not in the library code.

---

## Verification (per merger, when each one is executed later)

1. Read both source `run.py` files in full; identify the shared
   scaffolding and the unique pieces explicitly.
2. Pick the surviving folder per the proposals above.
3. Write the merged `run.py`. Keep both calibrations available so a
   reader can cross-check against the original tutorials' published
   numbers.
4. Run `python run.py` from the surviving folder; verify all figures
   regenerate.
5. Update the root `README.md` catalog: drop the absorbed row, expand
   the surviving description.
6. Move the absorbed folder's `_legacy/`-eligible content (if any) into
   `_legacy/`; otherwise `git rm` the folder.
7. Run `python scripts/validate_catalog.py` from repo root.
8. Re-render the merged README in a Markdown viewer and check the math.

Each merger is its own commit, separately reviewable.
