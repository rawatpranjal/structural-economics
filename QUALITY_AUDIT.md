# Quality Audit: Thin Tutorials That Deserve Deepening

The catalog has 86 active tutorials (excluding `_legacy/`). This audit
ranks them by combined size (run.py + README) and flags the ones whose
content is unusually thin relative to the rest of the catalog.

The premise: instead of merging adjacent tutorials, deepen the thin
ones. The recent `hjb-growth` and `huggett-incomplete-markets` updates
are the template — both moved from "states the equation" to "derives
the equation, justifies the discretisation, lists failure modes" and
roughly doubled in length. That made each piece more valuable as a
standalone reference.

This audit lists the tutorials below the median in size and notes,
for each cluster, what kind of deepening would help.

## Catalog size distribution

Median total size (run.py + README): roughly **624 lines**.
75th percentile: roughly **800 lines**.
Recent reference deepenings:
- `optimal-control/hjb-growth`: 1007 lines (deepened May 2026)
- `heterogeneous-agents/huggett-incomplete-markets`: 1274 lines (deepened May 2026)

Both were under 700 lines before the deepening.

## Tutorials below the median (deepening candidates)

Sorted by total size ascending. The **Why thin** column is a one-line
diagnosis based on the file structure (figure count, README length,
run.py density).

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

## Above the median (in good shape, no action)

The tutorials with **600+** total lines are mostly in the right
density range. The tutorials with **800+** total lines are dense
enough that any further additions would risk overloading them
(e.g. `industrial-organization/merger-simulation` at 1631 lines).

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
`add_solution_method`, and `add_takeaway` strings in run.py to
flesh out the prose around the equations, regenerate, validate.
The `hjb-growth` deepening commit (`5c04910`) is the template.

## Tutorials with no tables

This is not a defect by itself, but worth noting since most tutorials
present numerical results in a small comparison table.

| Tutorial | Note |
|---|---|
| `choice/bayesian-learning` | Could add a posterior-mass-by-iteration table |
| `choice/revealed-preference-afriat` | Could add a GARP-edges-by-budget table |
| `dynamic-programming/consumption-savings` | Could add a steady-state-moments table |
| `optimal-control/phase-diagrams` | Could add a steady-state-and-eigenvalues table |

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

## Why deepening beats merging

A deepened tutorial:

- Stays at the same URL (no link breakage).
- Stays in the catalog as one row (no discoverability loss).
- Becomes a stronger standalone reference for the specific topic.
- Carries no risk of regressing the other tutorials it would have
  been merged with.

A merged tutorial:

- Forces a reader interested in one method to read past the others.
- Risks losing prose during the merge.
- Cuts catalog rows but adds catalog complexity (descriptions have
  to cover multiple methods).

For the goals "trim the fat" and "make pieces meatier", deepening
serves both: each piece gains weight; the catalog gains depth without
losing surface area. The retirement of `continuous-cake-eating` to
`_legacy/` is a separate kind of trim — that tutorial earned its
removal because Pontryagin appears nowhere else; deepening would not
have helped.
