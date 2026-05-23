# Bullshit Detector Report: Wave 1 P2 Prelims
**Score: 22%**
**Date:** 2026-05-22
**Auditor stance:** hostile adversarial, read-only
**Claim sources:** spec.md sections 11-14, Notational alignment, Refactor ledger
**Artifact roots:** choice/weitzman-search-rule/, numerical-methods/quadrature/, numerical-methods/gaussian-processes/, game-theory/regret-matching/

---

## Pass 1: Claim extraction

From spec.md the auditable claims are:
1. Notation pinned to spec symbols in Model Setup tables.
2. Dense-tutorial cuts executed (pointer sentences replacing derivations).
3. Dense-tutorial Preliminary readings wired to the prelim.
4. Defining equations map to figures/tables in Results.
5. References match spec's reference list (year, journal, volume, pages).
6. Style: no em-dashes, no bare `$...$` math, Overview prose-only.

---

## Pass 2-4: Code/data grounding, gap categorization

---

## Section 1: `choice/weitzman-search-rule/` (#11)

**Notation pinning**

- HOLDS. Model Setup table (README.md line 37-49) has `z_j`, `c_j`, `F_j`, `b` (line 45: "Best inspected | b | Running maximum"). All four spec-pinned symbols present. `[from sequential-search-ursu/]` annotations present on `z_j`, `c_j`, `V_j`, `F_j`, `b`.
- NIT. Spec uses `F_j(v)` with lowercase `v` as the dummy variable. README.md line 43 Model Setup row shows `F_j` not `F_j(v)`. The equation block (line 20) correctly uses `dF_j(v)`. Inconsistency is NIT-level because `F_j` is a standard CDF shorthand and the equation block is correct.

**Cut execution**

- HOLDS. `choice/sequential-search-ursu/README.md` lines 62-78: the Weitzman derivation is NOT present. Replaced by pointer: "The Weitzman reservation value z_j (defined by c_j = E[max(u_ij - z_j, 0)] and derived in [choice/weitzman-search-rule/])". Spec specified cuts at ~lines 61-75 (reservation-value block) and ~lines 91-95 (index-ordering rule). Both cuts executed; the file now references the prelim inline.

**Prelim wiring**

- HOLDS. `choice/sequential-search-ursu/README.md` line 15: `[choice/weitzman-search-rule/]` is present in `## Preliminary readings`.

**Equation-to-result mapping**

- HOLDS. Three defining equations present: reservation-value equation (drives figure `reservation-values.png` in Results), index-ordering + stopping rule (drives all three result figures), policy comparison. All map to Results.

**References**

- HOLDS. Weitzman (1979) Econometrica 47(3), 641-654: matches spec exactly. README.md line 111.
- HOLDS. Kohn & Shavell (1974) JET 9(2), 93-123: matches spec. README.md line 112.
- HOLDS. Choi et al. (2018) Econometrica 86(4), 1257-1281: matches spec. README.md line 114.

**Style**

- HOLDS. No em-dashes found in README.md.
- HOLDS. No bare `$...$` math found.
- HOLDS. Overview (lines 3-12) is pure prose, no inline math.

**Bullshit score: 5%** (NIT on F_j vs F_j(v) is cosmetic; everything material holds.)

---

## Section 2: `numerical-methods/quadrature/` (#12)

**Notation pinning**

- HOLDS. Model Setup table (README.md lines 43-52) has `N` (nodes), `xi_n` (Hermite roots), `w_n` (weights), `rho` (AR(1) persistence), `sigma_varepsilon` (innovation sd). Spec-pinned symbols `{xi_n}`, `{w_n}`, `N`, `sigma_varepsilon` all present.
- HOLDS. AR(1) estimator formula (line 32) matches spec exactly: `E[f(z')|z] ≈ (1/√π) Σ w_n f(ρ z + σ_ε √2 ξ_n)`.
- NIT. Spec says node-count notation is `N` and symbol `H_n(x)` for Hermite polynomial. README uses `H_N` not `H_n` (uppercase subscript). In context the `N`-th polynomial is `H_N`, which is standard; spec's `H_n(x)` may have intended `n` as the order index. This is a convention ambiguity, not a false claim.

**Cut execution**

- HOLDS. `computational-methods/smolyak-sparse-grids/README.md` ~lines 199-213: GH nodes/weights are NOT defined inline. Instead: "where (z'_q, w_q) are the Gauss-Hermite nodes and weights derived in [numerical-methods/quadrature/]". Pointer present, derivation absent.
- HOLDS. `dynamic-programming/shock-discretization/README.md` line 13: `[numerical-methods/quadrature/]` in `## Preliminary readings`.
- HOLDS. `structural-econometrics/bayesian-dsge-hmc/README.md` line 15: `[numerical-methods/quadrature/]` in `## Preliminary readings`.
- HOLDS. `numerical-methods/bayesian-optimization/README.md` line 19: `[numerical-methods/quadrature/]` in `## Preliminary readings`. (Spec said "add to Preliminary readings if acquisition integral cites GH"; it does, and it's wired.)

**Equation-to-result mapping**

- HOLDS. GH identity -> `error-vs-nodes.png`. AR(1) change-of-variables -> `ar1-conditional-error.png`. GH nodes/weights -> `gh-nodes-weights.png`. Error-rate statement -> discussed in Results text. All four equations drive figures.

**References**

- HOLDS. Stroud & Secrest (1966) Prentice-Hall: matches spec. README line 105.
- HOLDS. Tauchen & Hussey (1991) Econometrica 59(2), 371-396: matches spec. README line 108.

**Style**

- BLOCKING. Overview (README.md lines 5-9) contains inline math in violation of CLAUDE.md Learned Rule "Overview is pure prose. No inline or display math anywhere in Overview." Verbatim violations:
  - Line 5: `$`f`$` (function symbol), `$`R^{-1/2}`$` (convergence rate), `$`f`$` (again).
  - Line 7: `$`f`$`, `$`2N - 1`$`, `$`N`$`, `$`f`$`, `$`\exp(-z^2/4)`$`.
  These are not borderline - they include symbolic math notation and LaTeX expressions.
- HOLDS. No em-dashes.
- HOLDS. No bare `$...$` math (all math correctly uses code-fence syntax, just misplaced in Overview).

**Bullshit score: 35%** (BLOCKING style violation: math in Overview. Material content otherwise correct.)

---

## Section 3: `numerical-methods/gaussian-processes/` (#13)

**Notation pinning**

- HOLDS. Model Setup table (README.md lines 54-66) has `x` (input), `sigma_n` (noise), `mu` (mean function), `k` (kernel), `ell` (length scale), `sigma_f` (output scale), `K` (training covariance), `mu_*` (posterior mean), `sigma_*^2` (posterior variance). All spec-pinned symbols present.
- NIT. Spec says "signal variance σ_f²" but Model Setup table (line 62) says "Output scale | sigma_f". The notation in equations correctly uses `sigma_f^2` in the RBF formula (line 28). The table labels the variable as `sigma_f` (the scale, not the variance), which is standard but differs from spec's label "signal variance σ_f²". The description column reads "Tuned jointly [from bayesian-optimization/]" with no indication whether it is the SD or variance. MEDIUM severity because it can confuse readers who expect σ_f vs σ_f².

**Cut execution**

- HOLDS. `numerical-methods/bayesian-optimization/README.md` Equations "### Method 1" (lines ~47-48): GP prior, kernel, and posterior conditioning derivation are NOT re-derived inline. Instead: "The GP prior, the squared-exponential kernel, and the closed-form posterior mean μ(x_*) and variance σ²(x_*) are derived in [numerical-methods/gaussian-processes/]". Pointer present, derivation absent.
- HOLDS. `numerical-methods/bayesian-optimization/README.md` Solution Method ~line 110: "the log-marginal-likelihood objective is derived in the prelim". Length-scale tuning paragraph replaced with pointer.
- HOLDS. `numerical-methods/bayesian-optimization/README.md` line 18: `[numerical-methods/gaussian-processes/]` in `## Preliminary readings`.

**Equation-to-result mapping**

- HOLDS. GP prior (`f ~ GP(mu, k)`) -> posterior figures and prose. RBF kernel -> `marginal-likelihood-curve.png` (ell sensitivity). Posterior mean/variance formulas -> `posterior-fit.png`. Log marginal likelihood -> `marginal-likelihood-curve.png`. All four equations drive figures.

**References**

- HOLDS. Rasmussen & Williams (2006) MIT Press Chapter 2 and 5: matches spec. README line 124.
- HOLDS. Kennedy & O'Hagan (2001) JRSS B 63(3), 425-464: matches spec. README line 126.
- HOLDS. Snoek et al. (2012) NIPS 25: matches spec. README line 127.

**Style**

- BLOCKING. Overview (README.md lines 3-11) contains inline math in violation of CLAUDE.md Learned Rule "Overview is pure prose. No inline or display math anywhere in Overview." Verbatim violations:
  - Line 5: `$`f`$` (function symbol) - "returns a whole function $`f`$".
  - Line 9: `$`f(x) = x \sin x`$` (full expression), `$`[0, 10]`$` (interval notation).
  These are symbolic math expressions, not natural language.
- HOLDS. No em-dashes.
- HOLDS. No bare `$...$` math.

**Bullshit score: 30%** (BLOCKING style violation: math in Overview. MEDIUM notation ambiguity on σ_f vs σ_f². Material content correct.)

---

## Section 4: `game-theory/regret-matching/` (#14)

**Notation pinning**

- HOLDS. Model Setup table (README.md lines 43-58) has `r_i^t(a)` (line 51), `R_i^T(a)` (line 52). Spec pins `r_i^t` and `R_i^T`.
- MEDIUM. Spec pins "time-average strategy π̄_i" but README uses `bar\pi^T` (with superscript T, not subscript i) at line 53: `bar\pi^T = (1/T) Σ_t (a_1^t, a_2^t)`. The spec's π̄_i is the per-player time-average; the README's `bar\pi^T` is the joint empirical distribution over A_1 × A_2. These are different objects. The per-player average appears in the pseudocode (line 71) as `bar_pi_i` but is not defined as a display symbol in the Model Setup table. This is a notation drift: spec says the pinned symbol is π̄_i, README uses a different symbol for a different (joint) object.
- HOLDS. Regret-matching probability formula (README.md line 34): `π_i^{T+1}(a) ∝ max(R_i^T(a), 0)`. Spec says `π_i^t ∝ max(R_i^t, 0)`. The README uses T+1 for the next-round strategy and T for the cumulative regret; spec uses lowercase t. The README's form is more precise (next-round strategy from current cumulative regret), not wrong - this is DILUTED relative to spec's shorthand.

**Cut execution**

- HOLDS. `game-theory/cfr-asymmetric-auction/README.md` Equations (line 47): "The Hart-Mas-Colell regret-matching primitive (instantaneous regret, cumulative regret, and the next-iteration strategy proportional to positive cumulative regret...) is derived in [game-theory/regret-matching/]". Primitive derivation absent; pointer present.
- HOLDS. `game-theory/cfr-asymmetric-auction/README.md` line 15: `[game-theory/regret-matching/]` in `## Preliminary readings`.

**Equation-to-result mapping**

- HOLDS. Instantaneous regret -> cumulative regret figure (`cumulative-regret.png`). Cumulative regret -> convergence figure (`time-average-convergence.png`). Regret-matching probability formula -> both preceding figures. Convergence theorem -> `time-average-convergence.png`. All drive Results.

**References**

- HOLDS. Hart & Mas-Colell (2000) Econometrica 68(5), 1127-1150: matches spec. README line 117.
- MEDIUM. Zinkevich et al. (2008) cited in regret-matching README (line 118). `cfr-asymmetric-auction/README.md` line 223 cites the same paper as "2007". The actual NIPS proceedings were published as a 2007 conference paper (NIPS 20 was held in December 2007; proceedings published in 2008). Both years appear in literature. The discrepancy between two tutorials in the same repo is a citation consistency problem. The spec says "2008" (aligning with the proceedings publication year); cfr uses 2007 (conference year). Not a fabrication, but a cross-tutorial inconsistency that muddies the catalog.
- HOLDS. Brown & Sandholm (2019) Science 365(6456), 885-890: matches spec. README line 120.

**Style**

- HOLDS. No em-dashes.
- HOLDS. No bare `$...$` math.
- HOLDS. Overview (lines 3-11) is pure prose, no inline math.
- NIT. Overview line 9 begins: "Regret matching is interesting because its convergence guarantee is unconditional." The CLAUDE.md Learned Rule says "One thought per sentence. One idea per paragraph." The sentence contains one thought; complies.

**Bullshit score: 15%** (MEDIUM notation drift on π̄_i vs bar\pi^T. MEDIUM citation year inconsistency across tutorials. Core content, cuts, and wiring correct.)

---

## Pass 5: Testable invariants

The following would catch the issues above in automated checks:

1. `INVARIANT[quadrature-overview-no-math]`: awk from `## Overview` to `## Equations` in `numerical-methods/quadrature/README.md`; assert no `$\`` tokens. Currently FAILS.
2. `INVARIANT[gp-overview-no-math]`: same awk in `numerical-methods/gaussian-processes/README.md`; assert no `$\`` tokens. Currently FAILS.
3. `INVARIANT[regret-pi-bar-subscript]`: Model Setup table in `game-theory/regret-matching/README.md` should have a row with symbol `\bar\pi_i` (subscript i, not superscript T) matching spec's π̄_i. Currently FAILS (table has `bar\pi^T`).
4. `INVARIANT[zinkevich-year-consistent]`: year "2008" in regret-matching prelim matches year "2007" in cfr-asymmetric-auction for the same paper. Currently FAILS (inconsistent across tutorials).

---

## Pass 6: Bullshit score

| Prelim | Score | Primary failure |
|--------|-------|-----------------|
| #11 weitzman-search-rule | 5% | NIT only (F_j vs F_j(v)) |
| #12 quadrature | 35% | BLOCKING: math in Overview |
| #13 gaussian-processes | 30% | BLOCKING: math in Overview + MEDIUM σ_f notation |
| #14 regret-matching | 15% | MEDIUM: π̄_i notation drift + citation year inconsistency |

**Aggregate score: 22%**

Score interpretation: material content, economic derivations, cut execution, and preliminary-readings wiring are mostly correct. The failures are rule compliance (math-in-Overview for two prelims) and precision failures in notation and citation consistency. No equations are fabricated. No cuts are missing. No references are invented.

Same disease (math in Overview) appears in 2 of 4 prelims (#12 and #13). Under the "same disease in >=2 siblings" rule, this bumps the project-level reading to note systemic quality-gate failure on Overview discipline. The fixes are surgical prose rewrites of two Overview sections only.
