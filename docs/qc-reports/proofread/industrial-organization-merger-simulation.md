# Proofread: industrial-organization/merger-simulation/

_Model: claude-sonnet-4-6. Generated: 2026-05-09T08:00:00Z._

## Paper / Source Verification

### Werden, G. and Froeb, L. (1994). "The Effects of Mergers in Differentiated Products Industries: Logit Demand and Merger Policy." *Journal of Law, Economics, & Organization*, 10(2).

- **Located:** https://academic.oup.com/jleo/article-abstract/10/2/407/842179
- **Tutorial claims:** The tutorial calibrates a logit demand system and solves post-merger Bertrand-Nash prices, citing this paper as the methodological source for logit-based merger simulation.
- **Source says:** The paper models mergers in differentiated-product industries using logit demand and derives how ownership changes translate into equilibrium price increases.
- **Verdict:** OK
- **Note:** The journal's full official name is "The Journal of Law, Economics, and Organization"; the tutorial's shortened form with an ampersand is a standard shorthand variant.

### Farrell, J. and Shapiro, C. (2010). "Antitrust Evaluation of Horizontal Mergers: An Economic Alternative to Market Definition." *The B.E. Journal of Theoretical Economics*, 10(1).

- **Located:** https://www.degruyterbrill.com/document/doi/10.2202/1935-1704.1563/html
- **Tutorial claims:** The tutorial cites this paper as the source for UPP, GUPPI, and CMCR as first-order merger screens based on diversion ratios and observed margins.
- **Source says:** The paper introduces Upward Pricing Pressure (UPP) as a merger screen for differentiated-product markets; the term GUPPI (UPP normalized by price) was coined by practitioners after this paper, but the underlying concept originates here.
- **Verdict:** OK
- **Note:** Journal, volume, and issue are correct; the article appears in the journal's "Policy and Perspective" section rather than the main "Contributions" section, but the journal citation is unchanged.

## Main Message Audit

> Merger simulation turns a change in control into an equilibrium price calculation. The ownership matrix is easy to change. The economic content sits in substitution and pass-through. Different calibrated demand systems can give different counterfactuals in the same observed market. UPP, GUPPI, and CMCR are useful triage tools. They point to products with strong internalized diversion at observed prices. The FOC solve adds rival reactions, demand curvature, and efficiency claims. Here all three systems raise prices and lower consumer surplus.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Merger simulation turns a change in control into an equilibrium price calculation | Equations (FOC system), Solution Method (algorithm) | OK |
| The ownership matrix is easy to change | Model Setup (ownership row), Solution Method (Omega\_pre / Omega\_post construction) | OK |
| The economic content sits in substitution and pass-through | Equations (diversion ratio, GUPPI gap reflects pass-through), Results (screen vs. solved gap in figure) | OK |
| Different calibrated demand systems can give different counterfactuals in the same observed market | Results table (logit 11.15%, linear 5.13%, log-linear 10.27%) | OK |
| UPP, GUPPI, and CMCR are useful triage tools pointing to products with strong internalized diversion | Equations (UPP, GUPPI, CMCR formulas), Results (UPP figure for newly co-owned products) | OK |
| The FOC solve adds rival reactions, demand curvature, and efficiency claims | Solution Method (algorithm includes rival-price solve and efficiency grid), Results (efficiency-frontier figure) | OK |
| Here all three systems raise prices and lower consumer surplus | Results table (all Delta CS negative, all Avg Actual Price Inc. positive) | OK |

Issues:
- None. All clauses are supported by the README's own equations, solution method, or results.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $J$ | Equations prose | Yes - prose | "There are $J$ inside products" |
| $p_j$ | Equations prose | Yes - prose | "Product $j$ has price $p_j$" |
| $c_j$ | Equations prose | Yes - prose | "Product $j$ has... marginal cost $c_j$" |
| $q_j(p)$ | Equations prose | Yes - prose | "quantity or share $q_j(p)$" |
| $f(j)$ | Equations prose | Yes - prose | "owner $f(j)$" |
| $\Omega_{jk}$ | Equations, ownership matrix equation | Yes - formula | Defined with indicator function immediately below |
| $\Delta_{kj}(p)$ | Equations, "With $\Delta_{kj}(p)=\partial q_k(p)/\partial p_j$" | Yes - inline | |
| $\circ$ | Equations, vector FOC | No | Hadamard (element-wise) product; not named anywhere in README |
| $s_j^{L}(p)$ | Equations, logit formula | Yes - formula | Superscript L not explained, but context identifies it as the logit system |
| $\xi_j$ | Equations, logit formula | No | Never defined in README; mean (indirect) utility from product characteristics in logit demand |
| $\alpha$ | Equations, logit formula | Partial | Constraint $\alpha < 0$ given at first use; named "Calibrated price coefficient" in Model Setup table (~35 lines later, within 50-line window) |
| $q_j^{A}(p)$ | Equations, linear demand formula | Yes - formula | Superscript A not explained, but context identifies it as the linear system |
| $a_j$ | Equations, linear demand formula | No | Demand intercept for linear system; no prose or table definition |
| $B_{jk}$ | Equations, linear demand formula | Partial | Model Setup describes the cross-slope ratio calibration value (0.10) but never formally introduces $B_{jk}$ as a price-response matrix |
| $q_j^{E}(p)$ | Equations, log-linear formula | Yes - formula | Superscript E not explained, but context identifies it as the log-linear system |
| $a_j^E$ | Equations, log-linear formula | No | Intercept for log-linear system; no prose or table definition |
| $E_{jk}$ | Equations, log-linear formula | Partial | Model Setup describes the cross-elasticity calibration value (0.15) but never formally introduces $E_{jk}$ as an elasticity matrix |
| $D_{j\to k}$ | Equations, diversion ratio formula | Yes - formula | Defined inline by the formula for the diversion ratio |
| $UPP_j$ | Equations, UPP summation formula | Yes - formula + prose | Defined by the summation over newly co-owned products |
| $\Omega^{post}_{jk}$, $\Omega^{pre}_{jk}$ | Equations, UPP formula | Partial | Pre/post superscripts follow from $\Omega_{jk}$ by context; no separate definition given |
| $GUPPI_j$ | Equations, GUPPI formula | Yes - formula + prose | "first-order screen for upward pricing pressure" |
| $CMCR_j$ | Equations, CMCR formula | Yes - formula + prose | "marginal-cost reduction that would offset that pressure at the observed price vector" |

Flagged issues:
- $\xi_j$: undefined. The logit mean utility parameter appears inside the logit share formula without any prose definition or table row anywhere in the README.
- $a_j$ (linear demand intercept): undefined. No prose definition and no table row.
- $a_j^E$ (log-linear intercept): undefined. No prose definition and no table row.
- $B_{jk}$: partially defined. The Model Setup table describes only the cross-slope ratio calibration value; $B_{jk}$ as a price-response matrix is never formally introduced.
- $E_{jk}$: partially defined. The Model Setup table describes only the cross-elasticity calibration value; $E_{jk}$ as an elasticity matrix is never formally introduced.
- $\circ$: not named. The operator in $(\Omega\circ\Delta(p)')$ is used without being identified as the Hadamard product.

## Summary

Both references locate to authoritative publisher pages and all bibliographic fields match the tutorial exactly. The main message is fully supported by the README's equations, solution method, and results, with no overreach or unsupported claims. The only class of issues is in notation completeness: $\xi_j$ (logit mean utility), $a_j$ and $a_j^E$ (demand intercepts for linear and log-linear systems), $B_{jk}$ and $E_{jk}$ (price-response and elasticity matrices, partially described only via calibration values in the Model Setup table), and the Hadamard product operator $\circ$ (unnamed) are all undefined or incompletely introduced in the Equations section where they first appear. Counts: 0 MAJOR, 0 MINOR, 0 NOT FOUND on references; 0 OVERREACH, 0 UNSUPPORTED on main message; 6 notation issues (3 undefined symbols, 2 partially defined, 1 unnamed operator). The single most important fix is adding a one-phrase definition of $\xi_j$ where the logit formula is introduced, since it is the only demand primitive with no definition at any point in the README.
