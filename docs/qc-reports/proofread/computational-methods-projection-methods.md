# Proofread: computational-methods/projection-methods/

_Model: claude-sonnet-4-6. Generated: 2026-05-08T22:49:07Z._

## Paper / Source Verification

### Judd, K. L. (1992). Projection Methods for Solving Aggregate Growth Models. *Journal of Economic Theory*, 58(2), 410-452.

- **Located:** https://doi.org/10.1016/0022-0531(92)90061-L (confirmed via REPEc and ScienceDirect)
- **Tutorial claims:** Paper cited as the foundational source for projection methods applied to aggregate growth models.
- **Source says:** Paper presents projection methods as a general numerical approach for solving operator equations in economic models, demonstrating efficiency gains over competing methods for aggregate growth models.
- **Verdict:** OK
- **Note:** All bibliographic details confirmed: author, year (1992), journal (Journal of Economic Theory), volume (58), issue (2), page range (410-452), DOI.

### Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.

- **Located:** https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/ (confirmed via Amazon, AbeBooks, Google Books)
- **Tutorial claims:** Cited as a reference for numerical methods in economics.
- **Source says:** Book by Kenneth L. Judd published by MIT Press in 1998, 633 pages, ISBN-13 9780262100717 for the hardcover edition.
- **Verdict:** OK
- **Note:** ISBN 9780262100717 corresponds to the hardcover; confirmed correct. MIT Press URL returns 403 but the book and ISBN are well-documented across multiple authoritative sources.

### Miranda, M. J. and Fackler, P. L. (2002). *Applied Computational Economics and Finance*. MIT Press.

- **Located:** https://mitpress.mit.edu/9780262633093/applied-computational-economics-and-finance/ (confirmed via AbeBooks, Amazon)
- **Tutorial claims:** Cited as a reference for computational methods in economics and finance.
- **Source says:** Book by Mario J. Miranda and Paul L. Fackler published by MIT Press in 2002, ISBN-13 9780262633093.
- **Verdict:** OK
- **Note:** All bibliographic details confirmed: authors, year (2002), publisher (MIT Press), ISBN (9780262633093).

## Main Message Audit

> Chebyshev projection works here because the saving rule is smooth. The economic check is still the Euler equation. Small off-node residuals mean the fitted policy preserves the planner's saving tradeoff.

| Clause | Supported by | Verdict |
|--------|--------------|---------|
| Chebyshev projection works here because the saving rule is smooth | Results (exponential accuracy gain with basis terms: max policy error drops from 0.0153 at 2 terms to 4.24e-05 at 8 terms); policy-functions figure shows smooth fit | OK |
| The economic check is still the Euler equation | Equations section (Euler equation defined); Solution Method (collocation sets Euler residuals to zero); Euler errors figure | OK |
| Small off-node residuals mean the fitted policy preserves the planner's saving tradeoff | Results (max Euler error 5.57e-04 at 8 terms on dense grid between collocation nodes) | OK |

Issues:
- None.

## Notation Completeness

| Symbol | First appearance | Defined? | Notes |
|--------|------------------|----------|-------|
| $k$ | Overview, sentence 1 | Yes — "A planner starts each period with capital" | |
| $g(k)$ | Overview, sentence 3 | Yes — "the policy $g(k)$ from current capital to capital tomorrow" | |
| $V(k)$ | Equations, Bellman eq. | Implicit — standard DP value-function notation | |
| $k'$ | Equations, Bellman eq. | Yes — budget constraint $c = Ak^\alpha - k'$ implies next-period capital | |
| $c$ | Equations, Bellman eq. | Yes — $c = Ak^\alpha - k'$ | Scalar form only; see flagged issue below |
| $\beta$ | Equations, Bellman eq. | Acceptable — Model Setup table defines "Discount factor $\beta$ \| 0.95" ~30 lines later | |
| $\alpha$ | Equations, Bellman eq. | Acceptable — Model Setup table defines "Capital share $\alpha$ \| 0.36" ~30 lines later | |
| $A$ | Equations, Bellman eq. | Acceptable — Model Setup table defines "Productivity $A$ \| 1.0" ~30 lines later | |
| $c_t$, $c_{t+1}$ | Equations, Euler eq. | Implicit — time-indexed form of $c$; relation to scalar $c$ unstated | |
| $k_{t+1}$ | Equations, Euler eq. | Implicit — next-period capital in time-indexed form | |
| $\theta_j$ | Equations, projection eq. | Implicit — coefficients in Chebyshev expansion; no surrounding prose definition | |
| $\theta$ | Equations, projection eq. | Implicit — coefficient vector; same symbol used for both vector and indexed element | |
| $T_j(x)$ | Equations, projection eq. | Partial — called "Chebyshev basis functions" in prose; $j$ index implicit | |
| $x(k)$ | Equations, projection eq. | Partial — constrained to $[-1,1]$; mapping formula not given in Equations prose | |
| $n$ | Equations, projection eq. | Acceptable — Model Setup table has "Main basis terms \| 8" ~28 lines later; pseudocode names it "basis size n" | |
| $j$ | Equations, projection eq. | Implicit — summation index running from 0 to $n-1$ | |
| $R_i(\theta)$ | Equations, residual eq. | Yes — "the log Euler residual" | |
| $k_i$ | Equations, residual eq. | Yes — "selected capital nodes $k_i$" | |
| $c(k_i;\theta)$ | Equations, residual eq. | Undefined as a function | See flagged issue below |
| $g^{\ast}(k)$ | Equations, after residual | Yes — "the exact policy" | |
| $g(k;\theta)$ | Equations, projection eq. | Implicit — parameterized version of $g(k)$; extension from $g(k)$ unstated | |

Flagged issues:
- **$c(k_i;\theta)$ and $c(g(k_i;\theta);\theta)$ undefined as functions.** The Bellman defines $c$ as a scalar via $c = Ak^\alpha - k'$. The residual formula uses $c$ as a function $c(k;\theta)$, implying $c(k;\theta) \equiv Ak^\alpha - g(k;\theta)$, but this functional form is never written down in the README.

## Summary

All three cited references are confirmed accurate with no bibliographic errors. The main message is fully supported by the tutorial's equations, results table, and figures. One notation issue is present: the function $c(k;\theta)$ appears in the collocation residual formula but is never explicitly defined in that form; the README only defines $c$ as a scalar via the budget constraint in the Bellman equation, leaving the reader to infer $c(k;\theta) \equiv Ak^\alpha - g(k;\theta)$. Overall: 0 MAJOR, 0 MINOR, 0 NOT FOUND reference issues; 0 OVERREACH or UNSUPPORTED message clauses; 1 flagged notation issue (undefined function form).
